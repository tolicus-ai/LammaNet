using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using LammaNet.Tensors;

namespace LammaNet.Compute;

/// <summary>
/// Batched (prefill) matrix multiplication Y[N, rows] = X[N, K] · W[rows, K]ᵀ for quantized weights.
/// Each task owns a tile of weight rows; the tile is dequantized once per K-block into a packed float panel
/// ([k][16 rows]) and swept over all tokens with a register-blocked 6×16 FMA micro-kernel.
/// Dequantization cost is amortized over the batch, and activations stay in full precision.
/// </summary>
internal sealed unsafe class Gemm
{
    public const int Nr = 16;      // rows per panel (two 8-wide vectors)
    public const int Mr = 6;       // tokens per micro-tile
    public const int RowTile = 64; // rows per task
    public const int KBlock = 256; // depth per packed block

    private readonly ComputePool _pool;
    private readonly GemmJob _job;

    public Gemm(ComputePool pool)
    {
        _pool = pool;
        _job = new GemmJob(pool.ThreadCount);
    }

    public static bool Supported(GgmlType t) => Dequantize.IsSupported(t) && Vector256.IsHardwareAccelerated;

    public void Multiply(WeightTensor w, float* x, int n, float* y, int xStride, int yStride, int rowStart, int rowCount)
    {
        lock (_job)   // one call at a time: the job's pointers and per-thread panels are shared by the call's tasks
        {
            _job.W = w; _job.X = x; _job.N = n; _job.Y = y; _job.XStride = xStride; _job.YStride = yStride;
            _job.RowStart = rowStart; _job.RowCount = rowCount;
            int tiles = (rowCount + RowTile - 1) / RowTile;
            // when there are few row tiles, also split tokens to keep all threads busy
            int tokenSplit = Math.Max(1, Math.Min((_pool.ThreadCount * 2 + tiles - 1) / tiles, (n + 47) / 48));
            _job.TokenSplit = tokenSplit;
            _job.Tiles = tiles;
            _pool.Run(_job, tiles * tokenSplit);
        }
    }

    private sealed class GemmJob : ParallelJob
    {
        public WeightTensor W = null!;
        public float* X; public float* Y; public int N, XStride, YStride, RowStart, RowCount, Tiles, TokenSplit;
        private readonly NativeBuffer[] _packed;
        private readonly NativeBuffer[] _rowTmp;

        public GemmJob(int threads)
        {
            _packed = new NativeBuffer[threads];
            _rowTmp = new NativeBuffer[threads];
            for (int i = 0; i < threads; i++)
            {
                _packed[i] = new NativeBuffer((long)RowTile * KBlock * 4, zero: false);
                _rowTmp[i] = new NativeBuffer((long)Nr * KBlock * 4 + 64, zero: false);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public override void Execute(int index, int threadId)
        {
            int tile = index % Tiles;
            int part = index / Tiles;
            int r0 = tile * RowTile;
            int rows = Math.Min(RowTile, RowCount - r0);
            int perPart = (N + TokenSplit - 1) / TokenSplit;
            perPart = (perPart + Mr - 1) / Mr * Mr;
            int n0 = part * perPart, n1 = Math.Min(N, n0 + perPart);
            if (n0 >= n1) return;

            var w = W;
            int K = w.Cols;
            int bs = GgmlTypeInfo.BlockSize(w.Type);
            int ts = GgmlTypeInfo.TypeSize(w.Type);
            float* packed = _packed[threadId].F;
            float* tmp = _rowTmp[threadId].F;
            int panels = (rows + Nr - 1) / Nr;

            for (int k0 = 0; k0 < K; k0 += KBlock)
            {
                int kb = Math.Min(KBlock, K - k0);
                // ---- dequantize + pack W[r0..r0+rows, k0..k0+kb] into panels [p][k][16]
                for (int p = 0; p < panels; p++)
                {
                    for (int lane = 0; lane < Nr; lane++)
                    {
                        int r = p * Nr + lane;
                        float* trow = tmp + (long)lane * kb;
                        if (r < rows)
                        {
                            byte* src = w.Row(RowStart + r0 + r) + (long)(k0 / bs) * ts;
                            DequantizeSimd.Row(w.Type, src, trow, kb);
                        }
                        else new Span<float>(trow, kb).Clear();
                    }
                    PackPanel(tmp, kb, packed + (long)p * kb * Nr);
                }
                bool accumulate = k0 > 0;
                // ---- sweep tokens
                for (int nn = n0; nn < n1; nn += Mr)
                {
                    int m = Math.Min(Mr, n1 - nn);
                    float* xa = X + (long)nn * XStride + k0;
                    for (int p = 0; p < panels; p++)
                    {
                        float* bp = packed + (long)p * kb * Nr;
                        float* yc = Y + (long)nn * YStride + r0 + p * Nr;
                        int validRows = Math.Min(Nr, rows - p * Nr);
                        if (m == Mr && validRows == Nr) Kernel6x16(xa, XStride, bp, kb, yc, YStride, accumulate);
                        else KernelEdge(xa, XStride, bp, kb, yc, YStride, accumulate, m, validRows);
                    }
                }
            }
        }

        /// <summary>Transposes a [16][kb] row block into [kb][16] (kb multiple of 8).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void PackPanel(float* src, int kb, float* dst)
        {
            if (System.Runtime.Intrinsics.X86.Avx.IsSupported && (kb & 7) == 0)
            {
                for (int half = 0; half < 2; half++)
                {
                    float* s0 = src + (long)half * 8 * kb;
                    for (int k = 0; k < kb; k += 8)
                    {
                        Transpose8x8(s0 + k, kb, dst + (long)k * Nr + half * 8, Nr);
                    }
                }
                return;
            }
            for (int lane = 0; lane < Nr; lane++)
                for (int k = 0; k < kb; k++) dst[k * Nr + lane] = src[(long)lane * kb + k];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Transpose8x8(float* s, int ls, float* d, int ld)
        {
            var r0 = Vector256.Load(s); var r1 = Vector256.Load(s + ls); var r2 = Vector256.Load(s + 2 * ls); var r3 = Vector256.Load(s + 3 * ls);
            var r4 = Vector256.Load(s + 4 * ls); var r5 = Vector256.Load(s + 5 * ls); var r6 = Vector256.Load(s + 6 * ls); var r7 = Vector256.Load(s + 7 * ls);
            var t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(r0, r1); var t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(r0, r1);
            var t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(r2, r3); var t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(r2, r3);
            var t4 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(r4, r5); var t5 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(r4, r5);
            var t6 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(r6, r7); var t7 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(r6, r7);
            var u0 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t0, t2, 0x44); var u1 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t0, t2, 0xEE);
            var u2 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t1, t3, 0x44); var u3 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t1, t3, 0xEE);
            var u4 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t4, t6, 0x44); var u5 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t4, t6, 0xEE);
            var u6 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t5, t7, 0x44); var u7 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t5, t7, 0xEE);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u0, u4, 0x20).Store(d);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u1, u5, 0x20).Store(d + ld);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u2, u6, 0x20).Store(d + 2 * ld);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u3, u7, 0x20).Store(d + 3 * ld);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u0, u4, 0x31).Store(d + 4 * ld);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u1, u5, 0x31).Store(d + 5 * ld);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u2, u6, 0x31).Store(d + 6 * ld);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(u3, u7, 0x31).Store(d + 7 * ld);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Kernel6x16(float* a, int lda, float* b, int kb, float* c, int ldc, bool acc)
        {
            Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector256<float>.Zero;
            float* a0 = a, a1 = a + lda, a2 = a + 2 * lda, a3 = a + 3 * lda, a4 = a + 4 * lda, a5 = a + 5 * lda;
            for (int k = 0; k < kb; k++)
            {
                var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 8);
                b += Nr;
                var v = Vector256.Create(a0[k]); c00 = Vector256.FusedMultiplyAdd(v, b0, c00); c01 = Vector256.FusedMultiplyAdd(v, b1, c01);
                v = Vector256.Create(a1[k]); c10 = Vector256.FusedMultiplyAdd(v, b0, c10); c11 = Vector256.FusedMultiplyAdd(v, b1, c11);
                v = Vector256.Create(a2[k]); c20 = Vector256.FusedMultiplyAdd(v, b0, c20); c21 = Vector256.FusedMultiplyAdd(v, b1, c21);
                v = Vector256.Create(a3[k]); c30 = Vector256.FusedMultiplyAdd(v, b0, c30); c31 = Vector256.FusedMultiplyAdd(v, b1, c31);
                v = Vector256.Create(a4[k]); c40 = Vector256.FusedMultiplyAdd(v, b0, c40); c41 = Vector256.FusedMultiplyAdd(v, b1, c41);
                v = Vector256.Create(a5[k]); c50 = Vector256.FusedMultiplyAdd(v, b0, c50); c51 = Vector256.FusedMultiplyAdd(v, b1, c51);
            }
            if (acc)
            {
                c00 += Vector256.Load(c); c01 += Vector256.Load(c + 8);
                c10 += Vector256.Load(c + ldc); c11 += Vector256.Load(c + ldc + 8);
                c20 += Vector256.Load(c + 2 * ldc); c21 += Vector256.Load(c + 2 * ldc + 8);
                c30 += Vector256.Load(c + 3 * ldc); c31 += Vector256.Load(c + 3 * ldc + 8);
                c40 += Vector256.Load(c + 4 * ldc); c41 += Vector256.Load(c + 4 * ldc + 8);
                c50 += Vector256.Load(c + 5 * ldc); c51 += Vector256.Load(c + 5 * ldc + 8);
            }
            c00.Store(c); c01.Store(c + 8);
            c10.Store(c + ldc); c11.Store(c + ldc + 8);
            c20.Store(c + 2 * ldc); c21.Store(c + 2 * ldc + 8);
            c30.Store(c + 3 * ldc); c31.Store(c + 3 * ldc + 8);
            c40.Store(c + 4 * ldc); c41.Store(c + 4 * ldc + 8);
            c50.Store(c + 5 * ldc); c51.Store(c + 5 * ldc + 8);
        }

        private static void KernelEdge(float* a, int lda, float* b, int kb, float* c, int ldc, bool acc, int m, int validRows)
        {
            float* res = stackalloc float[Nr];
            for (int i = 0; i < m; i++)
            {
                var c0 = Vector256<float>.Zero; var c1 = Vector256<float>.Zero;
                float* ai = a + (long)i * lda;
                float* bb = b;
                for (int k = 0; k < kb; k++, bb += Nr)
                {
                    var v = Vector256.Create(ai[k]);
                    c0 = Vector256.FusedMultiplyAdd(v, Vector256.Load(bb), c0);
                    c1 = Vector256.FusedMultiplyAdd(v, Vector256.Load(bb + 8), c1);
                }
                c0.Store(res); c1.Store(res + 8);
                float* ci = c + (long)i * ldc;
                for (int j = 0; j < validRows; j++) ci[j] = acc ? ci[j] + res[j] : res[j];
            }
        }
    }
}
