using LammaNet.Tensors;

namespace LammaNet.Compute;

/// <summary>
/// Multithreaded quantized matrix multiplication: Y[t, r] = dot(W[r, :], X[t, :]).
/// Activations are quantized once into the weight's vec-dot format (Q8_0 / Q8_1 / Q8_K), then
/// integer SIMD dot products run over (row-chunk x token-chunk) tiles sized to stay cache resident.
/// </summary>
public sealed unsafe class MatMul
{
    private readonly ComputePool _pool;
    private NativeBuffer? _qbuf;
    private readonly MatMulJob _job = new();
    private readonly QuantizeJob _qjob = new();

    private readonly Gemm _gemm;

    /// <summary>
    /// Batches of at least this many tokens use the packed float GEMM path, for every weight format; 0 (default): per
    /// format, see <see cref="GemmMinTokensFor"/>. LAMMA_GEMM_MIN sets it.
    /// </summary>
    public static int GemmMinTokens { get; set; } = int.TryParse(Environment.GetEnvironmentVariable("LAMMA_GEMM_MIN"), out var g) ? g : 0;

    /// <summary>
    /// Smallest batch that uses the float GEMM for <paramref name="type"/>. The int8 dot products (Q8_K activations) of
    /// the k-quants stay faster up to ~16-20 tokens on AVX2 (i7-9700K, Q4_K 2048x8192: 0.74 vs 1.23 ms at 8 tokens,
    /// 1.64 vs 1.93 at 16, 2.45 vs 1.95 at 24); for Q8_0 / Q4_0 the float GEMM catches up at ~8. Without VNNI an int8
    /// GEMM does not beat the float one at prompt sizes (vpmaddubsw + vpmaddwd + convert per 32 weights cost what
    /// four FMAs do; measured pp256 with the int8 path for every batch: 0.55-0.6x).
    /// </summary>
    public static int GemmMinTokensFor(GgmlType type) =>
        GemmMinTokens > 0 ? GemmMinTokens : type is GgmlType.Q4_K or GgmlType.Q5_K or GgmlType.Q6_K && VecDotMulti.Supports(type) ? 20 : 8;

    public MatMul(ComputePool pool)
    {
        _pool = pool;
        _gemm = new Gemm(pool);
    }

    public ComputePool Pool => _pool;

    private byte* EnsureQuantBuffer(long bytes)
    {
        if (_qbuf == null || _qbuf.Length < bytes)
        {
            _qbuf?.Dispose();
            _qbuf = new NativeBuffer(Math.Max(bytes, 1 << 16), zero: false);
        }
        return _qbuf.B;
    }

    /// <summary>
    /// Multiplies <paramref name="w"/> (Rows x Cols) with <paramref name="nTokens"/> activation rows of
    /// length Cols at <paramref name="x"/> (row stride <paramref name="xStride"/> floats) into
    /// <paramref name="y"/> (row stride <paramref name="yStride"/> floats).
    /// Optionally only rows [rowStart, rowStart+rowCount) of the weight are computed.
    /// </summary>
    public void Multiply(WeightTensor w, float* x, int nTokens, float* y, int xStride = -1, int yStride = -1,
        int rowStart = 0, int rowCount = -1)
    {
        // the job objects and the activation buffer are reused: concurrent callers would overwrite each other's
        // pointers (or free the buffer) while the pool's workers still read them
        lock (_job) MultiplyCore(w, x, nTokens, y, xStride, yStride, rowStart, rowCount);
    }

    private void MultiplyCore(WeightTensor w, float* x, int nTokens, float* y, int xStride, int yStride, int rowStart, int rowCount)
    {
        using var _prof = Profiler.Section("matmul");
        int cols = w.Cols;
        if (xStride < 0) xStride = cols;
        if (rowCount < 0) rowCount = w.Rows - rowStart;
        if (yStride < 0) yStride = rowCount;

        if (nTokens >= GemmMinTokensFor(w.Type) && Gemm.Supported(w.Type))
        {
            _gemm.Multiply(w, x, nTokens, y, xStride, yStride, rowStart, rowCount);
            return;
        }

        var vt = QuantizeActivations.VecDotType(w.Type);
        long qRowBytes = GgmlTypeInfo.RowSize(vt, cols);
        byte* q;
        if (vt == GgmlType.F32 && xStride == cols)
        {
            q = (byte*)x;
        }
        else
        {
            q = EnsureQuantBuffer(qRowBytes * nTokens);
            _qjob.Set(vt, x, xStride, q, qRowBytes, cols);
            if (nTokens >= 4) _pool.Run(_qjob, nTokens);
            else for (int t = 0; t < nTokens; t++) _qjob.Execute(t, 0);
        }

        // tile sizes: keep a block of weight rows hot in L1/L2 while sweeping tokens
        int rowsPerTask;
        int tokensPerTask;
        int nThreads = _pool.ThreadCount;
        if (nTokens == 1)
        {
            // decode: maximize parallelism, small chunks for load balancing
            rowsPerTask = Math.Max(8, rowCount / (nThreads * 8));
            tokensPerTask = 1;
        }
        else
        {
            long rowBytes = w.RowBytes;
            rowsPerTask = (int)Math.Clamp(32 * 1024 / Math.Max(rowBytes, 1), 4, 64);
            tokensPerTask = Math.Min(nTokens, 64);
        }
        int rowTasks = (rowCount + rowsPerTask - 1) / rowsPerTask;
        int tokTasks = (nTokens + tokensPerTask - 1) / tokensPerTask;

        _job.W = w; _job.Q = q; _job.QRowBytes = qRowBytes; _job.Y = y; _job.YStride = yStride;
        _job.RowStart = rowStart; _job.RowCount = rowCount; _job.RowsPerTask = rowsPerTask;
        _job.NTokens = nTokens; _job.TokensPerTask = tokensPerTask; _job.RowTasks = rowTasks;
        _pool.Run(_job, rowTasks * tokTasks);
    }

    private sealed class QuantizeJob : ParallelJob
    {
        private GgmlType _vt; private float* _x; private int _xStride; private byte* _q; private long _qRow; private int _cols;
        public void Set(GgmlType vt, float* x, int xStride, byte* q, long qRow, int cols)
        { _vt = vt; _x = x; _xStride = xStride; _q = q; _qRow = qRow; _cols = cols; }
        public override void Execute(int t, int threadId) =>
            QuantizeActivations.Row(_vt, _x + (long)t * _xStride, _q + t * _qRow, _cols);
    }

    internal static int PrefetchAhead = int.TryParse(Environment.GetEnvironmentVariable("LAMMA_PF"), out var pf) ? pf : 2;

    private sealed class MatMulJob : ParallelJob
    {
        public WeightTensor W = null!;
        public byte* Q; public long QRowBytes; public float* Y; public int YStride;
        public int RowStart, RowCount, RowsPerTask, NTokens, TokensPerTask, RowTasks;

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        public override void Execute(int index, int threadId)
        {
            int rt = index % RowTasks, tt = index / RowTasks;
            int r0 = rt * RowsPerTask, r1 = Math.Min(RowCount, r0 + RowsPerTask);
            int t0 = tt * TokensPerTask, t1 = Math.Min(NTokens, t0 + TokensPerTask);
            var type = W.Type;
            int cols = W.Cols;
            long rb = W.RowBytes;
            byte* wbase = W.Data + (long)RowStart * rb;
            if (t1 - t0 == 1)
            {
                // decode: stream weights once; software-prefetch ahead so the stream does not stall at page boundaries
                byte* qy = Q + t0 * QRowBytes;
                float* yrow = Y + (long)t0 * YStride;
                int Ahead = PrefetchAhead;
                for (int r = r0; r < r1; r++)
                {
                    if (r + Ahead < r1 && System.Runtime.Intrinsics.X86.Sse.IsSupported)
                    {
                        byte* pf = wbase + (r + Ahead) * rb;
                        for (long o = 0; o < rb; o += 64) System.Runtime.Intrinsics.X86.Sse.Prefetch0(pf + o);
                    }
                    yrow[r] = VecDot.Dot(type, wbase + r * rb, qy, cols);
                }
                return;
            }
            if (VecDotMulti.Supports(type))
            {
                // unpack each weight block once for up to 4 tokens
                float* res = stackalloc float[4];
                for (int r = r0; r < r1; r++)
                {
                    byte* wr = wbase + r * rb;
                    for (int t = t0; t < t1; t += 4)
                    {
                        int nt = Math.Min(4, t1 - t);
                        VecDotMulti.Dot(type, wr, Q + t * QRowBytes, QRowBytes, nt, cols, res);
                        for (int k = 0; k < nt; k++) Y[(long)(t + k) * YStride + r] = res[k];
                    }
                }
                return;
            }
            for (int t = t0; t < t1; t++)
            {
                byte* qy = Q + t * QRowBytes;
                float* yrow = Y + (long)t * YStride;
                for (int r = r0; r < r1; r++)
                    yrow[r] = VecDot.Dot(type, wbase + r * rb, qy, cols);
            }
        }
    }
}
