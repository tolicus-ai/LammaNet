using ILGPU;
using ILGPU.Runtime;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

/// <summary>Parameters of <see cref="Kernels.Gemm{T}"/>: Y[t, r] (op)= Σ_k W[r, k]·X[t, k] over all tokens.</summary>
public struct GemmParams
{
    public long WWord, XOff, YOff, BiasOff;
    public int Rows, RowWords, Cols, XStride, NTok, YStride;
    /// <summary>0 = store, 1 = accumulate (y += Scale·r).</summary>
    public int Mode;
    public float Scale;
}

/// <summary>Sixteen accumulators (a 4×4 register tile).</summary>
public struct Acc16 { public float C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22, C23, C30, C31, C32, C33; }

internal static partial class Kernels
{
    public const int GemmBM = 64, GemmBN = 64, GemmBK = 32, GemmThreads = 256;
    private const int GemmLd = GemmBM + 4; // padded shared row (floats); BM == BN

    /// <summary>
    /// Tiled SIMT GEMM for prompt processing: each 256-thread block computes 64 weight rows × 64 tokens; per 32-wide K
    /// step the block dequantizes its weight tile and stages the activations in shared memory (k-major), then every
    /// thread accumulates a 4×4 register tile. Grid: ceil(Rows/64) × ceil(NTok/64) blocks.
    /// </summary>
    public static void Gemm<T>(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, GemmParams p)
        where T : struct, IWeightFormat
    {
        var ws = SharedMemory.Allocate<float>(GemmBK * GemmLd);
        var xs = SharedMemory.Allocate<float>(GemmBK * GemmLd);
        var ws4 = ws.Cast<Float4>();
        var xs4 = xs.Cast<Float4>();
        var x4 = x.Cast<Float4>();
        var fmt = default(T);
        int tid = Group.IdxX;
        int tilesM = (p.Rows + GemmBM - 1) / GemmBM;
        int row0 = (Grid.IdxX % tilesM) * GemmBM, tok0 = (Grid.IdxX / tilesM) * GemmBN;
        int tr = tid >> 4, tc = tid & 15;
        // loader roles: weights 64 rows × 4 groups of 8 columns; activations 64 tokens × 8 float4
        int lr = tid >> 2, lk = (tid & 3) * 8;
        bool rowOk = row0 + lr < p.Rows;
        long wRow = p.WWord + (long)(row0 + lr) * p.RowWords;
        var c = new Acc16();
        for (int k0 = 0; k0 < p.Cols; k0 += GemmBK)
        {
            Float8 v = rowOk ? fmt.Dequant8(w, wRow, p.Cols, k0 + lk) : default;
            ws[(lk + 0) * GemmLd + lr] = v.W0; ws[(lk + 1) * GemmLd + lr] = v.W1;
            ws[(lk + 2) * GemmLd + lr] = v.W2; ws[(lk + 3) * GemmLd + lr] = v.W3;
            ws[(lk + 4) * GemmLd + lr] = v.W4; ws[(lk + 5) * GemmLd + lr] = v.W5;
            ws[(lk + 6) * GemmLd + lr] = v.W6; ws[(lk + 7) * GemmLd + lr] = v.W7;
            for (int e = 0; e < 2; e++)
            {
                int idx = tid * 2 + e, t = idx >> 3, q = idx & 7;
                Float4 xv = default;
                if (tok0 + t < p.NTok) xv = x4[(p.XOff + (long)(tok0 + t) * p.XStride + k0) / 4 + q];
                xs[(4 * q + 0) * GemmLd + t] = xv.X; xs[(4 * q + 1) * GemmLd + t] = xv.Y;
                xs[(4 * q + 2) * GemmLd + t] = xv.Z; xs[(4 * q + 3) * GemmLd + t] = xv.W;
            }
            Group.Barrier();
            for (int kk = 0; kk < GemmBK; kk++)
            {
                Float4 a = ws4[kk * (GemmLd / 4) + tr], b = xs4[kk * (GemmLd / 4) + tc];
                c.C00 += a.X * b.X; c.C01 += a.X * b.Y; c.C02 += a.X * b.Z; c.C03 += a.X * b.W;
                c.C10 += a.Y * b.X; c.C11 += a.Y * b.Y; c.C12 += a.Y * b.Z; c.C13 += a.Y * b.W;
                c.C20 += a.Z * b.X; c.C21 += a.Z * b.Y; c.C22 += a.Z * b.Z; c.C23 += a.Z * b.W;
                c.C30 += a.W * b.X; c.C31 += a.W * b.Y; c.C32 += a.W * b.Z; c.C33 += a.W * b.W;
            }
            Group.Barrier();
        }
        int r0 = row0 + tr * 4, t0 = tok0 + tc * 4;
        GemmStore(f, y, p, r0, t0, c.C00, c.C01, c.C02, c.C03);
        GemmStore(f, y, p, r0 + 1, t0, c.C10, c.C11, c.C12, c.C13);
        GemmStore(f, y, p, r0 + 2, t0, c.C20, c.C21, c.C22, c.C23);
        GemmStore(f, y, p, r0 + 3, t0, c.C30, c.C31, c.C32, c.C33);
    }

    /// <summary>Writes one weight row's results for 4 consecutive tokens.</summary>
    private static void GemmStore(ArrayView<float> f, ArrayView<float> y, GemmParams p, int r, int t0, float v0, float v1, float v2, float v3)
    {
        if (r >= p.Rows) return;
        float bias = p.BiasOff >= 0 ? f[p.BiasOff + r] : 0f;
        GemmStore1(y, p, r, t0, (v0 + bias) * p.Scale);
        GemmStore1(y, p, r, t0 + 1, (v1 + bias) * p.Scale);
        GemmStore1(y, p, r, t0 + 2, (v2 + bias) * p.Scale);
        GemmStore1(y, p, r, t0 + 3, (v3 + bias) * p.Scale);
    }

    private static void GemmStore1(ArrayView<float> y, GemmParams p, int r, int t, float v)
    {
        if (t >= p.NTok) return;
        long o = p.YOff + (long)t * p.YStride + r;
        if (p.Mode == 1) y[o] = y[o] + v;
        else y[o] = v;
    }

    /// <summary>u[t, j] = act(g[t, j]) * u[t, j] (gated FFN after two GEMMs); act 0 = SiLU, 1 = GELU.</summary>
    public static void ActMul(Index1D i, ArrayView<float> g, ArrayView<float> u, int act) =>
        u[i] = (act == 1 ? GeluTanh(g[i]) : Silu(g[i])) * u[i];
}

/// <summary>
/// Launches the prompt GEMM for any weight format (one kernel instantiation per format): the pipelined tensor-core
/// <see cref="Kernels.TcGemm2{T}"/> (activations converted to fp16 once per call), the earlier
/// <see cref="Kernels.TcGemm{T}"/>, or the SIMT <see cref="Kernels.Gemm{T}"/> without tensor cores.
/// </summary>
internal sealed class GemmLauncher : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Dictionary<int, Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, GemmParams>> _kernels = new();
    private readonly Dictionary<int, Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<Half>, ArrayView<float>, GemmParams>> _kernels2 = new();
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<Half>, long, int, int>? _toHalf;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<Half>, long, int, int>? _toHalfQ8;
    private MemoryBuffer1D<Half, Stride1D.Dense>? _xh;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, long, int, int, ArrayView<uint>, ArrayView<Float2>>? _quantQ8;
    private readonly Dictionary<int, Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<uint>, ArrayView<Float2>, ArrayView<float>, GemmParams>> _kernelsQ8 = new();
    private MemoryBuffer1D<uint, Stride1D.Dense>? _gq;
    private MemoryBuffer1D<Float2, Stride1D.Dense>? _gd;
    internal long ScratchBytes => (_xh?.LengthInBytes ?? 0) + (_gq?.LengthInBytes ?? 0) + (_gd?.LengthInBytes ?? 0);

    /// <summary>
    /// int8 tensor-core GEMMs (<see cref="Kernels.TcGemmQ4K"/>, <see cref="Kernels.TcGemmQ40"/>, <see cref="Kernels.TcGemmQ80"/>)
    /// for int8-activation calls on those weight formats (LAMMA_GPU_MMQ=0 disables).
    /// </summary>
    private static readonly bool s_mmq = Environment.GetEnvironmentVariable("LAMMA_GPU_MMQ") != "0";

    /// <summary>True when a tensor-core kernel (sm_80+) is used.</summary>
    public bool TensorCores { get; }

    /// <summary>True when the pipelined tensor-core kernel is used (LAMMA_GPU_GEMM=1 selects the earlier one).</summary>
    public bool Pipelined { get; }

    /// <param name="acc">Accelerator.</param>
    /// <param name="tensorCores">Force (true/false) or auto-detect (null: sm_80+, unless LAMMA_GPU_TC=0) the tensor-core kernel.</param>
    /// <param name="pipelined">Force (true/false) or auto-select (null: unless LAMMA_GPU_GEMM=1) the pipelined tensor-core kernel.</param>
    public GemmLauncher(Accelerator acc, bool? tensorCores = null, bool? pipelined = null)
    {
        _acc = acc;
        bool capable = acc is ILGPU.Runtime.Cuda.CudaAccelerator cu && cu.Architecture is { } arch && arch.Major >= 8;
        TensorCores = capable && (tensorCores ?? Environment.GetEnvironmentVariable("LAMMA_GPU_TC") != "0");
        Pipelined = TensorCores && (pipelined ?? Environment.GetEnvironmentVariable("LAMMA_GPU_GEMM") != "1");
    }

    private Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<Half>, ArrayView<float>, GemmParams> Kernel2(int type)
    {
        if (_kernels2.TryGetValue(type, out var k)) return k;
        var mi = typeof(Kernels).GetMethod(nameof(Kernels.TcGemm2))!.MakeGenericMethod(MatVecLauncher.FormatType(type));
        var del = (Action<ArrayView<uint>, ArrayView<float>, ArrayView<Half>, ArrayView<float>, GemmParams>)Delegate.CreateDelegate(
            typeof(Action<ArrayView<uint>, ArrayView<float>, ArrayView<Half>, ArrayView<float>, GemmParams>), mi);
        k = _acc.LoadKernel(del);
        _kernels2[type] = k;
        return k;
    }

    private Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, GemmParams> Kernel(int type)
    {
        if (_kernels.TryGetValue(type, out var k)) return k;
        var mi = typeof(Kernels).GetMethod(TensorCores ? nameof(Kernels.TcGemm) : nameof(Kernels.Gemm))!.MakeGenericMethod(MatVecLauncher.FormatType(type));
        var del = (Action<ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, GemmParams>)Delegate.CreateDelegate(
            typeof(Action<ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, GemmParams>), mi);
        k = _acc.LoadKernel(del);
        _kernels[type] = k;
        return k;
    }

    /// <summary>
    /// y[t, :] (op)= W·x[t, :] + bias, scaled, for all n tokens. <paramref name="q8"/> rounds the activations to int8 blocks
    /// first, as the matvec kernels do for the same matrices (pipelined GEMM only).
    /// </summary>
    public void Run(AcceleratorStream stream, ArrayView<float> f, ArrayView<float> x, long xOff, int xStride, int n, ArrayView<float> y, int mode, float scale,
        DMat m, long yOff, int yStride, bool q8 = false)
    {
        var p = new GemmParams
        {
            WWord = m.Word, XOff = xOff, YOff = yOff, BiasOff = m.BiasOff, Rows = m.Rows, RowWords = m.RowWords, Cols = m.Cols,
            XStride = xStride, NTok = n, YStride = yStride, Mode = mode, Scale = scale,
        };
        if (q8 && Pipelined && s_mmq && HasInt8Gemm(m.Type) && m.Cols % 256 == 0)
        {
            // int8 activations (the rounding of ToHalfRowsQ8, kept as integers) and int8 mma on the raw nibbles
            long need = (long)n * m.Cols;
            EnsureQ8(need);
            _quantQ8 ??= _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, long, int, int, ArrayView<uint>, ArrayView<Float2>>(Kernels.QuantizeQ8);
            _quantQ8(stream, (int)(need / 4), x, xOff, xStride, m.Cols, _gq!.View, _gd!.View);
            if (!_kernelsQ8.TryGetValue(m.Type, out var kq))
                _kernelsQ8[m.Type] = kq = _acc.LoadKernel<ArrayView<uint>, ArrayView<float>, ArrayView<uint>, ArrayView<Float2>, ArrayView<float>, GemmParams>(
                    m.Type switch { DevFormat.Q4_K => Kernels.TcGemmQ4K, DevFormat.Q4_0 => Kernels.TcGemmQ40, _ => Kernels.TcGemmQ80 });
            int tilesQ = (m.Rows + Kernels.TcQ8BM - 1) / Kernels.TcQ8BM * ((n + Kernels.TcQ8BN - 1) / Kernels.TcQ8BN);
            kq(stream, new KernelConfig(tilesQ, Kernels.TcQ8Threads), m.Buf.View, f, _gq.View, _gd!.View, y, p);
            return;
        }
        if (Pipelined)
        {
            long need = (long)n * m.Cols;
            if (_xh == null || _xh.Length < need)
            {
                _xh?.Dispose();
                _xh = _acc.Allocate1D<Half>(Math.Max(need, 1L << 20));
            }
            if (q8)
            {
                _toHalfQ8 ??= _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<Half>, long, int, int>(Kernels.ToHalfRowsQ8);
                _toHalfQ8(stream, (int)(need / 4), x, _xh.View, xOff, xStride, m.Cols);
            }
            else
            {
                _toHalf ??= _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<Half>, long, int, int>(Kernels.ToHalfRows);
                _toHalf(stream, (int)(need / 4), x, _xh.View, xOff, xStride, m.Cols);
            }
            int tiles2 = (m.Rows + Kernels.Tc2BM - 1) / Kernels.Tc2BM * ((n + Kernels.Tc2BN - 1) / Kernels.Tc2BN);
            Kernel2(m.Type)(stream, new KernelConfig(tiles2, Kernels.Tc2Threads), m.Buf.View, f, _xh.View, y, p);
            return;
        }
        int tiles = (m.Rows + Kernels.GemmBM - 1) / Kernels.GemmBM * ((n + Kernels.GemmBN - 1) / Kernels.GemmBN);
        Kernel(m.Type)(stream, new KernelConfig(tiles, TensorCores ? Kernels.TcThreads : Kernels.GemmThreads), m.Buf.View, f, x, y, p);
    }

    /// <summary>
    /// Allocates the fp16 activation scratch for up to <paramref name="rows"/> x <paramref name="cols"/> now, so it is never
    /// reallocated later (a captured CUDA graph would keep the old pointer);
    /// <paramref name="int8"/>: also the int8 activations of the int8 tensor-core GEMM.
    /// </summary>
    public void Reserve(long rows, int cols, bool int8 = false)
    {
        if (!Pipelined) return;
        if (int8 && s_mmq) EnsureQ8(rows * cols);
        if (_xh != null && _xh.Length >= rows * cols) return;
        _xh?.Dispose();
        _xh = _acc.Allocate1D<Half>(Math.Max(rows * cols, 1L << 20));
    }

    /// <summary>Weight formats with an int8 tensor-core GEMM.</summary>
    public static bool HasInt8Gemm(int type) => type is DevFormat.Q4_K or DevFormat.Q4_0 or DevFormat.Q8_0;

    private void EnsureQ8(long values)
    {
        if (_gq != null && _gq.Length * 4 >= values) return;
        _gq?.Dispose(); _gd?.Dispose();
        _gq = _acc.Allocate1D<uint>(Math.Max(values / 4, 1L << 18));
        _gd = _acc.Allocate1D<Float2>(Math.Max(values / 32, 1L << 15));
    }

    public void Dispose()
    {
        _xh?.Dispose();
        _gq?.Dispose();
        _gd?.Dispose();
    }
}
