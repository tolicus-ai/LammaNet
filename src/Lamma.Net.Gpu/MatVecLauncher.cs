using ILGPU;
using ILGPU.Runtime;

namespace LammaNet.Gpu;

/// <summary>A matrix resident in a device weight buffer.</summary>
internal sealed class DMat
{
    public required MemoryBuffer1D<uint, Stride1D.Dense> Buf;
    public long Word;
    public int Rows, Cols, RowWords, Type;
    public long BiasOff = -1;
}

/// <summary>
/// Launches <see cref="Kernels.MatVec{TA, TB, TC}"/>, instantiating (and caching) the kernel for each combination of
/// weight formats, and splitting the tokens into JIT-specialized tiles of up to <see cref="Kernels.MatVecTokens"/>.
/// </summary>
internal sealed class MatVecLauncher
{
    private readonly Accelerator _acc;
    private readonly Dictionary<(int, int, int), Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, MatVecParams, SpecializedValue<int>>> _kernels = new();
    private readonly Dictionary<(int, int, int), Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<uint>, ArrayView<Float2>, ArrayView<float>, MatVecParams, SpecializedValue<int>>> _q8Kernels = new();
    private Action<AcceleratorStream, Index1D, ArrayView<float>, long, int, int, ArrayView<uint>, ArrayView<Float2>>? _kQuant;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _xq;
    private MemoryBuffer1D<Float2, Stride1D.Dense>? _xd;
    private long _q8Rows, _q8Cols;
    internal long ScratchBytes => (_xq?.LengthInBytes ?? 0) + (_xd?.LengthInBytes ?? 0);

    /// <summary>
    /// Which calls use int8 activations with dp4a (only after <see cref="Reserve"/>). Measured on an RTX 3070: 4 tokens
    /// cost 1.0-1.7x one token instead of 2.7-3.7x, and one token is as fast or faster. Inputs that are not normalized
    /// (attention output, FFN hidden) carry outliers that make 8-bit blocks coarse: with them Gemma 4 moves 1.6x further
    /// from llama.cpp's reference probabilities, without them (<see cref="Int8Activations.Normalized"/>) it stays at float
    /// accuracy. Every batch size up to the reservation uses the same kernel, and the kernel is batch-invariant (a token's
    /// row is bit-identical whatever batch it is in); attention chunking and the prompt GEMM (16+ tokens) are not.
    /// </summary>
    public Int8Activations Int8 { get; init; } = Int8Activations.Normalized;

    /// <summary>True when some calls use int8 activations (buffers reserved and <see cref="Int8"/> not off).</summary>
    public bool Int8Enabled => _xq != null && Int8 != Int8Activations.Off;

    public MatVecLauncher(Accelerator acc) => _acc = acc;

    /// <summary>
    /// Whether a call on these matrices uses int8 activations (any batch the reservation holds): the prompt GEMM rounds
    /// its activations the same way for them (<see cref="GemmLauncher.Run"/>). <paramref name="normedInput"/>: the input
    /// is a normalized activation (see <see cref="Int8"/>).
    /// </summary>
    public bool UsesQ8(int ta, int tb, int tc, int cols, long xOff, int xStride, bool normedInput) =>
        _xq != null && (Int8 == Int8Activations.All || (Int8 == Int8Activations.Normalized && normedInput)) && HasQ8(ta) && HasQ8(tb) && HasQ8(tc)
        && cols % 256 == 0 && cols <= _q8Cols && xStride % 4 == 0 && xOff % 4 == 0;

    /// <summary>Formats with an int8-activation (dp4a) kernel.</summary>
    public static bool HasQ8(int type) => type is DevFormat.Q4_0 or DevFormat.Q8_0 or DevFormat.Q4_K or DevFormat.Q6_K;

    /// <summary>
    /// Allocates the int8 activation buffers for batches of up to <paramref name="rows"/> tokens (including the padding
    /// of a partial last tile) of up to <paramref name="cols"/> inputs now: a captured CUDA graph keeps their address.
    /// Without a reservation (media encoders) every call uses the float kernel.
    /// </summary>
    public void Reserve(long rows, int cols)
    {
        if (_acc is not ILGPU.Runtime.Cuda.CudaAccelerator || (_xq != null && _q8Rows >= rows && _q8Cols >= cols)) return;
        rows = Math.Max(Math.Max(rows, _q8Rows), Kernels.MatVecTokens);
        cols = (int)Math.Max(cols, _q8Cols);
        _q8Rows = rows; _q8Cols = cols;
        _xq?.Dispose(); _xd?.Dispose();
        _xq = _acc.Allocate1D<uint>(Math.Max(rows * cols / 4, 4));
        _xd = _acc.Allocate1D<Float2>(Math.Max(rows * cols / 32, 4));
    }

    private Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<uint>, ArrayView<Float2>, ArrayView<float>, MatVecParams, SpecializedValue<int>> KernelQ8(int a, int b, int c)
    {
        if (_q8Kernels.TryGetValue((a, b, c), out var k)) return k;
        var mi = typeof(Kernels).GetMethod(nameof(Kernels.MatVecQ8))!.MakeGenericMethod(FormatType(a), FormatType(b), FormatType(c));
        var del = (Action<ArrayView<uint>, ArrayView<float>, ArrayView<uint>, ArrayView<Float2>, ArrayView<float>, MatVecParams, SpecializedValue<int>>)Delegate.CreateDelegate(
            typeof(Action<ArrayView<uint>, ArrayView<float>, ArrayView<uint>, ArrayView<Float2>, ArrayView<float>, MatVecParams, SpecializedValue<int>>), mi);
        k = _acc.LoadKernel(del);
        _q8Kernels[(a, b, c)] = k;
        return k;
    }

    public static Type FormatType(int type) => type switch
    {
        DevFormat.Q4_K => typeof(FmtQ4K),
        DevFormat.Q5_K => typeof(FmtQ5K),
        DevFormat.Q6_K => typeof(FmtQ6K),
        DevFormat.Q8_0 => typeof(FmtQ80),
        DevFormat.Q4_0 => typeof(FmtQ40),
        DevFormat.F16 => typeof(FmtF16),
        DevFormat.BF16 => typeof(FmtBF16),
        DevFormat.F32 => typeof(FmtF32),
        _ => throw new NotSupportedException($"device format {type}"),
    };

    private Action<AcceleratorStream, KernelConfig, ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, MatVecParams, SpecializedValue<int>> Kernel(int a, int b, int c)
    {
        if (_kernels.TryGetValue((a, b, c), out var k)) return k;
        var mi = typeof(Kernels).GetMethod(nameof(Kernels.MatVec))!.MakeGenericMethod(FormatType(a), FormatType(b), FormatType(c));
        var del = (Action<ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, MatVecParams, SpecializedValue<int>>)Delegate.CreateDelegate(
            typeof(Action<ArrayView<uint>, ArrayView<float>, ArrayView<float>, ArrayView<float>, MatVecParams, SpecializedValue<int>>), mi);
        k = _acc.LoadKernel(del);
        _kernels[(a, b, c)] = k;
        return k;
    }

    private static MatDesc Desc(DMat m, long yOff, int yStride) => new()
    {
        WWord = m.Word, Rows = m.Rows, RowWords = m.RowWords, YOff = yOff, YStride = yStride, BiasOff = m.BiasOff,
    };

    /// <summary>Token tile of the matvec kernel for a batch of n tokens: 1, 2, 4 or 8.</summary>
    public static int TileSize(int n) => n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 4 : Kernels.MatVecTokens;

    /// <summary>
    /// y[t, :] (op)= W·x[t, :] for up to three matrices of the same buffer and input (mode 2: gated pair a/b).
    /// <paramref name="normedInput"/>: x is a normalized activation (int8 activations by default, see <see cref="UsesQ8"/>).
    /// </summary>
    public void Run(AcceleratorStream stream, ArrayView<float> f, ArrayView<float> x, long xOff, int xStride, int n, ArrayView<float> y, int mode, float scale,
        (DMat M, long YOff, int YStride) a, (DMat M, long YOff, int YStride)? b, (DMat M, long YOff, int YStride)? c, int act, float softCap,
        long mulOff = 0, int mulStride = 0, bool normedInput = false)
    {
        int ta = a.M.Type, tb = b?.M.Type ?? ta, tc = c?.M.Type ?? ta;
        int cols = a.M.Cols;
        int rows = a.M.Rows;
        if (b is { } bb && mode != 2) rows += bb.M.Rows;
        if (c is { } cc) rows += cc.M.Rows;
        bool q8 = UsesQ8(ta, tb, tc, cols, xOff, xStride, normedInput) && n + Kernels.MatVecTokens <= _q8Rows;   // the engine's batches always fit
        // tile sizes are limited to 1, 2, 4 and 8 tokens (few JIT specializations, all compiled at warm-up); the last
        // tile may be partial: its padding rows are read (scratch buffers have MatVecTokens rows of slack) but not stored
        int nt = TileSize(n);
        {
            int tiles = (n + nt - 1) / nt;
            long t0 = 0;
            var p = new MatVecParams
            {
                M0 = Desc(a.M, a.YOff + t0 * a.YStride, a.YStride), NMat = 1, Cols = a.M.Cols, XStride = xStride, XOff = xOff + t0 * xStride,
                Tiles = tiles, NTok = n, Mode = mode, Act = act, Scale = scale, SoftCap = softCap, TotalRows = rows,
                MulOff = mulOff + t0 * mulStride, MulStride = mulStride,
            };
            if (b is { } b2) { p.M1 = Desc(b2.M, b2.YOff + t0 * b2.YStride, b2.YStride); p.NMat = 2; }
            if (c is { } c2) { p.M2 = Desc(c2.M, c2.YOff + t0 * c2.YStride, c2.YStride); p.NMat = 3; }
            long warps = (long)rows * tiles;
            var grid = new KernelConfig((int)((warps + Kernels.WarpsPerGroup - 1) / Kernels.WarpsPerGroup), Kernels.GroupSize);
            if (q8)
            {
                // int8 activations once per call (all matrices share the input), then dp4a dot products
                _kQuant ??= _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, long, int, int, ArrayView<uint>, ArrayView<Float2>>(Kernels.QuantizeQ8);
                _kQuant(stream, n * (cols >> 2), x, xOff, xStride, cols, _xq!.View, _xd!.View);
                KernelQ8(ta, tb, tc)(stream, grid, a.M.Buf.View, f, _xq.View, _xd!.View, y, p, SpecializedValue.New(nt));
            }
            else Kernel(ta, tb, tc)(stream, grid, a.M.Buf.View, f, x, y, p, SpecializedValue.New(nt));
        }
    }
}
