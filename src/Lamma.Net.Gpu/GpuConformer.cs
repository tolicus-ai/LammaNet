using ILGPU;
using ILGPU.Runtime;
using LammaNet.Multimodal;

namespace LammaNet.Gpu;

/// <summary>
/// The Gemma 4 audio Conformer on the GPU (<see cref="ConformerSpec"/>): every block (macaron half-step FFNs, chunked
/// relative-position attention, the GLU / depthwise-conv module) runs on the device with clamped linear layers through
/// the tiled GEMM; only the projected frames go in and the encoded frames come out.
/// </summary>
internal sealed unsafe class GpuConformer : GpuEncoderBase, IConformerRunner
{
    private readonly ConformerSpec _spec;
    private readonly int _d, _heads, _hd, _ff, _kz;
    private readonly Block[] _blocks;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, long, int, float, float> _kScaleQK;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ChunkAttnParams> _kAttn;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int> _kGlu;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, long, int, int> _kDw;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>> _kSilu;

    private int _capRows = -1;
    private MemoryBuffer1D<float, Stride1D.Dense>? _x, _cur, _q, _k, _v, _att, _ffb, _t2, _scr;

    private sealed class Block
    {
        public Lin Up = null!, Down = null!, Up1 = null!, Down1 = null!, Q = null!, K = null!, V = null!, O = null!, Pw1 = null!, Pw2 = null!;
        public long FfnNorm, FfnPost, FfnNorm1, FfnPost1, AttnPre, AttnPost, PerDim, ConvPre, ConvMid, Ln2, ConvDw, RelK;
    }

    public GpuConformer(Accelerator acc, string deviceName, ConformerSpec spec) : base(acc, deviceName)
    {
        _spec = spec;
        _d = spec.Dim; _heads = spec.Heads; _hd = _d / _heads;
        _ff = spec.Blocks[0].Up.W.Rows;
        _kz = spec.Blocks[0].ConvDw.Length / _d;
        if (_d % 32 != 0 || _ff % 4 != 0 || spec.ChunkSize + spec.LeftContext > Kernels.Warp || spec.Blocks[0].Pw1.W.Rows != 2 * _d)
            throw new NotSupportedException($"Conformer shape d={_d} ff={_ff} window={spec.ChunkSize + spec.LeftContext} not supported on the GPU");

        var layers = spec.Blocks.SelectMany(b => new[] { b.Up, b.Down, b.Up1, b.Down1, b.Q, b.K, b.V, b.O, b.Pw1, b.Pw2 }).ToList();
        var lins = UploadLinears(layers);
        _blocks = new Block[spec.Blocks.Count];
        for (int i = 0, m = 0; i < _blocks.Length; i++)
        {
            var b = spec.Blocks[i];
            _blocks[i] = new Block
            {
                Up = lins[m++], Down = lins[m++], Up1 = lins[m++], Down1 = lins[m++], Q = lins[m++], K = lins[m++], V = lins[m++], O = lins[m++], Pw1 = lins[m++], Pw2 = lins[m++],
                FfnNorm = Param(b.FfnNorm), FfnPost = Param(b.FfnPost), FfnNorm1 = Param(b.FfnNorm1), FfnPost1 = Param(b.FfnPost1),
                AttnPre = Param(b.AttnPre), AttnPost = Param(b.AttnPost), PerDim = Param(b.PerDimScale),
                ConvPre = Param(b.ConvPreNorm), ConvMid = Param(b.ConvMidNorm), Ln2 = Param(b.Ln2), ConvDw = Param(b.ConvDw), RelK = Param(b.RelK),
            };
        }
        UploadParams();

        _kScaleQK = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, long, int, float, float>(Kernels.ScaleQK);
        _kAttn = Acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ChunkAttnParams>(Kernels.ChunkRelAttn);
        _kGlu = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(Kernels.Glu);
        _kDw = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, long, int, int>(Kernels.DwConvCausal);
        _kSilu = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>>(Kernels.SiluInPlace);

        // compile every kernel now (JIT) so the first clip is not slow
        var warm = new float[32 * _d];
        fixed (float* w = warm) Run(w, 32);
    }

    private void Ensure(int rows)
    {
        if (rows <= _capRows) return;
        FreeActivations();
        long n = (long)rows * _d;
        _x = Acc.Allocate1D<float>(n); _cur = Acc.Allocate1D<float>(n);
        _q = Acc.Allocate1D<float>(n); _k = Acc.Allocate1D<float>(n); _v = Acc.Allocate1D<float>(n); _att = Acc.Allocate1D<float>(n);
        _ffb = Acc.Allocate1D<float>((long)rows * _ff);
        _t2 = Acc.Allocate1D<float>(2 * n);
        _scr = Acc.Allocate1D<float>((long)rows * Math.Max(_ff, 2 * _d));
        _capRows = rows;
    }

    public void Run(float* x, int rows)
    {
        Acc.Bind();
        Ensure(rows);
        long n = (long)rows * _d;
        _x!.View.SubView(0, n).CopyFromCPU(Stream, ref *x, n);
        float eps = _spec.Eps;
        var X = _x.View;
        var scr = _scr!.View;
        foreach (var B in _blocks)
        {
            HalfFfn(B.FfnNorm, B.Up, B.Down, B.FfnPost, rows);

            // chunked local self-attention with relative positions
            RmsNorm(X, _cur!.View, B.AttnPre, rows, _d, eps);
            Linear(B.Q, _cur.View, _d, rows, _q!.View, scr);
            Linear(B.K, _cur.View, _d, rows, _k!.View, scr);
            Linear(B.V, _cur.View, _d, rows, _v!.View, scr);
            _kScaleQK(Stream, (int)n, _q.View, _k.View, P, B.PerDim, _hd, _spec.QScale, _spec.KScale);
            var ap = new ChunkAttnParams
            {
                RelOff = B.RelK, N = rows, Heads = _heads, Hd = _hd, D = _d, C = _spec.ChunkSize, P = _spec.LeftContext,
                R = _spec.LeftContext + 1, SoftCap = _spec.SoftCap,
            };
            long warps = (long)rows * _heads;
            _kAttn(Stream, new KernelConfig((int)((warps + Kernels.WarpsPerGroup - 1) / Kernels.WarpsPerGroup), Kernels.GroupSize),
                _q.View, _k.View, _v.View, P, _att!.View, ap);
            Linear(B.O, _att.View, _d, rows, _cur.View, scr);
            RmsNorm(_cur.View, X, B.AttnPost, rows, _d, eps, mode: 1);

            // convolution module: norm → pw1 → GLU → causal depthwise conv → norm → silu → pw2, added to x
            RmsNorm(X, _cur.View, B.ConvPre, rows, _d, eps);
            Linear(B.Pw1, _cur.View, _d, rows, _t2!.View, scr);
            _kGlu(Stream, (int)n, _t2.View, _cur.View, _d);
            _kDw(Stream, (int)n, _cur.View, _t2.View, P, B.ConvDw, _d, _kz);
            RmsNorm(_t2.View, _t2.View, B.ConvMid, rows, _d, eps);
            _kSilu(Stream, (int)n, _t2.View);
            Linear(B.Pw2, _t2.View, _d, rows, _cur.View, scr);
            KAdd(Stream, (int)n, _cur.View, X);

            HalfFfn(B.FfnNorm1, B.Up1, B.Down1, B.FfnPost1, rows);
            RmsNorm(X, X, B.Ln2, rows, _d, eps);
        }
        _x.View.SubView(0, n).CopyToCPU(Stream, ref *x, n);
        Stream.Synchronize();
    }

    /// <summary>x += ½ · post_norm(Down · silu(Up · norm(x))).</summary>
    private void HalfFfn(long norm, Lin up, Lin down, long post, int rows)
    {
        RmsNorm(_x!.View, _cur!.View, norm, rows, _d, _spec.Eps);
        Linear(up, _cur.View, _d, rows, _ffb!.View, _scr!.View);
        _kSilu(Stream, rows * _ff, _ffb.View);
        Linear(down, _ffb.View, _ff, rows, _cur.View, _scr.View);
        RmsNorm(_cur.View, _x.View, post, rows, _d, _spec.Eps, mode: 1, scale: 0.5f);
    }

    private void FreeActivations()
    {
        foreach (var b in new MemoryBuffer?[] { _x, _cur, _q, _k, _v, _att, _ffb, _t2, _scr }) b?.Dispose();
    }

    public override void Dispose()
    {
        Acc.Bind();
        FreeActivations();
        base.Dispose();
    }
}
