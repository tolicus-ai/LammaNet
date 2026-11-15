using ILGPU;
using ILGPU.Runtime;
using LammaNet.Multimodal;
using LammaNet.Tensors;

namespace LammaNet.Gpu;

/// <summary>
/// Shared plumbing of the GPU media encoders (<see cref="GpuVit"/>, <see cref="GpuConformer"/>): its own stream on the
/// layer offload's accelerator, the tiled GEMM, matrices uploaded in device formats, small float parameters (norm weights,
/// biases) in one buffer, clamped linear layers and RMS norms.
/// </summary>
internal abstract unsafe class GpuEncoderBase : IDisposable
{
    protected readonly Accelerator Acc;
    protected readonly AcceleratorStream Stream;
    protected readonly GemmLauncher Gemm;
    private GemmLauncher? _exactGemm;

    /// <summary>fp32 GEMM (no tensor cores): for inputs whose rounding to fp16 the following blocks would amplify.</summary>
    protected GemmLauncher ExactGemm => _exactGemm ??= new GemmLauncher(Acc, tensorCores: false);
    private readonly List<float> _params = [];
    private MemoryBuffer1D<uint, Stride1D.Dense>? _weights;
    private MemoryBuffer1D<float, Stride1D.Dense>? _paramBuf;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, NormParams> _kRms;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, float, float> _kClamp;
    protected readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>> KAdd;

    public string DeviceName { get; }

    /// <summary>The parameter buffer (valid after <see cref="UploadParams"/>).</summary>
    protected ArrayView<float> P => _paramBuf!.View;

    /// <summary>LAMMA_GPU_PROFILE=1 with LAMMA_PROFILE=1: stages are timed (synchronizing the stream around each).</summary>
    private static readonly bool s_stageProfile = LammaNet.Compute.Profiler.Enabled && Environment.GetEnvironmentVariable("LAMMA_GPU_PROFILE") == "1";

    /// <summary>Times a device stage into the profiler (no-op unless stage profiling is on).</summary>
    protected StageScope Stage(string name) => s_stageProfile ? new StageScope(Stream, name) : default;

    protected readonly struct StageScope : IDisposable
    {
        private readonly AcceleratorStream? _stream;
        private readonly LammaNet.Compute.Profiler.Scope _scope;

        public StageScope(AcceleratorStream stream, string name)
        {
            stream.Synchronize();
            _stream = stream;
            _scope = LammaNet.Compute.Profiler.Section(name);
        }

        public void Dispose()
        {
            if (_stream == null) return;
            _stream.Synchronize();
            _scope.Dispose();
        }
    }

    /// <summary>A device matrix with its source layer (bias offset in <see cref="P"/>, clamps).</summary>
    protected sealed class Lin
    {
        public required DMat M;
        public required VitLinear Src;
    }

    protected GpuEncoderBase(Accelerator acc, string deviceName)
    {
        Acc = acc;
        Acc.Bind();
        DeviceName = deviceName;
        Stream = Acc.CreateStream();
        Gemm = new GemmLauncher(Acc);
        _kRms = Acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, NormParams>(Kernels.RmsNorm);
        _kClamp = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, float, float>(Kernels.Clamp);
        KAdd = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>>(Kernels.AddInPlace);
    }

    /// <summary>Appends a float array to the parameter buffer; returns its offset (16-byte aligned) or -1 for null.</summary>
    protected long Param(float[]? data)
    {
        if (data == null) return -1;
        while (_params.Count % 4 != 0) _params.Add(0);
        long off = _params.Count;
        _params.AddRange(data);
        return off;
    }

    protected void UploadParams()
    {
        while (_params.Count % 4 != 0) _params.Add(0);
        _paramBuf = Acc.Allocate1D<float>(Math.Max(4, _params.Count));
        _paramBuf.View.CopyFromCPU(_params.ToArray());
    }

    /// <summary>Uploads the matrices of <paramref name="layers"/> (one buffer) and records their biases as parameters.</summary>
    protected Lin[] UploadLinears(IReadOnlyList<VitLinear> layers)
    {
        var types = new int[layers.Count];
        long words = 0;
        for (int i = 0; i < layers.Count; i++)
        {
            var w = layers[i].W;
            if (w.Cols % 32 != 0) throw new NotSupportedException($"{w.Name}: row length {w.Cols} is not a multiple of 32");
            types[i] = DevFormat.For(w.Type, w.Cols);
            words = (words + 3) & ~3L;
            words += (long)DevFormat.RowWords(types[i], w.Cols) * w.Rows;
        }
        var host = new uint[Math.Max(words, 4)];
        _weights = Acc.Allocate1D<uint>(host.Length);
        var lins = new Lin[layers.Count];
        long off = 0;
        fixed (uint* hp = host)
            for (int i = 0; i < layers.Count; i++)
            {
                var w = layers[i].W;
                off = (off + 3) & ~3L;
                int rw = DevFormat.RowWords(types[i], w.Cols);
                DevFormat.Pack(w, types[i], hp + off);
                var m = new DMat { Buf = _weights, Word = off, Rows = w.Rows, Cols = w.Cols, RowWords = rw, Type = types[i] };
                m.BiasOff = Param(layers[i].Bias);
                lins[i] = new Lin { M = m, Src = layers[i] };
                off += (long)rw * w.Rows;
            }
        _weights.View.CopyFromCPU(host);
        return lins;
    }

    /// <summary>y (mode 0: =, 1: +=) W·clamp(x) + b, then clamped; <paramref name="scratch"/> holds the clamped input when needed.</summary>
    protected void Linear(Lin l, ArrayView<float> x, int width, int rows, ArrayView<float> y, ArrayView<float> scratch, int mode = 0, bool exact = false)
    {
        var input = x;
        if (l.Src.ClampsInput)
        {
            _kClamp(Stream, rows * width, x, scratch, l.Src.InMin, l.Src.InMax);
            input = scratch;
        }
        if (mode != 0 && l.Src.ClampsOutput) throw new InvalidOperationException("an output clamp cannot be fused with accumulation");
        (exact ? ExactGemm : Gemm).Run(Stream, P, input, 0, width, rows, y, mode, 1f, l.M, 0, l.M.Rows);
        if (l.Src.ClampsOutput) _kClamp(Stream, rows * l.M.Rows, y, y, l.Src.OutMin, l.Src.OutMax);
    }

    /// <summary>RMS norm of rows of <paramref name="d"/>: dst = v (mode 0) or dst += scale·v (mode 1), v = norm(src)·w.</summary>
    protected void RmsNorm(ArrayView<float> src, ArrayView<float> dst, long w, int rows, int d, float eps, int mode = 0, float scale = 1f) =>
        _kRms(Stream, new KernelConfig(rows, Kernels.GroupSize), P, src, dst,
            new NormParams { XOff = 0, YOff = 0, WOff = w, Rows = rows, D = d, XStride = d, YStride = d, Mode = mode, Eps = eps, Scale = scale, InScale = 1f });

    public virtual void Dispose()
    {
        Acc.Bind();
        Gemm.Dispose();
        _exactGemm?.Dispose();
        _paramBuf?.Dispose();
        _weights?.Dispose();
        Stream.Dispose();
    }
}
