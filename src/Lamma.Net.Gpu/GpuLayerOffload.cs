using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using LammaNet.Compute;
using LammaNet.Diagnostics;
using LammaNet.Inference;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Tensors;
using System.Numerics;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

/// <summary>Which accelerator to use.</summary>
public enum GpuDevice
{
    /// <summary>The CUDA device with the most memory; fails if none is present.</summary>
    Auto,
    Cuda,
    /// <summary>ILGPU CPU emulation (for testing kernels without a GPU; slow).</summary>
    CpuEmulation,
}

/// <summary>Which matrix-vector products of the language model use 8-bit integer activations (dp4a).</summary>
public enum Int8Activations
{
    /// <summary>Float activations everywhere (most stable numerics across batch shapes and prompt / decode paths).</summary>
    Off,
    /// <summary>
    /// Projections of RMS-norm outputs (attention q/k/v, FFN gate/up, LM head): most of the speed of int8 at float
    /// accuracy. Inputs that are not normalized (attention output, FFN hidden) keep float activations.
    /// </summary>
    Normalized,
    /// <summary>Every supported projection, as llama.cpp does: fastest, with measurably noisier probabilities on Gemma 4.</summary>
    All,
    /// <summary>Use normalized int8 projections by default and float activations for Gemma 3 with Q4_0 projections.</summary>
    Auto,
}

/// <summary>Options for GPU offloading.</summary>
public sealed class GpuOffloadOptions
{
    /// <summary>
    /// Number of leading layers to run on the GPU. -1 (default): as many supported layers as fit in the device's free
    /// memory (keeping <see cref="ReserveBytes"/> free); the LM head is offloaded too when every layer fits.
    /// </summary>
    public int Layers { get; init; } = -1;
    /// <summary>Device memory left free when fitting layers automatically (other applications, driver, fragmentation).</summary>
    public long ReserveBytes { get; init; } = 384L << 20;
    public GpuDevice Device { get; init; } = GpuDevice.Auto;
    /// <summary>Zero-based index among devices of the selected kind.</summary>
    public int DeviceIndex { get; init; }
    /// <summary>Also run the final norm + LM head on the device when every layer is offloaded.</summary>
    public bool OffloadHead { get; init; } = true;
    /// <summary>Replay small batches (decode steps) through captured CUDA graphs.</summary>
    public bool UseGraphs { get; init; } = true;
    /// <summary>
    /// 8-bit integer activations for quantized weights (Q4_0, Q8_0, Q4_K, Q6_K): decode +2-18% (Gemma 4 the most) and batches of 2-8 tokens
    /// (speculative verification, concurrent requests) 2-6x faster per matrix. LAMMA_GPU_Q8 (0, normalized, all) overrides.
    /// </summary>
    public Int8Activations Int8 { get; init; } = Int8Activations.Auto;
    /// <summary>
    /// Compile the specialized kernels and capture the decode graph while loading (a scratch prompt and decode step),
    /// so the first request does not pay for it.
    /// </summary>
    public bool Warmup { get; init; } = true;
}

/// <summary>
/// Runs the first N decoder layers of a model on a GPU through ILGPU, with their FP16 or Q8_0 KV cache in device memory.
/// Attention layers of the Llama / Qwen2-3.5 / Granite / Gemma 2-4 families and the recurrent layers of hybrid models
/// (Gated DeltaNet of Qwen3.5, Mamba2 of Granite 4-H) are supported; recurrent states of live sequences stay in device
/// slots (<see cref="IRecurrentStateStore"/>), snapshots for prefix reuse are host copies.
/// <para>
/// All work goes to one stream. Batches of up to <see cref="Kernels.MatVecTokens"/> tokens (decode, speculative
/// verification, small continuous batches) are captured once per shape into a CUDA graph and replayed: every
/// per-step value (positions, page tables, rotary tables, inputs) lives in device buffers that are updated before
/// the replay, and the launch configuration depends only on the batch shape.
/// </para>
/// </summary>
public sealed unsafe partial class GpuLayerOffload : ILayerOffload, IRecurrentStateStore, LammaNet.Multimodal.IVisionAccelerator
{
    private readonly Context _ctx;
    private readonly Accelerator _acc;
    private readonly Microsoft.Extensions.Logging.ILogger _log;
    private readonly AcceleratorStream _stream;
    private readonly ModelConfig _c;
    private readonly int _layers;
    private readonly int _maxBatch;
    private readonly int _pageSize;
    private readonly int _pageCount;
    private readonly bool _kvQ8;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<int>, int, int, int, float>? _scaleQueryTemperature;
    private readonly MemoryBuffer1D<Half, Stride1D.Dense>? _keyRows, _valueRows;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, int, int>? _packKv;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<Half>, ArrayView<int>, ArrayView<float>, int, int, int, int, int>? _rotateQ8;
    private readonly DLayer[] _dl;
    private readonly List<MemoryBuffer> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _f; // all small float parameters (norm weights, biases)
    private readonly DMat? _output;
    private readonly long _outputNormOff = -1;
    private readonly DMat? _pleProj;          // Gemma 3n/4 per-layer input projection (computed on the device)
    private readonly long _pleNormOff = -1;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>? _pleProjOut;
    private readonly bool _pleTokenPart; // Gemma 3n/4 add per-layer token embeddings; EmbeddingGemma 2 has only the projection
    private readonly int _maxHeads, _maxHdV;

    // scratch
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _x, _xb, _qkv, _att, _ff, _ff2, _mix, _logits, _rope;
    private readonly long _kBase, _vBase; // K and V regions inside _qkv (multi-matrix launches share one output buffer)
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _infoFull, _infoSwa, _slots, _rows;
    private MemoryBuffer1D<float, Stride1D.Dense>? _part, _ple;
    private MemoryBuffer1D<int, Stride1D.Dense> _pages;

    // CUDA graphs per batch shape (invalidated when a buffer they reference is reallocated)
    private readonly bool _graphsEnabled;
    private readonly Dictionary<(int N, int Rows, bool Ple, int Chunks, int ChunksSwa), CudaGraph> _graphs = new();

    // kernels
    private readonly MatVecLauncher _matVec;
    private readonly GemmLauncher _gemm;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int> _kActMul;
    /// <summary>Batches of at least this many tokens use the tiled GEMM instead of matvec tiles.</summary>
    private static readonly int s_gemmMinTokens = int.TryParse(Environment.GetEnvironmentVariable("LAMMA_GPU_GEMM_MIN"), out var gm) ? gm : 16;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, NormParams> _kRms;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, QkvParams> _kQkv;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams> _kAttn;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ResidualNormParams> _kResNorm;
    private static readonly bool s_fuseNorms = Environment.GetEnvironmentVariable("LAMMA_GPU_FUSE_NORMS") != "0";
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams, SpecializedValue<int>, SpecializedValue<int>> _kAttnGqa;
    private static readonly bool s_gqaAttn = Environment.GetEnvironmentVariable("LAMMA_GPU_ATTN_GQA") != "0";
    /// <summary>Decode attention uses <see cref="Kernels.AttnDecodeGqa"/> for every attention layer.</summary>
    private readonly bool _gqaDecode;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, AttnParams> _kCombine;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>> _kPrefill;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>>? _kTcPrefill;
    private MemoryBuffer1D<int, Stride1D.Dense>? _tiles;
    private int _tileCount; // query tiles of the current (eager) batch, 0 when the decode kernel is used
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float> _kScale;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int> _kGather;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int> _kGeluMul;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<Half>, long, long> _kPageCopy;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<Half>, ArrayView<int>, ArrayView<float>, int, int, int, int, int> _kRotateK;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ConvParams> _kConv;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, DeltaParams> _kDelta;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ScanParams> _kScan;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, GatedNormParams> _kGatedNorm;

    // recurrent layers: scratch (A: in-projections, B: beta/alpha, C: conv output, D: mixer output) and state arenas
    private readonly bool _hasRec;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>? _recA, _recB, _recC, _recD;
    private readonly long _recA2; // offset of the second in-projection output (DeltaNet z) inside _recA
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _seqInfo, _tokList;
    private int _slotCap;
    private readonly Stack<int> _freeSlots = new();

    // optional on-device profiling (LAMMA_GPU_PROFILE=1, disables graphs): stream markers between launches
    private static readonly bool s_profile = Environment.GetEnvironmentVariable("LAMMA_GPU_PROFILE") == "1";
    private readonly List<(string Label, ProfilingMarker Marker)> _marks = new();
    private readonly Dictionary<string, (double Ms, long Count)> _profile = new();
    private readonly Dictionary<string, double> _cpuProfile = new();
    private long _lastMark;

    public int LayerCount => _layers;
    public string DeviceName { get; }

    /// <summary>
    /// Runs a vision encoder on this offload's device (same CUDA context, own stream and buffers); null when it does
    /// not fit or the shape is unsupported (the encoder then stays on the CPU).
    /// </summary>
    public LammaNet.Multimodal.IVitRunner? CreateVit(LammaNet.Multimodal.VitSpec spec)
    {
        if (_acc.AcceleratorType != AcceleratorType.Cuda) return null;
        try
        {
            long need = spec.Blocks.Sum(b => new[] { b.Q, b.K, b.V, b.O, b.Up, b.Down, b.Gate }.Sum(l => l == null ? 0 : MatrixBytes(l.W)));
            if (!FitsEncoder("vision encoder", need + (256L << 20))) return null; // weights + ~4-tile activations
            return new GpuVit(_acc, DeviceName, spec);
        }
        catch (Exception e) when (e is NotSupportedException or AcceleratorException or OutOfMemoryException)
        {
            Log.EncoderCreateFailed(_log, e, "vision encoder", DeviceName);
            return null;
        }
    }

    public LammaNet.Multimodal.IVitPipelineRunner? CreateVitPipeline(LammaNet.Multimodal.VitPipelineSpec spec)
    {
        if (_acc.AcceleratorType != AcceleratorType.Cuda) return null;
        try
        {
            long need = spec.Vit.Blocks.Sum(b => new[] { b.Q, b.K, b.V, b.O, b.Up, b.Down, b.Gate }.Sum(l => l == null ? 0 : MatrixBytes(l.W)))
                + MatrixBytes(spec.PatchEmbedding.W) + spec.Projector.Sum(l => MatrixBytes(l.W));
            if (!FitsEncoder("vision pipeline", need + (512L << 20))) return null; // weights + activations of an 8-tile pass
            return new GpuVit(_acc, DeviceName, spec.Vit, spec);
        }
        catch (Exception e) when (e is NotSupportedException or AcceleratorException or OutOfMemoryException)
        {
            Log.EncoderCreateFailed(_log, e, "vision pipeline", DeviceName);
            return null;
        }
    }

    public LammaNet.Multimodal.IConformerRunner? CreateConformer(LammaNet.Multimodal.ConformerSpec spec)
    {
        if (_acc.AcceleratorType != AcceleratorType.Cuda) return null;
        try
        {
            long need = spec.Blocks.Sum(b => new[] { b.Up, b.Down, b.Up1, b.Down1, b.Q, b.K, b.V, b.O, b.Pw1, b.Pw2 }.Sum(l => MatrixBytes(l.W)));
            if (!FitsEncoder("audio encoder", need + (256L << 20))) return null; // weights + activations of a long clip
            return new GpuConformer(_acc, DeviceName, spec);
        }
        catch (Exception e) when (e is NotSupportedException or AcceleratorException or OutOfMemoryException)
        {
            Log.EncoderCreateFailed(_log, e, "audio encoder", DeviceName);
            return null;
        }
    }

    private bool FitsEncoder(string encoder, long need)
    {
        long free = FreeDeviceMemory();
        if (free < 0 || need <= free) return true;
        Log.EncoderNoDeviceMemory(_log, encoder, need >> 20, free >> 20);
        return false;
    }
    /// <summary>Whether projections use 8-bit integer activations (<see cref="GpuOffloadOptions.Int8"/>, LAMMA_GPU_Q8).</summary>
    public bool UsesInt8Activations => _matVec.Int8Enabled;

    /// <summary>How the layer split was chosen (estimated sizes and free memory).</summary>
    public string Plan { get; }
    public bool ComputesHead => _output != null;
    public bool ComputesPerLayerInputs => _pleProj != null;
    public IRecurrentStateStore? RecurrentStore => _hasRec ? this : null;

    private sealed class DLayer
    {
        public required MemoryBuffer1D<uint, Stride1D.Dense> W;
        public DMat? Wq, Wk, Wv, Wqkv, Wo, Gate, Up, Down, PleGate, PleProj;
        public long AttnNorm = -1, AttnPost = -1, FfnNorm = -1, FfnPost = -1, QNorm = -1, KNorm = -1, PlePost = -1;
        public MemoryBuffer1D<Half, Stride1D.Dense>? KCache, VCache;
        public float OutScale = 1f;
        // recurrent layers
        public LayerKind Kind;
        public DMat? SsmQkv, SsmZ, SsmBeta, SsmAlpha, SsmIn, SsmOut;
        public long ConvW = -1, ConvB = -1, DtBias = -1, A = -1, D = -1, SsmNorm = -1;
        public MemoryBuffer1D<float, Stride1D.Dense>? RecConv, RecSsm;
        public int ConvSize, SsmSize;
    }

    /// <summary>Lists the accelerators ILGPU can see.</summary>
    public static IReadOnlyList<string> ListDevices()
    {
        using var ctx = Context.Create(b => b.Default());
        return ctx.Devices.Select(d => $"{d.AcceleratorType}: {d.Name} ({d.MemorySize >> 20} MB)").ToList();
    }

    /// <summary>
    /// True if the kernels can run on <paramref name="d"/>. They reduce with 32-lane warp shuffles, which ILGPU's
    /// OpenCL backend only provides for Intel sub-groups, so OpenCL devices are not supported.
    /// </summary>
    public static bool IsSupported(Device d) => d is CudaDevice or CPUDevice;

    /// <summary>
    /// Adjusts a split so that no host layer reads KV owned by the device (KV sharing, e.g. Gemma 4):
    /// valid splits either offload every layer or stop before the earliest shared KV source.
    /// </summary>
    public static int ValidSplit(ModelConfig c, int k)
    {
        while (k > 0 && k < c.NLayer)
        {
            int bad = int.MaxValue;
            for (int il = k; il < c.NLayer; il++)
                if (c.KvSourceLayer[il] < k) bad = Math.Min(bad, c.KvSourceLayer[il]);
            if (bad == int.MaxValue) break;
            k = bad;
        }
        return k;
    }

    /// <summary>Number of leading layers of <paramref name="config"/> that this backend can execute.</summary>
    public static int SupportedLayerPrefix(ModelConfig config)
    {
        if (config.Traits.Phi2) return 0; // affine layer norm and parallel, biased MLP run on the host
        int n = 0;
        while (n < config.NLayer && LayerSupported(config, n)) n++;
        return n;
    }

    private static bool LayerSupported(ModelConfig c, int il) => c.Kinds[il] switch
    {
        LayerKind.Attention => Math.Max(c.HeadDimK[il], c.HeadDimV[il]) <= Kernels.MaxHeadDim && c.HeadDimK[il] % 32 == 0 && c.HeadDimV[il] % 64 == 0,
        LayerKind.GatedDeltaNet => c.SsmStateSize % 32 == 0 && c.SsmStateSize <= Kernels.MaxStateRow && c.SsmConvKernel is >= 2 and <= Kernels.MaxConvKernel,
        LayerKind.Mamba2 => c.SsmStateSize % 32 == 0 && c.SsmStateSize <= Kernels.MaxStateRow && c.SsmConvKernel is >= 2 and <= Kernels.MaxConvKernel,
        LayerKind.ShortConv => c.SsmConvKernel is >= 2 and <= Kernels.MaxConvKernel,
        _ => false,
    };

    /// <summary>Creates an offload; pass it to <see cref="EngineOptions.Offload"/> via <see cref="Factory"/>.</summary>
    public GpuLayerOffload(Model model, OffloadGeometry geometry, GpuOffloadOptions? options = null)
    {
        lock (s_createLock)   // engines created at once would otherwise plan their layers with the same free memory
        {
            try
            {
                options ??= new GpuOffloadOptions();
                _log = LammaLogging.Create<GpuLayerOffload>(geometry.LoggerFactory);
                _c = model.Config;
                _maxBatch = geometry.MaxBatch;
                _pageSize = geometry.PageSize;
                _pageCount = geometry.PageCount;
                _kvQ8 = geometry.KvCacheType == KvCacheType.Q8_0;
                _ctx = Context.Create(b => { b.Default().Math(MathMode.Fast32BitOnly).EnableAlgorithms(); if (s_profile) b.Profiling(); });
                Device dev = options.Device switch
                {
                    GpuDevice.Cuda => _ctx.GetCudaDevice(options.DeviceIndex),
                    GpuDevice.CpuEmulation => new CPUDevice(Kernels.Warp, Kernels.WarpsPerGroup, Math.Max(1, Environment.ProcessorCount / 2)),
                    _ => _ctx.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).OrderByDescending(d => d.MemorySize).FirstOrDefault()
                         ?? throw new InvalidOperationException("No CUDA GPU found (use GpuDevice.CpuEmulation to test kernels on the CPU)"),
                };
                _acc = dev.CreateAccelerator(_ctx);
                _stream = _acc.CreateStream();
                DeviceName = $"{_acc.AcceleratorType}: {_acc.Name}";
                (_layers, bool fitsHead, Plan) = ChooseLayers(model, geometry, options, FreeDeviceMemory());
                _graphsEnabled = options.UseGraphs && !s_profile && _acc is CudaAccelerator
                                 && Environment.GetEnvironmentVariable("LAMMA_GPU_GRAPHS") != "0";

                _matVec = new MatVecLauncher(_acc)
                {
                    Int8 = Environment.GetEnvironmentVariable("LAMMA_GPU_Q8") switch
                    {
                        "0" or "off" => Int8Activations.Off, "1" or "normalized" => Int8Activations.Normalized, "all" => Int8Activations.All,
                        _ when options.Int8 != Int8Activations.Auto => options.Int8,
                        // Gemma 3 Q4_0 accumulates appreciable logit error from int8 projection inputs.
                        _ when _c.Architecture == "gemma3" && model.Layers[0].Wq?.Type == GgmlType.Q4_0 => Int8Activations.Off,
                        _ => Int8Activations.Normalized,
                    },
                };
                _gemm = new GemmLauncher(_acc);
                if (_gemm.TensorCores)
                    _kTcPrefill = _acc.LoadKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>>(_kvQ8 ? Kernels.TcAttnPrefillQ8 : Kernels.TcAttnPrefill);
                _kActMul = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(Kernels.ActMul);
                _kRms = _acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, NormParams>(Kernels.RmsNorm);
                _kResNorm = _acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ResidualNormParams>(Kernels.RmsNormResidualNorm);
                _kQkv = _acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, QkvParams>(Kernels.QkvPost);
                _kAttn = _acc.LoadKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams>(_kvQ8 ? Kernels.AttnKvQ8 : Kernels.AttnDecode);
                _kAttnGqa = _acc.LoadKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams, SpecializedValue<int>, SpecializedValue<int>>(_kvQ8 ? Kernels.AttnDecodeGqaQ8 : Kernels.AttnDecodeGqa);
                _gqaDecode = s_gqaAttn && Enumerable.Range(0, _layers).Where(l => _c.Kinds[l] == LayerKind.Attention).All(GqaFits);
                _minChunk = s_minChunk ?? (_gqaDecode ? Kernels.GqaPass : 32);
                _kCombine = _acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, AttnParams>(Kernels.AttnCombine2);
                _kPrefill = _acc.LoadKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>>(Kernels.AttnPrefill);
                _kScale = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float>(Kernels.Scale);
                if (_c.AttentionTemperatureScale != 0)
                    _scaleQueryTemperature = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<int>, int, int, int, float>(Kernels.ScaleQueryTemperature);
                _kGather = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int>(Kernels.GatherRows);
                _kGeluMul = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(Kernels.GeluMul);
                _kPageCopy = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<Half>, long, long>(Kernels.PageCopy);
                _kRotateK = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<Half>, ArrayView<int>, ArrayView<float>, int, int, int, int, int>(Kernels.RotateK);
                _kConv = _acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ConvParams>(Kernels.Conv1dSilu);
                _kDelta = _acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, DeltaParams>(Kernels.DeltaRule);
                _kScan = _acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ScanParams>(Kernels.SsmScan);
                _kGatedNorm = _acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, GatedNormParams>(Kernels.GatedNorm);

                // ---- small float parameters (norm weights, biases) packed into one buffer
                var fp = new List<float>();
                long F(float[]? data)
                {
                    if (data == null) return -1;
                    while (fp.Count % 4 != 0) fp.Add(0);
                    long off = fp.Count;
                    fp.AddRange(data);
                    return off;
                }

                // ---- weights: one device buffer per layer
                _dl = new DLayer[_layers];
                int maxQ = 0, maxKv = 64, maxAtt = 0, maxFf = _c.NEmbd;
                for (int il = 0; il < _layers; il++)
                {
                    var L = model.Layers[il];
                    var mats = new (WeightTensor? W, float[]? Bias)[] { (L.Wq, L.Bq), (L.Wk, L.Bk), (L.Wv, L.Bv), (L.Wqkv, null), (L.Wo, L.Bo),
                        (L.FfnGate, L.FfnGateBias), (L.FfnUp, L.FfnUpBias), (L.FfnDown, L.FfnDownBias), (L.PleInpGate, null), (L.PleProj, null),
                        (L.SsmInProjQkv, null), (L.SsmGate, null), (L.SsmBeta, null), (L.SsmAlpha, null), (L.SsmIn, null), (L.SsmOut, null) };
                    var buf = UploadMatrices(mats.Select(m => m.W).ToArray(), out var dm);
                    for (int i = 0; i < dm.Length; i++) if (dm[i] != null) dm[i]!.BiasOff = F(mats[i].Bias);
                    var d = new DLayer
                    {
                        W = buf, Wq = dm[0], Wk = dm[1], Wv = dm[2], Wqkv = dm[3], Wo = dm[4], Gate = dm[5], Up = dm[6], Down = dm[7], PleGate = dm[8], PleProj = dm[9],
                        AttnNorm = F(L.AttnNorm), AttnPost = F(L.AttnPostNorm), FfnNorm = F(L.FfnNorm), FfnPost = F(L.FfnPostNorm),
                        QNorm = F(L.QNorm), KNorm = F(L.KNorm), PlePost = F(L.PlePostNorm), OutScale = L.OutScale,
                        Kind = L.Kind, SsmQkv = dm[10], SsmZ = dm[11], SsmBeta = dm[12], SsmAlpha = dm[13], SsmIn = dm[14], SsmOut = dm[15],
                        ConvW = F(L.SsmConv1d), ConvB = F(L.SsmConv1dBias), DtBias = F(L.SsmDtBias), A = F(L.SsmA), D = F(L.SsmD), SsmNorm = F(L.SsmNorm),
                    };
                    if (L.Kind != LayerKind.Attention)
                    {
                        var rs = RecurrentSizes(_c, il);
                        d.ConvSize = rs.Conv; d.SsmSize = rs.Ssm;
                        _hasRec = true;
                    }
                    else if (_c.KvSourceLayer[il] == il)
                    {
                        long rowsTotal = (long)_pageCount * _pageSize;
                        d.KCache = Alloc<Half>(CacheWords(rowsTotal * _c.NHeadKv[il] * _c.HeadDimK[il], _kvQ8));
                        d.VCache = Alloc<Half>(CacheWords(rowsTotal * _c.NHeadKv[il] * _c.HeadDimV[il], _kvQ8));
                    }
                    _dl[il] = d;
                    int qMul = _c.GatedAttention ? 2 : 1;
                    maxQ = Math.Max(maxQ, _c.NHead[il] * _c.HeadDimK[il] * qMul + (L.Wqkv != null ? 2 * _c.NHeadKv[il] * Math.Max(_c.HeadDimK[il], _c.HeadDimV[il]) : 0));
                    maxKv = Math.Max(maxKv, _c.NHeadKv[il] * Math.Max(_c.HeadDimK[il], _c.HeadDimV[il]));
                    maxAtt = Math.Max(maxAtt, _c.NHead[il] * _c.HeadDimV[il]);
                    maxFf = Math.Max(maxFf, _c.NFf[il]);
                    if (L.Kind == LayerKind.Attention)
                    {
                        _maxHeads = Math.Max(_maxHeads, _c.NHead[il]);
                        _maxHdV = Math.Max(_maxHdV, _c.HeadDimV[il]);
                    }
                }
                _maxHeads = Math.Max(_maxHeads, 1);
                maxFf = Math.Max(maxFf, Math.Max(maxAtt, _c.NEmbdPerLayer));

                if (fitsHead && _layers == _c.NLayer && model.PerLayerModelProj != null && model.PerLayerProjNorm != null && _c.NEmbdPerLayer > 0)
                {
                    UploadMatrices([model.PerLayerModelProj], out var pm);
                    _pleProj = pm[0];
                    _pleNormOff = F(model.PerLayerProjNorm);
                    _pleTokenPart = model.PerLayerTokenEmbedding != null;
                }
                if (fitsHead && options.OffloadHead && _layers == _c.NLayer && !_c.IsEmbeddingModel)
                {
                    UploadMatrices([model.Output], out var hm);
                    _output = hm[0];
                    _outputNormOff = F(model.OutputNorm);
                }
                _f = _acc.Allocate1D(fp.Count == 0 ? [0f] : fp.ToArray());
                _owned.Add(_f);

                long nb = _maxBatch + Kernels.MatVecTokens; // slack rows: partial matvec tiles read padding rows
                _x = Alloc<float>(nb * _c.NEmbd); _xb = Alloc<float>(nb * _c.NEmbd); _mix = Alloc<float>(nb * _c.NEmbd);
                _kBase = nb * maxQ; _vBase = _kBase + nb * maxKv;
                _qkv = Alloc<float>(_vBase + nb * maxKv);
                if (_kvQ8 && maxKv > 0)
                {
                    _keyRows = Alloc<Half>((long)_maxBatch * maxKv);
                    _valueRows = Alloc<Half>((long)_maxBatch * maxKv);
                    _packKv = _acc.LoadKernel<ArrayView<Half>, ArrayView<Half>, ArrayView<int>, int, int>(Kernels.PackKvQ8);
                    _rotateQ8 = _acc.LoadKernel<ArrayView<Half>, ArrayView<int>, ArrayView<float>, int, int, int, int, int>(Kernels.RotateKvQ8);
                }
                _att = Alloc<float>(nb * Math.Max(maxAtt, _c.NEmbd));
                _ff = Alloc<float>(nb * maxFf);
                _ff2 = Alloc<float>(nb > Kernels.MatVecTokens ? nb * maxFf : 4);
                _logits = _output != null ? Alloc<float>(Kernels.MatVecTokens * (long)_c.NVocab) : Alloc<float>(4);
                int ropeDims = _c.Rope.Where(r => r != null).Distinct().Sum(r => r!.Dims);
                _rope = Alloc<float>(nb * Math.Max(ropeDims, 4));
                // widest GEMM input; fixed address for graph replays (int8 activations: buffers of the int8 GEMM too)
                bool int8 = _matVec.Int8 != Int8Activations.Off;
                _gemm.Reserve(nb, Math.Max(maxFf, _c.NEmbd), int8);
                // int8 activations of any matvec batch; only with the fp16 prompt GEMM, which then rounds the same way
                if (int8 && _gemm.Pipelined) _matVec.Reserve(nb, Math.Max(Math.Max(maxFf, maxAtt), _c.NEmbd));
                _infoFull = Alloc<int>(3 * nb); _infoSwa = Alloc<int>(3 * nb); _slots = Alloc<int>(nb); _rows = Alloc<int>(nb);
                _pages = Alloc<int>((long)_pageCount * 4 + 64);
                if (_pleProj != null) _pleProjOut = Alloc<float>(nb * _pleProj.Rows);
                // attention partials for graph-sized batches are allocated up front so captured graphs stay valid
                _part = Alloc<float>((long)Kernels.MatVecTokens * _maxHeads * GraphChunks(1) * (2 + _maxHdV));

                _seqInfo = Alloc<int>(3 * nb);
                _tokList = Alloc<int>(nb);
                if (_hasRec)
                {
                    var (convDim, inA, inA2, inB, outD) = RecurrentScratch(_c);
                    _recA2 = nb * inA;
                    _recA = Alloc<float>(nb * (inA + inA2));
                    _recB = Alloc<float>(nb * Math.Max(inB, 4));
                    _recC = Alloc<float>(nb * convDim);
                    _recD = Alloc<float>(nb * outD);
                    GrowSlots(4);
                }
                if (options.Warmup && _layers > 0) Warm(model);
                if (_layers > 0) Log.AcceleratorSelected(_log, DeviceName, Plan);
            }
            catch
            {
                ReleasePartial();   // a half-built offload would keep its device context (and memory) alive
                throw;
            }
        }
    }

    private static readonly object s_createLock = new();

    /// <summary>Frees what a failed constructor allocated.</summary>
    private void ReleasePartial()
    {
        try
        {
            foreach (var b in _owned) b.Dispose();
            _stream?.Dispose();
            _acc?.Dispose();
            _ctx?.Dispose();
        }
        catch (Exception e) { Log.EncoderCreateFailed(_log, e, "releasing a failed offload", DeviceName ?? "the device"); }
    }

    /// <summary>
    /// Runs batches of every matvec tile size (1, 2, 4, 8+) and a GEMM-sized prompt on a scratch sequence so the JIT
    /// specializations exist and the decode graph is captured (KV pages written here are overwritten later).
    /// </summary>
    private void Warm(Model model)
    {
        int capacity = (int)Math.Min(int.MaxValue, (long)_pageCount * _pageSize);
        int n = Math.Min(Math.Min(Math.Max(16, s_gemmMinTokens), _maxBatch), capacity), d = _c.NEmbd;
        int warmTokens = (int)Math.Min((long)n + 9, capacity);
        using var kv = new KvCache(_c, warmTokens, KvCacheType.F16, _pageSize, _ => false, _hasRec ? this : null);
        using var seq = new SequenceState(kv, _c);
        if (!seq.EnsureCapacity(warmTokens)) return;
        var ropes = new Dictionary<RopeConfig, nint>();
        var mem = new List<NativeBuffer>();
        try
        {
            foreach (var rc in _c.Rope)
                if (rc != null && !ropes.ContainsKey(rc)) { var b = new NativeBuffer((long)n * rc.Dims * 4); mem.Add(b); ropes[rc] = (nint)b.F; }
            var x = new NativeBuffer((long)n * d * 4); mem.Add(x);
            NativeBuffer? ple = _c.NEmbdPerLayer > 0 ? new NativeBuffer((long)n * _c.NLayer * _c.NEmbdPerLayer * 4) : null;
            if (ple != null) mem.Add(ple);
            using var logits = _output != null ? new NativeBuffer((long)_c.NVocab * sizeof(float), zero: false) : null;
            foreach (int count in new[] { n, 9, 4, 2, 1 }.Distinct())
            {
                if (count > n) continue;
                var seqs = Enumerable.Repeat(seq, count).ToArray();
                var pos = Enumerable.Range(count == n ? 0 : Math.Min(n, warmTokens - count), count).ToArray();
                Forward(new OffloadBatch
                {
                    X = x.F, N = count, Sequences = seqs, Positions = pos, RopeCaches = ropes, PerLayerInputs = ple != null ? ple.F : null,
                    OutputRows = logits != null ? [count - 1] : null, Logits = logits != null ? logits.F : null,
                });
            }
        }
        finally
        {
            foreach (var b in mem) b.Dispose();
        }
    }

    /// <summary>Per-slot conv-window and state sizes (floats) of a recurrent layer, in the host layout.</summary>
    private static (int Conv, int Ssm) RecurrentSizes(ModelConfig c, int il)
    {
        if (c.Kinds[il] == LayerKind.ShortConv) return ((c.SsmConvKernel - 1) * c.NEmbd, 0);
        if (c.Kinds[il] == LayerKind.GatedDeltaNet)
        {
            int hv = c.SsmInnerSize / c.SsmTimeStepRank, convDim = c.SsmStateSize * c.SsmGroupCount * 2 + c.SsmInnerSize;
            return ((c.SsmConvKernel - 1) * convDim, hv * hv * c.SsmTimeStepRank);
        }
        int cd = c.SsmInnerSize + 2 * c.SsmGroupCount * c.SsmStateSize;
        return ((c.SsmConvKernel - 1) * cd, c.SsmStateSize * c.SsmInnerSize);
    }

    /// <summary>Scratch widths per token: conv channels, in-projection A (+ second output), beta/alpha, mixer output.</summary>
    private static (int ConvDim, int InA, int InA2, int InB, int OutD) RecurrentScratch(ModelConfig c)
    {
        int convDim = 0, inA = 0, inA2 = 0, inB = 0, outD = 0;
        foreach (var k in c.Kinds.Distinct())
        {
            if (k == LayerKind.GatedDeltaNet)
            {
                int cd = c.SsmStateSize * c.SsmGroupCount * 2 + c.SsmInnerSize;
                convDim = Math.Max(convDim, cd); inA = Math.Max(inA, cd); inA2 = Math.Max(inA2, c.SsmInnerSize);
                inB = Math.Max(inB, 2 * c.SsmTimeStepRank); outD = Math.Max(outD, c.SsmInnerSize);
            }
            else if (k == LayerKind.ShortConv)
            {
                inA = Math.Max(inA, 3 * c.NEmbd);
                outD = Math.Max(outD, c.NEmbd);
            }
            else if (k == LayerKind.Mamba2)
            {
                int cd = c.SsmInnerSize + 2 * c.SsmGroupCount * c.SsmStateSize;
                convDim = Math.Max(convDim, cd);
                inA = Math.Max(inA, 2 * c.SsmInnerSize + 2 * c.SsmGroupCount * c.SsmStateSize + c.SsmTimeStepRank);
                outD = Math.Max(outD, c.SsmInnerSize);
            }
        }
        return (convDim, inA, inA2, inB, outD);
    }

    // ------------------------------------------------------------------ recurrent state store

    /// <summary>Grows every recurrent layer's state arena to <paramref name="cap"/> slots (existing slots are kept).</summary>
    private void GrowSlots(int cap)
    {
        for (int il = 0; il < _layers; il++)
        {
            var L = _dl[il];
            if (L.Kind == LayerKind.Attention) continue;
            var conv = _acc.Allocate1D<float>((long)cap * Math.Max(L.ConvSize, 1));
            var ssm = _acc.Allocate1D<float>((long)cap * Math.Max(1, L.SsmSize));
            conv.MemSetToZero(_stream);
            ssm.MemSetToZero(_stream);
            if (L.RecConv != null)
            {
                conv.View.SubView(0, L.RecConv.Length).CopyFrom(_stream, L.RecConv.View);
                ssm.View.SubView(0, L.RecSsm!.Length).CopyFrom(_stream, L.RecSsm.View);
                _stream.Synchronize();
                _owned.Remove(L.RecConv); _owned.Remove(L.RecSsm);
                L.RecConv.Dispose(); L.RecSsm.Dispose();
            }
            _owned.Add(conv); _owned.Add(ssm);
            L.RecConv = conv; L.RecSsm = ssm;
        }
        for (int sl = cap - 1; sl >= _slotCap; sl--) _freeSlots.Push(sl);
        _slotCap = cap;
        InvalidateGraphs();
    }

    public bool OwnsLayer(int layer) => layer < _layers && _dl[layer].Kind != LayerKind.Attention;

    public int Allocate()
    {
        _acc.Bind();
        if (_freeSlots.Count == 0) GrowSlots(_slotCap * 2);
        int slot = _freeSlots.Pop();
        Reset(slot);
        return slot;
    }

    public void Free(int slot) => _freeSlots.Push(slot);

    public void Reset(int slot)
    {
        _acc.Bind();
        foreach (var L in _dl)
        {
            if (L.Kind == LayerKind.Attention) continue;
            if (L.ConvSize > 0) L.RecConv!.View.SubView((long)slot * L.ConvSize, L.ConvSize).MemSetToZero(_stream);
            if (L.SsmSize > 0) L.RecSsm!.View.SubView((long)slot * L.SsmSize, L.SsmSize).MemSetToZero(_stream);
        }
    }

    public void Copy(int srcSlot, int dstSlot)
    {
        _acc.Bind();
        foreach (var L in _dl)
        {
            if (L.Kind == LayerKind.Attention) continue;
            if (L.ConvSize > 0)
                L.RecConv!.View.SubView((long)dstSlot * L.ConvSize, L.ConvSize).CopyFrom(_stream, L.RecConv.View.SubView((long)srcSlot * L.ConvSize, L.ConvSize));
            if (L.SsmSize > 0)
                L.RecSsm!.View.SubView((long)dstSlot * L.SsmSize, L.SsmSize).CopyFrom(_stream, L.RecSsm.View.SubView((long)srcSlot * L.SsmSize, L.SsmSize));
        }
    }

    public void Download(int slot, int layer, float* conv, float* ssm)
    {
        _acc.Bind();
        var L = _dl[layer];
        if (L.ConvSize > 0) L.RecConv!.View.SubView((long)slot * L.ConvSize, L.ConvSize).CopyToCPU(_stream, ref *conv, L.ConvSize);
        if (L.SsmSize > 0) L.RecSsm!.View.SubView((long)slot * L.SsmSize, L.SsmSize).CopyToCPU(_stream, ref *ssm, L.SsmSize);
        _stream.Synchronize();
    }

    public void Upload(int slot, int layer, float* conv, float* ssm)
    {
        _acc.Bind();
        var L = _dl[layer];
        if (L.ConvSize > 0) L.RecConv!.View.SubView((long)slot * L.ConvSize, L.ConvSize).CopyFromCPU(_stream, ref *conv, L.ConvSize);
        if (L.SsmSize > 0) L.RecSsm!.View.SubView((long)slot * L.SsmSize, L.SsmSize).CopyFromCPU(_stream, ref *ssm, L.SsmSize);
        _stream.Synchronize();
    }

    // ------------------------------------------------------------------ memory planning

    [System.Runtime.InteropServices.DllImport("nvcuda", EntryPoint = "cuMemGetInfo_v2")]
    private static extern int CuMemGetInfo(out nuint free, out nuint total);

    private long FreeDeviceMemory()
    {
        if (_acc is not CudaAccelerator) return long.MaxValue;
        _acc.Bind();
        return CuMemGetInfo(out var free, out _) == 0 ? (long)free : _acc.MemorySize;
    }

    /// <summary>Device bytes of a weight matrix in its device format.</summary>
    private static long MatrixBytes(WeightTensor? w) =>
        w == null ? 0 : (long)DevFormat.RowWords(DevFormat.For(w.Type, w.Cols), w.Cols) * w.Rows * 4;

    private static long CacheWords(long elements, bool q8) => q8 ? elements / 32 * 17 : elements;

    /// <summary>Estimated device bytes for a layer's weights, KV pages or recurrent arenas.</summary>
    public static long EstimateLayerBytes(Model model, int il, OffloadGeometry g)
    {
        var c = model.Config;
        var L = model.Layers[il];
        long b = new[] { L.Wq, L.Wk, L.Wv, L.Wqkv, L.Wo, L.FfnGate, L.FfnUp, L.FfnDown, L.PleInpGate, L.PleProj,
            L.SsmInProjQkv, L.SsmGate, L.SsmBeta, L.SsmAlpha, L.SsmIn, L.SsmOut }.Sum(MatrixBytes);
        if (c.Kinds[il] == LayerKind.Attention)
        {
            if (c.KvSourceLayer[il] == il)
                b += CacheWords((long)g.PageCount * g.PageSize * c.NHeadKv[il] * (c.HeadDimK[il] + c.HeadDimV[il]), g.KvCacheType == KvCacheType.Q8_0) * 2;
        }
        else
        {
            var (conv, ssm) = RecurrentSizes(c, il);
            b += 8L * (conv + ssm) * 4; // initial arena (slots grow on demand)
        }
        return b;
    }

    /// <summary>
    /// Layers to offload: the requested count, or with -1 the longest supported prefix that fits in free memory
    /// (after scratch buffers and the reserve); the head (LM head, per-layer input projection) only when all layers fit.
    /// </summary>
    private (int Layers, bool Head, string Plan) ChooseLayers(Model model, OffloadGeometry g, GpuOffloadOptions o, long free)
    {
        int supported = SupportedLayerPrefix(_c);
        long head = MatrixBytes(model.Output) + MatrixBytes(model.PerLayerModelProj);
        if (o.Layers >= 0)
        {
            int k = ValidSplit(_c, Math.Min(o.Layers, supported));
            return (k, true, $"{k} layers requested");
        }
        // scratch: activations for the max batch, logits tile, attention partials (generous upper bound)
        long scratch = (long)g.MaxBatch * (8L * _c.NEmbd + 4L * _c.NFf.DefaultIfEmpty(0).Max() + 16384) * 4 + 8L * _c.NVocab * 4 + (64L << 20);
        if (_kvQ8)
        {
            int width = Enumerable.Range(0, supported).Select(i => _c.NHeadKv[i] * Math.Max(_c.HeadDimK[i], _c.HeadDimV[i])).DefaultIfEmpty().Max();
            scratch += (long)g.MaxBatch * 4 * width;
        }
        long budget = free == long.MaxValue ? long.MaxValue : free - o.ReserveBytes - scratch;
        if ((g.Resources ?? ResourceLimiter.Default).MaxDeviceMemoryBytes is long cap) budget = Math.Min(budget, cap - scratch);   // ResourceLimiter
        long used = 0;
        int n = 0;
        while (n < supported)
        {
            long lb = EstimateLayerBytes(model, n, g);
            if (used + lb > budget) break;
            used += lb;
            n++;
        }
        n = ValidSplit(_c, n);
        bool fitsHead = n == _c.NLayer && used + head <= budget;
        string mb(long v) => v == long.MaxValue ? "unlimited" : $"{v >> 20} MB";
        return (n, fitsHead, $"auto: {n}/{_c.NLayer} layers ({mb(used)}){(fitsHead ? $" + head ({mb(head)})" : n == _c.NLayer ? " (head on CPU)" : "")}; free {mb(free)}, scratch {mb(scratch)}, reserve {mb(o.ReserveBytes)}");
    }

    /// <summary>Factory for <see cref="EngineOptions.Offload"/> (throws if no device is available).</summary>
    public static Func<Model, OffloadGeometry, ILayerOffload?> Factory(GpuOffloadOptions? options = null) =>
        (model, geometry) => new GpuLayerOffload(model, geometry, options);

    /// <summary>Why the last <see cref="TryCreate(Model, OffloadGeometry)"/> returned null (diagnostics).</summary>
    public static string? LastUnavailableReason { get; private set; }

    /// <summary>
    /// Creates an offload on the default CUDA device, or returns null (run on the CPU) when no device is usable, the
    /// driver is missing, or no layer fits. Used by <see cref="Accelerators"/> for engines without an explicit offload.
    /// </summary>
    public static ILayerOffload? TryCreate(Model model, OffloadGeometry geometry) => TryCreate(model, geometry, null);

    /// <inheritdoc cref="TryCreate(Model, OffloadGeometry)"/>
    public static ILayerOffload? TryCreate(Model model, OffloadGeometry geometry, GpuOffloadOptions? options)
    {
        var log = LammaLogging.Create<GpuLayerOffload>(geometry.LoggerFactory);
        if (SupportedLayerPrefix(model.Config) == 0)
        {
            LastUnavailableReason = $"no supported GPU layers for {model.Config.Architecture}";
            Log.AcceleratorUnavailable(log, LastUnavailableReason);
            return null;
        }
        try
        {
            using (var probe = Context.Create(b => b.Cuda()))
                if (!probe.Devices.Any(d => d.AcceleratorType == AcceleratorType.Cuda))
                {
                    LastUnavailableReason = "no CUDA device";
                    Log.AcceleratorUnavailable(log, LastUnavailableReason);
                    return null;
                }
            var off = new GpuLayerOffload(model, geometry, options);
            if (off.LayerCount > 0) { LastUnavailableReason = null; return off; }
            LastUnavailableReason = $"no layer fits in device memory or is supported ({off.Plan})";
            off.Dispose();
            Log.AcceleratorUnavailable(log, LastUnavailableReason);
        }
        catch (Exception e)
        {
            LastUnavailableReason = $"{e.GetType().Name}: {e.Message}";
            Log.AcceleratorFailed(log, e);
        }
        return null;
    }

    /// <summary>Factory that falls back to the CPU (null) instead of throwing; see <see cref="TryCreate(Model, OffloadGeometry)"/>.</summary>
    public static Func<Model, OffloadGeometry, ILayerOffload?> Auto(GpuOffloadOptions? options = null) =>
        (model, geometry) => TryCreate(model, geometry, options);

    /// <summary>True for layers whose KV lives on this device (the host cache skips them).</summary>
    public bool OwnsKv(int layer) => layer < _layers && _c.KvSourceLayer[layer] == layer;

    /// <summary>Device memory used by weights, caches and scratch buffers.</summary>
    public long DeviceBytes => _owned.Sum(b => b.LengthInBytes) + _gemm.ScratchBytes + _matVec.ScratchBytes;

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged
    {
        var b = _acc.Allocate1D<T>(Math.Max(n, 4));
        b.MemSetToZero(_stream);
        _owned.Add(b);
        return b;
    }

    /// <summary>Packs matrices into one device buffer (each matrix 16-byte aligned).</summary>
    private MemoryBuffer1D<uint, Stride1D.Dense> UploadMatrices(WeightTensor?[] ws, out DMat?[] mats)
    {
        mats = new DMat?[ws.Length];
        long words = 0;
        var types = new int[ws.Length];
        for (int i = 0; i < ws.Length; i++)
        {
            var w = ws[i];
            if (w == null) continue;
            if (w.Cols % 32 != 0) throw new NotSupportedException($"{w.Name}: row length {w.Cols} is not a multiple of 32");
            types[i] = DevFormat.For(w.Type, w.Cols);
            words = (words + 3) & ~3L;
            words += (long)DevFormat.RowWords(types[i], w.Cols) * w.Rows;
        }
        var host = new uint[Math.Max(words, 4)];
        var buf = Alloc<uint>(host.Length);
        long off = 0;
        fixed (uint* hp = host)
            for (int i = 0; i < ws.Length; i++)
            {
                var w = ws[i];
                if (w == null) continue;
                off = (off + 3) & ~3L;
                int rw = DevFormat.RowWords(types[i], w.Cols);
                DevFormat.Pack(w, types[i], hp + off);
                mats[i] = new DMat { Buf = buf, Word = off, Rows = w.Rows, Cols = w.Cols, RowWords = rw, Type = types[i] };
                off += (long)rw * w.Rows;
            }
        buf.View.CopyFromCPU(_stream, host);
        return buf;
    }

    // ------------------------------------------------------------------ launch helpers

    private static KernelConfig Warps(long warps) => new((int)((warps + Kernels.WarpsPerGroup - 1) / Kernels.WarpsPerGroup), Kernels.GroupSize);

    /// <summary>
    /// y[t, :] (op)= W·x[t, :] for up to three matrices of the same buffer and input. <paramref name="normed"/>: x is an
    /// RMS-norm output (int8 activations by default, <see cref="MatVecLauncher.UsesQ8"/>).
    /// </summary>
    private void MatVec(ArrayView<float> x, long xOff, int xStride, int n, ArrayView<float> y, int mode, float scale,
        (DMat M, long YOff, int YStride) a, (DMat M, long YOff, int YStride)? b = null, (DMat M, long YOff, int YStride)? c = null,
        int act = 0, float softCap = 0, string label = "matvec", ArrayView<float>? mul = null, long mulOff = 0, int mulStride = 0, bool normed = false)
    {
        if (n >= s_gemmMinTokens && softCap == 0)
        {
            // prompt processing: tiled GEMM per matrix (epilogues that need both pair outputs run separately); activations
            // rounded to int8 blocks where the matvec kernels use them, so prompt and decode numerics match
            int ta = a.M.Type, tb = b?.M.Type ?? ta, tc = c?.M.Type ?? ta;
            bool q8 = _matVec.UsesQ8(ta, tb, tc, a.M.Cols, xOff, xStride, normed);
            switch (mode)
            {
                case 2:
                    _gemm.Run(_stream, _f.View, x, xOff, xStride, n, _ff2.View, 0, 1f, a.M, a.YOff, a.YStride, q8);
                    _gemm.Run(_stream, _f.View, x, xOff, xStride, n, y, 0, 1f, b!.Value.M, b.Value.YOff, b.Value.YStride, q8);
                    _kActMul(_stream, n * a.YStride, _ff2.View.SubView(a.YOff, (long)n * a.YStride), y.SubView(b.Value.YOff, (long)n * a.YStride), act);
                    break;
                case 3:
                    _gemm.Run(_stream, _f.View, x, xOff, xStride, n, y, 0, 1f, a.M, a.YOff, a.YStride, q8);
                    _kGeluMul(_stream, n * a.M.Rows, y.SubView(a.YOff, (long)n * a.YStride), mul!.Value, a.M.Rows, mulStride, (int)mulOff);
                    break;
                default:
                    _gemm.Run(_stream, _f.View, x, xOff, xStride, n, y, mode, scale, a.M, a.YOff, a.YStride, q8);
                    if (b is { } bb) _gemm.Run(_stream, _f.View, x, xOff, xStride, n, y, mode, scale, bb.M, bb.YOff, bb.YStride, q8);
                    if (c is { } cc) _gemm.Run(_stream, _f.View, x, xOff, xStride, n, y, mode, scale, cc.M, cc.YOff, cc.YStride, q8);
                    break;
            }
            Mark(label);
            return;
        }
        _matVec.Run(_stream, mul ?? _f.View, x, xOff, xStride, n, y, mode, scale, a, b, c, act, softCap, mulOff, mulStride, normed);
        Mark(label);
    }

    private void Norm(ArrayView<float> x, long xOff, int xStride, ArrayView<float> y, long yOff, int yStride, long wOff, int rows, int dim,
        int mode = 0, float scale = 1f, float inScale = 1f)
    {
        var p = new NormParams { XOff = xOff, YOff = yOff, WOff = wOff, Rows = rows, D = dim, XStride = xStride, YStride = yStride, Mode = mode, Eps = _c.RmsEps, Scale = scale, InScale = inScale };
        _kRms(_stream, new KernelConfig(rows, Kernels.GroupSize), _f.View, x, y, p);
        Mark("norm");
    }

    /// <summary>
    /// Residual post-norm of <c>_mix</c> into x (mode 1: x += scale·norm, mode 2: x = (x + norm)·scale); with a next
    /// pre-norm (<paramref name="nextPre"/> ≥ 0) also normalizes the updated x into <c>_xb</c> in the same launch.
    /// Returns true when <c>_xb</c> holds that pre-norm.
    /// </summary>
    private bool ResidualNorm(long post, int mode, float scale, long nextPre, int n)
    {
        int d = _c.NEmbd;
        if (nextPre < 0 || !s_fuseNorms || d % 4 != 0)
        {
            Norm(_mix.View, 0, d, _x.View, 0, d, post, n, d, mode: mode, scale: scale);
            return false;
        }
        var p = new ResidualNormParams { XOff = 0, YOff = 0, ZOff = 0, WOff = post, W2Off = nextPre, Rows = n, D = d, Mode = mode, Eps = _c.RmsEps, Scale = scale };
        _kResNorm(_stream, new KernelConfig(n, Kernels.GroupSize), _f.View, _mix.View, _x.View, _xb.View, p);
        Mark("norm");
        return true;
    }

    /// <summary>Grows a scratch buffer; reallocation invalidates captured graphs.</summary>
    private MemoryBuffer1D<T, Stride1D.Dense> Ensure<T>(MemoryBuffer1D<T, Stride1D.Dense>? buf, long n) where T : unmanaged
    {
        if (buf != null && buf.Length >= n) return buf;
        if (buf != null) { _owned.Remove(buf); buf.Dispose(); }
        InvalidateGraphs();
        var b = _acc.Allocate1D<T>(Math.Max(n, 64));
        _owned.Add(b);
        return b;
    }

    private void InvalidateGraphs()
    {
        foreach (var g in _graphs.Values) g.Dispose();
        _graphs.Clear();
    }

    private void Upload(MemoryBuffer1D<int, Stride1D.Dense> buf, int[] data) => buf.View.SubView(0, data.Length).CopyFromCPU(_stream, data);

    /// <summary>Most context chunks per (token, head) for a batch of n tokens: enough warps to fill the device.</summary>
    private int GraphChunks(int n)
    {
        int target = Math.Max(1, _acc.NumMultiprocessors) * 16;
        return Math.Clamp((target + n * _maxHeads - 1) / (n * _maxHeads), 1, 64);
    }

    /// <summary>
    /// Flash-decoding split: at least <see cref="_minChunk"/> positions per chunk, at most
    /// <see cref="GraphChunks"/> chunks. Graph replays use the length rounded up to a power of two, so a graph serves
    /// a whole range of context lengths (the kernel derives each token's chunk length from its own range).
    /// </summary>
    private int AttnChunks(int n, int len) => Math.Clamp((len + _minChunk - 1) / _minChunk, 1, GraphChunks(n));

    // positions per chunk: 128 for AttnDecodeGqa (a thread per position), 32 for AttnDecode (one 32-position step per
    // warp keeps enough warps busy at short contexts); LAMMA_ATTN_CHUNK overrides
    private static readonly int? s_minChunk = int.TryParse(Environment.GetEnvironmentVariable("LAMMA_ATTN_CHUNK"), out var mc) && mc > 0 ? mc : null;
    private readonly int _minChunk = 32;

    /// <summary>Shapes <see cref="Kernels.AttnDecodeGqa"/> supports: up to 8 query heads per KV head, value dims in 64-slices.</summary>
    private bool GqaFits(int il)
    {
        int nh = _c.NHead[il], nkv = _c.NHeadKv[il], hk = _c.HeadDimK[il], hv = _c.HeadDimV[il];
        return nkv > 0 && nh % nkv == 0 && nh / nkv <= Kernels.GqaMaxGroup && hk == hv && hk % 64 == 0 && hk <= Kernels.MaxHeadDim;
    }

    // ------------------------------------------------------------------ profiling

    private void Mark(string label)
    {
        if (!s_profile) return;
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        _cpuProfile[label] = _cpuProfile.GetValueOrDefault(label) + (now - _lastMark) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        _marks.Add((label, _stream.AddProfilingMarker()));
        _lastMark = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    private void CollectProfile()
    {
        if (!s_profile || _marks.Count == 0) return;
        _stream.Synchronize();
        for (int i = 1; i < _marks.Count; i++)
        {
            string label = _marks[i].Label;
            double ms = _marks[i].Marker.MeasureFrom(_marks[i - 1].Marker).TotalMilliseconds;
            _profile.TryGetValue(label, out var v);
            _profile[label] = (v.Ms + ms, v.Count + 1);
        }
        foreach (var (_, m) in _marks) m.Dispose();
        _marks.Clear();
    }

    /// <summary>Clears the accumulated profile (e.g. after warm-up / JIT).</summary>
    public void ResetProfile() { _profile.Clear(); _cpuProfile.Clear(); }

    /// <summary>Per-kernel device time accumulated with LAMMA_GPU_PROFILE=1.</summary>
    public void ReportProfile(TextWriter w)
    {
        double total = _profile.Values.Sum(v => v.Ms);
        foreach (var (k, v) in _profile.OrderByDescending(kv => kv.Value.Ms))
            w.WriteLine($"gpu {k,-14} {v.Ms,9:F2} ms {100 * v.Ms / total,5:F1}%  {v.Count,7} launches  {v.Ms * 1000 / v.Count,8:F1} us/launch   cpu {_cpuProfile.GetValueOrDefault(k) * 1000 / v.Count,7:F1} us/launch");
    }

    // ------------------------------------------------------------------ forward

    public void Forward(OffloadBatch batch)
    {
        _acc.Bind();
        int n = batch.N, d = _c.NEmbd;
        if (n > _maxBatch) throw new ArgumentException("Batch too large for GPU scratch buffers");
        Mark("start");

        // ---- page tables and per-token attention ranges (full and sliding-window)
        var seqIndex = new Dictionary<SequenceState, int>();
        var pages = new List<int>();
        var pageOffsets = new List<int>();
        foreach (var s in batch.Sequences)
            if (!seqIndex.ContainsKey(s)) { seqIndex[s] = pageOffsets.Count; pageOffsets.Add(pages.Count); pages.AddRange(s.PageTable); }
        var infoFull = new int[3 * n];
        var infoSwa = new int[3 * n];
        var slots = new int[n];
        int maxLen = 1, maxLenSwa = 1;
        // bidirectional models: tokens see their sequence's whole input (all in this batch)
        int[]? seqEnd = null;
        if (!_c.Causal)
        {
            seqEnd = new int[pageOffsets.Count];
            for (int t = 0; t < n; t++) seqEnd[seqIndex[batch.Sequences[t]]] = Math.Max(seqEnd[seqIndex[batch.Sequences[t]]], batch.Positions[t]);
        }
        for (int t = 0; t < n; t++)
        {
            var s = batch.Sequences[t];
            int p = batch.Positions[t], off = pageOffsets[seqIndex[s]];
            int last = seqEnd != null ? seqEnd[seqIndex[s]] : p;
            infoFull[3 * t] = off; infoFull[3 * t + 1] = 0; infoFull[3 * t + 2] = last;
            int ws = 0, we = last;
            if (_c.SlidingWindow > 0)
            {
                if (seqEnd != null && _c.SymmetricSwa) { ws = Math.Max(0, p - _c.SlidingWindow / 2); we = Math.Min(last, p + _c.SlidingWindow / 2); }
                else ws = Math.Max(0, p - _c.SlidingWindow + 1);
            }
            infoSwa[3 * t] = off; infoSwa[3 * t + 1] = ws; infoSwa[3 * t + 2] = we;
            slots[t] = s.PageTable[p / _pageSize] * _pageSize + p % _pageSize;
            maxLen = Math.Max(maxLen, last + 1);
            maxLenSwa = Math.Max(maxLenSwa, we - ws + 1);
        }
        _pages = Ensure(_pages, pages.Count + 1);
        Upload(_pages, pages.Count == 0 ? [0] : pages.ToArray());
        Upload(_infoFull, infoFull);
        Upload(_infoSwa, infoSwa);
        Upload(_slots, slots);

        // ---- rope caches (host-computed, exact), inputs
        var ropeOffsets = new Dictionary<RopeConfig, long>();
        long ropeTotal = 0;
        foreach (var (rc, _) in batch.RopeCaches) { ropeOffsets[rc] = ropeTotal; ropeTotal += (long)n * rc.Dims; }
        foreach (var (rc, ptr) in batch.RopeCaches)
            _rope.View.SubView(ropeOffsets[rc], (long)n * rc.Dims).CopyFromCPU(_stream, ref *(float*)ptr, (long)n * rc.Dims);
        bool ple = batch.PerLayerInputs != null && _c.NEmbdPerLayer > 0;
        if (ple)
        {
            long len = (long)n * _c.NLayer * _c.NEmbdPerLayer;
            _ple = Ensure(_ple, Math.Max(len, (long)Kernels.MatVecTokens * _c.NLayer * _c.NEmbdPerLayer));
            _ple.View.SubView(0, len).CopyFromCPU(_stream, ref *batch.PerLayerInputs, len);
        }
        _x.View.SubView(0, (long)n * d).CopyFromCPU(_stream, ref *batch.X, (long)n * d);
        int[]? outRows = _output != null ? batch.OutputRows : null;
        if (outRows != null) Upload(_rows, outRows);
        if (_hasRec)
        {
            // tokens grouped by sequence (position order kept), with each sequence's recurrent-state slot
            var info = new int[3 * n];
            var toks = new int[n];
            int ns = 0, k = 0;
            var order = new List<SequenceState>();
            foreach (var s in batch.Sequences) if (!order.Contains(s)) order.Add(s);
            foreach (var s in order)
            {
                info[3 * ns] = k;
                for (int t = 0; t < n; t++) if (batch.Sequences[t] == s) toks[k++] = t;
                info[3 * ns + 1] = k - info[3 * ns];
                info[3 * ns + 2] = s.Recurrent?.Slot ?? throw new InvalidOperationException("Sequence has no device recurrent state");
                ns++;
            }
            Upload(_seqInfo, info);
            Upload(_tokList, toks);
        }
        Mark("upload");

        // ---- layers (+ head): replay a captured graph for small batches, otherwise launch directly
        bool graph = _graphsEnabled && n <= Kernels.MatVecTokens && (outRows == null || outRows.Length <= Kernels.MatVecTokens);
        if (graph)
        {
            int chunks = AttnChunks(n, (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(maxLen, _minChunk)));
            int chunksSwa = AttnChunks(n, (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(maxLenSwa, _minChunk)));
            var key = (n, outRows?.Length ?? -1, ple, chunks, chunksSwa);
            if (!_graphs.TryGetValue(key, out var g))
            {
                // the first run of a shape compiles its kernels (JIT) eagerly, then the shape is captured
                LaunchAll(n, outRows?.Length ?? 0, ple, ropeOffsets, chunks, chunksSwa);
                _stream.Synchronize();
                g = CudaGraph.Capture(((CudaStream)_stream).StreamPtr,
                    () => LaunchAll(n, outRows?.Length ?? 0, ple, ropeOffsets, chunks, chunksSwa));
                _graphs[key] = g;
                // the eager run above already produced the results for this step
            }
            else g.Launch(((CudaStream)_stream).StreamPtr);
        }
        else
        {
            // query tiles for the prefill attention kernel: consecutive positions of one sequence, up to PrefillBQ each
            var tiles = new List<int>();
            for (int t = 0; t < n;)
            {
                int c = 1;
                while (c < Kernels.PrefillBQ && t + c < n && batch.Sequences[t + c] == batch.Sequences[t] && batch.Positions[t + c] == batch.Positions[t + c - 1] + 1) c++;
                tiles.Add(t); tiles.Add(c);
                t += c;
            }
            _tiles = Ensure(_tiles, tiles.Count);
            Upload(_tiles, tiles.ToArray());
            _tileCount = tiles.Count / 2;
            int chunks = AttnChunks(n, maxLen), chunksSwa = AttnChunks(n, maxLenSwa);
            _part = Ensure(_part, (long)n * _maxHeads * Math.Max(chunks, chunksSwa) * (2 + _maxHdV));
            LaunchAll(n, outRows?.Length ?? 0, ple, ropeOffsets, chunks, chunksSwa);
            _tileCount = 0;
        }

        if (outRows != null)
        {
            int v = _c.NVocab;
            for (int r0 = 0; r0 < outRows.Length; r0 += Kernels.MatVecTokens)
            {
                int nr = Math.Min(Kernels.MatVecTokens, outRows.Length - r0);
                if (r0 > 0) Head(r0, nr); // rows beyond the first tile (large batches only; never inside a graph)
                _logits.View.SubView(0, (long)nr * v).CopyToCPU(_stream, ref *(batch.Logits + (long)r0 * v), (long)nr * v);
            }
            Mark("download");
            CollectProfile();
            return;
        }
        _x.View.SubView(0, (long)n * d).CopyToCPU(_stream, ref *batch.X, (long)n * d);
        _stream.Synchronize();
        Mark("download");
        CollectProfile();
    }

    /// <summary>Enqueues every kernel of the offloaded layers (and the first head tile) for a batch of n tokens.</summary>
    private void LaunchAll(int n, int outRows, bool ple, Dictionary<RopeConfig, long> ropeOffsets, int chunks, int chunksSwa)
    {
        int d = _c.NEmbd;
        var x = _x.View;
        if (ple && _pleProj != null)
        {
            // per-layer inputs: ple = (tokenPart + norm(proj(x) / sqrt(d)) * w) / sqrt(2), per layer segment of NEmbdPerLayer
            int pd = _c.NEmbdPerLayer;
            MatVec(x, 0, d, n, _pleProjOut!.View, 0, 1f, (_pleProj, 0, _pleProj.Rows), label: "ple");
            if (_pleTokenPart)
                Norm(_pleProjOut.View, 0, pd, _ple!.View, 0, pd, _pleNormOff, n * _c.NLayer, pd, mode: 2, scale: 1f / MathF.Sqrt(2f), inScale: 1f / MathF.Sqrt(d));
            else
                Norm(_pleProjOut.View, 0, pd, _ple!.View, 0, pd, _pleNormOff, n * _c.NLayer, pd, inScale: 1f / MathF.Sqrt(d));
        }
        bool xbReady = false;   // _xb already holds this layer's pre-norm (fused into the previous layer's last norm)
        for (int il = 0; il < _layers; il++)
        {
            var L = _dl[il];
            if (!xbReady) Norm(x, 0, d, _xb.View, 0, d, L.AttnNorm, n, d);
            xbReady = false;
            ArrayView<float> mixIn;
            int mixWidth;
            DMat outMat;
            switch (L.Kind)
            {
                case LayerKind.GatedDeltaNet:
                    DeltaNet(L, n);
                    mixIn = _recD!.View; mixWidth = _c.SsmInnerSize; outMat = L.SsmOut!;
                    break;
                case LayerKind.Mamba2:
                    Mamba2(L, n);
                    mixIn = _recD!.View; mixWidth = _c.SsmInnerSize; outMat = L.SsmOut!;
                    break;
                case LayerKind.ShortConv:
                    ShortConv(L, n);
                    mixIn = _recD!.View; mixWidth = d; outMat = L.SsmOut!;
                    break;
                default:
                    Attention(il, L, n, ropeOffsets, _c.IsSwa[il] ? chunksSwa : chunks);
                    mixIn = _att.View; mixWidth = _c.NHead[il] * _c.HeadDimV[il]; outMat = L.Wo!;
                    break;
            }
            // mixer output → residual (post-norm variant goes through _mix)
            bool ffnReady = false;
            if (L.AttnPost >= 0)
            {
                MatVec(mixIn, 0, mixWidth, n, _mix.View, 0, 1f, (outMat, 0, d), label: "wo");
                ffnReady = ResidualNorm(L.AttnPost, 1, _c.ResidualScale, L.Down != null ? L.FfnNorm : -1, n);
            }
            else MatVec(mixIn, 0, mixWidth, n, x, 1, _c.ResidualScale, (outMat, 0, d), label: "wo");

            bool pleNext = ple && L.PleGate != null;
            long nextPre = il + 1 < _layers ? _dl[il + 1].AttnNorm : -1;
            if (L.Down != null)
            {
                if (!ffnReady) Norm(x, 0, d, _xb.View, 0, d, L.FfnNorm, n, d);
                int ff = L.Down.Cols;
                if (L.Gate != null)
                    MatVec(_xb.View, 0, d, n, _ff.View, 2, 1f, (L.Gate, 0, ff), (L.Up!, 0, ff), act: _c.Activation == FfnActivation.GeGlu ? 1 : 0, normed: true, label: "gate_up");
                else
                    MatVec(_xb.View, 0, d, n, _ff.View, 0, 1f, (L.Up!, 0, ff), normed: true, label: "gate_up");
                if (L.FfnPost >= 0)
                {
                    MatVec(_ff.View, 0, ff, n, _mix.View, 0, 1f, (L.Down, 0, d), label: "down");
                    // the next layer's pre-norm joins when nothing else changes x before it
                    xbReady = ResidualNorm(L.FfnPost, 1, _c.ResidualScale, !pleNext && L.OutScale == 1f ? nextPre : -1, n);
                }
                else MatVec(_ff.View, 0, ff, n, x, 1, _c.ResidualScale, (L.Down, 0, d), label: "down");
            }

            bool scaled = false;
            if (ple && L.PleGate != null)
            {
                // g = gelu(gate * x) * ple[t, layer]; x = (x + norm(proj * g)) * outScale
                int pd = _c.NEmbdPerLayer;
                MatVec(x, 0, d, n, _ff.View, 3, 1f, (L.PleGate, 0, pd), act: 1, label: "ple", mul: _ple!.View, mulOff: il * pd, mulStride: _c.NLayer * pd);
                MatVec(_ff.View, 0, pd, n, _mix.View, 0, 1f, (L.PleProj!, 0, d), label: "ple");
                xbReady = ResidualNorm(L.PlePost, 2, L.OutScale, nextPre, n);
                scaled = true;
            }
            if (L.OutScale != 1f && !scaled) { _kScale(_stream, n * d, x.SubView(0, (long)n * d), L.OutScale); Mark("scale"); }
        }
        if (outRows > 0)
        {
            _kGather(_stream, outRows * d, x, _mix.View, _rows.View, d);
            Mark("gather");
            Norm(_mix.View, 0, d, _xb.View, 0, d, _outputNormOff, outRows, d);
            Head(0, Math.Min(outRows, Kernels.MatVecTokens));
        }
    }

    /// <summary>Logits of normalized output rows [r0, r0 + nr) of <c>_xb</c> into <c>_logits</c>.</summary>
    private void Head(int r0, int nr) =>
        MatVec(_xb.View, (long)r0 * _c.NEmbd, _c.NEmbd, nr, _logits.View, 0, 1f / _c.LogitScale, (_output!, 0, _c.NVocab), softCap: _c.FinalLogitSoftcap, normed: true, label: "lm_head");

    /// <summary>Gated DeltaNet mixer: in-projections, conv + SiLU, delta rule, gated norm → <c>_recD</c>.</summary>
    private void DeltaNet(DLayer L, int n)
    {
        int d = _c.NEmbd, hk = _c.SsmStateSize, nK = _c.SsmGroupCount, nV = _c.SsmTimeStepRank, hv = _c.SsmInnerSize / nV;
        int keyDim = hk * nK, valueDim = hv * nV, convDim = 2 * keyDim + valueDim;
        var a = _recA!.View;
        MatVec(_xb.View, 0, d, n, a, 0, 1f, (L.SsmQkv!, 0, convDim), (L.SsmZ!, _recA2, valueDim), normed: true, label: "ssm_in");
        MatVec(_xb.View, 0, d, n, _recB!.View, 0, 1f, (L.SsmBeta!, 0, 2 * nV), (L.SsmAlpha!, nV, 2 * nV), normed: true, label: "ssm_in");
        var cp = new ConvParams
        {
            InOff = 0, OutOff = 0, WOff = L.ConvW, BiasOff = L.ConvB, StateStride = L.ConvSize, MulInOff = -1, MulOutOff = -1,
            Channels = convDim, KSize = _c.SsmConvKernel, InStride = convDim, OutStride = convDim, MaxSeqs = n, Silu = 1,
        };
        _kConv(_stream, n * convDim, _f.View, a, _recC!.View, L.RecConv!.View, _seqInfo.View, _tokList.View, cp);
        Mark("ssm_conv");
        var dp = new DeltaParams
        {
            ConvOff = 0, BaOff = 0, OutOff = 0, DtBiasOff = L.DtBias, AOff = L.A, StateStride = L.SsmSize,
            MaxSeqs = n, NV = nV, NK = nK, HK = hk, HV = hv, ConvStride = convDim, BaStride = 2 * nV, OutStride = valueDim,
            Scale = 1f / MathF.Sqrt(hv), Eps = _c.RmsEps,
        };
        _kDelta(_stream, Warps((long)n * nV * hv), _f.View, _recC.View, _recB.View, _recD!.View, L.RecSsm!.View, _seqInfo.View, _tokList.View, dp);
        Mark("ssm_scan");
        var gp = new GatedNormParams
        {
            YOff = 0, ZOff = _recA2, WOff = L.SsmNorm, Rows = n * nV, Groups = nV, D = hv, YStride = valueDim, ZStride = valueDim, WStride = 0,
            NormFirst = 1, Eps = _c.RmsEps,
        };
        _kGatedNorm(_stream, Warps((long)n * nV), _f.View, _recD.View, a, gp);
        Mark("ssm_norm");
    }

    /// <summary>LFM2 gated short convolution: in-projection (b | c | x), y = c · conv(b · x) → <c>_recD</c>.</summary>
    private void ShortConv(DLayer L, int n)
    {
        int d = _c.NEmbd;
        var a = _recA!.View;
        MatVec(_xb.View, 0, d, n, a, 0, 1f, (L.SsmIn!, 0, 3 * d), normed: true, label: "ssm_in");
        var cp = new ConvParams
        {
            InOff = 2 * d, MulInOff = 0, MulOutOff = d, OutOff = 0, WOff = L.ConvW, BiasOff = -1, StateStride = L.ConvSize,
            Channels = d, KSize = _c.SsmConvKernel, InStride = 3 * d, OutStride = d, MaxSeqs = n, Silu = 0,
        };
        _kConv(_stream, n * d, _f.View, a, _recD!.View, L.RecConv!.View, _seqInfo.View, _tokList.View, cp);
        Mark("ssm_conv");
    }

    /// <summary>Mamba2 mixer: in-projection, conv + SiLU, selective scan, gated group norm → <c>_recD</c>.</summary>
    private void Mamba2(DLayer L, int n)
    {
        int d = _c.NEmbd, dInner = _c.SsmInnerSize, dState = _c.SsmStateSize, nHead = _c.SsmTimeStepRank, nGroup = _c.SsmGroupCount;
        int headDim = dInner / nHead, convDim = dInner + 2 * nGroup * dState, dInProj = 2 * dInner + 2 * nGroup * dState + nHead;
        var a = _recA!.View;
        MatVec(_xb.View, 0, d, n, a, 0, 1f, (L.SsmIn!, 0, dInProj), normed: true, label: "ssm_in");
        var cp = new ConvParams
        {
            InOff = dInner, OutOff = 0, WOff = L.ConvW, BiasOff = L.ConvB, StateStride = L.ConvSize, MulInOff = -1, MulOutOff = -1,
            Channels = convDim, KSize = _c.SsmConvKernel, InStride = dInProj, OutStride = convDim, MaxSeqs = n, Silu = 1,
        };
        _kConv(_stream, n * convDim, _f.View, a, _recC!.View, L.RecConv!.View, _seqInfo.View, _tokList.View, cp);
        Mark("ssm_conv");
        var sp = new ScanParams
        {
            XbcOff = 0, DtOff = 2 * dInner + 2 * nGroup * dState, YOff = 0, DtBiasOff = L.DtBias, AOff = L.A, DOff = L.D, StateStride = L.SsmSize,
            MaxSeqs = n, NHead = nHead, HeadDim = headDim, DState = dState, NGroup = nGroup, DInner = dInner,
            XbcStride = convDim, DtStride = dInProj, YStride = dInner,
        };
        _kScan(_stream, Warps((long)n * nHead * headDim), _f.View, _recC.View, a, _recD!.View, L.RecSsm!.View, _seqInfo.View, _tokList.View, sp);
        Mark("ssm_scan");
        int gsz = dInner / nGroup;
        var gp = new GatedNormParams
        {
            YOff = 0, ZOff = 0, WOff = L.SsmNorm, Rows = n * nGroup, Groups = nGroup, D = gsz, YStride = dInner, ZStride = dInProj, WStride = gsz,
            NormFirst = 0, Eps = _c.RmsEps,
        };
        _kGatedNorm(_stream, Warps((long)n * nGroup), _f.View, _recD.View, a, gp);
        Mark("ssm_norm");
    }

    private void Attention(int il, DLayer L, int n, Dictionary<RopeConfig, long> ropeOffsets, int chunks)
    {
        int d = _c.NEmbd, nh = _c.NHead[il], nkv = _c.NHeadKv[il], hk = _c.HeadDimK[il], hv = _c.HeadDimV[il];
        int src = _c.KvSourceLayer[il];
        bool own = src == il;
        bool gated = _c.GatedAttention;
        int qDim = nh * hk * (gated ? 2 : 1), kDim = nkv * hk, vDim = nkv * hv;

        long qOff = 0, kOff = _kBase, vOff = _vBase;
        int qStride = qDim, kStride = kDim, vStride = vDim;
        var qBuf = _qkv.View;
        if (L.Wqkv != null)
        {
            int tot = L.Wqkv.Rows;
            MatVec(_xb.View, 0, d, n, qBuf, 0, 1f, (L.Wqkv, 0, tot), normed: true, label: "qkv");
            kOff = qDim; vOff = qDim + kDim;
            qStride = kStride = vStride = tot;
        }
        else if (!own) MatVec(_xb.View, 0, d, n, qBuf, 0, 1f, (L.Wq!, 0, qDim), normed: true, label: "qkv");
        else if (L.Wv != null) MatVec(_xb.View, 0, d, n, qBuf, 0, 1f, (L.Wq!, 0, qDim), (L.Wk!, _kBase, kDim), (L.Wv, _vBase, vDim), normed: true, label: "qkv");
        else
        {
            // no V projection (Gemma 4 full-attention layers): V is the raw K projection
            MatVec(_xb.View, 0, d, n, qBuf, 0, 1f, (L.Wq!, 0, qDim), (L.Wk!, _kBase, kDim), normed: true, label: "qkv");
            vOff = _kBase; vStride = kDim;
        }
        var rope = _c.Rope[il];
        var S = _dl[src];
        var qp = new QkvParams
        {
            QOff = qOff, KOff = kOff, VOff = vOff, QNormOff = L.QNorm, KNormOff = L.KNorm,
            RopeOff = rope != null ? ropeOffsets[rope] : 0,
            N = n, NHead = nh, NKv = nkv, Hd = hk, HdV = hv, QStride = qStride, KStride = kStride, VStride = vStride,
            QHeadPitch = gated ? 2 * hk : hk, KRow = kDim, VRow = vDim, OwnKv = own ? 1 : 0, VNorm = _c.VNorm ? 1 : 0,
            RopeDims = rope?.Dims ?? 0, RopeNeox = rope != null && rope.Mode != RopeMode.Normal ? 1 : 0, RopeStride = rope?.Dims ?? 0,
            Eps = _c.RmsEps, DenseKvRows = _kvQ8 ? 1 : 0,
        };
        int units = n * (nh + (own ? 2 * nkv : 0));
        var keys = S.KCache!.View;
        var values = S.VCache!.View;
        _kQkv(_stream, Warps(units), _f.View, qBuf, qBuf, qBuf, _rope.View,
            _kvQ8 ? _keyRows!.View : keys, _kvQ8 ? _valueRows!.View : values, _slots.View, qp);
        if (_kvQ8 && own)
        {
            _packKv!(_stream, Warps((long)n * kDim / 32), _keyRows!.View, keys, _slots.View, kDim, n);
            _packKv!(_stream, Warps((long)n * vDim / 32), _valueRows!.View, values, _slots.View, vDim, n);
        }
        Mark("qkv_post");
        _scaleQueryTemperature?.Invoke(_stream, n * qDim, qBuf, _infoFull.View, qDim, qStride, rope?.NCtxOrig ?? 1, _c.AttentionTemperatureScale);

        float scale = _c.AttentionScale != 0 ? _c.AttentionScale : 1f / MathF.Sqrt(hk);
        var ap = new AttnParams
        {
            QOff = qOff, OutOff = 0, N = n, NHead = nh, NKv = nkv, Hd = hk, HdV = hv, PageSize = _pageSize,
            QStride = qStride, QHeadPitch = gated ? 2 * hk : hk, OutStride = nh * hv, KRow = kDim, VRow = vDim,
            NChunks = chunks, Gated = gated ? 1 : 0, Scale = scale, SoftCap = _c.AttnLogitSoftcap,
        };
        var info = _c.IsSwa[il] ? _infoSwa : _infoFull;
        if (_tileCount > 0 && _kTcPrefill != null && hk == hv && hk <= Kernels.PrefillMaxHd && hk % 32 == 0)
        {
            _kTcPrefill(_stream, new KernelConfig(_tileCount * nh, 128), qBuf, S.KCache.View, S.VCache.View, _pages.View, info.View, _tiles!.View, _att.View, ap, SpecializedValue.New(hk));
            Mark("attn");
            return;
        }
        if (!_kvQ8 && _tileCount > 0 && hk == hv && hk <= Kernels.PrefillMaxHd && hk % 8 == 0)
        {
            _kPrefill(_stream, new KernelConfig(_tileCount * nh, 128), qBuf, S.KCache.View, S.VCache.View, _pages.View, info.View, _tiles!.View, _att.View, ap, SpecializedValue.New(hk));
            Mark("attn");
            return;
        }
        if (_gqaDecode)
            _kAttnGqa(_stream, new KernelConfig(n * nkv * chunks, Kernels.GroupSize), qBuf, S.KCache.View, S.VCache.View, _pages.View, info.View, _att.View, _part!.View, ap,
                SpecializedValue.New(nh / nkv), SpecializedValue.New(hk));
        else
        {
            int headGroups = (nh + Kernels.WarpsPerGroup - 1) / Kernels.WarpsPerGroup;
            _kAttn(_stream, new KernelConfig(n * headGroups * chunks, Kernels.GroupSize), qBuf, S.KCache.View, S.VCache.View, _pages.View, info.View, _att.View, _part!.View, ap);
        }
        Mark("attn");
        if (chunks > 1)
        {
            _kCombine(_stream, new KernelConfig(n * nh * ((hv + Kernels.GroupSize - 1) / Kernels.GroupSize), Kernels.GroupSize), qBuf, _part.View, _att.View, ap);
            Mark("attn_combine");
        }
    }

    public void RotateKeys(IReadOnlyList<int> pageTable, int from, int to, int delta)
    {
        if (to <= from) return;
        _acc.Bind();
        var slots = new int[to - from];
        for (int pos = from; pos < to; pos++) slots[pos - from] = pageTable[pos / _pageSize] * _pageSize + pos % _pageSize;
        using var ds = _acc.Allocate1D(slots);
        var rots = new Dictionary<RopeConfig, MemoryBuffer1D<float, Stride1D.Dense>>();
        try
        {
            for (int il = 0; il < _layers; il++)
            {
                var L = _dl[il];
                var rope = _c.Rope[il];
                if (L.KCache == null || rope == null) continue;
                if (!rots.TryGetValue(rope, out var rot)) rots[rope] = rot = _acc.Allocate1D(rope.ShiftRotation(delta));
                int nkv = _c.NHeadKv[il];
                if (_kvQ8)
                {
                    _rotateQ8!(_stream, Warps((long)slots.Length * nkv), L.KCache.View, ds.View, rot.View,
                        slots.Length, nkv, _c.HeadDimK[il], rope.Dims, rope.Mode != RopeMode.Normal ? 1 : 0);
                    continue;
                }
                _kRotateK(_stream, slots.Length * nkv * (rope.Dims / 2), L.KCache.View, ds.View, rot.View, slots.Length, nkv, _c.HeadDimK[il], rope.Dims,
                    rope.Mode != RopeMode.Normal ? 1 : 0);
            }
            _stream.Synchronize();
        }
        finally
        {
            foreach (var b in rots.Values) b.Dispose();
        }
    }

    /// <summary>Reads one K row (all KV heads) of a device-owned layer (tests / diagnostics).</summary>
    internal float[] ReadKeyRow(int layer, int slot)
    {
        _acc.Bind();
        int n = _c.NHeadKv[layer] * _c.HeadDimK[layer];
        int words = (int)CacheWords(n, _kvQ8);
        var h = new Half[words];
        _dl[layer].KCache!.View.SubView((long)slot * words, words).CopyToCPU(_stream, h);
        _stream.Synchronize();
        if (_kvQ8)
        {
            var bytes = System.Runtime.InteropServices.MemoryMarshal.Cast<Half, sbyte>(h.AsSpan());
            var result = new float[n];
            for (int i = 0; i < n; i++) result[i] = (float)h[i / 32 * 17] * bytes[i / 32 * 34 + 2 + i % 32];
            return result;
        }
        return h.Select(v => (float)v).ToArray();
    }

    public void CopyPage(int src, int dst, int slots)
    {
        _acc.Bind();
        for (int il = 0; il < _layers; il++)
        {
            var L = _dl[il];
            if (L.KCache == null) continue;
            long kRow = (long)_c.NHeadKv[il] * _c.HeadDimK[il], vRow = (long)_c.NHeadKv[il] * _c.HeadDimV[il];
            kRow = CacheWords(kRow, _kvQ8);
            vRow = CacheWords(vRow, _kvQ8);
            _kPageCopy(_stream, (int)(slots * kRow), L.KCache.View, (long)src * _pageSize * kRow, (long)dst * _pageSize * kRow);
            _kPageCopy(_stream, (int)(slots * vRow), L.VCache!.View, (long)src * _pageSize * vRow, (long)dst * _pageSize * vRow);
        }
        _stream.Synchronize();
    }

    private readonly List<PinnedHost> _pins = [];
    private static readonly bool s_pinHost = Environment.GetEnvironmentVariable("LAMMA_GPU_PIN") != "0";

    /// <summary>A host range registered with the driver (page-locked) for direct DMA; unregistered on dispose.</summary>
    private sealed class PinnedHost(GpuLayerOffload owner, IDisposable scope) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            lock (owner._pins) owner._pins.Remove(this);
            owner._acc.Bind();
            scope.Dispose();
        }
    }

    public unsafe IDisposable? PinHostMemory(void* pointer, long bytes)
    {
        if (!s_pinHost || _acc.AcceleratorType != AcceleratorType.Cuda) return null;
        try
        {
            _acc.Bind();
            var handle = new PinnedHost(this, _acc.CreatePageLockFromPinned<byte>((IntPtr)pointer, bytes));
            lock (_pins) _pins.Add(handle);
            return handle;
        }
        catch (Exception e) when (e is AcceleratorException or NotSupportedException or ArgumentException or InvalidOperationException)
        {
            Log.PinFailed(_log, e, bytes);
            return null;   // registration is an optimization only
        }
    }

    public void Dispose()
    {
        InvalidateGraphs();
        PinnedHost[] pins;
        lock (_pins) pins = [.. _pins];
        foreach (var p in pins) p.Dispose();   // still registered: the runner outlived us
        foreach (var b in _owned) b.Dispose();
        _stream.Dispose();
        _acc.Dispose();
        _ctx.Dispose();
    }
}
