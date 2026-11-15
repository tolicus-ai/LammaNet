using LammaNet.Chat;
using LammaNet.Compute;
using LammaNet.Gguf;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Multimodal;

/// <summary>Options for multimodal encoders.</summary>
public sealed record ProjectorOptions
{
    /// <summary>
    /// Maximum image tokens per image (lower = faster). Default per projector: Gemma 4 280 (the model's default soft-token
    /// budget), Qwen-VL 1024 or the mmproj's clip.vision.image_max_pixels (llama.cpp: 4096).
    /// </summary>
    public int? MaxImageTokens { get; init; }
    /// <summary>Minimum image tokens per image (small images are upscaled). Default: Gemma 4 70, Qwen-VL 8.</summary>
    public int? MinImageTokens { get; init; }
    /// <summary>Idefics3: longest image edge before tiling (default: the model's preprocessing size; lower = fewer tiles).</summary>
    public int? ImageLongestEdge { get; init; }
    /// <summary>
    /// Device for the vision and audio transformers. <see cref="MultimodalProjector.Load(string, Inference.InferenceEngine, ProjectorOptions?)"/>
    /// uses the engine's GPU offload by default; null runs on the CPU.
    /// </summary>
    public IVisionAccelerator? Accelerator { get; init; }
    /// <summary>Compute threads for the encoders.</summary>
    public int? Threads { get; init; }
    /// <summary>
    /// Logging of the encoders (device fallbacks). <see cref="MultimodalProjector.Load(string, Inference.InferenceEngine, ProjectorOptions?)"/>
    /// uses the engine's by default; otherwise <see cref="Diagnostics.LammaLogging.LoggerFactory"/>.
    /// </summary>
    public Microsoft.Extensions.Logging.ILoggerFactory? LoggerFactory { get; init; }
    /// <summary>
    /// Resource limits of the encoders' compute threads. <see cref="MultimodalProjector.Load(string, Inference.InferenceEngine, ProjectorOptions?)"/>
    /// uses the engine's by default; otherwise <see cref="Compute.ResourceLimiter.Default"/>.
    /// </summary>
    public Compute.ResourceLimiter? Resources { get; init; }
}

/// <summary>
/// Gemma 4 multimodal projector (mmproj GGUF): SigLIP-style vision transformer with 2D RoPE + 3×3 pooling,
/// and a Conformer audio encoder with chunked local attention. Produces embeddings in the LLM's input space,
/// wrapped in the model's begin/end media tokens.
/// </summary>
public sealed unsafe class Gemma4Projector : IMediaEncoder, IDisposable
{
    private readonly GgufFile _f;
    private readonly ComputePool _pool;
    private readonly MatMul _mm;
    private readonly ProjectorOptions _opt;
    private readonly Microsoft.Extensions.Logging.ILogger _log;
    // one encode at a time: the encoders' pool, matmul scratch and device runners are shared (concurrent encodes from a
    // realtime session's transcription and response, or agents sharing a client, corrupted each other's buffers)
    private readonly object _encodeLock = new();
    private bool _disposed;
    private readonly int _llmDim;
    private readonly int _imgBeg = -1, _imgEnd = -1, _audBeg = -1, _audEnd = -1;

    // ---- vision
    private readonly bool _hasVision;
    private readonly int _vDim, _vHeads, _vLayers, _patch = 16, _merge = 3;
    private readonly float _vEps;
    private readonly WeightTensor? _patchEmbd;
    private readonly float[]? _posEmbd; // [2][10240][dim]
    private readonly int _posSize;
    private readonly VLayer[] _vl = [];
    private readonly Linear? _vProj;
    // device runners, created on first use (an image-only conversation never uploads the audio encoder)
    private Lazy<IVitRunner?>? _vDevice;
    private Lazy<IConformerRunner?>? _aDevice;

    // ---- audio
    private readonly bool _hasAudio;
    private readonly int _aDim, _aHeads, _aLayers, _nMel;
    private readonly ALayer[] _al = [];
    private readonly float[]? _conv0Norm, _conv1Norm;
    private readonly WeightTensor? _conv0W, _conv1W;   // [cout][cin·9 (padded to 32)] for im2col
    private readonly int _conv0Out, _conv1Out;
    private readonly Linear? _aInpProj, _aOutProj, _aMmProj;

    private sealed class VLayer
    {
        public float[] Ln1 = [], Ln2 = [], QNorm = [], KNorm = [], AttnPost = [], FfnPost = [];
        public Linear Q = null!, K = null!, V = null!, O = null!, Gate = null!, Up = null!, Down = null!;
    }

    private sealed class ALayer
    {
        public float[] FfnNorm = [], FfnPost = [], FfnNorm1 = [], FfnPost1 = [], AttnPre = [], AttnPost = [], PerDimScale = [], ConvPreNorm = [], ConvMidNorm = [], Ln2 = [], ConvDw = [];
        public Linear Up = null!, Down = null!, Up1 = null!, Down1 = null!, Q = null!, K = null!, V = null!, O = null!, KRel = null!, Pw1 = null!, Pw2 = null!;
    }

    public bool SupportsImages => _hasVision;
    public bool SupportsAudio => _hasAudio;
    /// <summary>Where the vision transformer runs ("CPU" or the accelerator's name).</summary>
    public string VisionDevice => _vDevice?.Value?.DeviceName ?? "CPU";
    /// <summary>Where the audio encoder runs ("CPU" or the accelerator's name).</summary>
    public string AudioDevice => _aDevice?.Value?.DeviceName ?? "CPU";
    /// <summary>Why an encoder fell back from the accelerator to the CPU (e.g. out of device memory), if it did.</summary>
    public string? DeviceError { get; private set; }

    public Gemma4Projector(string mmprojPath, Model llm, ProjectorOptions? options = null)
    {
        _opt = options ?? new ProjectorOptions();
        _log = Diagnostics.LammaLogging.Create<Gemma4Projector>(_opt.LoggerFactory);
        _f = new GgufFile(mmprojPath);
        _pool = new ComputePool(_opt.Threads ?? ComputePool.DefaultThreadCount(), _opt.Resources ?? ResourceLimiter.Default);
        _mm = new MatMul(_pool);
        _llmDim = llm.Config.NEmbd;
        var f = _f;
        string vt = f.Get("clip.vision.projector_type", "");
        string at = f.Get("clip.audio.projector_type", "");
        if (vt != "gemma4v" && at != "gemma4a") throw new NotSupportedException($"Not a Gemma 4 projector (vision={vt}, audio={at})");

        int Tok(string s) => llm.Vocab.Find(s) ?? -1;
        _imgBeg = Tok("<|image>"); _imgEnd = Tok("<image|>");
        _audBeg = Tok("<|audio>"); _audEnd = Tok("<audio|>");

        Linear L(string n) => Linear.Load(f, n) ?? throw new InvalidDataException($"mmproj tensor {n} missing");
        float[] R(string n) => EncoderOps.F(f, n) ?? throw new InvalidDataException($"mmproj tensor {n} missing");

        if (vt == "gemma4v" && f.Get("clip.has_vision_encoder", false))
        {
            _hasVision = true;
            _vDim = f.Get<int>("clip.vision.embedding_length");
            _vHeads = f.Get<int>("clip.vision.attention.head_count");
            _vLayers = f.Get<int>("clip.vision.block_count");
            _vEps = f.Get("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            _patch = f.Get("clip.vision.patch_size", 16);
            _merge = f.Get("clip.vision.projector.scale_factor", 3);
            var pe = f.GetTensor("v.patch_embd.weight");
            // [16,16,3,dim] → matrix [dim rows][3*16*16 cols] in (ci, ky, kx) order
            _patchEmbd = new WeightTensor("v.patch_embd", pe.Type, [pe.Dims[0] * pe.Dims[1] * pe.Dims[2], pe.Dims[3]], f.GetTensorData(pe));
            var pos = f.GetTensor("v.position_embd.weight");
            _posSize = (int)pos.Dims[1];
            _posEmbd = WeightTensor.FromGguf(f, pos).ToFloatArray();
            _vl = new VLayer[_vLayers];
            for (int i = 0; i < _vLayers; i++)
            {
                string p = $"v.blk.{i}.";
                _vl[i] = new VLayer
                {
                    Ln1 = R(p + "ln1.weight"), Ln2 = R(p + "ln2.weight"),
                    QNorm = R(p + "attn_q_norm.weight"), KNorm = R(p + "attn_k_norm.weight"),
                    AttnPost = R(p + "attn_post_norm.weight"), FfnPost = R(p + "ffn_post_norm.weight"),
                    Q = L(p + "attn_q"), K = L(p + "attn_k"), V = L(p + "attn_v"), O = L(p + "attn_out"),
                    Gate = L(p + "ffn_gate"), Up = L(p + "ffn_up"), Down = L(p + "ffn_down"),
                };
            }
            _vProj = L("mm.input_projection");
            if (_opt.Accelerator is { } acc) _vDevice = new(() => acc.CreateVit(VisionSpec()));
        }

        if (at == "gemma4a" && f.Get("clip.has_audio_encoder", false))
        {
            _hasAudio = true;
            _aDim = f.Get<int>("clip.audio.embedding_length");
            _aHeads = f.Get<int>("clip.audio.attention.head_count");
            _aLayers = f.Get<int>("clip.audio.block_count");
            _nMel = f.Get("clip.audio.num_mel_bins", 128);
            _conv0Out = (int)f.GetTensor("a.conv1d.0.weight").Dims[3];
            _conv1Out = (int)f.GetTensor("a.conv1d.1.weight").Dims[3];
            _conv0W = ConvMatrix("a.conv1d.0", R("a.conv1d.0.weight"), _conv0Out);
            _conv1W = ConvMatrix("a.conv1d.1", R("a.conv1d.1.weight"), _conv1Out);
            _conv0Norm = EncoderOps.F(f, "a.conv1d.0.norm.weight"); _conv1Norm = EncoderOps.F(f, "a.conv1d.1.norm.weight");
            _aInpProj = L("a.input_projection");
            _aOutProj = Linear.Load(f, "a.pre_encode.out");
            _aMmProj = Linear.Load(f, "mm.a.input_projection");
            _al = new ALayer[_aLayers];
            for (int i = 0; i < _aLayers; i++)
            {
                string p = $"a.blk.{i}.";
                _al[i] = new ALayer
                {
                    FfnNorm = R(p + "ffn_norm.weight"), FfnPost = R(p + "ffn_post_norm.weight"),
                    FfnNorm1 = R(p + "ffn_norm_1.weight"), FfnPost1 = R(p + "ffn_post_norm_1.weight"),
                    AttnPre = R(p + "attn_pre_norm.weight"), AttnPost = R(p + "attn_post_norm.weight"),
                    PerDimScale = R(p + "per_dim_scale.weight"),
                    // note: conv_norm / norm_conv are swapped in GGUF (llama.cpp loads them reversed)
                    ConvPreNorm = R(p + "conv_norm.weight"), ConvMidNorm = R(p + "norm_conv.weight"),
                    Ln2 = R(p + "ln2.weight"), ConvDw = R(p + "conv_dw.weight"),
                    Up = L(p + "ffn_up"), Down = L(p + "ffn_down"), Up1 = L(p + "ffn_up_1"), Down1 = L(p + "ffn_down_1"),
                    Q = L(p + "attn_q"), K = L(p + "attn_k"), V = L(p + "attn_v"), O = L(p + "attn_out"),
                    KRel = L(p + "attn_k_rel"), Pw1 = L(p + "conv_pw1"), Pw2 = L(p + "conv_pw2"),
                };
            }
            if (_opt.Accelerator is { } acc) _aDevice = new(() => acc.CreateConformer(AudioSpec()));
        }
    }

    /// <summary>The vision transformer for a device: RMS norms, q/k/v head norms, sandwich norms, GeGLU, 2D rope, clamped linears.</summary>
    private VitSpec VisionSpec()
    {
        static VitLinear V(Linear l) => new(l.W, l.Bias) { InMin = l.InMin, InMax = l.InMax, OutMin = l.OutMin, OutMax = l.OutMax };
        var blocks = _vl.Select(L => new VitBlock(L.Ln1, null, L.Ln2, null, V(L.Q), V(L.K), V(L.V), V(L.O), V(L.Up), V(L.Down))
        {
            Gate = V(L.Gate), QNorm = L.QNorm, KNorm = L.KNorm, AttnPostNorm = L.AttnPost, FfnPostNorm = L.FfnPost,
        }).ToList();
        return new VitSpec(_vDim, _vHeads, _vEps, blocks, null, null)
        {
            Norm = VitNorm.RmsNorm, VNorm = true, AttentionScale = 1f, Rope = VitRope.NeoxHalves,
        };
    }

    // audio attention: chunks of C frames attend to the P previous frames; R relative distances P..0
    private const int AudioChunk = 12, AudioLeft = 12, AudioRel = AudioLeft + 1;
    private const float AudioSoftCap = 50f, AudioEps = 1e-6f;
    private float AudioQScale => (float)(1.0 / Math.Sqrt(_aDim / _aHeads) / Math.Log(2.0));
    private static float AudioKScale => (float)(Math.Log(1.0 + Math.E) / Math.Log(2.0));

    /// <summary>Sinusoidal relative position embeddings for distances P..0 ([R][dim]).</summary>
    private static float[] RelativePositions(int d)
    {
        var posEmb = new float[AudioRel * d];
        int half = d / 2;
        double inc = Math.Log(10000.0) / Math.Max(half - 1, 1);
        for (int p = 0; p < AudioRel; p++)
        {
            double position = AudioLeft - p;
            for (int i = 0; i < half; i++)
            {
                double s = position * Math.Exp(-i * inc);
                posEmb[p * d + i] = (float)Math.Sin(s);
                posEmb[p * d + i + half] = (float)Math.Cos(s);
            }
        }
        return posEmb;
    }

    /// <summary>The audio Conformer for a device; the relative-position keys are constant, so they are projected here once.</summary>
    private ConformerSpec AudioSpec()
    {
        static VitLinear V(Linear l) => new(l.W, l.Bias) { InMin = l.InMin, InMax = l.InMax, OutMin = l.OutMin, OutMax = l.OutMax };
        int d = _aDim;
        var posEmb = RelativePositions(d);
        var blocks = _al.Select(L =>
        {
            var relK = new float[AudioRel * d];
            fixed (float* pe = posEmb, rk = relK) L.KRel.Forward(_mm, pe, AudioRel, rk);
            return new ConformerBlock
            {
                FfnNorm = L.FfnNorm, FfnPost = L.FfnPost, FfnNorm1 = L.FfnNorm1, FfnPost1 = L.FfnPost1, AttnPre = L.AttnPre, AttnPost = L.AttnPost,
                PerDimScale = L.PerDimScale, ConvPreNorm = L.ConvPreNorm, ConvMidNorm = L.ConvMidNorm, Ln2 = L.Ln2, ConvDw = L.ConvDw,
                Up = V(L.Up), Down = V(L.Down), Up1 = V(L.Up1), Down1 = V(L.Down1), Q = V(L.Q), K = V(L.K), V = V(L.V), O = V(L.O),
                Pw1 = V(L.Pw1), Pw2 = V(L.Pw2), RelK = relK,
            };
        }).ToList();
        return new ConformerSpec(d, _aHeads, AudioEps, blocks)
        {
            ChunkSize = AudioChunk, LeftContext = AudioLeft, SoftCap = AudioSoftCap, QScale = AudioQScale, KScale = AudioKScale,
        };
    }

    /// <summary>
    /// Runs <paramref name="run"/> on the device runner, if there is one. A device failure (e.g. out of memory for a large
    /// input) disables the runner, is kept in <see cref="DeviceError"/>, and returns false so the caller uses the CPU path;
    /// the runner only writes its input buffer after a complete run.
    /// </summary>
    private bool OnDevice<T>(ref Lazy<T?>? device, Action<T> run) where T : class, IDisposable
    {
        if (device?.Value is not { } runner) return false;
        try
        {
            run(runner);
            return true;
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            DeviceError = $"{typeof(T).Name}: {e.GetType().Name}: {e.Message}";
            string deviceName = runner switch { IVitRunner v => v.DeviceName, IConformerRunner c => c.DeviceName, _ => "the device" };
            Diagnostics.Log.EncoderDeviceFailed(_log, e, runner is IConformerRunner ? "Gemma 4 audio encoder" : "Gemma 4 vision encoder", deviceName);
            device = null;
            try { runner.Dispose(); }
            catch (Exception de) { Diagnostics.Log.EncoderDeviceFailed(_log, de, "disposing the failed encoder", deviceName); }
            return false;
        }
    }

    // ===================================================================== IMediaEncoder

    public bool CanEncode(ChatPart part) => part switch
    {
        ImagePart => _hasVision,
        AudioPart => _hasAudio,
        _ => false,
    };

    public IReadOnlyList<PromptSegment> Encode(ChatPart part)
    {
        switch (part)
        {
            case ImagePart img:
            {
                var rgb = img.EncodedBytes != null ? RgbImage.Decode(img.EncodedBytes) : new RgbImage(img.Width, img.Height, img.Rgb!);
                var (data, rows) = EncodeImage(rgb);
                long hash = ContentHash(img.EncodedBytes ?? img.Rgb!);
                return Wrap(_imgBeg, new EmbeddingSegment(data, rows, _llmDim, hash, "image"), _imgEnd);
            }
            case AudioPart aud:
            {
                float[] pcm;
                if (aud.Samples != null) pcm = aud.SampleRate == 16000 ? aud.Samples : AudioProcessing.Resample(aud.Samples, aud.SampleRate, 16000);
                else
                {
                    var (s, rate) = AudioProcessing.DecodeWav(aud.EncodedBytes!);
                    pcm = AudioProcessing.Resample(s, rate, 16000);
                }
                var (data, rows) = EncodeAudio(pcm);
                long hash = ContentHash(System.Runtime.InteropServices.MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray());
                return Wrap(_audBeg, new EmbeddingSegment(data, rows, _llmDim, hash, "audio"), _audEnd);
            }
            default: throw new NotSupportedException();
        }
    }

    private static IReadOnlyList<PromptSegment> Wrap(int beg, EmbeddingSegment emb, int end)
    {
        var list = new List<PromptSegment>();
        if (beg >= 0) list.Add(new TokenSegment([beg]));
        list.Add(emb);
        if (end >= 0) list.Add(new TokenSegment([end]));
        return list;
    }

    /// <summary>FNV-1a 64-bit content hash (identifies media for prompt-prefix caching).</summary>
    private static long ContentHash(byte[] data)
    {
        ulong h = 14695981039346656037UL;
        foreach (byte b in data) { h ^= b; h *= 1099511628211UL; }
        return (long)h;
    }

    // ===================================================================== vision

    /// <summary>Computes the resized image dimensions (multiples of patch×merge) within the token budget.</summary>
    public (int W, int H) TargetSize(int width, int height)
    {
        int align = _patch * _merge;
        long patchArea = (long)align * align;
        long maxPx = (_opt.MaxImageTokens ?? 280) * patchArea, minPx = (_opt.MinImageTokens ?? 70) * patchArea;
        int Round(double x) => (int)Math.Round(x / align) * align;
        int Ceil(double x) => (int)Math.Ceiling(x / align) * align;
        int Floor(double x) => (int)Math.Floor(x / align) * align;
        int w = Math.Max(align, Round(width)), h = Math.Max(align, Round(height));
        if ((long)w * h > maxPx)
        {
            double beta = Math.Sqrt((double)height * width / maxPx);
            h = Math.Max(align, Floor(height / beta));
            w = Math.Max(align, Floor(width / beta));
        }
        else if ((long)w * h < minPx)
        {
            double beta = Math.Sqrt(minPx / ((double)height * width));
            h = Ceil(height * beta);
            w = Ceil(width * beta);
        }
        return (w, h);
    }

    /// <summary>Encodes an image into LLM-space embeddings [tokens × llmDim].</summary>
    public (float[] Data, int Tokens) EncodeImage(RgbImage image)
    {
        lock (_encodeLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return EncodeImageCore(image);
        }
    }

    private (float[] Data, int Tokens) EncodeImageCore(RgbImage image)
    {
        if (!_hasVision) throw new NotSupportedException("This projector has no vision encoder");
        var (tw, th) = TargetSize(image.Width, image.Height);
        RgbImage img;
        using (Profiler.Section("vision.preprocess"))
            img = image.ResizePilPadded(tw, th, ResampleFilter.Bicubic); // as llama.cpp mtmd (Pillow bicubic, aspect kept, centered)
        int nx = tw / _patch, ny = th / _patch, n = nx * ny;
        int d = _vDim, pp = 3 * _patch * _patch;

        using var patches = new NativeBuffer((long)n * pp * 4);
        float* pt = patches.F;
        for (int py = 0; py < ny; py++)
            for (int px = 0; px < nx; px++)
            {
                float* row = pt + (long)(py * nx + px) * pp;
                for (int c = 0; c < 3; c++)
                    for (int ky = 0; ky < _patch; ky++)
                        for (int kx = 0; kx < _patch; kx++)
                        {
                            int ix = px * _patch + kx, iy = py * _patch + ky;
                            float pv = img.Pixels[(iy * tw + ix) * 3 + c] / 255f;
                            row[(c * _patch + ky) * _patch + kx] = pv * 2f - 1f;
                        }
            }

        using var xb = new NativeBuffer((long)n * d * 4);
        float* x = xb.F;
        using (Profiler.Section("vision.patch_embd"))
        {
            _mm.Multiply(_patchEmbd!, pt, n, x, pp, d);
            fixed (float* pe = _posEmbd)
                for (int i = 0; i < n; i++)
                {
                    int pxi = i % nx, pyi = i / nx;
                    Ops.Add(x + (long)i * d, pe + (long)pxi * d, x + (long)i * d, d);
                    Ops.Add(x + (long)i * d, pe + ((long)_posSize + pyi) * d, x + (long)i * d, d);
                }
        }

        int dh = d / _vHeads, ff = _vl[0].Up.Out;
        using var cur = new NativeBuffer((long)n * d * 4);
        using var q = new NativeBuffer((long)n * d * 4);
        using var k = new NativeBuffer((long)n * d * 4);
        using var v = new NativeBuffer((long)n * d * 4);
        using var att = new NativeBuffer((long)n * d * 4);
        using var g = new NativeBuffer((long)n * ff * 4);
        using var u = new NativeBuffer((long)n * ff * 4);
        var rope = new RopeConfig { Mode = RopeMode.Neox, Dims = dh / 2, FreqBase = 100f };
        var ropeX = new float[n * (dh / 2)];
        var ropeY = new float[n * (dh / 2)];
        fixed (float* rx = ropeX, ry = ropeY)
            for (int i = 0; i < n; i++)
            {
                rope.BuildCache([i % nx], rx + (long)i * (dh / 2));
                rope.BuildCache([i / nx], ry + (long)i * (dh / 2));
            }

        // per patch: the x-position pairs (first half of each head), then the y-position pairs (second half)
        var table = _vDevice == null ? null : new float[(long)n * dh];
        for (int i = 0; table != null && i < n; i++)
        {
            Array.Copy(ropeX, (long)i * (dh / 2), table, (long)i * dh, dh / 2);
            Array.Copy(ropeY, (long)i * (dh / 2), table, (long)i * dh + dh / 2, dh / 2);
        }
        bool onDevice;
        using (Profiler.Section("vision.device"))
            onDevice = OnDevice(ref _vDevice, dev => { fixed (float* tp = table) dev.Run(x, n, n, tp); });
        if (!onDevice) foreach (var L in _vl)
        {
            EncoderOps.RmsNormRows(x, L.Ln1, cur.F, n, d, _vEps, _mm.Pool);
            L.Q.Forward(_mm, cur.F, n, q.F);
            L.K.Forward(_mm, cur.F, n, k.F);
            L.V.Forward(_mm, cur.F, n, v.F);
            float* qp = q.F, kp = k.F, vp = v.F;
            int heads = _vHeads;
            fixed (float* qn = L.QNorm, kn = L.KNorm, rx = ropeX, ry = ropeY)
            {
                float* qnp = qn, knp = kn, rxp = rx, ryp = ry;
                float eps = _vEps;
                EncoderOps.For(_mm.Pool, n, i =>
                {
                    for (int h = 0; h < heads; h++)
                    {
                        float* qh = qp + (long)i * d + h * dh, kh = kp + (long)i * d + h * dh, vh = vp + (long)i * d + h * dh;
                        Ops.RmsNorm(qh, qnp, qh, dh, eps);
                        Ops.RmsNorm(kh, knp, kh, dh, eps);
                        Ops.RmsNorm(vh, null, vh, dh, eps);
                        // 2D rope: first half rotated with x position, second half with y position (neox within each half)
                        rope.Apply(qh, 1, dh / 2, rxp + (long)i * (dh / 2));
                        rope.Apply(kh, 1, dh / 2, rxp + (long)i * (dh / 2));
                        rope.Apply(qh + dh / 2, 1, dh / 2, ryp + (long)i * (dh / 2));
                        rope.Apply(kh + dh / 2, 1, dh / 2, ryp + (long)i * (dh / 2));
                    }
                });
            }
            EncoderOps.Attention(q.F, k.F, v.F, att.F, n, _vHeads, dh, 1.0f, _mm);
            L.O.Forward(_mm, att.F, n, cur.F);
            EncoderOps.RmsNormRows(cur.F, L.AttnPost, cur.F, n, d, _vEps, _mm.Pool);
            Ops.Add(x, cur.F, x, n * d);

            EncoderOps.RmsNormRows(x, L.Ln2, cur.F, n, d, _vEps, _mm.Pool);
            L.Gate.Forward(_mm, cur.F, n, g.F);
            L.Up.Forward(_mm, cur.F, n, u.F);
            Simd.GeGlu(g.F, u.F, u.F, n * ff);
            L.Down.Forward(_mm, u.F, n, cur.F);
            EncoderOps.RmsNormRows(cur.F, L.FfnPost, cur.F, n, d, _vEps, _mm.Pool);
            Ops.Add(x, cur.F, x, n * d);
        }

        using var _proj = Profiler.Section("vision.projection");
        // 3×3 average pooling over the patch grid
        int ox = nx / _merge, oy = ny / _merge, tokens = ox * oy;
        using var pooled = new NativeBuffer((long)tokens * d * 4);
        float sc = MathF.Sqrt(d) / (_merge * _merge);
        for (int ty = 0; ty < oy; ty++)
            for (int tx = 0; tx < ox; tx++)
            {
                float* o = pooled.F + (long)(ty * ox + tx) * d;
                for (int dy = 0; dy < _merge; dy++)
                    for (int dx = 0; dx < _merge; dx++)
                        Ops.Add(o, x + (long)((ty * _merge + dy) * nx + tx * _merge + dx) * d, o, d);
                Ops.Scale(o, sc, o, d);
            }
        EncoderOps.RmsNormRows(pooled.F, null, pooled.F, tokens, d, _vEps, _mm.Pool);
        var result = new float[tokens * _llmDim];
        fixed (float* r = result) _vProj!.Forward(_mm, pooled.F, tokens, r);
        return (result, tokens);
    }

    // ===================================================================== audio

    /// <summary>Encodes 16 kHz mono PCM into LLM-space embeddings [tokens × llmDim] (one token per 40 ms).</summary>
    public (float[] Data, int Tokens) EncodeAudio(float[] pcm16k)
    {
        lock (_encodeLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return EncodeAudioCore(pcm16k);
        }
    }

    private (float[] Data, int Tokens) EncodeAudioCore(float[] pcm16k)
    {
        if (!_hasAudio) throw new NotSupportedException("This projector has no audio encoder");
        float[] mel;
        int frames;
        using (Profiler.Section("audio.features")) (mel, frames) = AudioProcessing.Gemma4LogMel(pcm16k, _nMel);
        if (frames < 4) throw new ArgumentException("Audio clip is too short");

        // ---- subsampling: two stride-2 3×3 convs over (time, freq), LayerNorm over channels, ReLU.
        // Activations are [time][freq][channel], so the second conv's output is already the per-time-step flattening
        // (channel fastest, then freq) the input projection expects.
        var subsample = Profiler.Section("audio.subsample");
        var (c0, t0, f0) = Conv2dS2(mel, 1, frames, _nMel, melInput: true, _conv0W!, _conv0Out);
        ChannelNormRelu(c0, _conv0Out, t0 * f0, _conv0Norm);
        var (c1, t1, f1) = Conv2dS2(c0, _conv0Out, t0, f0, melInput: false, _conv1W!, _conv1Out);
        ChannelNormRelu(c1, _conv1Out, t1 * f1, _conv1Norm);
        subsample.Dispose();
        int n = t1;

        int d = _aDim;
        using var xb = new NativeBuffer((long)n * d * 4);
        float* x = xb.F;
        fixed (float* flat = c1) _aInpProj!.Forward(_mm, flat, n, x);
        bool onDevice;
        using (Profiler.Section("audio.device")) onDevice = OnDevice(ref _aDevice, dev => dev.Run(x, n));
        if (!onDevice) ConformerBlocks(x, n);

        const float eps = AudioEps;
        int outDim = _aOutProj?.Out ?? d;
        using var proj = new NativeBuffer((long)n * outDim * 4);
        if (_aOutProj != null) _aOutProj.Forward(_mm, x, n, proj.F);
        else Buffer.MemoryCopy(x, proj.F, (long)n * d * 4, (long)n * d * 4);
        EncoderOps.RmsNormRows(proj.F, null, proj.F, n, outDim, eps, _mm.Pool);
        var result = new float[n * _llmDim];
        fixed (float* r = result)
        {
            if (_aMmProj != null) _aMmProj.Forward(_mm, proj.F, n, r);
            else Buffer.MemoryCopy(proj.F, r, (long)n * outDim * 4, (long)n * outDim * 4);
        }
        return (result, n);
    }

    /// <summary>The Conformer blocks on the CPU: x [n, dim] in place.</summary>
    private void ConformerBlocks(float* x, int n)
    {
        int d = _aDim;
        const float eps = AudioEps;
        int dh = d / _aHeads, ff = _al[0].Up.Out;
        using var cur = new NativeBuffer((long)n * Math.Max(2 * d, ff) * 4);
        using var tmp = new NativeBuffer((long)n * Math.Max(2 * d, ff) * 4);
        using var q = new NativeBuffer((long)n * d * 4);
        using var k = new NativeBuffer((long)n * d * 4);
        using var v = new NativeBuffer((long)n * d * 4);
        using var att = new NativeBuffer((long)n * d * 4);

        const int C = AudioChunk, P = AudioLeft, S = C + P, R = AudioRel;
        var posEmb = RelativePositions(d);
        using var relK = new NativeBuffer((long)R * d * 4);
        float qScale = AudioQScale, kScale = AudioKScale;
        const float softcap = AudioSoftCap;

        foreach (var L in _al)
        {
            // FFN 1 (half-step)
            HalfFfn(x, n, d, L.FfnNorm, L.Up, L.Down, L.FfnPost, cur.F, tmp.F, eps);

            // chunked local self-attention with relative positions
            EncoderOps.RmsNormRows(x, L.AttnPre, cur.F, n, d, eps, _mm.Pool);
            L.Q.Forward(_mm, cur.F, n, q.F);
            L.K.Forward(_mm, cur.F, n, k.F);
            L.V.Forward(_mm, cur.F, n, v.F);
            fixed (float* pe = posEmb) L.KRel.Forward(_mm, pe, R, relK.F);
            float* qp = q.F, kp = k.F, vp = v.F, ap = att.F, rk = relK.F;
            int heads = _aHeads;
            fixed (float* pds = L.PerDimScale)
            {
                float* pdsp = pds;
                for (int i = 0; i < n; i++)
                    for (int h = 0; h < heads; h++)
                    {
                        float* qh = qp + (long)i * d + h * dh;
                        Ops.Scale(qh, qScale, qh, dh);
                        Ops.Mul(qh, pdsp, qh, dh);
                        Ops.Scale(kp + (long)i * d + h * dh, kScale, kp + (long)i * d + h * dh, dh);
                    }
            }
            var attnSection = Profiler.Section("audio.attention");
            EncoderOps.For(_mm.Pool, n * heads, task =>
            {
                int h = task % heads, gq = task / heads;
                int b = gq / C, c = gq % C;
                float* qh = qp + (long)gq * d + h * dh;
                Span<float> sc = stackalloc float[S];
                Span<bool> valid = stackalloc bool[S];
                float mx = float.NegativeInfinity;
                for (int s = 0; s < S; s++)
                {
                    int gk = b * C - P + s;
                    bool ok = gk >= 0 && gk < n && gk <= gq && (gq - gk) < P;
                    float score = 0;
                    if (gk >= 0 && gk < n) score = Simd.Dot(qh, kp + (long)gk * d + h * dh, dh);
                    int r = s - c; // relative-position term (Transformer-XL shift)
                    if (r >= 0 && r < R) score += Simd.Dot(qh, rk + (long)r * d + h * dh, dh);
                    score = MathF.Tanh(score / softcap) * softcap;
                    if (!ok) score += -1e9f;
                    sc[s] = score; valid[s] = ok;
                    if (score > mx) mx = score;
                }
                float sum = 0;
                for (int s = 0; s < S; s++) { sc[s] = MathF.Exp(sc[s] - mx); sum += sc[s]; }
                float* o = ap + (long)gq * d + h * dh;
                new Span<float>(o, dh).Clear();
                for (int s = 0; s < S; s++)
                {
                    int gk = b * C - P + s;
                    if (gk < 0 || gk >= n || sc[s] == 0) continue;
                    Simd.AddScaled(vp + (long)gk * d + h * dh, sc[s] / sum, o, dh);
                }
            });
            attnSection.Dispose();
            L.O.Forward(_mm, att.F, n, cur.F);
            EncoderOps.RmsNormRows(cur.F, L.AttnPost, cur.F, n, d, eps, _mm.Pool);
            Ops.Add(x, cur.F, x, n * d);

            // convolution module: norm → pw1 → GLU → causal depthwise conv(5) → norm → silu → pw2
            EncoderOps.RmsNormRows(x, L.ConvPreNorm, cur.F, n, d, eps, _mm.Pool);
            L.Pw1.Forward(_mm, cur.F, n, tmp.F); // [n, 2d]
            float* glu = cur.F;
            for (int t = 0; t < n; t++)
            {
                float* src = tmp.F + (long)t * 2 * d;
                float* dst = glu + (long)t * d;
                for (int i = 0; i < d; i++) dst[i] = src[i] / (1f + MathF.Exp(-src[d + i]));
            }
            var convSection = Profiler.Section("audio.conv");
            int kz = L.ConvDw.Length / d;
            fixed (float* w = L.ConvDw)
            {
                float* wp = w, tp = tmp.F;
                EncoderOps.For(_mm.Pool, n, t =>
                {
                    float* o = tp + (long)t * d;
                    for (int ch = 0; ch < d; ch++)
                    {
                        float acc = 0;
                        for (int j = 0; j < kz; j++)
                        {
                            int ts = t - (kz - 1) + j;
                            if (ts >= 0) acc += glu[(long)ts * d + ch] * wp[ch * kz + j];
                        }
                        o[ch] = acc;
                    }
                });
            }
            convSection.Dispose();
            EncoderOps.RmsNormRows(tmp.F, L.ConvMidNorm, tmp.F, n, d, eps, _mm.Pool);
            Simd.Silu(tmp.F, tmp.F, n * d);
            L.Pw2.Forward(_mm, tmp.F, n, cur.F);
            Ops.Add(x, cur.F, x, n * d);

            // FFN 2 (half-step)
            HalfFfn(x, n, d, L.FfnNorm1, L.Up1, L.Down1, L.FfnPost1, cur.F, tmp.F, eps);

            EncoderOps.RmsNormRows(x, L.Ln2, x, n, d, eps, _mm.Pool);
        }
    }

    private void HalfFfn(float* x, int n, int d, float[] norm, Linear up, Linear down, float[] post, float* cur, float* tmp, float eps)
    {
        EncoderOps.RmsNormRows(x, norm, cur, n, d, eps, _mm.Pool);
        up.Forward(_mm, cur, n, tmp);
        Simd.Silu(tmp, tmp, n * up.Out);
        down.Forward(_mm, tmp, n, cur);
        EncoderOps.RmsNormRows(cur, post, cur, n, d, eps, _mm.Pool);
        Ops.AddScaled(cur, 0.5f, x, n * d);
    }

    /// <summary>A 3×3 conv kernel (ggml [kw=freq][kh=time][cin][cout]) as a GEMM matrix [cout][cin·9], columns padded to 32.</summary>
    private static WeightTensor ConvMatrix(string name, float[] w, int cout)
    {
        int k = w.Length / cout, kp = (k + 31) / 32 * 32;
        var buf = new NativeBuffer((long)cout * kp * 4);
        for (int co = 0; co < cout; co++) w.AsSpan(co * k, k).CopyTo(new Span<float>(buf.F + (long)co * kp, k));
        return new WeightTensor(name, GgmlType.F32, [kp, cout], buf.B, buf);
    }

    /// <summary>
    /// 3×3 conv, stride 2, padding 1 over (time, freq) as im2col + GEMM. Output layout [time][freq][cout].
    /// Input: the log-mel spectrogram [freq][time] (<paramref name="melInput"/>) or a previous conv output [time][freq][cin].
    /// </summary>
    private (float[] Out, int T, int F) Conv2dS2(float[] input, int cin, int tIn, int fIn, bool melInput, WeightTensor w, int cout)
    {
        int tOut = (tIn - 1) / 2 + 1, fOut = (fIn - 1) / 2 + 1, positions = tOut * fOut, k = w.Cols;
        var output = new float[(long)positions * cout];
        const int Chunk = 4096;
        using var cols = new NativeBuffer((long)Math.Min(Chunk, positions) * k * 4);
        fixed (float* inp = input, outp = output)
        {
            float* ip = inp, cp = cols.F;
            for (int p0 = 0; p0 < positions; p0 += Chunk)
            {
                int np = Math.Min(Chunk, positions - p0), first = p0;
                EncoderOps.For(_mm.Pool, np, i =>
                {
                    int pos = first + i, t = pos / fOut, f = pos % fOut;
                    float* row = cp + (long)i * k;
                    new Span<float>(row, k).Clear();
                    for (int ky = 0; ky < 3; ky++) // time
                    {
                        int ti = t * 2 - 1 + ky;
                        if (ti < 0 || ti >= tIn) continue;
                        for (int kx = 0; kx < 3; kx++) // freq
                        {
                            int fi = f * 2 - 1 + kx;
                            if (fi < 0 || fi >= fIn) continue;
                            if (melInput) row[ky * 3 + kx] = ip[(long)fi * tIn + ti];
                            else
                            {
                                float* src = ip + ((long)ti * fIn + fi) * cin;
                                for (int ci = 0; ci < cin; ci++) row[ci * 9 + ky * 3 + kx] = src[ci];
                            }
                        }
                    }
                });
                _mm.Multiply(w, cp, np, outp + (long)p0 * cout);
            }
        }
        return (output, tOut, fOut);
    }

    /// <summary>LayerNorm across channels at each position (no bias), then ReLU. Layout [pos][channel].</summary>
    private void ChannelNormRelu(float[] x, int channels, int positions, float[]? w)
    {
        fixed (float* xp0 = x)
        {
            float* xp = xp0;
            EncoderOps.For(_mm.Pool, positions, p =>
            {
                float* v = xp + (long)p * channels;
                double mean = 0;
                for (int c = 0; c < channels; c++) mean += v[c];
                mean /= channels;
                double var = 0;
                for (int c = 0; c < channels; c++) { double dlt = v[c] - mean; var += dlt * dlt; }
                var /= channels;
                float inv = (float)(1 / Math.Sqrt(var + 1e-6));
                for (int c = 0; c < channels; c++)
                {
                    float o = (float)(v[c] - mean) * inv;
                    if (w != null) o *= w[c];
                    v[c] = MathF.Max(0, o);
                }
            });
        }
    }

    public void Dispose()
    {
        lock (_encodeLock)   // waits for an encode in progress
        {
            if (_disposed) return;
            _disposed = true;
            if (_vDevice is { IsValueCreated: true }) _vDevice.Value?.Dispose();
            if (_aDevice is { IsValueCreated: true }) _aDevice.Value?.Dispose();
            _pool.Dispose();
            _f.Dispose();
        }
    }
}
