using LammaNet.Chat;
using LammaNet.Compute;
using LammaNet.Gguf;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Multimodal;

/// <summary>
/// Qwen3-VL style vision encoder ("qwen3vl_merger" mmproj), used by Qwen3.5 / Qwen3.6 multimodal models:
/// temporal-pair patch embedding, bilinearly resized learned position embeddings, ViT with 2D rotary embeddings,
/// and a 2×2 patch-merger MLP. Image tokens carry 2D M-RoPE positions (t, row, col) for the language model.
/// </summary>
public sealed unsafe class Qwen3VLProjector : IMediaEncoder, IDisposable
{
    private readonly GgufFile _f;
    private readonly ComputePool _pool;
    private readonly MatMul _mm;
    private readonly ProjectorOptions _opt;
    private readonly int _llmDim, _dim, _heads, _layers, _patch, _merge, _posSide;
    private readonly float _eps;
    private readonly float[] _mean, _std;
    private readonly WeightTensor _patchW;
    private readonly float[] _patchB, _pos, _postLnW, _postLnB;
    private readonly Linear _mm0, _mm2;
    private readonly Layer[] _l;
    private readonly int _beg, _end;
    private readonly long _minPixels, _maxPixels;
    private IVitRunner? _device;
    private readonly Microsoft.Extensions.Logging.ILogger _log;
    // one encode at a time: the pool, matmul scratch and device runner are shared
    private readonly object _encodeLock = new();
    private bool _disposed;

    private sealed class Layer
    {
        public float[] Ln1W = [], Ln1B = [], Ln2W = [], Ln2B = [];
        public Linear Qkv = null!, O = null!, Up = null!, Down = null!;
    }

    public Qwen3VLProjector(string mmprojPath, Model llm, ProjectorOptions? options = null)
    {
        _opt = options ?? new ProjectorOptions();
        _log = Diagnostics.LammaLogging.Create<Qwen3VLProjector>(_opt.LoggerFactory);
        _f = new GgufFile(mmprojPath);
        var f = _f;
        string type = f.Get("clip.projector_type", f.Get("clip.vision.projector_type", ""));
        if (type != "qwen3vl_merger") throw new NotSupportedException($"Not a Qwen3-VL projector ({type})");
        if (f.Get<bool[]>("clip.vision.is_deepstack_layers", []).Any(b => b))
            throw new NotSupportedException("Deepstack vision layers are not supported yet");
        _pool = new ComputePool(_opt.Threads ?? ComputePool.DefaultThreadCount(), _opt.Resources ?? ResourceLimiter.Default);
        _mm = new MatMul(_pool);
        _llmDim = llm.Config.NEmbd;
        _dim = f.Get<int>("clip.vision.embedding_length");
        _heads = f.Get<int>("clip.vision.attention.head_count");
        _layers = f.Get<int>("clip.vision.block_count");
        _patch = f.Get("clip.vision.patch_size", 16);
        _merge = f.Get("clip.vision.spatial_merge_size", 2);
        _eps = f.Get("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        // token limits: 8..1024 merged tokens unless the mmproj specifies pixel limits (llama.cpp allows up to 4096, which
        // fills a 4K context and costs minutes of CPU encoding for a phone photo; raise MaxImageTokens for fine detail)
        long tokenArea = (long)_patch * _merge * _patch * _merge;
        _minPixels = f.Get("clip.vision.image_min_pixels", 0) is int minP and > 0 ? minP : 8 * tokenArea;
        _maxPixels = f.Get("clip.vision.image_max_pixels", 0) is int maxP and > 0 ? maxP : 1024 * tokenArea;
        _mean = f.Get("clip.vision.image_mean", new[] { 0.5f, 0.5f, 0.5f });
        _std = f.Get("clip.vision.image_std", new[] { 0.5f, 0.5f, 0.5f });

        float[] R(string n) => EncoderOps.F(f, n) ?? throw new InvalidDataException($"mmproj tensor {n} missing");
        Linear L(string n) => Linear.Load(f, n) ?? throw new InvalidDataException($"mmproj tensor {n} missing");

        // temporal patch pair: a still image is fed as two identical frames -> the two kernels can be summed
        var w0 = WeightTensor.FromGguf(f, f.GetTensor("v.patch_embd.weight")).ToFloatArray();
        var w1 = WeightTensor.FromGguf(f, f.GetTensor("v.patch_embd.weight.1")).ToFloatArray();
        for (int i = 0; i < w0.Length; i++) w0[i] += w1[i];
        int pp = 3 * _patch * _patch;
        var buf = new NativeBuffer((long)w0.Length * 4);
        w0.AsSpan().CopyTo(buf.AsFloatSpan());
        _patchW = new WeightTensor("v.patch_embd", GgmlType.F32, [pp, _dim], buf.B, buf);
        _patchB = R("v.patch_embd.bias");
        _pos = R("v.position_embd.weight");
        _posSide = (int)Math.Sqrt(_pos.Length / _dim);
        _postLnW = R("v.post_ln.weight"); _postLnB = R("v.post_ln.bias");
        _mm0 = L("mm.0"); _mm2 = L("mm.2");
        _l = new Layer[_layers];
        for (int i = 0; i < _layers; i++)
        {
            string p = $"v.blk.{i}.";
            _l[i] = new Layer
            {
                Ln1W = R(p + "ln1.weight"), Ln1B = R(p + "ln1.bias"), Ln2W = R(p + "ln2.weight"), Ln2B = R(p + "ln2.bias"),
                Qkv = L(p + "attn_qkv"), O = L(p + "attn_out"), Up = L(p + "ffn_up"), Down = L(p + "ffn_down"),
            };
        }
        _beg = llm.Vocab.Find("<|vision_start|>") ?? -1;
        _end = llm.Vocab.Find("<|vision_end|>") ?? -1;
        if (_opt.Accelerator != null) _device = _opt.Accelerator.CreateVit(DeviceSpec());
    }

    /// <summary>Where the vision transformer runs ("CPU" or the accelerator's name).</summary>
    public string EncoderDevice => _device?.DeviceName ?? "CPU";

    /// <summary>The ViT for a device: LayerNorm blocks with the fused QKV split into row views, 2D rope, final LayerNorm.</summary>
    internal VitSpec DeviceSpec()
    {
        int d = _dim;
        VitLinear Rows(Linear l, int part) => new(
            new WeightTensor(l.W.Name + "." + part, l.W.Type, [l.W.Cols, d], l.W.Data + (long)part * d * l.W.RowBytes),
            l.Bias?[(part * d)..((part + 1) * d)]);
        static VitLinear V(Linear l) => new(l.W, l.Bias);
        var blocks = _l.Select(L => new VitBlock(L.Ln1W, L.Ln1B, L.Ln2W, L.Ln2B, Rows(L.Qkv, 0), Rows(L.Qkv, 1), Rows(L.Qkv, 2), V(L.O), V(L.Up), V(L.Down))).ToList();
        return new VitSpec(d, _heads, _eps, blocks, _postLnW, _postLnB)
        {
            Rope = VitRope.Neox,
            Merger = new VitMerger(_merge, [V(_mm0), V(_mm2)]),
            PatchEmbedding = new VitPatchEmbedding(_patch, _merge, new VitLinear(_patchW, _patchB), _mean, _std, _pos, _posSide),
        };
    }

    public bool CanEncode(ChatPart part) => part is ImagePart;

    public IReadOnlyList<PromptSegment> Encode(ChatPart part)
    {
        var img = (ImagePart)part;
        var rgb = img.EncodedBytes != null ? RgbImage.Decode(img.EncodedBytes) : new RgbImage(img.Width, img.Height, img.Rgb!);
        var (data, gw, gh) = EncodeImage(rgb);
        int n = gw * gh;
        var rope = new int[n][];
        for (int i = 0; i < n; i++) rope[i] = [0, i / gw, i % gw, 0];
        long hash = (long)Fnv(img.EncodedBytes ?? img.Rgb!);
        var seg = new EmbeddingSegment(data, n, _llmDim, hash, "image") { RopePositions = rope, RopeSpan = Math.Max(gw, gh) };
        var list = new List<PromptSegment>();
        if (_beg >= 0) list.Add(new TokenSegment([_beg]));
        list.Add(seg);
        if (_end >= 0) list.Add(new TokenSegment([_end]));
        return list;
    }

    private static ulong Fnv(byte[] d)
    {
        ulong h = 14695981039346656037UL;
        foreach (var b in d) { h ^= b; h *= 1099511628211UL; }
        return h;
    }

    /// <summary>Encodes an image; returns LLM-space embeddings and the merged token grid (width, height).</summary>
    public (float[] Data, int GridW, int GridH) EncodeImage(RgbImage image)
    {
        lock (_encodeLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return EncodeImageCore(image);
        }
    }

    /// <summary>The image resized for the encoder (sides multiples of patch·merge, token budget, Pillow bicubic, aspect kept, centered).</summary>
    internal RgbImage PrepareImage(RgbImage image)
    {
        int align = _patch * _merge;
        long area = (long)align * align;
        long maxPx = _opt.MaxImageTokens is int mx ? mx * area : _maxPixels, minPx = _opt.MinImageTokens is int mn ? mn * area : _minPixels;
        int Round(double x) => (int)Math.Round(x / align) * align;
        int Ceil(double x) => (int)Math.Ceiling(x / align) * align;
        int Floor(double x) => (int)Math.Floor(x / align) * align;
        int tw = Math.Max(align, Round(image.Width)), th = Math.Max(align, Round(image.Height));
        if ((long)tw * th > maxPx)
        {
            double beta = Math.Sqrt((double)image.Height * image.Width / maxPx);
            th = Math.Max(align, Floor(image.Height / beta)); tw = Math.Max(align, Floor(image.Width / beta));
        }
        else if ((long)tw * th < minPx)
        {
            double beta = Math.Sqrt(minPx / ((double)image.Height * image.Width));
            th = Ceil(image.Height * beta); tw = Ceil(image.Width * beta);
        }
        using (Profiler.Section("vision.preprocess"))
            return image.ResizePilPadded(tw, th, ResampleFilter.Bicubic); // as llama.cpp mtmd (Pillow bicubic, aspect kept, centered)
    }

    /// <summary>
    /// Patch embedding of a prepared image on the CPU: <paramref name="x"/> [patches, dim] in merge-block order (+ bias,
    /// + resized positions); returns the 2D rotary table [patches, head_dim].
    /// </summary>
    internal float[] EmbedOnCpu(RgbImage img, float* x)
    {
        int tw = img.Width, th = img.Height;
        int pw = tw / _patch, ph = th / _patch, n = pw * ph, d = _dim, pp = 3 * _patch * _patch;
        // patch order: merge blocks (by, bx), then (dy, dx) inside each 2×2 block
        var order = new (int Y, int X)[n];
        int k = 0;
        for (int by = 0; by < ph / _merge; by++)
            for (int bx = 0; bx < pw / _merge; bx++)
                for (int dy = 0; dy < _merge; dy++)
                    for (int dx = 0; dx < _merge; dx++)
                        order[k++] = (by * _merge + dy, bx * _merge + dx);

        using var patches = new NativeBuffer((long)n * pp * 4);
        for (int i = 0; i < n; i++)
        {
            var (py, px) = order[i];
            float* row = patches.F + (long)i * pp;
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < _patch; ky++)
                    for (int kx = 0; kx < _patch; kx++)
                    {
                        float v = img.Pixels[((py * _patch + ky) * tw + px * _patch + kx) * 3 + c] / 255f;
                        row[(c * _patch + ky) * _patch + kx] = (v - _mean[c]) / _std[c];
                    }
        }
        _mm.Multiply(_patchW, patches.F, n, x, pp, d);
        var posGrid = ResizePositions(pw, ph);
        fixed (float* b = _patchB, pg = posGrid)
            for (int i = 0; i < n; i++)
            {
                var (py, px) = order[i];
                Ops.Add(x + (long)i * d, b, x + (long)i * d, d);
                Ops.Add(x + (long)i * d, pg + ((long)py * pw + px) * d, x + (long)i * d, d);
            }

        // 2D rope tables: first half of the rotated pairs uses the row, second half the column
        int dh = d / _heads, half = dh / 2;
        var cache = new float[n * dh];
        double scaleBase = Math.Pow(10000.0, -2.0 / half);
        for (int i = 0; i < n; i++)
        {
            var (py, px) = order[i];
            for (int s = 0; s < half; s++)
            {
                int sec = s < half / 2 ? s : s - half / 2;
                double theta = (s < half / 2 ? py : px) * Math.Pow(scaleBase, sec);
                cache[i * dh + 2 * s] = (float)Math.Cos(theta);
                cache[i * dh + 2 * s + 1] = (float)Math.Sin(theta);
            }
        }

        return cache;
    }

    private (float[] Data, int GridW, int GridH) EncodeImageCore(RgbImage image)
    {
        var img = PrepareImage(image);
        int tw = img.Width, th = img.Height;
        int pw = tw / _patch, ph = th / _patch, n = pw * ph, d = _dim;
        if (_device is { } dev)
        {
            // the whole image on the device: patches, embedding, positions, rotary angles, blocks and merger
            var result = new float[n / (_merge * _merge) * _llmDim];
            try
            {
                using (Profiler.Section("vision.device"))
                    fixed (byte* px = img.Pixels) fixed (float* r = result) dev.RunImage(px, tw, th, r);
                return (result, pw / _merge, ph / _merge);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                // e.g. out of device memory for a large image: this and later images use the CPU
                Diagnostics.Log.EncoderDeviceFailed(_log, e, "Qwen-VL vision encoder", dev.DeviceName);
                _device = null;
                try { dev.Dispose(); }
                catch (Exception de) { Diagnostics.Log.EncoderDeviceFailed(_log, de, "disposing the failed encoder", dev.DeviceName); }
            }
        }

        using var xb = new NativeBuffer((long)n * d * 4);
        float* x = xb.F;
        var cache = EmbedOnCpu(img, x);
        int dh = d / _heads, half = dh / 2;

        // CPU path (no device, or the device failed)
        int ff = _l[0].Up.Out;
        using var cur = new NativeBuffer((long)n * d * 4);
        using var qkv = new NativeBuffer((long)n * 3 * d * 4);
        using var q = new NativeBuffer((long)n * d * 4);
        using var kk = new NativeBuffer((long)n * d * 4);
        using var vv = new NativeBuffer((long)n * d * 4);
        using var att = new NativeBuffer((long)n * d * 4);
        using var up = new NativeBuffer((long)n * ff * 4);
        foreach (var L in _l)
        {
            LayerNormRows(x, L.Ln1W, L.Ln1B, cur.F, n, d);
            L.Qkv.Forward(_mm, cur.F, n, qkv.F);
            float* qp = q.F, kp = kk.F, vp = vv.F, src = qkv.F;
            fixed (float* rc = cache)
            {
                float* rcp = rc;
                EncoderOps.For(_mm.Pool, n, i =>
                {
                    Buffer.MemoryCopy(src + (long)i * 3 * d, qp + (long)i * d, d * 4L, d * 4L);
                    Buffer.MemoryCopy(src + (long)i * 3 * d + d, kp + (long)i * d, d * 4L, d * 4L);
                    Buffer.MemoryCopy(src + (long)i * 3 * d + 2 * d, vp + (long)i * d, d * 4L, d * 4L);
                    for (int h = 0; h < _heads; h++)
                    {
                        Rotate(qp + (long)i * d + h * dh, rcp + (long)i * dh, half);
                        Rotate(kp + (long)i * d + h * dh, rcp + (long)i * dh, half);
                    }
                });
            }
            EncoderOps.Attention(q.F, kk.F, vv.F, att.F, n, _heads, dh, 1f / MathF.Sqrt(dh), _mm);
            L.O.Forward(_mm, att.F, n, cur.F);
            Ops.Add(x, cur.F, x, n * d);
            LayerNormRows(x, L.Ln2W, L.Ln2B, cur.F, n, d);
            L.Up.Forward(_mm, cur.F, n, up.F);
            Simd.Gelu(up.F, up.F, n * ff);
            L.Down.Forward(_mm, up.F, n, cur.F);
            Ops.Add(x, cur.F, x, n * d);
        }
        LayerNormRows(x, _postLnW, _postLnB, x, n, d);
        return Merge(x, n, pw, ph);
    }

    /// <summary>2×2 merger: consecutive groups of merge² patches are concatenated → MLP.</summary>
    private (float[] Data, int GridW, int GridH) Merge(float* x, int n, int pw, int ph)
    {
        using var _ = Profiler.Section("vision.projection");
        int tokens = n / (_merge * _merge);
        using var hidden = new NativeBuffer((long)tokens * _mm0.Out * 4);
        _mm0.Forward(_mm, x, tokens, hidden.F);
        Simd.Gelu(hidden.F, hidden.F, tokens * _mm0.Out);
        var result = new float[tokens * _llmDim];
        fixed (float* r = result) _mm2.Forward(_mm, hidden.F, tokens, r);
        return (result, pw / _merge, ph / _merge);
    }

    /// <summary>Vision rope: rotates pairs (i, i + half) over the whole head.</summary>
    private static void Rotate(float* v, float* cache, int half)
    {
        for (int i = 0; i < half; i++)
        {
            float c = cache[2 * i], s = cache[2 * i + 1];
            float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }

    /// <summary>Bilinear (align-corners) resize of the learned square position table to the patch grid; layout [y][x][d].</summary>
    private float[] ResizePositions(int w, int h)
    {
        int s = _posSide, d = _dim;
        var outp = new float[w * h * d];
        if (w == s && h == s) { Array.Copy(_pos, outp, outp.Length); return outp; }
        for (int y = 0; y < h; y++)
        {
            double fy = h > 1 ? (double)y * (s - 1) / (h - 1) : 0;
            int y0 = (int)Math.Floor(fy), y1 = Math.Min(s - 1, y0 + 1);
            double wy = fy - y0;
            for (int x = 0; x < w; x++)
            {
                double fx = w > 1 ? (double)x * (s - 1) / (w - 1) : 0;
                int x0 = (int)Math.Floor(fx), x1 = Math.Min(s - 1, x0 + 1);
                double wx = fx - x0;
                for (int c = 0; c < d; c++)
                {
                    double v = (1 - wy) * ((1 - wx) * _pos[(y0 * s + x0) * d + c] + wx * _pos[(y0 * s + x1) * d + c])
                             + wy * ((1 - wx) * _pos[(y1 * s + x0) * d + c] + wx * _pos[(y1 * s + x1) * d + c]);
                    outp[(y * w + x) * d + c] = (float)v;
                }
            }
        }
        return outp;
    }

    private void LayerNormRows(float* x, float[] w, float[] b, float* y, int rows, int d)
    {
        fixed (float* wp = w, bp = b)
        {
            float* wpp = wp, bpp = bp;
            float eps = _eps;
            EncoderOps.For(_mm.Pool, rows, r => Ops.LayerNorm(x + (long)r * d, wpp, bpp, y + (long)r * d, d, eps));
        }
    }

    public void Dispose()
    {
        lock (_encodeLock)   // waits for an encode in progress
        {
            if (_disposed) return;
            _disposed = true;
            _device?.Dispose();
            _pool.Dispose();
            _f.Dispose();
        }
    }
}
