using LammaNet.Compute;
using LammaNet.Gguf;
using LammaNet.Tensors;

namespace LammaNet.Multimodal;

/// <summary>
/// SigLIP / SigLIP2 vision transformer ("v.*" tensors of a clip mmproj): 2D patch embedding, learned absolute
/// position embeddings, pre-norm blocks (LayerNorm, biased multi-head attention, GELU MLP) and an optional final
/// LayerNorm. Shared by projector heads that only differ after the ViT (Idefics3 / granite-docling, ...).
/// </summary>
internal sealed unsafe class SiglipEncoder : IDisposable
{
    public int Dim { get; }
    public int Heads { get; }
    public int Patch { get; }
    /// <summary>Square input resolution (clip.vision.image_size).</summary>
    public int ImageSize { get; }
    /// <summary>Patches per tile side.</summary>
    public int Side => ImageSize / Patch;

    private readonly MatMul _mm;
    private readonly float _eps;
    private readonly float[] _mean, _std;
    private readonly WeightTensor _patchW;
    private readonly float[] _patchB, _pos;
    private readonly float[]? _postLnW, _postLnB;
    private readonly Layer[] _l;
    private IVitRunner? _device;

    /// <summary>Device running the transformer blocks, or null for the CPU.</summary>
    public string? DeviceName => _device?.DeviceName;

    private sealed class Layer
    {
        public float[] Ln1W = [], Ln1B = [], Ln2W = [], Ln2B = [];
        public Linear Q = null!, K = null!, V = null!, O = null!, Up = null!, Down = null!;
    }

    public SiglipEncoder(GgufFile f, MatMul mm, IVisionAccelerator? accelerator = null)
    {
        _mm = mm;
        Dim = f.Get<int>("clip.vision.embedding_length");
        Heads = f.Get<int>("clip.vision.attention.head_count");
        Patch = f.Get("clip.vision.patch_size", 16);
        ImageSize = f.Get<int>("clip.vision.image_size");
        _eps = f.Get("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        _mean = f.Get("clip.vision.image_mean", new[] { 0.5f, 0.5f, 0.5f });
        _std = f.Get("clip.vision.image_std", new[] { 0.5f, 0.5f, 0.5f });

        float[] R(string n) => EncoderOps.F(f, n) ?? throw new InvalidDataException($"mmproj tensor {n} missing");
        Linear L(string n) => Linear.Load(f, n) ?? throw new InvalidDataException($"mmproj tensor {n} missing");

        // conv2d kernel [kw, kh, 3, dim] = a [dim, 3·kh·kw] matrix over patches flattened as (c, ky, kx)
        var w = R("v.patch_embd.weight");
        int pp = 3 * Patch * Patch;
        var buf = new NativeBuffer((long)w.Length * 4);
        w.AsSpan().CopyTo(buf.AsFloatSpan());
        _patchW = new WeightTensor("v.patch_embd", GgmlType.F32, [pp, Dim], buf.B, buf);
        _patchB = R("v.patch_embd.bias");
        _pos = R("v.position_embd.weight");
        if (_pos.Length != Side * Side * Dim)
            throw new NotSupportedException($"position table of {_pos.Length / Dim} entries does not match the {Side}×{Side} patch grid");
        _postLnW = EncoderOps.F(f, "v.post_ln.weight");
        _postLnB = EncoderOps.F(f, "v.post_ln.bias");
        int layers = f.Get<int>("clip.vision.block_count");
        _l = new Layer[layers];
        for (int i = 0; i < layers; i++)
        {
            string p = $"v.blk.{i}.";
            _l[i] = new Layer
            {
                Ln1W = R(p + "ln1.weight"), Ln1B = R(p + "ln1.bias"), Ln2W = R(p + "ln2.weight"), Ln2B = R(p + "ln2.bias"),
                Q = L(p + "attn_q"), K = L(p + "attn_k"), V = L(p + "attn_v"), O = L(p + "attn_out"),
                Up = L(p + "ffn_up"), Down = L(p + "ffn_down"),
            };
            // some converters store the MLP matrices swapped (llama.cpp checks the same)
            if (_l[i].Up.In != Dim) (_l[i].Up, _l[i].Down) = (_l[i].Down, _l[i].Up);
        }
        if (accelerator != null) UseDevice(accelerator);
    }

    /// <summary>The transformer blocks for a device.</summary>
    public VitSpec DeviceSpec()
    {
        static VitLinear V(Linear l) => new(l.W, l.Bias);
        var blocks = _l.Select(L => new VitBlock(L.Ln1W, L.Ln1B, L.Ln2W, L.Ln2B, V(L.Q), V(L.K), V(L.V), V(L.O), V(L.Up), V(L.Down))).ToArray();
        return new VitSpec(Dim, Heads, _eps, blocks, _postLnW, _postLnB);
    }

    /// <summary>The whole encoder (patchify, embedding, positions, blocks) followed by <paramref name="projector"/>, for a device.</summary>
    public VitPipelineSpec PipelineSpec(int merge, IReadOnlyList<VitLinear> projector) =>
        new(DeviceSpec(), ImageSize, Patch, merge, new VitLinear(_patchW, _patchB), _pos, _mean, _std, projector);

    /// <summary>Runs the transformer blocks of <see cref="Encode"/> on <paramref name="accelerator"/> (if it can).</summary>
    public void UseDevice(IVisionAccelerator accelerator) => _device ??= accelerator.CreateVit(DeviceSpec());

    /// <summary>
    /// Encodes square <see cref="ImageSize"/> tiles into <paramref name="output"/> [tiles · Side², Dim]. Within a tile
    /// the patches are ordered by <paramref name="merge"/>×<paramref name="merge"/> blocks (block row, block col, dy, dx),
    /// so a pixel-shuffle merge of the result is a plain reshape: merged token = merge² consecutive rows, feature
    /// index c + Dim·dx + Dim·merge·dy (Idefics3 / llama.cpp order). Attention is per tile.
    /// </summary>
    public void Encode(IReadOnlyList<RgbImage> tiles, int merge, float* output)
    {
        int side = Side, n = side * side, d = Dim, pp = 3 * Patch * Patch, ps = Patch;
        if (side % merge != 0) throw new NotSupportedException($"patch grid {side} is not divisible by the merge factor {merge}");
        var order = new (int Y, int X)[n];
        int o = 0;
        for (int by = 0; by < side / merge; by++)
            for (int bx = 0; bx < side / merge; bx++)
                for (int dy = 0; dy < merge; dy++)
                    for (int dx = 0; dx < merge; dx++)
                        order[o++] = (by * merge + dy, bx * merge + dx);

        const int TilesPerChunk = 4;
        int dh = d / Heads, ff = _l[0].Up.Out;
        float scale = 1f / MathF.Sqrt(dh);
        int maxRows = Math.Min(tiles.Count, TilesPerChunk) * n;
        using var patches = new NativeBuffer((long)maxRows * pp * 4, zero: false);
        using var cur = new NativeBuffer((long)maxRows * d * 4, zero: false);
        using var q = new NativeBuffer((long)maxRows * d * 4, zero: false);
        using var k = new NativeBuffer((long)maxRows * d * 4, zero: false);
        using var v = new NativeBuffer((long)maxRows * d * 4, zero: false);
        using var att = new NativeBuffer((long)maxRows * d * 4, zero: false);
        using var up = new NativeBuffer((long)maxRows * ff * 4, zero: false);
        for (int t0 = 0; t0 < tiles.Count; t0 += TilesPerChunk)
        {
            int nt = Math.Min(TilesPerChunk, tiles.Count - t0), rows = nt * n;
            float* x = output + (long)t0 * n * d;
            float* pb = patches.F;
            EncoderOps.For(_mm.Pool, rows, i =>
            {
                var img = tiles[t0 + i / n];
                if (img.Width != ImageSize || img.Height != ImageSize) throw new ArgumentException($"tiles must be {ImageSize}×{ImageSize}");
                var (py, px) = order[i % n];
                float* row = pb + (long)i * pp;
                for (int c = 0; c < 3; c++)
                    for (int ky = 0; ky < ps; ky++)
                        for (int kx = 0; kx < ps; kx++)
                        {
                            float val = img.Pixels[((py * ps + ky) * ImageSize + px * ps + kx) * 3 + c] / 255f;
                            row[(c * ps + ky) * ps + kx] = (val - _mean[c]) / _std[c];
                        }
            });
            _mm.Multiply(_patchW, patches.F, rows, x, pp, d);
            fixed (float* b = _patchB, pos = _pos)
                for (int i = 0; i < rows; i++)
                {
                    var (py, px) = order[i % n];
                    float* xi = x + (long)i * d;
                    Ops.Add(xi, b, xi, d);
                    Ops.Add(xi, pos + ((long)py * side + px) * d, xi, d);
                }
            if (_device != null)
            {
                using (Profiler.Section("vit.device")) _device.Run(x, rows, n);
                continue;
            }

            foreach (var L in _l)
            {
                EncoderOps.LayerNormRows(x, L.Ln1W, L.Ln1B, cur.F, rows, d, _eps, _mm.Pool);
                L.Q.Forward(_mm, cur.F, rows, q.F);
                L.K.Forward(_mm, cur.F, rows, k.F);
                L.V.Forward(_mm, cur.F, rows, v.F);
                using (Profiler.Section("vit.attention"))
                for (int t = 0; t < nt; t++)
                {
                    long off = (long)t * n * d;
                    EncoderOps.Attention(q.F + off, k.F + off, v.F + off, att.F + off, n, Heads, dh, scale, _mm);
                }
                L.O.Forward(_mm, att.F, rows, cur.F);
                Ops.Add(x, cur.F, x, rows * d);
                EncoderOps.LayerNormRows(x, L.Ln2W, L.Ln2B, cur.F, rows, d, _eps, _mm.Pool);
                L.Up.Forward(_mm, cur.F, rows, up.F);
                Simd.Gelu(up.F, up.F, rows * ff);
                L.Down.Forward(_mm, up.F, rows, cur.F);
                Ops.Add(x, cur.F, x, rows * d);
            }
            if (_postLnW != null && _postLnB != null) EncoderOps.LayerNormRows(x, _postLnW, _postLnB, x, rows, d, _eps, _mm.Pool);
        }
    }

    public void Dispose() => _device?.Dispose();
}
