using LammaNet.Chat;
using LammaNet.Compute;
using LammaNet.Gguf;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Tokenization;

namespace LammaNet.Multimodal;

/// <summary>
/// Idefics3 projector ("idefics3" mmproj; SmolVLM, granite-docling): the image is resized so its longest edge is
/// clip.vision.preproc_image_size, split into a grid of square SigLIP tiles plus a downscaled overview, each tile is
/// encoded by the ViT, pixel-shuffled by the scale factor and projected by one linear layer. Tiles are framed with
/// &lt;fake_token_around_image&gt;&lt;row_i_col_j&gt; tokens and the overview with &lt;global-img&gt;, as in
/// transformers' Idefics3Processor and llama.cpp's mtmd.
/// </summary>
public sealed unsafe class Idefics3Projector : IMediaEncoder, IDisposable
{
    private readonly GgufFile _f;
    private readonly ComputePool _pool;
    private readonly MatMul _mm;
    private readonly SiglipEncoder _vit;
    private readonly Linear _fc;
    private readonly int _llmDim, _merge, _longestEdge;
    private readonly Vocabulary _vocab;
    private readonly int[] _rowEnd, _overviewStart, _overviewEnd;
    private readonly object _lock = new();
    private bool _disposed;
    private readonly Microsoft.Extensions.Logging.ILogger _log;
    private IVitPipelineRunner? _pipeline;

    public Idefics3Projector(string mmprojPath, Model llm, ProjectorOptions? options = null)
    {
        var opt = options ?? new ProjectorOptions();
        _log = Diagnostics.LammaLogging.Create<Idefics3Projector>(opt.LoggerFactory);
        _f = new GgufFile(mmprojPath);
        string type = _f.Get("clip.projector_type", _f.Get("clip.vision.projector_type", ""));
        if (type != "idefics3") throw new NotSupportedException($"Not an Idefics3 projector ({type})");
        _pool = new ComputePool(opt.Threads ?? ComputePool.DefaultThreadCount(), opt.Resources ?? ResourceLimiter.Default);
        _mm = new MatMul(_pool);
        _vit = new SiglipEncoder(_f, _mm);
        _fc = Linear.Load(_f, "mm.model.fc") ?? throw new InvalidDataException("mmproj tensor mm.model.fc missing");
        _llmDim = llm.Config.NEmbd;
        _merge = _f.Get("clip.vision.projector.scale_factor", 2);
        _longestEdge = opt.ImageLongestEdge ?? _f.Get("clip.vision.preproc_image_size", 4 * _vit.ImageSize);
        if (_fc.In != _vit.Dim * _merge * _merge || _fc.Out != _llmDim)
            throw new InvalidDataException($"projector shape {_fc.In}→{_fc.Out} does not match ViT {_vit.Dim}·{_merge}² → LLM {_llmDim}");
        _vocab = llm.Vocab;
        _rowEnd = Single("\n");
        _overviewStart = [.. Single("\n\n"), .. Single("<fake_token_around_image>"), .. Single("<global-img>")];
        _overviewEnd = Single("<fake_token_around_image>");
        if (opt.Accelerator is { } acc)
        {
            // preferably the whole encoder on the device (pixels in, tokens out), else only the transformer blocks
            _pipeline = acc.CreateVitPipeline(_vit.PipelineSpec(_merge, [new VitLinear(_fc.W, _fc.Bias)]));
            if (_pipeline == null) _vit.UseDevice(acc);
        }
    }

    /// <summary>Tokens per encoded tile (image tile or overview).</summary>
    public int TokensPerTile => _vit.Side * _vit.Side / (_merge * _merge);

    private int[] Single(string piece)
    {
        var t = _vocab.Tokenize(piece, addSpecial: false, parseSpecial: true);
        if (t.Count != 1) throw new InvalidDataException($"'{piece.Replace("\n", "\\n")}' is not a single token in this vocabulary");
        return [t[0]];
    }

    /// <summary>Device running the vision transformer ("CPU" without an accelerator).</summary>
    public string EncoderDevice => _pipeline?.DeviceName ?? _vit.DeviceName ?? "CPU";

    /// <summary>Why the encoder fell back from the accelerator to the CPU (e.g. out of device memory), if it did.</summary>
    public string? DeviceError { get; private set; }

    public bool CanEncode(ChatPart part) => part is ImagePart;

    public IReadOnlyList<PromptSegment> Encode(ChatPart part)
    {
        var img = (ImagePart)part;
        RgbImage rgb;
        using (Profiler.Section("vision.decode"))
            rgb = img.EncodedBytes != null ? RgbImage.Decode(img.EncodedBytes) : new RgbImage(img.Width, img.Height, img.Rgb!);
        List<RgbImage> tiles; int cols, rows;
        using (Profiler.Section("vision.preprocess")) (tiles, cols, rows) = Preprocess(rgb);
        float[] emb;
        emb = EncodeTiles(tiles);
        ulong hash = Fnv(img.EncodedBytes ?? img.Rgb!);
        int per = TokensPerTile;
        EmbeddingSegment Seg(int i) =>
            new(emb.AsSpan(i * per * _llmDim, per * _llmDim).ToArray(), per, _llmDim, (long)(hash ^ ((ulong)(i + 1) * 0x9E3779B97F4A7C15UL)), "image");

        // tiles row by row (each "<fake_token_around_image><row_y_col_x>" + tile, "\n" between rows), then the overview
        var list = new List<PromptSegment>();
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                list.Add(new TokenSegment(_vocab.Tokenize($"<fake_token_around_image><row_{y + 1}_col_{x + 1}>", addSpecial: false, parseSpecial: true)));
                list.Add(Seg(y * cols + x));
            }
            if (y != rows - 1) list.Add(new TokenSegment(_rowEnd));
        }
        list.Add(new TokenSegment(_overviewStart));
        list.Add(Seg(tiles.Count - 1));
        list.Add(new TokenSegment(_overviewEnd));
        return list;
    }

    /// <summary>
    /// Resizes (Lanczos, aspect preserved) to the longest edge, then up to whole tiles, and cuts the grid; the overview
    /// (the refined image scaled to one tile) is last. Returns the tiles and the grid size.
    /// </summary>
    public (List<RgbImage> Tiles, int Cols, int Rows) Preprocess(RgbImage image)
    {
        int tile = _vit.ImageSize, longest = _longestEdge;
        double aspect = (double)image.Width / image.Height;
        int rw, rh;
        if (image.Width >= image.Height)
        {
            rw = longest;
            rh = (int)(longest / aspect);
            rh += rh % 2;
        }
        else
        {
            rh = longest;
            rw = (int)(longest * aspect);
            rw += rw % 2;
        }
        rw = Math.Max(rw, 1); rh = Math.Max(rh, 1);
        int cols = (rw + tile - 1) / tile, rows = (rh + tile - 1) / tile;
        var refined = image.ResizePil(rw, rh, ResampleFilter.Lanczos).ResizePil(cols * tile, rows * tile, ResampleFilter.Lanczos);
        var tiles = new List<RgbImage>(rows * cols + 1);
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                tiles.Add(refined.Crop(x * tile, y * tile, tile, tile));
        tiles.Add(refined.ResizePil(tile, tile, ResampleFilter.Lanczos));
        return (tiles, cols, rows);
    }

    /// <summary>Encodes tiles to LLM-space embeddings [tiles · TokensPerTile, n_embd].</summary>
    public float[] EncodeTiles(IReadOnlyList<RgbImage> tiles)
    {
        lock (_lock)   // the encoder's buffers, pool and device stream are shared
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return EncodeTilesCore(tiles);
        }
    }

    private float[] EncodeTilesCore(IReadOnlyList<RgbImage> tiles)
    {
        int n = _vit.Side * _vit.Side, merged = tiles.Count * TokensPerTile;
        if (_pipeline != null)
        {
            int size = _vit.ImageSize, tileBytes = size * size * 3;
            var pixels = new byte[(long)tiles.Count * tileBytes];
            for (int t = 0; t < tiles.Count; t++)
            {
                if (tiles[t].Width != size || tiles[t].Height != size) throw new ArgumentException($"tiles must be {size}×{size}");
                tiles[t].Pixels.CopyTo(pixels, (long)t * tileBytes);
            }
            var tokens = new float[(long)merged * _llmDim];
            try
            {
                using (Profiler.Section("vision.device"))
                    fixed (byte* px = pixels) fixed (float* r = tokens) _pipeline.Encode(px, tiles.Count, r);
                return tokens;
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                // a device failure (e.g. out of memory) disables the device; this and later images use the CPU
                DeviceError = $"{e.GetType().Name}: {e.Message}";
                var failed = _pipeline;
                _pipeline = null;
                Diagnostics.Log.EncoderDeviceFailed(_log, e, "Idefics3 vision pipeline", failed.DeviceName);
                try { failed.Dispose(); }
                catch (Exception de) { Diagnostics.Log.EncoderDeviceFailed(_log, de, "disposing the failed encoder", failed.DeviceName); }
            }
        }
        using var feats = new Tensors.NativeBuffer((long)tiles.Count * n * _vit.Dim * 4, zero: false);
        _vit.Encode(tiles, _merge, feats.F);
        // pixel shuffle: merge² consecutive patches (block order) form one token of Dim·merge² features
        var result = new float[(long)merged * _llmDim];
        using (Profiler.Section("vision.projection"))
        fixed (float* r = result) _fc.Forward(_mm, feats.F, merged, r);
        return result;
    }

    private static ulong Fnv(byte[] d)
    {
        ulong h = 14695981039346656037UL;
        foreach (var b in d) { h ^= b; h *= 1099511628211UL; }
        return h;
    }

    public void Dispose()
    {
        lock (_lock)   // waits for an encode in progress
        {
            if (_disposed) return;
            _disposed = true;
            _pipeline?.Dispose();
            _vit.Dispose();
            _pool.Dispose();
            _f.Dispose();
        }
    }
}
