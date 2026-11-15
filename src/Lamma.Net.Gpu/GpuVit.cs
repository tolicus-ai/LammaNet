using ILGPU;
using ILGPU.Runtime;
using LammaNet.Multimodal;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

/// <summary>
/// Vision transformer on the GPU (<see cref="VitSpec"/>: SigLIP, Gemma 4, Qwen-VL): weights resident in device formats,
/// activations stay on the device for all blocks (only the patch embeddings go in and the features come out). Linear
/// layers use the tiled GEMM (tensor cores on sm_80+) with bias and residual-add epilogues; q/k/v head norms, the 2D
/// rotary embedding (angle table per call) and the fp16 K/V conversion are one kernel; attention reuses the prefill
/// kernel with one fp16 "page" per image tile and full (bidirectional) ranges. With a <see cref="VitPipelineSpec"/> the
/// whole tiled encoder runs here: tile pixels are patchified, embedded and positioned on the device, and the projector
/// runs before the download, so only pixels go in and tokens come out.
/// </summary>
internal sealed unsafe class GpuVit : GpuEncoderBase, IVitRunner, IVitPipelineRunner
{
    /// <summary>Tiles per device pass of <see cref="Encode"/> (bounds the activation memory).</summary>
    private const int MaxTilesPerPass = 8;
    private readonly VitSpec _spec;
    private readonly int _d, _heads, _hd, _ff;
    private readonly bool _gated, _clampedInputs;
    private readonly Block[] _blocks;
    private readonly long _postW, _postB;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, LnParams> _kLn;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>> _kGelu;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>> _kGeGlu;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, QkvParams> _kQkv;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>> _kAttn;
    private readonly int _attnThreads;
    /// <summary>Query rows per attention tile: 64 for the FlashAttention-2 style kernel, else 16.</summary>
    private readonly int _tileRows;

    // activations, sized for the largest call so far
    private int _capRows = -1, _groupRows = -1, _rows = -1, _tileCount;
    private MemoryBuffer1D<float, Stride1D.Dense>? _x, _cur, _q, _k, _v, _att, _up, _gate, _scr, _rope;
    private MemoryBuffer1D<Half, Stride1D.Dense>? _kh, _vh;
    private MemoryBuffer1D<int, Stride1D.Dense>? _pages, _info, _tiles, _slots;

    // pipeline stages (pixels → patches → embedding; projector after the blocks)
    private readonly VitPipelineSpec? _pipe;
    private readonly Lin? _patchEmbd;
    // projector over merge² consecutive rows: the pipeline's, or the spec's merger
    private readonly Lin[] _projector = [];
    private readonly int _merge;
    private int _capTokens = -1;
    private readonly long _posOff, _meanOff, _stdOff;
    private readonly MemoryBuffer1D<int, Stride1D.Dense>? _order;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<int>, ArrayView<float>, PatchParams>? _kPatchify;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, long, int, int>? _kAddPos;
    private int _capTiles = -1;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _pix;
    // whole images of a dynamic-resolution ViT (RunImage)
    private readonly VitPatchEmbedding? _imgEmbd;
    private readonly Lin? _imgLin;
    private readonly long _imgMeanOff, _imgStdOff, _imgPosOff;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<float>, GridPatchParams>? _kPatchGrid;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, GridPatchParams, long, int, int>? _kPosGrid;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, GridPatchParams, int>? _kRopeGrid;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _imgPix;
    private MemoryBuffer1D<float, Stride1D.Dense>? _imgPatches;
    private MemoryBuffer1D<float, Stride1D.Dense>? _patches, _proj0, _proj1;

    private sealed class Block
    {
        public Lin Q = null!, K = null!, V = null!, O = null!, Up = null!, Down = null!;
        public Lin? Gate;
        public long Ln1W, Ln1B, Ln2W, Ln2B, QNorm, KNorm, AttnPost, FfnPost;
    }

    public GpuVit(Accelerator acc, string deviceName, VitSpec spec, VitPipelineSpec? pipeline = null) : base(acc, deviceName)
    {
        _spec = spec;
        _d = spec.Dim; _heads = spec.Heads; _hd = _d / _heads;
        _ff = spec.Blocks[0].Up.W.Rows;
        _gated = spec.Blocks.Any(b => b.Gate != null);
        if (_d % 32 != 0 || _ff % 4 != 0 || _hd % 8 != 0 || _hd > Kernels.PrefillMaxHd || (spec.Rope == VitRope.NeoxHalves && _hd % 4 != 0))
            throw new NotSupportedException($"ViT shape d={_d} heads={_heads} ff={_ff} not supported on the GPU");

        var layers = new List<VitLinear>();
        foreach (var b in spec.Blocks)
        {
            layers.AddRange([b.Q, b.K, b.V, b.O, b.Up, b.Down]);
            if (b.Gate != null) layers.Add(b.Gate);
        }
        int blockLayers = layers.Count;
        if (pipeline != null)
        {
            if (spec.Rope != VitRope.None || pipeline.PatchEmbedding.W.Rows != _d || pipeline.Side % pipeline.Merge != 0)
                throw new NotSupportedException("vision pipeline shape not supported on the GPU");
            layers.Add(pipeline.PatchEmbedding);
        }
        if (spec.PatchEmbedding is { } pe)
        {
            if (pipeline != null || spec.Merger == null || pe.Linear.W.Rows != _d || pe.Linear.W.Cols != 3 * pe.Patch * pe.Patch || pe.Merge != spec.Merger.Merge
                || spec.Rope != VitRope.Neox || pe.Positions.Length != pe.PositionSide * pe.PositionSide * _d)
                throw new NotSupportedException("image patch embedding shape not supported on the GPU");
            layers.Add(pe.Linear);
        }
        int embdLayers = layers.Count;
        var projector = pipeline?.Projector ?? spec.Merger?.Projector;
        _merge = pipeline?.Merge ?? spec.Merger?.Merge ?? 0;
        if (projector != null)
        {
            if (projector.Count is < 1 or > 2 || projector[0].W.Cols != _d * _merge * _merge || projector.Any(l => l.ClampsInput || l.ClampsOutput))
                throw new NotSupportedException("vision projector shape not supported on the GPU");
            layers.AddRange(projector);
        }
        _clampedInputs = layers.Any(l => l.ClampsInput);
        var lins = UploadLinears(layers);
        _blocks = new Block[spec.Blocks.Count];
        int m = 0;
        for (int i = 0; i < _blocks.Length; i++)
        {
            var b = spec.Blocks[i];
            _blocks[i] = new Block
            {
                Q = lins[m++], K = lins[m++], V = lins[m++], O = lins[m++], Up = lins[m++], Down = lins[m++],
                Gate = b.Gate != null ? lins[m++] : null,
                Ln1W = Param(b.Ln1W), Ln1B = Param(b.Ln1B), Ln2W = Param(b.Ln2W), Ln2B = Param(b.Ln2B),
                QNorm = Param(b.QNorm), KNorm = Param(b.KNorm), AttnPost = Param(b.AttnPostNorm), FfnPost = Param(b.FfnPostNorm),
            };
        }
        _postW = Param(spec.PostLnW); _postB = Param(spec.PostLnB);
        _projector = lins[embdLayers..];
        if (spec.PatchEmbedding is { } img)
        {
            _imgEmbd = img;
            _imgLin = lins[embdLayers - 1];
            _imgMeanOff = Param(img.Mean);
            _imgStdOff = Param(img.Std);
            _imgPosOff = Param(img.Positions);
            _kPatchGrid = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<float>, GridPatchParams>(Kernels.PatchifyGrid);
            _kPosGrid = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, GridPatchParams, long, int, int>(Kernels.AddGridPositions);
            _kRopeGrid = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, GridPatchParams, int>(Kernels.GridRope);
        }
        if (pipeline != null)
        {
            _pipe = pipeline;
            _patchEmbd = lins[blockLayers];
            _posOff = Param(pipeline.Positions);
            _meanOff = Param(pipeline.Mean);
            _stdOff = Param(pipeline.Std);
            // patch rows in merge-block order (block row, block col, dy, dx): the pixel shuffle becomes a reshape
            int side = pipeline.Side, merge = pipeline.Merge;
            var order = new int[side * side];
            int o = 0;
            for (int by = 0; by < side / merge; by++)
                for (int bx = 0; bx < side / merge; bx++)
                    for (int dy = 0; dy < merge; dy++)
                        for (int dx = 0; dx < merge; dx++)
                            order[o++] = (by * merge + dy) * side + bx * merge + dx;
            _order = Acc.Allocate1D<int>(order.Length);
            _order.View.CopyFromCPU(order);
            _kPatchify = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<int>, ArrayView<float>, PatchParams>(Kernels.Patchify);
            _kAddPos = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, long, int, int>(Kernels.AddPositions);
        }
        UploadParams();

        _kLn = Acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, LnParams>(Kernels.LayerNorm);
        _kGelu = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>>(Kernels.GeluInPlace);
        _kGeGlu = Acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>>(Kernels.GeGluInPlace);
        _kQkv = Acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, QkvParams>(Kernels.QkvPost);
        // 64-query tensor-core tiles for head dims <= 64 (the vision encoders); the 128-wide variant spills registers
        // and was slower where measured (text prefill), so larger heads use the 16-query kernels
        bool tc64 = Gemm.TensorCores && _hd % 16 == 0 && _hd <= Kernels.Fa2SmallHd;
        bool tc = Gemm.TensorCores && _hd % 32 == 0;
        _kAttn = Acc.LoadKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>>(
            tc64 ? Kernels.TcAttnPrefill64Small : tc ? Kernels.TcAttnPrefill : Kernels.AttnPrefill);
        _tileRows = tc64 ? Kernels.Fa2Q : Kernels.PrefillBQ;
        _attnThreads = 128;

        // compile every kernel now (JIT) so the first image is not slow
        var warm = new float[32 * _d];
        var rope = new float[32 * _hd];
        fixed (float* w = warm, r = rope) Run(w, 32, 16, spec.Rope != VitRope.None ? r : null);
        if (_pipe != null)
        {
            var pixels = new byte[_pipe.ImageSize * _pipe.ImageSize * 3];
            var tokens = new float[_pipe.TokensPerTile * _pipe.OutDim];
            fixed (byte* px = pixels) fixed (float* t = tokens) Encode(px, 1, t);
        }
    }

    private void Ensure(int rows, int groupRows)
    {
        if (rows > _capRows)
        {
            FreeActivations();
            long n = (long)rows * _d;
            _x = Acc.Allocate1D<float>(n); _cur = Acc.Allocate1D<float>(n);
            _q = Acc.Allocate1D<float>(n); _k = Acc.Allocate1D<float>(n); _v = Acc.Allocate1D<float>(n);
            _att = Acc.Allocate1D<float>(n); _up = Acc.Allocate1D<float>((long)rows * _ff);
            _gate = _gated ? Acc.Allocate1D<float>((long)rows * _ff) : null;
            _scr = Acc.Allocate1D<float>(_clampedInputs ? (long)rows * Math.Max(Math.Max(_d, _ff), 3 * (_pipe?.Patch ?? 0) * (_pipe?.Patch ?? 0)) : 4);
            _rope = Acc.Allocate1D<float>(Math.Max(4, (long)rows * _hd));
            _kh = Acc.Allocate1D<Half>(n); _vh = Acc.Allocate1D<Half>(n);
            _pages = Acc.Allocate1D<int>(rows); _info = Acc.Allocate1D<int>(3L * rows);
            _tiles = Acc.Allocate1D<int>(2L * (rows + _tileRows));
            _slots = Acc.Allocate1D<int>(rows);
            _slots.View.CopyFromCPU(Enumerable.Range(0, rows).ToArray());
            _capRows = rows;
            _rows = -1;
        }
        if (rows == _rows && groupRows == _groupRows) return;
        // one page per image tile; token t of tile g attends to all positions [0, groupRows) of page g
        int groups = (rows + groupRows - 1) / groupRows;
        var pages = new int[groups];
        var info = new int[3 * rows];
        var tiles = new List<int>();
        for (int g = 0; g < groups; g++)
        {
            pages[g] = g;
            int r0 = g * groupRows, len = Math.Min(groupRows, rows - r0);
            for (int t = 0; t < len; t++) { info[3 * (r0 + t)] = g; info[3 * (r0 + t) + 1] = 0; info[3 * (r0 + t) + 2] = len - 1; }
            for (int q0 = 0; q0 < len; q0 += _tileRows) { tiles.Add(r0 + q0); tiles.Add(Math.Min(_tileRows, len - q0)); }
        }
        _pages!.View.SubView(0, groups).CopyFromCPU(Stream, pages);
        _info!.View.SubView(0, info.Length).CopyFromCPU(Stream, info);
        _tiles!.View.SubView(0, tiles.Count).CopyFromCPU(Stream, tiles.ToArray());
        _tileCount = tiles.Count / 2;
        _rows = rows; _groupRows = groupRows;
    }

    public void Run(float* x, int rows, int groupRows, float* rope = null)
    {
        Upload(x, rows, groupRows, rope);
        RunBlocks(rows, groupRows);
        long n = (long)rows * _d;
        _x!.View.SubView(0, n).CopyToCPU(Stream, ref *x, n);
        Stream.Synchronize();
    }

    public void RunProjected(float* x, int rows, int groupRows, float* rope, float* output)
    {
        if (_pipe != null || _projector.Length == 0) throw new InvalidOperationException("created without a merger");
        if (rows % (_merge * _merge) != 0) throw new ArgumentException($"rows must be a multiple of {_merge * _merge}", nameof(rows));
        Upload(x, rows, groupRows, rope);
        RunBlocks(rows, groupRows);
        Project(rows / (_merge * _merge), output);
        Stream.Synchronize();
    }

    public void RunImage(byte* rgb, int width, int height, float* output)
    {
        int rows = EmbedImageOnDevice(rgb, width, height);
        RunBlocks(rows, rows);
        Project(rows / (_imgEmbd!.Merge * _imgEmbd.Merge), output);
        Stream.Synchronize();
    }

    /// <summary>The embedding stage of <see cref="RunImage"/> only: x [patches, dim] and the rotary table (tests).</summary>
    internal void EmbedImage(byte* rgb, int width, int height, float* x, float* rope)
    {
        int rows = EmbedImageOnDevice(rgb, width, height);
        _x!.View.SubView(0, (long)rows * _d).CopyToCPU(Stream, ref *x, (long)rows * _d);
        _rope!.View.SubView(0, (long)rows * _hd).CopyToCPU(Stream, ref *rope, (long)rows * _hd);
        Stream.Synchronize();
    }

    /// <summary>Uploads the image and computes patches, embedding, positions and rotary angles into x / rope; returns the patch count.</summary>
    private int EmbedImageOnDevice(byte* rgb, int width, int height)
    {
        var pe = _imgEmbd ?? throw new InvalidOperationException("created without a patch embedding");
        int align = pe.Patch * pe.Merge;
        if (width % align != 0 || height % align != 0) throw new ArgumentException($"image sides must be multiples of {align}");
        var gp = new GridPatchParams
        {
            Pw = width / pe.Patch, Ph = height / pe.Patch, Ps = pe.Patch, Merge = pe.Merge, Pp = 3 * pe.Patch * pe.Patch, Width = width,
            MeanOff = _imgMeanOff, StdOff = _imgStdOff,
        };
        int rows = gp.Pw * gp.Ph;
        long bytes = (long)width * height * 3, patchFloats = (long)rows * gp.Pp;
        Acc.Bind();
        Ensure(rows, rows);
        if (_imgPix == null || _imgPix.Length < bytes) { _imgPix?.Dispose(); _imgPix = Acc.Allocate1D<byte>(bytes); }
        if (_imgPatches == null || _imgPatches.Length < patchFloats) { _imgPatches?.Dispose(); _imgPatches = Acc.Allocate1D<float>(patchFloats); }
        using (Stage("vit.upload")) _imgPix.View.SubView(0, bytes).CopyFromCPU(Stream, ref *rgb, bytes);
        using (Stage("vit.embed"))
        {
            _kPatchGrid!(Stream, (int)patchFloats, _imgPix.View, _imgPatches.View, P, gp);
            // fp32: the padded border patches are near-constant rows whose layer norms amplify fp16 input rounding
            Linear(_imgLin!, _imgPatches.View, gp.Pp, rows, _x!.View, _scr!.View, exact: true);
            _kPosGrid!(Stream, rows * _d, _x.View, P, gp, _imgPosOff, pe.PositionSide, _d);
            _kRopeGrid!(Stream, rows * (_hd / 2), _rope!.View, gp, _hd);
        }
        return rows;
    }

    private void Upload(float* x, int rows, int groupRows, float* rope)
    {
        if (_spec.Rope != VitRope.None && rope == null) throw new ArgumentNullException(nameof(rope), "This encoder needs a rotary angle table");
        Acc.Bind();
        Ensure(rows, groupRows);
        long n = (long)rows * _d;
        _x!.View.SubView(0, n).CopyFromCPU(Stream, ref *x, n);
        if (_spec.Rope != VitRope.None) _rope!.View.SubView(0, (long)rows * _hd).CopyFromCPU(Stream, ref *rope, (long)rows * _hd);
    }

    /// <summary>
    /// Encodes tiles end to end (see <see cref="VitPipelineSpec"/>): per pass of up to <see cref="MaxTilesPerPass"/> tiles,
    /// upload pixels → patchify → patch embedding + bias + positions → blocks → projector → download tokens.
    /// </summary>
    public void Encode(byte* pixels, int tiles, float* output)
    {
        var pipe = _pipe ?? throw new InvalidOperationException("created without a vision pipeline");
        Acc.Bind();
        int n = pipe.Side * pipe.Side, pp = 3 * pipe.Patch * pipe.Patch;
        int tileBytes = pipe.ImageSize * pipe.ImageSize * 3, perTile = pipe.TokensPerTile, outDim = pipe.OutDim;
        for (int t0 = 0; t0 < tiles; t0 += MaxTilesPerPass)
        {
            int nt = Math.Min(MaxTilesPerPass, tiles - t0), rows = nt * n, tokens = nt * perTile;
            Ensure(rows, n);
            EnsurePipeline(nt, n, pp);
            using (Stage("vit.upload"))
                _pix!.View.SubView(0, (long)nt * tileBytes).CopyFromCPU(Stream, ref pixels[(long)t0 * tileBytes], (long)nt * tileBytes);
            using (Stage("vit.embed"))
            {
                _kPatchify!(Stream, rows * pp, _pix.View, _patches!.View, _order!.View, P,
                    new PatchParams { N = n, Side = pipe.Side, Ps = pipe.Patch, ImageSize = pipe.ImageSize, Pp = pp, MeanOff = _meanOff, StdOff = _stdOff });
                Linear(_patchEmbd!, _patches.View, pp, rows, _x!.View, _scr!.View);
                _kAddPos!(Stream, rows * _d, _x.View, P, _order.View, _posOff, n, _d);
            }
            RunBlocks(rows, n);
            Project(tokens, output + (long)t0 * perTile * outDim);
        }
        Stream.Synchronize();
    }

    /// <summary>
    /// Projector over merge² consecutive rows of x (a [tokens, d·merge²] view), GELU between stages; downloads
    /// [tokens, OutDim] to <paramref name="output"/> (asynchronously: the caller synchronizes).
    /// </summary>
    private void Project(int tokens, float* output)
    {
        using var _proj = Stage("vit.projector");
        if (tokens > _capTokens)
        {
            _proj0?.Dispose(); _proj1?.Dispose();
            _proj0 = Acc.Allocate1D<float>((long)tokens * _projector[0].M.Rows);
            _proj1 = _projector.Length > 1 ? Acc.Allocate1D<float>((long)tokens * _projector[1].M.Rows) : null;
            _capTokens = tokens;
        }
        int outDim = _projector[^1].M.Rows;
        var outBuf = _proj0!.View;
        Gemm.Run(Stream, P, _x!.View, 0, _d * _merge * _merge, tokens, outBuf, 0, 1f, _projector[0].M, 0, _projector[0].M.Rows);
        if (_projector.Length > 1)
        {
            _kGelu(Stream, tokens * _projector[0].M.Rows, outBuf);
            Gemm.Run(Stream, P, outBuf, 0, _projector[0].M.Rows, tokens, _proj1!.View, 0, 1f, _projector[1].M, 0, outDim);
            outBuf = _proj1.View;
        }
        outBuf.SubView(0, (long)tokens * outDim).CopyToCPU(Stream, ref *output, (long)tokens * outDim);
    }

    private void EnsurePipeline(int tiles, int n, int pp)
    {
        if (tiles <= _capTiles) return;
        foreach (var b in new MemoryBuffer?[] { _pix, _patches }) b?.Dispose();
        var pipe = _pipe!;
        _pix = Acc.Allocate1D<byte>((long)tiles * pipe.ImageSize * pipe.ImageSize * 3);
        _patches = Acc.Allocate1D<float>((long)tiles * n * pp);
        _capTiles = tiles;
    }

    /// <summary>All blocks and the final norm on <see cref="_x"/> (rope table already uploaded).</summary>
    private void RunBlocks(int rows, int groupRows)
    {
        float scale = _spec.AttentionScale != 0 ? _spec.AttentionScale : 1f / MathF.Sqrt(_hd);
        var ap = new AttnParams
        {
            QOff = 0, OutOff = 0, N = rows, NHead = _heads, NKv = _heads, Hd = _hd, HdV = _hd, PageSize = groupRows,
            QStride = _d, QHeadPitch = _hd, OutStride = _d, KRow = _d, VRow = _d, NChunks = 1, Gated = 0, Scale = scale, SoftCap = 0,
        };
        var scr = _scr!.View;
        foreach (var B in _blocks)
        {
            using (Stage("vit.norm")) Norm(_x.View, _cur!.View, B.Ln1W, B.Ln1B, rows);
            using (Stage("vit.gemm.qkv"))
            {
                Linear(B.Q, _cur.View, _d, rows, _q!.View, scr);
                Linear(B.K, _cur.View, _d, rows, _k!.View, scr);
                Linear(B.V, _cur.View, _d, rows, _v!.View, scr);
            }
            var qp = new QkvParams
            {
                QOff = 0, KOff = 0, VOff = 0, QNormOff = B.QNorm, KNormOff = B.KNorm, RopeOff = 0,
                N = rows, NHead = _heads, NKv = _heads, Hd = _hd, HdV = _hd, QStride = _d, KStride = _d, VStride = _d, QHeadPitch = _hd,
                KRow = _d, VRow = _d, OwnKv = 1, VNorm = _spec.VNorm ? 1 : 0,
                RopeDims = _spec.Rope != VitRope.None ? _hd : 0, RopeNeox = 1, RopeStride = _hd,
                RopeSegPairs = _spec.Rope == VitRope.NeoxHalves ? _hd / 4 : 0, Eps = _spec.Eps,
            };
            long units = (long)rows * 3 * _heads;
            using (Stage("vit.qkv_post"))
                _kQkv(Stream, new KernelConfig((int)((units + Kernels.WarpsPerGroup - 1) / Kernels.WarpsPerGroup), Kernels.GroupSize),
                    P, _q.View, _k.View, _v.View, _rope!.View, _kh!.View, _vh!.View, _slots!.View, qp);
            using (Stage("vit.attention"))
                _kAttn(Stream, new KernelConfig(_tileCount * _heads, _attnThreads), _q.View, _kh.View, _vh.View, _pages!.View, _info!.View, _tiles!.View, _att!.View, ap, SpecializedValue.New(_hd));
            using (Stage("vit.gemm.o")) Residual(B.O, _att.View, _d, rows, B.AttnPost);

            using (Stage("vit.norm")) Norm(_x.View, _cur.View, B.Ln2W, B.Ln2B, rows);
            using (Stage("vit.gemm.up")) Linear(B.Up, _cur.View, _d, rows, _up!.View, scr);
            using (Stage("vit.act"))
            {
                if (B.Gate != null)
                {
                    Linear(B.Gate, _cur.View, _d, rows, _gate!.View, scr);
                    _kGeGlu(Stream, rows * _ff, _gate.View, _up.View);
                }
                else _kGelu(Stream, rows * _ff, _up.View);
            }
            using (Stage("vit.gemm.down")) Residual(B.Down, _up.View, _ff, rows, B.FfnPost);
        }
        if (_postW >= 0) Norm(_x!.View, _x.View, _postW, _postB, rows);
    }

    /// <summary>x += [rms_norm](W·src): accumulated in the GEMM epilogue unless a post-norm or a clamp sits in between.</summary>
    private void Residual(Lin l, ArrayView<float> src, int width, int rows, long postNorm)
    {
        if (postNorm < 0 && !l.Src.ClampsOutput)
        {
            Linear(l, src, width, rows, _x!.View, _scr!.View, mode: 1);
            return;
        }
        Linear(l, src, width, rows, _cur!.View, _scr!.View);
        if (postNorm >= 0) RmsNorm(_cur.View, _x!.View, postNorm, rows, _d, _spec.Eps, mode: 1);
        else KAdd(Stream, rows * _d, _cur.View, _x!.View);
    }

    private void Norm(ArrayView<float> src, ArrayView<float> dst, long w, long b, int rows)
    {
        if (_spec.Norm == VitNorm.RmsNorm) RmsNorm(src, dst, w, rows, _d, _spec.Eps);
        else
            _kLn(Stream, new KernelConfig(rows, Kernels.GroupSize), P, src, dst,
                new LnParams { XOff = 0, YOff = 0, WOff = w, BOff = b, Rows = rows, D = _d, XStride = _d, YStride = _d, Eps = _spec.Eps });
    }

    private void FreeActivations()
    {
        foreach (var b in new MemoryBuffer?[] { _x, _cur, _q, _k, _v, _att, _up, _gate, _scr, _rope, _kh, _vh, _pages, _info, _tiles, _slots }) b?.Dispose();
    }

    public override void Dispose()
    {
        Acc.Bind();
        FreeActivations();
        foreach (var b in new MemoryBuffer?[] { _pix, _patches, _proj0, _proj1, _order, _imgPix, _imgPatches }) b?.Dispose();
        base.Dispose();
    }
}
