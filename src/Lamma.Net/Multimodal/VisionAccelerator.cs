using LammaNet.Tensors;

namespace LammaNet.Multimodal;

/// <summary>A linear layer of a vision transformer: y = clamp(W·clamp(x) + bias) (clamps: Gemma 4 "clippable" linears).</summary>
public sealed record VitLinear(WeightTensor W, float[]? Bias)
{
    public float InMin { get; init; } = float.NegativeInfinity;
    public float InMax { get; init; } = float.PositiveInfinity;
    public float OutMin { get; init; } = float.NegativeInfinity;
    public float OutMax { get; init; } = float.PositiveInfinity;
    public bool ClampsInput => !float.IsNegativeInfinity(InMin) || !float.IsPositiveInfinity(InMax);
    public bool ClampsOutput => !float.IsNegativeInfinity(OutMin) || !float.IsPositiveInfinity(OutMax);
}

/// <summary>
/// One pre-norm ViT block: x += [post-norm](O·attn(norm1(x))); x += [post-norm](Down·act(norm2(x))).
/// The activation is GELU(Up·x), or GELU(Gate·x) ⊙ Up·x when <see cref="Gate"/> is set.
/// </summary>
public sealed record VitBlock(float[] Ln1W, float[]? Ln1B, float[] Ln2W, float[]? Ln2B,
    VitLinear Q, VitLinear K, VitLinear V, VitLinear O, VitLinear Up, VitLinear Down)
{
    /// <summary>Gated FFN (GeGLU): Down·(gelu(Gate·x) ⊙ Up·x).</summary>
    public VitLinear? Gate { get; init; }
    /// <summary>Per-head RMS norm weights of Q and K (applied before the rotary embedding).</summary>
    public float[]? QNorm { get; init; }
    public float[]? KNorm { get; init; }
    /// <summary>RMS norm applied to the attention / FFN output before the residual add ("sandwich" norm, Gemma).</summary>
    public float[]? AttnPostNorm { get; init; }
    public float[]? FfnPostNorm { get; init; }
}

/// <summary>Normalization used by a <see cref="VitSpec"/>.</summary>
public enum VitNorm { LayerNorm, RmsNorm }

/// <summary>Rotary embedding of a <see cref="VitSpec"/>; the angles come per call as a table (see <see cref="IVitRunner.Run"/>).</summary>
public enum VitRope
{
    None,
    /// <summary>Pairs (i, i + hd/2) over the whole head (Qwen-VL 2D rope).</summary>
    Neox,
    /// <summary>Each half of the head rotated on its own with pairs (i, i + hd/4) (Gemma 4: x position, then y position).</summary>
    NeoxHalves,
}

/// <summary>
/// The transformer part of a vision encoder: SigLIP (LayerNorm, biased attention, GELU MLP), Gemma 4 (RMS norms with
/// q/k/v head norms, sandwich norms, GeGLU, 2D rope, unit attention scale, clamped linears) and Qwen-VL (2D rope).
/// </summary>
public sealed record VitSpec(int Dim, int Heads, float Eps, IReadOnlyList<VitBlock> Blocks, float[]? PostLnW, float[]? PostLnB)
{
    public VitNorm Norm { get; init; } = VitNorm.LayerNorm;
    /// <summary>Weight-less RMS norm of each V head.</summary>
    public bool VNorm { get; init; }
    /// <summary>Attention score scale (0 = 1/√head_dim).</summary>
    public float AttentionScale { get; init; }
    public VitRope Rope { get; init; }
    /// <summary>Optional merger after the final norm, run by <see cref="IVitRunner.RunProjected"/>.</summary>
    public VitMerger? Merger { get; init; }
    /// <summary>Optional patch embedding for whole images on the device (<see cref="IVitRunner.RunImage"/>).</summary>
    public VitPatchEmbedding? PatchEmbedding { get; init; }
}

/// <summary>
/// Patch embedding of a dynamic-resolution ViT (Qwen-VL): an RGB image whose sides are multiples of Patch·Merge is cut into
/// Patch×Patch patches in merge-block order (row-major blocks of Merge×Merge patches), normalized per channel ((v/255 −
/// mean) / std, rows laid out (channel, y, x)), embedded with <see cref="Linear"/> (+ bias), and given the learned square
/// position table (<see cref="PositionSide"/>² rows) resized to the patch grid (bilinear, align corners). The rotary
/// angles are the Qwen-VL 2D ones (first half of the pairs from the row, second half from the column, base 10000).
/// </summary>
public sealed record VitPatchEmbedding(int Patch, int Merge, VitLinear Linear, float[] Mean, float[] Std, float[] Positions, int PositionSide);

/// <summary>
/// Projector over <see cref="Merge"/>² consecutive rows (rows already in merge-block order, so the merge is a reshape to
/// [rows / merge², Dim · merge²]), GELU between stages (Qwen-VL 2×2 merger).
/// </summary>
public sealed record VitMerger(int Merge, IReadOnlyList<VitLinear> Projector)
{
    public int OutDim => Projector[^1].W.Rows;
}

/// <summary>Runs a <see cref="VitSpec"/> on a device.</summary>
public unsafe interface IVitRunner : IDisposable
{
    string DeviceName { get; }

    /// <summary>
    /// Applies all blocks (and the final norm) to <paramref name="x"/> [rows, Dim] in place. Attention is bidirectional
    /// within consecutive groups of <paramref name="groupRows"/> rows (one image tile each). With a rotary embedding,
    /// <paramref name="rope"/> holds per row head_dim/2 interleaved (cos, sin) pairs in rotation-pair order.
    /// </summary>
    void Run(float* x, int rows, int groupRows, float* rope = null);

    /// <summary>
    /// <see cref="Run"/> followed by the spec's <see cref="VitSpec.Merger"/> on the device: <paramref name="output"/> receives
    /// [rows / merge², OutDim]; <paramref name="x"/> is only read.
    /// </summary>
    void RunProjected(float* x, int rows, int groupRows, float* rope, float* output);

    /// <summary>
    /// A whole image on the device: patches, embedding, positions and rotary angles (<see cref="VitSpec.PatchEmbedding"/>),
    /// the blocks, and the <see cref="VitSpec.Merger"/>: <paramref name="rgb"/> (width × height × 3 bytes) in,
    /// <paramref name="output"/> [patches / merge², OutDim] out.
    /// </summary>
    void RunImage(byte* rgb, int width, int height, float* output) => throw new NotSupportedException("This runner has no patch embedding");
}

/// <summary>
/// A whole tiled vision encoder for a device: square RGB tiles → normalized patches in merge-block order → patch
/// embedding (+ bias) + learned positions → <see cref="Vit"/> → projector over <see cref="Merge"/>² consecutive rows
/// (the pixel shuffle is a reshape in block order), GELU between projector stages. Only pixels go in and LLM-space
/// tokens come out.
/// </summary>
public sealed record VitPipelineSpec(VitSpec Vit, int ImageSize, int Patch, int Merge, VitLinear PatchEmbedding, float[] Positions,
    float[] Mean, float[] Std, IReadOnlyList<VitLinear> Projector)
{
    public int Side => ImageSize / Patch;
    public int TokensPerTile => Side * Side / (Merge * Merge);
    public int OutDim => Projector[^1].W.Rows;
}

/// <summary>Runs a <see cref="VitPipelineSpec"/> on a device.</summary>
public unsafe interface IVitPipelineRunner : IDisposable
{
    string DeviceName { get; }

    /// <summary>Encodes tiles (RGB bytes, tiles × ImageSize × ImageSize × 3) into <paramref name="output"/> [tiles · TokensPerTile, OutDim].</summary>
    void Encode(byte* pixels, int tiles, float* output);
}

/// <summary>
/// One block of the Gemma 4 audio Conformer: half-step FFN, chunked local self-attention with relative positions,
/// convolution module (pointwise GLU → causal depthwise conv → norm → SiLU → pointwise), half-step FFN, final norm.
/// </summary>
public sealed record ConformerBlock
{
    public required float[] FfnNorm { get; init; }
    public required float[] FfnPost { get; init; }
    public required float[] FfnNorm1 { get; init; }
    public required float[] FfnPost1 { get; init; }
    public required float[] AttnPre { get; init; }
    public required float[] AttnPost { get; init; }
    /// <summary>Per-dimension query scale (one head's worth, shared by all heads).</summary>
    public required float[] PerDimScale { get; init; }
    public required float[] ConvPreNorm { get; init; }
    public required float[] ConvMidNorm { get; init; }
    public required float[] Ln2 { get; init; }
    /// <summary>Depthwise causal conv weights [Dim][kernel].</summary>
    public required float[] ConvDw { get; init; }
    public required VitLinear Up { get; init; }
    public required VitLinear Down { get; init; }
    public required VitLinear Up1 { get; init; }
    public required VitLinear Down1 { get; init; }
    public required VitLinear Q { get; init; }
    public required VitLinear K { get; init; }
    public required VitLinear V { get; init; }
    public required VitLinear O { get; init; }
    public required VitLinear Pw1 { get; init; }
    public required VitLinear Pw2 { get; init; }
    /// <summary>Relative-position keys (the k_rel projection of the sinusoidal embeddings for distances LeftContext..0): [LeftContext + 1][Dim].</summary>
    public required float[] RelK { get; init; }
}

/// <summary>The Gemma 4 audio Conformer stack (after subsampling and input projection, before the output projections).</summary>
public sealed record ConformerSpec(int Dim, int Heads, float Eps, IReadOnlyList<ConformerBlock> Blocks)
{
    /// <summary>Queries are processed in chunks of this many frames ...</summary>
    public int ChunkSize { get; init; } = 12;
    /// <summary>... attending causally to at most this many previous frames.</summary>
    public int LeftContext { get; init; } = 12;
    public float SoftCap { get; init; } = 50f;
    public float QScale { get; init; }
    public float KScale { get; init; }
}

/// <summary>Runs a <see cref="ConformerSpec"/> on a device.</summary>
public unsafe interface IConformerRunner : IDisposable
{
    string DeviceName { get; }

    /// <summary>Applies all blocks to <paramref name="x"/> [rows, Dim] in place.</summary>
    void Run(float* x, int rows);
}

/// <summary>
/// A device that can run media encoders (vision transformers and the audio Conformer). GPU layer offloads implement it,
/// so an encoder shares the accelerator (and CUDA context) the language model already uses; see <see cref="ProjectorOptions.Accelerator"/>.
/// </summary>
public interface IVisionAccelerator
{
    /// <summary>Uploads the encoder; null when the device cannot run it (unsupported shape, not enough memory).</summary>
    IVitRunner? CreateVit(VitSpec spec);

    /// <summary>Uploads a whole tiled vision encoder (pixels in, tokens out); null when the device cannot run it.</summary>
    IVitPipelineRunner? CreateVitPipeline(VitPipelineSpec spec) => null;

    /// <summary>Uploads an audio Conformer; null when the device cannot run it.</summary>
    IConformerRunner? CreateConformer(ConformerSpec spec) => null;
}
