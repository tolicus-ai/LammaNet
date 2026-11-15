using LammaNet.Tensors;

namespace LammaNet.Models;

/// <summary>Weights of one decoder block, with explicit roles independent of GGUF naming quirks.</summary>
public sealed class LayerWeights
{
    public int Index { get; init; }
    public LayerKind Kind { get; init; }

    // ---- normalization
    public float[]? AttnNorm;
    /// <summary>Applied to the mixer output before the residual add (Gemma "post_attention_norm").</summary>
    public float[]? AttnPostNorm;
    /// <summary>Pre-FFN norm.</summary>
    public float[]? FfnNorm;
    /// <summary>Applied to the FFN output before the residual add (Gemma "post_feedforward_norm").</summary>
    public float[]? FfnPostNorm;

    // ---- attention
    public WeightTensor? Wq, Wk, Wv, Wqkv, Wo;
    public float[]? Bq, Bk, Bv, Bo;
    public float[]? QNorm, KNorm;

    // ---- FFN
    public WeightTensor? FfnGate, FfnUp, FfnDown;
    public float[]? FfnGateBias, FfnUpBias, FfnDownBias;

    // ---- per-layer scalar output scale (Gemma 4)
    public float OutScale = 1f;

    // ---- per-layer embeddings (Gemma 3n/4)
    public WeightTensor? PleInpGate, PleProj;
    public float[]? PlePostNorm;

    // ---- Gated DeltaNet (Qwen3.5)
    public WeightTensor? SsmInProjQkv, SsmGate, SsmBeta, SsmAlpha, SsmOut;
    public float[]? SsmConv1d;   // [channels][kernel]
    public float[]? SsmConv1dBias;
    public float[]? SsmDtBias, SsmA, SsmNorm, SsmD;

    // ---- Mamba2
    public WeightTensor? SsmIn;
}
