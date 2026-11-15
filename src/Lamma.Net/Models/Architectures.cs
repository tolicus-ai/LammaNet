namespace LammaNet.Models;

/// <summary>
/// Per-architecture behavior that GGUF metadata does not spell out. Adding a model whose layers the engine already
/// implements (attention, Gated DeltaNet, Mamba2, Gemma per-layer inputs, ...) means adding one entry here; generic
/// hyper-parameters (heads, dims, rope, sliding window, causality, pooling) come from the GGUF metadata.
/// </summary>
public sealed record ArchitectureTraits
{
    /// <summary>Gemma family: embeddings scaled by √d, GELU-gated FFN, rope scaling only on full-attention layers.</summary>
    public bool Gemma { get; init; }
    /// <summary>Attention scores unscaled (q/k RMS norms make 1/√d unnecessary; Gemma 4 family).</summary>
    public bool AttentionScaleOne { get; init; }
    /// <summary>Weight-less RMS norm of V (Gemma 4 family).</summary>
    public bool VNorm { get; init; }
    /// <summary>rope_freqs factors apply only to full-attention layers ("proportional rope", Gemma 4).</summary>
    public bool ProportionalRope { get; init; }
    /// <summary>q_proj outputs [q, gate] per head; the output is gated with sigmoid(gate) (Qwen3.5 / Qwen3-Next).</summary>
    public bool GatedAttention { get; init; }
    /// <summary>The pre-FFN norm is stored as post_attention_norm (Qwen3.5 / Qwen3-Next naming).</summary>
    public bool FfnNormNamedPostAttention { get; init; }
    /// <summary>Hybrid layer layout: recurrent layers marked by attention.recurrent_layers / full_attention_interval.</summary>
    public bool DeltaNetHybrid { get; init; }
    /// <summary>Hybrid layout: layers with head_count_kv == 0 are Mamba2.</summary>
    public bool MambaHybrid { get; init; }
    /// <summary>Hybrid layout: layers with head_count_kv == 0 are gated short convolutions (LFM2).</summary>
    public bool ShortConvHybrid { get; init; }
    /// <summary>rope.scaling.finetuned=false means no positional encoding (Granite 4.0-H).</summary>
    public bool RopeSwitchByFinetuned { get; init; }
    /// <summary>Non-causal sliding windows are symmetric (|Δpos| ≤ window/2) rather than trailing.</summary>
    public bool SymmetricSwa { get; init; }
    /// <summary>Default for tokenizer.ggml.add_bos_token when the metadata omits it.</summary>
    public bool AddBosDefault { get; init; }
    /// <summary>Sliding-window layer period when the metadata has a window but no pattern (gemma2: 2, gemma3: 6).</summary>
    public int SwaPeriod { get; init; }

    private static readonly ArchitectureTraits Gemma4 = new()
    {
        Gemma = true, AttentionScaleOne = true, VNorm = true, ProportionalRope = true, AddBosDefault = true,
    };

    private static readonly Dictionary<string, ArchitectureTraits> s_known = new()
    {
        ["llama"] = new() { AddBosDefault = true },
        ["gemma"] = new() { Gemma = true, AddBosDefault = true },
        ["gemma2"] = new() { Gemma = true, AddBosDefault = true, SwaPeriod = 2 },
        ["gemma3"] = new() { Gemma = true, AddBosDefault = true, SwaPeriod = 6 },
        ["gemma3n"] = new() { Gemma = true, AddBosDefault = true },
        ["gemma4"] = Gemma4,
        // EmbeddingGemma 2: Gemma 4 blocks, bidirectional attention with symmetric windows, mean pooling + projection
        ["gemma-embedding2"] = Gemma4 with { ProportionalRope = false, SymmetricSwa = true },
        ["qwen35"] = new() { GatedAttention = true, FfnNormNamedPostAttention = true, DeltaNetHybrid = true },
        ["qwen3next"] = new() { GatedAttention = true, FfnNormNamedPostAttention = true, DeltaNetHybrid = true },
        ["granite"] = new() { RopeSwitchByFinetuned = true },
        ["granitemoe"] = new() { RopeSwitchByFinetuned = true },
        ["granitehybrid"] = new() { RopeSwitchByFinetuned = true, MambaHybrid = true },
        ["falcon-h1"] = new() { MambaHybrid = true },
        ["nemotron_h"] = new() { MambaHybrid = true },
        ["lfm2"] = new() { ShortConvHybrid = true },
    };

    /// <summary>Traits of <paramref name="architecture"/> (defaults for unknown llama-like architectures).</summary>
    public static ArchitectureTraits For(string architecture) =>
        s_known.TryGetValue(architecture, out var t) ? t
        : architecture.StartsWith("gemma", StringComparison.Ordinal) ? new() { Gemma = true, AddBosDefault = true }
        : new();
}
