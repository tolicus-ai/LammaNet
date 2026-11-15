namespace LammaNet.Sampling;

/// <summary>Token sampling configuration. Defaults follow llama.cpp.</summary>
public sealed record SamplingOptions
{
    /// <summary>Temperature; 0 (or less) means greedy decoding.</summary>
    public float Temperature { get; init; } = 0.8f;
    public int TopK { get; init; } = 40;
    public float TopP { get; init; } = 0.95f;
    public float MinP { get; init; } = 0.05f;
    /// <summary>Locally typical sampling (1 disables).</summary>
    public float TypicalP { get; init; } = 1f;
    /// <summary>Penalty applied to logits of tokens seen in the last <see cref="PenaltyLastN"/> tokens (1 disables).</summary>
    public float RepetitionPenalty { get; init; } = 1f;
    public float FrequencyPenalty { get; init; }
    public float PresencePenalty { get; init; }
    public int PenaltyLastN { get; init; } = 64;
    /// <summary>DRY (don't repeat yourself) multiplier; 0 disables.</summary>
    public float DryMultiplier { get; init; }
    public float DryBase { get; init; } = 1.75f;
    public int DryAllowedLength { get; init; } = 2;
    /// <summary>Random seed; null uses a random seed.</summary>
    public int? Seed { get; init; }
    /// <summary>Additive logit biases per token id.</summary>
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; }

    public static SamplingOptions Greedy { get; } = new() { Temperature = 0, TopK = 1 };
}
