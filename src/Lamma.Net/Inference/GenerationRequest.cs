using LammaNet.Sampling;

namespace LammaNet.Inference;

/// <summary>Why generation ended.</summary>
public enum FinishReason
{
    None,
    /// <summary>An end-of-generation token was produced.</summary>
    Stop,
    /// <summary>The maximum token count was reached.</summary>
    Length,
    /// <summary>A configured stop sequence was produced.</summary>
    StopSequence,
    /// <summary>The constrained output (grammar/JSON) completed.</summary>
    ConstraintComplete,
    /// <summary>The request was cancelled.</summary>
    Cancelled,
    /// <summary>The context window or KV cache is full.</summary>
    ContextFull,
    Error,
}

/// <summary>A text-generation request for the <see cref="InferenceEngine"/>.</summary>
public sealed class GenerationRequest
{
    public required Prompt Prompt { get; init; }
    public SamplingOptions Sampling { get; init; } = new();
    /// <summary>Maximum generated tokens (null = until stop or context end).</summary>
    public int? MaxTokens { get; init; }
    /// <summary>Text sequences that end generation (not included in the output).</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];
    /// <summary>Additional token ids that end generation.</summary>
    public IReadOnlySet<int>? StopTokens { get; init; }
    /// <summary>Optional decoding constraint (e.g. JSON).</summary>
    public ITokenConstraint? Constraint { get; init; }
    /// <summary>Render control/special tokens in the output text.</summary>
    public bool RenderSpecialTokens { get; init; }
    /// <summary>Do not stop on end-of-generation tokens (e.g. to continue a raw completion).</summary>
    public bool IgnoreEos { get; init; }
    /// <summary>Keep the KV state cached after completion for prompt-prefix reuse (default true).</summary>
    public bool CachePrompt { get; init; } = true;
}

/// <summary>One streamed generation event.</summary>
public readonly record struct GeneratedToken(int TokenId, string Text, FinishReason FinishReason = FinishReason.None)
{
    public bool IsFinal => FinishReason != FinishReason.None;
}

/// <summary>Usage statistics of a finished request.</summary>
public sealed record GenerationStats
{
    public int PromptTokens { get; init; }
    /// <summary>Prompt positions served from the prefix cache (not recomputed).</summary>
    public int CachedPromptTokens { get; init; }
    public int CompletionTokens { get; init; }
    public TimeSpan PromptTime { get; init; }
    public TimeSpan GenerationTime { get; init; }
    public FinishReason FinishReason { get; init; }

    public double PromptTokensPerSecond => PromptTime.TotalSeconds > 0 ? (PromptTokens - CachedPromptTokens) / PromptTime.TotalSeconds : 0;
    public double TokensPerSecond => GenerationTime.TotalSeconds > 0 ? CompletionTokens / GenerationTime.TotalSeconds : 0;
}
