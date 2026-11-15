namespace LammaNet.Inference;

/// <summary>A piece of model input: either text tokens or a block of pre-computed embeddings (image/audio).</summary>
public abstract record PromptSegment
{
    /// <summary>Number of positions this segment occupies.</summary>
    public abstract int Length { get; }
}

/// <summary>Token ids.</summary>
public sealed record TokenSegment(IReadOnlyList<int> Tokens) : PromptSegment
{
    public override int Length => Tokens.Count;
}

/// <summary>
/// Embedding rows injected directly into the residual stream (output of a vision / audio encoder).
/// <see cref="ContentHash"/> identifies the media so prompt-prefix caching can reuse it safely.
/// </summary>
public sealed record EmbeddingSegment(float[] Data, int Rows, int Dim, long ContentHash, string Modality = "image") : PromptSegment
{
    public override int Length => Rows;

    /// <summary>Optional per-row M-RoPE positions (t, h, w, e offsets relative to the segment's first rope position).</summary>
    public int[][]? RopePositions { get; init; }

    /// <summary>Rope positions consumed by this segment for M-RoPE models (e.g. max(grid w, h) for Qwen-VL images); defaults to Rows.</summary>
    public int? RopeSpan { get; init; }

    public float[] Row(int i) => Data.AsSpan(i * Dim, Dim).ToArray();
}

/// <summary>A full model prompt made of token and embedding segments.</summary>
public sealed class Prompt
{
    private readonly List<PromptSegment> _segments = new();

    public IReadOnlyList<PromptSegment> Segments => _segments;
    public int Length => _segments.Sum(s => s.Length);

    public Prompt() { }
    public Prompt(IEnumerable<int> tokens) => Add(tokens);

    public Prompt Add(IEnumerable<int> tokens)
    {
        var list = tokens as IReadOnlyList<int> ?? tokens.ToList();
        if (list.Count == 0) return this;
        if (_segments.Count > 0 && _segments[^1] is TokenSegment ts)
            _segments[^1] = new TokenSegment(ts.Tokens.Concat(list).ToList());
        else _segments.Add(new TokenSegment(list.ToList()));
        return this;
    }

    public Prompt Add(EmbeddingSegment embeddings)
    {
        _segments.Add(embeddings);
        return this;
    }

    /// <summary>
    /// Identity of each position for prefix matching: token id, or a negative pseudo-id derived
    /// from the media hash and row index for embedding positions.
    /// </summary>
    public List<int> PositionKeys()
    {
        var keys = new List<int>(Length);
        foreach (var s in _segments)
        {
            switch (s)
            {
                case TokenSegment t: keys.AddRange(t.Tokens); break;
                case EmbeddingSegment e:
                    for (int i = 0; i < e.Rows; i++)
                        keys.Add(-(int)((((ulong)e.ContentHash * 31UL + (ulong)i) & 0x3fffffffUL) + 1));
                    break;
            }
        }
        return keys;
    }
}
