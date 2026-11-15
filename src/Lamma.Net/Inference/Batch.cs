using LammaNet.Memory;

namespace LammaNet.Inference;

/// <summary>
/// A micro-batch of positions to evaluate. Positions may belong to different sequences (continuous batching)
/// and each position is either a token id or a raw embedding row (multimodal input).
/// Tokens of the same sequence must appear in increasing position order.
/// </summary>
public sealed class Batch
{
    private readonly List<int> _tokens = new();
    private readonly List<int> _positions = new();
    private readonly List<SequenceState> _seqs = new();
    private readonly List<bool> _logits = new();
    private readonly List<float[]?> _embeddings = new();
    private readonly List<int[]?> _mpos = new();

    public int Count => _tokens.Count;
    public IReadOnlyList<int> Tokens => _tokens;
    public IReadOnlyList<int> Positions => _positions;
    public IReadOnlyList<SequenceState> Sequences => _seqs;
    public IReadOnlyList<bool> WantLogits => _logits;
    public IReadOnlyList<float[]?> Embeddings => _embeddings;
    /// <summary>Optional multi-dimensional rope positions (t, h, w, e) per entry (M-RoPE models).</summary>
    public IReadOnlyList<int[]?> MultiPositions => _mpos;
    public int LogitCount { get; private set; }

    public void Clear()
    {
        _tokens.Clear(); _positions.Clear(); _seqs.Clear(); _logits.Clear(); _embeddings.Clear(); _mpos.Clear();
        LogitCount = 0;
    }

    public void Add(SequenceState seq, int token, int position, bool logits, float[]? embedding = null, int[]? mropePosition = null)
    {
        _tokens.Add(token);
        _positions.Add(position);
        _seqs.Add(seq);
        _logits.Add(logits);
        _embeddings.Add(embedding);
        _mpos.Add(mropePosition);
        if (logits) LogitCount++;
    }
}
