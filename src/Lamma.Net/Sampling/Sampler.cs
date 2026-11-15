namespace LammaNet.Sampling;

/// <summary>
/// Restricts which tokens may be generated next (e.g. JSON grammar, tool-call schemas).
/// </summary>
public interface ITokenConstraint
{
    /// <summary>Returns true if <paramref name="token"/> may be emitted in the current state.</summary>
    bool IsAllowed(int token);
    /// <summary>Advances the constraint state with an emitted token.</summary>
    void Accept(int token);
    /// <summary>True once the constrained output is complete (generation should stop).</summary>
    bool IsComplete { get; }
    /// <summary>Tokens that are allowed only when the constrained output is complete (e.g. EOS).</summary>
    bool AllowsEndOfGeneration { get; }
}

/// <summary>
/// Stateful token sampler implementing the llama.cpp sampling chain:
/// logit bias → penalties → DRY → top-k → typical → top-p → min-p → temperature → draw,
/// with optional constrained decoding (lazy: only re-samples when a constraint rejects the pick).
/// </summary>
public sealed class Sampler
{
    private readonly SamplingOptions _o;
    private readonly Random _rng;
    private readonly List<int> _history = new();
    private (int Id, float Logit)[] _cand = [];
    private float[] _buf = [];

    public ITokenConstraint? Constraint { get; set; }
    public SamplingOptions Options => _o;
    public IReadOnlyList<int> History => _history;

    /// <summary>End-of-generation tokens (used to gate them by the constraint).</summary>
    public IReadOnlySet<int>? EndTokens { get; set; }

    public Sampler(SamplingOptions? options = null)
    {
        _o = options ?? new SamplingOptions();
        _rng = _o.Seed is int s ? new Random(s) : new Random();
    }

    /// <summary>Feeds previously known tokens (prompt) into the penalty history.</summary>
    public void AcceptPrompt(IEnumerable<int> tokens) => _history.AddRange(tokens);

    /// <summary>Records a generated token.</summary>
    public void Accept(int token)
    {
        _history.Add(token);
        Constraint?.Accept(token);
    }

    private bool Allowed(int id)
    {
        var c = Constraint;
        if (c == null) return true;
        if (EndTokens != null && EndTokens.Contains(id)) return c.AllowsEndOfGeneration;
        return c.IsAllowed(id);
    }

    /// <summary>Samples the next token from <paramref name="logits"/> (not modified).</summary>
    public int Sample(ReadOnlySpan<float> logits)
    {
        int n = logits.Length;
        if (_buf.Length < n) _buf = new float[n];
        var l = _buf.AsSpan(0, n);
        logits.CopyTo(l);
        if (_o.LogitBias != null)
            foreach (var (id, b) in _o.LogitBias) if ((uint)id < (uint)n) l[id] += b;
        ApplyPenalties(l);
        if (_o.DryMultiplier > 0) ApplyDry(l);

        int pick = SampleFrom(l, constrained: false);
        if (Constraint != null && !Allowed(pick))
        {
            for (int i = 0; i < n; i++) if (!Allowed(i)) l[i] = float.NegativeInfinity;
            pick = SampleFrom(l, constrained: true);
        }
        return pick;
    }

    private int SampleFrom(Span<float> l, bool constrained)
    {
        int n = l.Length;
        if (_o.Temperature <= 0 || _o.TopK == 1)
        {
            int best = 0;
            for (int i = 1; i < n; i++) if (l[i] > l[best]) best = i;
            return best;
        }

        // ---- top-k (selection without full sort)
        int k = _o.TopK > 0 ? Math.Min(_o.TopK, n) : n;
        if (_cand.Length < Math.Max(k, 1)) _cand = new (int, float)[Math.Max(k, 64)];
        int m;
        if (k < n) m = TopK(l, k);
        else
        {
            if (_cand.Length < n) _cand = new (int, float)[n];
            for (int i = 0; i < n; i++) _cand[i] = (i, l[i]);
            m = n;
        }
        var cand = _cand.AsSpan(0, m);
        cand.Sort((a, b) => b.Logit.CompareTo(a.Logit));
        while (m > 1 && float.IsNegativeInfinity(cand[m - 1].Logit)) m--;
        cand = cand[..m];

        // probabilities at temperature 1 for the truncation filters
        Span<float> p = m <= 4096 ? stackalloc float[m] : new float[m];
        Softmax(cand, p, 1f);

        // ---- typical
        if (_o.TypicalP < 1f && m > 1)
        {
            double ent = 0;
            for (int i = 0; i < m; i++) if (p[i] > 0) ent -= p[i] * Math.Log(p[i]);
            var pa = p.ToArray();
            var idx = Enumerable.Range(0, m).OrderBy(i => Math.Abs(-Math.Log(Math.Max(pa[i], 1e-30)) - ent)).ToArray();
            double cum = 0; int keep = 0;
            var keepSet = new bool[m];
            foreach (var i in idx) { keepSet[i] = true; cum += p[i]; keep++; if (cum >= _o.TypicalP && keep >= 1) break; }
            int w = 0;
            for (int i = 0; i < m; i++) if (keepSet[i]) cand[w++] = cand[i];
            m = w; cand = cand[..m];
            p = p[..m];
            Softmax(cand, p, 1f);
        }

        // ---- top-p
        if (_o.TopP < 1f && m > 1)
        {
            float cum = 0; int keep = m;
            for (int i = 0; i < m; i++) { cum += p[i]; if (cum >= _o.TopP) { keep = i + 1; break; } }
            m = keep; cand = cand[..m]; p = p[..m];
        }

        // ---- min-p (relative to the top token)
        if (_o.MinP > 0f && m > 1)
        {
            float thr = p[0] * _o.MinP;
            int keep = 1;
            while (keep < m && p[keep] >= thr) keep++;
            m = keep; cand = cand[..m]; p = p[..m];
        }

        // ---- temperature + draw
        Softmax(cand, p, _o.Temperature);
        double r = _rng.NextDouble();
        double acc = 0;
        for (int i = 0; i < m; i++)
        {
            acc += p[i];
            if (r < acc) return cand[i].Id;
        }
        return cand[m - 1].Id;
    }

    private static void Softmax(Span<(int Id, float Logit)> c, Span<float> p, float temp)
    {
        float max = float.NegativeInfinity;
        foreach (var x in c) max = Math.Max(max, x.Logit);
        double sum = 0;
        for (int i = 0; i < c.Length; i++) { p[i] = MathF.Exp((c[i].Logit - max) / temp); sum += p[i]; }
        for (int i = 0; i < c.Length; i++) p[i] = (float)(p[i] / sum);
    }

    /// <summary>Selects the k largest logits into <see cref="_cand"/> using a size-k min-heap.</summary>
    private int TopK(Span<float> l, int k)
    {
        var heap = new PriorityQueue<int, float>(k + 1);
        for (int i = 0; i < l.Length; i++)
        {
            float v = l[i];
            if (heap.Count < k) heap.Enqueue(i, v);
            else if (heap.TryPeek(out _, out var min) && v > min) heap.EnqueueDequeue(i, v);
        }
        int m = 0;
        while (heap.TryDequeue(out int id, out float v)) _cand[m++] = (id, v);
        return m;
    }

    private void ApplyPenalties(Span<float> l)
    {
        if (_o.RepetitionPenalty == 1f && _o.FrequencyPenalty == 0 && _o.PresencePenalty == 0) return;
        int start = Math.Max(0, _history.Count - (_o.PenaltyLastN < 0 ? _history.Count : _o.PenaltyLastN));
        var counts = new Dictionary<int, int>();
        for (int i = start; i < _history.Count; i++) counts[_history[i]] = counts.GetValueOrDefault(_history[i]) + 1;
        foreach (var (id, cnt) in counts)
        {
            if ((uint)id >= (uint)l.Length) continue;
            ref float v = ref l[id];
            if (_o.RepetitionPenalty != 1f) v = v <= 0 ? v * _o.RepetitionPenalty : v / _o.RepetitionPenalty;
            v -= cnt * _o.FrequencyPenalty + (cnt > 0 ? 1 : 0) * _o.PresencePenalty;
        }
    }

    /// <summary>DRY: penalizes tokens that would extend a repeated sequence beyond the allowed length.</summary>
    private void ApplyDry(Span<float> l)
    {
        var h = _history;
        int n = h.Count;
        if (n < 2) return;
        int last = h[n - 1];
        var maxLen = new Dictionary<int, int>();
        int window = Math.Min(n - 1, 1024);
        for (int i = n - 2; i >= n - 1 - window; i--)
        {
            if (h[i] != last) continue;
            // length of match ending at i vs ending at n-1
            int len = 1;
            while (len <= i && h[i - len] == h[n - 1 - len] && len < 64) len++;
            int next = h[i + 1];
            if (!maxLen.TryGetValue(next, out var cur) || len > cur) maxLen[next] = len;
        }
        foreach (var (tok, len) in maxLen)
        {
            if (len < _o.DryAllowedLength || (uint)tok >= (uint)l.Length) continue;
            l[tok] -= _o.DryMultiplier * MathF.Pow(_o.DryBase, len - _o.DryAllowedLength);
        }
    }
}
