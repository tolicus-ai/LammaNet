namespace LammaNet.Inference;

/// <summary>
/// Streams text while detecting stop sequences: text that could be the beginning of a stop sequence is
/// held back until it is either confirmed (generation stops, the stop text is dropped) or ruled out.
/// </summary>
internal sealed class StopDetector(IReadOnlyList<string> stops)
{
    private readonly IReadOnlyList<string> _stops = stops.Where(s => s.Length > 0).ToList();
    private string _pending = "";

    public bool Stopped { get; private set; }

    /// <summary>Adds text; returns the text that is safe to emit.</summary>
    public string Push(string text)
    {
        if (Stopped) return "";
        if (_stops.Count == 0) return text;
        _pending += text;
        int best = -1;
        foreach (var s in _stops)
        {
            int idx = _pending.IndexOf(s, StringComparison.Ordinal);
            if (idx >= 0 && (best < 0 || idx < best)) best = idx;
        }
        if (best >= 0)
        {
            Stopped = true;
            var emit = _pending[..best];
            _pending = "";
            return emit;
        }
        // hold back the longest suffix that is a prefix of a stop sequence
        int hold = 0;
        foreach (var s in _stops)
            for (int k = Math.Min(s.Length - 1, _pending.Length); k > hold; k--)
                if (_pending.AsSpan(_pending.Length - k).SequenceEqual(s.AsSpan(0, k))) { hold = k; break; }
        var safe = _pending[..^hold];
        _pending = _pending[^hold..];
        return safe;
    }

    /// <summary>Returns any held-back text at the end of generation.</summary>
    public string Flush()
    {
        var p = Stopped ? "" : _pending;
        _pending = "";
        return p;
    }
}
