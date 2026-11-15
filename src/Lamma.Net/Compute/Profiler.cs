using System.Diagnostics;

namespace LammaNet.Compute;

/// <summary>Lightweight opt-in section profiler (set LAMMA_PROFILE=1). Zero cost when disabled.</summary>
public static class Profiler
{
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("LAMMA_PROFILE") == "1";
    private static readonly Dictionary<string, (long Ticks, long Count)> s_sections = new();

    public readonly struct Scope : IDisposable
    {
        private readonly string? _name;
        private readonly long _start;
        public Scope(string name) { _name = name; _start = Stopwatch.GetTimestamp(); }
        public void Dispose()
        {
            if (_name == null) return;
            long el = Stopwatch.GetTimestamp() - _start;
            lock (s_sections)
            {
                s_sections.TryGetValue(_name, out var v);
                s_sections[_name] = (v.Ticks + el, v.Count + 1);
            }
        }
    }

    public static Scope Section(string name) => Enabled ? new Scope(name) : default;

    public static void Report(TextWriter w)
    {
        if (!Enabled) return;
        lock (s_sections)
        {
            foreach (var (k, v) in s_sections.OrderByDescending(kv => kv.Value.Ticks))
                w.WriteLine($"{k,-28} {v.Ticks * 1000.0 / Stopwatch.Frequency,10:F1} ms  {v.Count,8} calls  {v.Ticks * 1e6 / Stopwatch.Frequency / Math.Max(1, v.Count),8:F1} us/call");
            s_sections.Clear();
        }
    }
}
