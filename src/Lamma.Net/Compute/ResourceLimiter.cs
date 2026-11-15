using System.Diagnostics;
using System.Runtime.InteropServices;
using LammaNet.Diagnostics;
using Microsoft.Extensions.Logging;

namespace LammaNet.Compute;

/// <summary>
/// Limits how much of the machine Lamma.Net takes, so it can share it with the application it runs in. One limiter is
/// shared by every engine and media encoder created with it (<see cref="Default"/> unless <c>EngineOptions.Resources</c> /
/// <c>ProjectorOptions.Resources</c> name another), and every limit can be changed at any time; running engines follow
/// within a step.
/// <list type="bullet">
/// <item><see cref="MaxThreads"/>: compute threads taking part in a parallel step (the others sleep, they do not spin).</item>
/// <item><see cref="MaxUtilization"/>: the share of time an engine computes; after each step it pauses for the rest
/// (bounds CPU and GPU use alike, and the heat and power that come with them).</item>
/// <item><see cref="MaxConcurrentSteps"/>: engines computing at the same moment (several agents on several models take
/// turns instead of oversubscribing the cores).</item>
/// <item><see cref="ThreadPriority"/>, <see cref="SpinIterations"/>: how hard idle and busy compute threads compete with
/// the application's threads.</item>
/// <item><see cref="Adaptive"/>: watches the machine's CPU load and lowers the thread count while other processes need
/// the cores, raising it again when they do not.</item>
/// <item><see cref="MaxDeviceMemoryBytes"/>: device memory an accelerator offload may plan with (applies to engines
/// created afterwards).</item>
/// </list>
/// </summary>
public sealed class ResourceLimiter : IDisposable
{
    /// <summary>
    /// The limiter of components created without one. Initialized from the environment: LAMMA_MAX_THREADS,
    /// LAMMA_MAX_UTILIZATION (0.05-1), LAMMA_MAX_STEPS, LAMMA_ADAPTIVE=1, LAMMA_THREAD_PRIORITY (Lowest..Highest),
    /// LAMMA_SPIN, LAMMA_MAX_DEVICE_MB.
    /// </summary>
    public static ResourceLimiter Default { get; } = FromEnvironment();

    private volatile int _maxThreads;          // 0 = unlimited
    private volatile int _adaptiveThreads;     // 0 = no adaptive cap
    private double _maxUtilization = 1;
    private volatile int _maxSteps;            // 0 = unlimited
    private volatile int _priorityVersion;
    private ThreadPriority _priority = ThreadPriority.Normal;
    private volatile int _spin = 20000;
    private volatile bool _yielding;           // adaptive: other processes are busy
    private long _maxDeviceMemory = -1;
    private readonly object _stepLock = new();
    private int _runningSteps, _peakSteps;
    private Timer? _monitor;
    private CpuSampler? _sampler;
    private double _otherCores;

    /// <summary>Logging of adaptive changes (default: <see cref="LammaLogging.LoggerFactory"/>).</summary>
    public ILoggerFactory? LoggerFactory { get; set; }

    /// <summary>Compute threads that take part in one parallel step of a pool (engine or encoder); null = all of them.</summary>
    public int? MaxThreads
    {
        get => _maxThreads > 0 ? _maxThreads : null;
        set => _maxThreads = value is int v ? Math.Max(1, v) : 0;
    }

    /// <summary>
    /// Share of time (0.05-1) an engine spends computing: after a step of duration t it pauses t·(1/u − 1). 1 (default)
    /// never pauses.
    /// </summary>
    public double MaxUtilization
    {
        get => Volatile.Read(ref _maxUtilization);
        set => Volatile.Write(ref _maxUtilization, double.IsNaN(value) ? 1 : Math.Clamp(value, 0.05, 1));
    }

    /// <summary>Engines sharing this limiter that may compute a step at the same moment; null = unlimited.</summary>
    public int? MaxConcurrentSteps
    {
        get => _maxSteps > 0 ? _maxSteps : null;
        set
        {
            _maxSteps = value is int v ? Math.Max(1, v) : 0;
            lock (_stepLock) Monitor.PulseAll(_stepLock);
        }
    }

    /// <summary>Priority of the compute threads (default Normal, so the application's own threads are not starved).</summary>
    public ThreadPriority ThreadPriority
    {
        get => _priority;
        set { _priority = value; Interlocked.Increment(ref _priorityVersion); }
    }

    /// <summary>
    /// Spin iterations of an idle compute thread before it sleeps (default 20000, a few hundred microseconds: keeps
    /// back-to-back kernels fast). Lower values free the cores sooner between steps.
    /// </summary>
    public int SpinIterations
    {
        get => _spin;
        set => _spin = Math.Max(0, value);
    }

    /// <summary>
    /// Watch the machine's CPU load (every 0.5 s) and use fewer compute threads while other processes need the cores
    /// (down to one), spinning only briefly meanwhile; the limits above still apply.
    /// </summary>
    public bool Adaptive
    {
        get => _monitor != null;
        set
        {
            lock (_stepLock)
            {
                if (value && _monitor == null)
                {
                    _sampler = CpuSampler.Create();
                    if (_sampler == null)
                    {
                        Log.AdaptiveUnavailable(LammaLogging.Create<ResourceLimiter>(LoggerFactory));
                        return;
                    }
                    _monitor = new Timer(_ => Sample(), null, 0, 500);
                }
                else if (!value && _monitor != null)
                {
                    _monitor.Dispose();
                    _monitor = null;
                    _adaptiveThreads = 0;
                    _yielding = false;
                }
            }
        }
    }

    /// <summary>Device memory an accelerator offload may use for weights, caches and scratch; null = all free memory but its reserve.</summary>
    public long? MaxDeviceMemoryBytes
    {
        get => Interlocked.Read(ref _maxDeviceMemory) is var v && v >= 0 ? v : null;
        set => Interlocked.Exchange(ref _maxDeviceMemory, value is long v ? Math.Max(0, v) : -1);
    }

    /// <summary>The thread cap in effect: <see cref="MaxThreads"/>, lowered by <see cref="Adaptive"/> load control (null = none).</summary>
    public int? EffectiveMaxThreads
    {
        get
        {
            int a = _maxThreads, b = _adaptiveThreads;
            int c = a > 0 && b > 0 ? Math.Min(a, b) : Math.Max(a, b);
            return c > 0 ? c : null;
        }
    }

    /// <summary>CPU cores other processes used recently (adaptive mode; 0 otherwise).</summary>
    public double OtherProcessesCores => Volatile.Read(ref _otherCores);

    /// <summary>Most engines seen computing at the same moment (diagnostics).</summary>
    public int PeakConcurrentSteps => Volatile.Read(ref _peakSteps);

    // ------------------------------------------------------------------ used by pools and engines

    /// <summary>Threads of a pool of <paramref name="poolThreads"/> that take part in the next step.</summary>
    internal int ActiveThreads(int poolThreads) => EffectiveMaxThreads is int c ? Math.Clamp(c, 1, poolThreads) : poolThreads;

    /// <summary>Spin iterations for idle compute threads (short while other processes are busy).</summary>
    internal int Spin(int poolDefault) => _yielding ? Math.Min(_spin, 200) : Math.Min(_spin, poolDefault);

    internal int PriorityVersion => _priorityVersion;

    /// <summary>Waits for a step slot (<see cref="MaxConcurrentSteps"/>); dispose the result when the step is done.</summary>
    internal StepSlot EnterStep(CancellationToken ct)
    {
        lock (_stepLock)
        {
            while (_maxSteps > 0 && _runningSteps >= _maxSteps)
            {
                ct.ThrowIfCancellationRequested();
                Monitor.Wait(_stepLock, 50);
            }
            _runningSteps++;
            if (_runningSteps > _peakSteps) _peakSteps = _runningSteps;
        }
        return new StepSlot(this);
    }

    private void ExitStep()
    {
        lock (_stepLock)
        {
            _runningSteps--;
            Monitor.Pulse(_stepLock);
        }
    }

    /// <summary>Pauses after a step of <paramref name="stepTime"/> to keep the engine at <see cref="MaxUtilization"/>.</summary>
    internal void PauseAfterStep(TimeSpan stepTime, CancellationToken ct)
    {
        double u = MaxUtilization;
        if (u >= 1) return;
        var pause = TimeSpan.FromTicks(Math.Min((long)(stepTime.Ticks * (1 / u - 1)), TimeSpan.TicksPerSecond));
        if (pause > TimeSpan.Zero) ct.WaitHandle.WaitOne(pause);
    }

    internal readonly struct StepSlot(ResourceLimiter owner) : IDisposable
    {
        public void Dispose() => owner.ExitStep();
    }

    // ------------------------------------------------------------------ adaptive load control

    /// <summary>Feeds one load sample: <paramref name="otherCores"/> cores busy in other processes (tests drive this directly).</summary>
    internal void ApplyLoad(double otherCores)
    {
        int cores = CpuTopology.PhysicalCores, logical = Environment.ProcessorCount;
        // other processes' load, smoothed (~2 s), in physical cores
        double prev = Volatile.Read(ref _otherCores);
        double smooth = prev + (otherCores - prev) * 0.35;
        Volatile.Write(ref _otherCores, smooth);
        double physicalBusy = smooth * cores / Math.Max(1, logical);
        int target = Math.Clamp(cores - (int)Math.Ceiling(physicalBusy - 0.25), 1, cores);
        int old = _adaptiveThreads;
        // hysteresis: give threads back one at a time, take them away at once
        int next = old == 0 ? target : target < old ? target : Math.Min(target, old + 1);
        _yielding = physicalBusy > 0.5;
        if (next != old)
        {
            _adaptiveThreads = next >= cores ? 0 : next;
            Log.AdaptiveThreads(LammaLogging.Create<ResourceLimiter>(LoggerFactory), next, smooth);
        }
    }

    private void Sample()
    {
        try
        {
            if (_sampler?.Sample() is double other) ApplyLoad(other);
        }
        catch (Exception e)
        {
            Log.AdaptiveSampleFailed(LammaLogging.Create<ResourceLimiter>(LoggerFactory), e);
        }
    }

    public void Dispose() => Adaptive = false;

    private static ResourceLimiter FromEnvironment()
    {
        var l = new ResourceLimiter();
        static string? E(string n) => Environment.GetEnvironmentVariable(n) is { Length: > 0 } v ? v : null;
        if (int.TryParse(E("LAMMA_MAX_THREADS"), out int t) && t > 0) l.MaxThreads = t;
        if (double.TryParse(E("LAMMA_MAX_UTILIZATION"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double u)) l.MaxUtilization = u;
        if (int.TryParse(E("LAMMA_MAX_STEPS"), out int s) && s > 0) l.MaxConcurrentSteps = s;
        if (Enum.TryParse<ThreadPriority>(E("LAMMA_THREAD_PRIORITY"), true, out var pr)) l.ThreadPriority = pr;
        if (int.TryParse(E("LAMMA_SPIN"), out int sp)) l.SpinIterations = sp;
        if (long.TryParse(E("LAMMA_MAX_DEVICE_MB"), out long mb) && mb > 0) l.MaxDeviceMemoryBytes = mb << 20;
        if (E("LAMMA_ADAPTIVE") is "1" or "true") l.Adaptive = true;
        return l;
    }

    /// <summary>CPU time of the whole machine and of this process between two samples, as cores busy in other processes.</summary>
    private sealed class CpuSampler
    {
        private readonly Func<(long Busy, long Total)?> _system;
        private (long Busy, long Total) _lastSys;
        private TimeSpan _lastOwn;
        private long _lastTicks;

        private CpuSampler(Func<(long, long)?> system, (long, long) first)
        {
            _system = system;
            _lastSys = first;
            _lastOwn = Process.GetCurrentProcess().TotalProcessorTime;
            _lastTicks = Stopwatch.GetTimestamp();
        }

        public static CpuSampler? Create()
        {
            Func<(long, long)?>? f = OperatingSystem.IsWindows() ? WindowsTimes : OperatingSystem.IsLinux() ? LinuxTimes : null;
            return f?.Invoke() is { } first ? new CpuSampler(f, first) : null;
        }

        /// <summary>Cores busy in other processes since the previous sample.</summary>
        public double? Sample()
        {
            if (_system() is not { } sys) return null;
            var own = Process.GetCurrentProcess().TotalProcessorTime;
            long now = Stopwatch.GetTimestamp();
            double wall = Stopwatch.GetElapsedTime(_lastTicks, now).TotalSeconds;
            long dBusy = sys.Busy - _lastSys.Busy, dTotal = sys.Total - _lastSys.Total;
            double ownCores = (own - _lastOwn).TotalSeconds / Math.Max(wall, 1e-3);
            _lastSys = sys; _lastOwn = own; _lastTicks = now;
            if (dTotal <= 0) return null;
            double busyCores = (double)dBusy / dTotal * Environment.ProcessorCount;
            return Math.Max(0, busyCores - ownCores);
        }

        private static (long, long)? WindowsTimes()
        {
            if (!GetSystemTimes(out long idle, out long kernel, out long user)) return null;
            long total = kernel + user;   // kernel time includes idle time
            return (total - idle, total);
        }

        private static (long, long)? LinuxTimes()
        {
            try
            {
                var parts = File.ReadLines("/proc/stat").First().Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1).Select(long.Parse).ToArray();
                long idle = parts[3] + (parts.Length > 4 ? parts[4] : 0), total = parts.Sum();
                return (total - idle, total);
            }
            catch (Exception e) when (e is IOException or FormatException or UnauthorizedAccessException) { return null; }
        }

        [DllImport("kernel32", SetLastError = true)]
        private static extern bool GetSystemTimes(out long idle, out long kernel, out long user);
    }
}
