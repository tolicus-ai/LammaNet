using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;

namespace LammaNet.Compute;

/// <summary>A unit of parallel work. <see cref="Execute"/> is called once per task index.</summary>
public abstract class ParallelJob
{
    public abstract void Execute(int index, int threadId);
}

/// <summary>
/// Low-latency fork/join thread pool for compute kernels. Workers spin briefly on a generation
/// counter before sleeping, so dispatching a kernel costs well under a microsecond while hot
/// (vs ~5-20µs for <see cref="Parallel.For(int,int,Action{int})"/>). Work items are claimed with an atomic
/// counter for natural load balancing. The calling thread participates as worker 0. An exception thrown by a task on any
/// thread stops the remaining tasks and is rethrown by <see cref="Run"/> once every worker has left the job.
/// With a <see cref="ResourceLimiter"/>, each step uses at most its thread cap (the other workers sleep until the cap
/// includes them again), idle workers spin as long as the limiter allows, and the threads take its priority.
/// </summary>
public sealed class ComputePool : IDisposable
{
    private readonly Thread[] _threads;
    private readonly int _nThreads;
    private ParallelJob? _job;
    private int _taskCount;
    private int _nextTask;
    private int _pending;
    private long _generation;
    private volatile bool _disposed;
    private readonly object _sleepLock = new();
    private int _sleepers;
    private readonly int _spinIterations;
    private ExceptionDispatchInfo? _error;
    private readonly ResourceLimiter? _limiter;
    private volatile int _active;              // workers (incl. the caller) taking part in the current step
    private readonly object _parkLock = new(); // workers outside the cap wait here
    private int _priorityVersion = -1;
    [ThreadStatic] private static int t_threadId;

    public int ThreadCount => _nThreads;

    /// <summary>Global default pool sized to the physical core count.</summary>
    public static ComputePool Default => s_default.Value;
    private static readonly Lazy<ComputePool> s_default = new(() => new ComputePool(DefaultThreadCount()));

    /// <summary>One compute thread per physical core (SMT siblings do not help the bandwidth-bound kernels).</summary>
    public static int DefaultThreadCount() => CpuTopology.PhysicalCores;

    public ComputePool(int threads, int spinIterations = 20000) : this(threads, null, spinIterations) { }

    /// <summary>A pool of <paramref name="threads"/> threads that follows <paramref name="limiter"/> (null: all threads, AboveNormal priority).</summary>
    public ComputePool(int threads, ResourceLimiter? limiter, int spinIterations = 20000)
    {
        _nThreads = Math.Max(1, threads);
        _spinIterations = spinIterations;
        _limiter = limiter;
        _active = _nThreads;
        _threads = new Thread[_nThreads - 1];
        for (int i = 0; i < _threads.Length; i++)
        {
            int id = i + 1;
            _threads[i] = new Thread(() => WorkerLoop(id)) { IsBackground = true, Name = $"lamma-compute-{id}", Priority = limiter?.ThreadPriority ?? ThreadPriority.AboveNormal };
            _threads[i].Start();
        }
        _priorityVersion = limiter?.PriorityVersion ?? 0;
    }

    /// <summary>The limiter this pool follows, if any.</summary>
    public ResourceLimiter? Limiter => _limiter;

    /// <summary>Runs <paramref name="job"/> for task indices [0, <paramref name="tasks"/>), blocking until done.</summary>
    public void Run(ParallelJob job, int tasks)
    {
        if (tasks <= 0) return;
        int active = _limiter?.ActiveThreads(_nThreads) ?? _nThreads;
        if (tasks == 1 || active == 1)
        {
            if (active != _active) lock (this) _active = active;   // the workers park (never while a step counts them)
            for (int i = 0; i < tasks; i++) job.Execute(i, 0);
            return;
        }
        lock (this) // one dispatch at a time
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ApplyPriority();
            _job = job;
            _taskCount = tasks;
            _nextTask = 0;
            _error = null;
            int before = _active;
            _active = active;
            _pending = active - 1;
            if (active > before)
                lock (_parkLock) Monitor.PulseAll(_parkLock);
            Interlocked.Increment(ref _generation);
            if (Volatile.Read(ref _sleepers) > 0)
                lock (_sleepLock) Monitor.PulseAll(_sleepLock);

            try { DoWork(0); }
            catch (Exception e) { Fault(e); }

            // the workers may still be inside the job (and its buffers) until they have all checked out
            var sw = new SpinWait();
            while (Volatile.Read(ref _pending) > 0) sw.SpinOnce(-1);
            _job = null;
            Interlocked.Exchange(ref _error, null)?.Throw();
        }
    }

    /// <summary>Keeps the first exception of a job and lets the remaining tasks go unclaimed.</summary>
    private void Fault(Exception e)
    {
        Interlocked.CompareExchange(ref _error, ExceptionDispatchInfo.Capture(e), null);
        Volatile.Write(ref _nextTask, _taskCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DoWork(int threadId)
    {
        var job = _job!;
        int n = _taskCount;
        int i;
        while ((i = Interlocked.Increment(ref _nextTask) - 1) < n) job.Execute(i, threadId);
    }

    /// <summary>Takes the limiter's thread priority when it changed (under the dispatch lock).</summary>
    private void ApplyPriority()
    {
        if (_limiter == null || _limiter.PriorityVersion == _priorityVersion) return;
        _priorityVersion = _limiter.PriorityVersion;
        var pr = _limiter.ThreadPriority;
        foreach (var t in _threads)
            try { t.Priority = pr; } catch (ThreadStateException) { /* exited (disposed) */ }
    }

    private void WorkerLoop(int id)
    {
        t_threadId = id;
        long seen = 0;
        while (!_disposed)
        {
            if (id >= _active)
            {
                // outside the thread cap: sleep (no spinning) until a step includes this worker again; `seen` is kept,
                // so a step that already counts this worker is not missed
                lock (_parkLock)
                    while (id >= _active && !_disposed) Monitor.Wait(_parkLock, 250);
                continue;
            }
            int spins = 0, spinMax = _limiter?.Spin(_spinIterations) ?? _spinIterations;
            long g;
            while ((g = Volatile.Read(ref _generation)) == seen)
            {
                if (_disposed) return;
                if (id >= _active) break;
                if (++spins < spinMax) { Thread.SpinWait(8); continue; }
                lock (_sleepLock)
                {
                    Interlocked.Increment(ref _sleepers);
                    if (Volatile.Read(ref _generation) == seen && !_disposed) Monitor.Wait(_sleepLock, 50);
                    Interlocked.Decrement(ref _sleepers);
                }
                // stay parked (no new spin phase) until work arrives: Run pulses sleepers, the timeout is a safety net
            }
            if (g == seen) continue;   // left the wait to park
            // the step's participant count belongs to generation g only if the generation has not moved on meanwhile
            int active = _active;
            if (Volatile.Read(ref _generation) != g) continue;
            seen = g;
            if (id >= active) continue;   // not part of this step
            try { DoWork(id); }
            catch (Exception e) { Fault(e); }   // rethrown on the dispatching thread (an escaping exception would end the process)
            finally { Interlocked.Decrement(ref _pending); }
        }
    }

    public void Dispose()
    {
        _disposed = true;
        lock (_sleepLock) Monitor.PulseAll(_sleepLock);
        lock (_parkLock) Monitor.PulseAll(_parkLock);
    }
}

/// <summary>Allocation-free parallel-for over a delegate stored in a reusable job object.</summary>
public sealed class ActionJob : ParallelJob
{
    public Action<int, int>? Body;
    public override void Execute(int index, int threadId) => Body!(index, threadId);
}
