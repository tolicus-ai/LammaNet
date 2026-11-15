using System.Diagnostics;
using LammaNet.Compute;
using LammaNet.Inference;

namespace LammaNet.Tests;

/// <summary>Tests that compare wall-clock timings: run alone, after the parallel test classes (which load the CPU and GPU).</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TimingCollection
{
    public const string Name = "Timing";
}

[Collection(TimingCollection.Name)]
public class ResourceLimiterTests
{
    [Fact]
    public void PoolUsesOnlyTheThreadCapAndFollowsChanges()
    {
        using var limiter = new ResourceLimiter { MaxThreads = 2 };
        using var pool = new ComputePool(8, limiter);
        int maxId = -1;
        var job = new ActionJob { Body = (_, tid) => { Thread.SpinWait(200); InterlockedMax(ref maxId, tid); } };
        for (int i = 0; i < 20; i++) pool.Run(job, 256);
        Assert.InRange(maxId, 0, 1);

        limiter.MaxThreads = null;   // all threads again
        maxId = -1;
        for (int i = 0; i < 50 && maxId < 2; i++) pool.Run(job, 256);
        Assert.True(maxId >= 2, $"max thread id {maxId}");

        limiter.MaxThreads = 3;
        maxId = -1;
        for (int i = 0; i < 20; i++) pool.Run(job, 256);
        Assert.InRange(maxId, 0, 2);
    }

    /// <summary>Caps changing between (and racing with) steps never lose or double-count a worker: every step completes with the right result.</summary>
    [Fact]
    public void ChangingCapsDuringSteadyWorkIsSafe()
    {
        using var limiter = new ResourceLimiter();
        using var pool = new ComputePool(8, limiter);
        var rnd = new Random(7);
        using var cts = new CancellationTokenSource();
        var changer = Task.Run(() =>
        {
            var r = new Random(11);
            while (!cts.IsCancellationRequested) { limiter.MaxThreads = r.Next(0, 9) is var v && v == 0 ? null : v; Thread.SpinWait(r.Next(10, 5000)); }
        });
        long sum = 0;
        var job = new ActionJob { Body = (i, _) => Interlocked.Add(ref sum, i) };
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < 3000; it++)
        {
            int n = rnd.Next(1, 200);
            sum = 0;
            pool.Run(job, n);
            Assert.Equal((long)n * (n - 1) / 2, sum);
        }
        cts.Cancel();
        changer.Wait();
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(60), $"took {sw.Elapsed}");
    }

    [Fact]
    public void AdaptiveLoadLowersAndRestoresTheThreadCap()
    {
        using var limiter = new ResourceLimiter();
        int cores = CpuTopology.PhysicalCores;
        if (cores < 2) return;
        for (int i = 0; i < 20; i++) limiter.ApplyLoad(Environment.ProcessorCount);   // other processes use the whole machine
        Assert.Equal(1, limiter.EffectiveMaxThreads);
        for (int i = 0; i < 60; i++) limiter.ApplyLoad(0);                            // and then nothing
        Assert.Null(limiter.EffectiveMaxThreads);
        limiter.MaxThreads = 2;                                                        // the explicit cap still applies
        Assert.Equal(2, limiter.EffectiveMaxThreads);
    }

    [Fact]
    public async Task EnginesTakeTurnsWithOneConcurrentStep()
    {
        var path = TestModels.Find(TestModels.Lfm2Q4);
        if (path == null) return;
        using var limiter = new ResourceLimiter { MaxConcurrentSteps = 1 };
        EngineOptions Options() => new() { ContextSize = 256, Threads = 2, AutoAccelerator = false, Resources = limiter };
        using var a = InferenceEngine.Load(path, Options());
        using var b = InferenceEngine.Load(path, Options());
        var req = (InferenceEngine e) => e.GenerateTextAsync(new GenerationRequest { Prompt = new Prompt(e.Tokenize("Once upon a time")), MaxTokens = 24, IgnoreEos = true });
        var results = await Task.WhenAll(Task.Run(() => req(a)), Task.Run(() => req(b)), Task.Run(() => req(a)), Task.Run(() => req(b)));
        Assert.All(results, r => Assert.Equal(24, r.Stats.CompletionTokens));
        Assert.Equal(1, limiter.PeakConcurrentSteps);
    }

    [Fact]
    public async Task UtilizationLimitPausesBetweenSteps()
    {
        var path = TestModels.Find(TestModels.Lfm2Q4);
        if (path == null) return;
        using var limiter = new ResourceLimiter();
        using var engine = InferenceEngine.Load(path, new EngineOptions { ContextSize = 256, Threads = 4, AutoAccelerator = false, Resources = limiter });
        async Task<TimeSpan> Generate()
        {
            var (_, s) = await engine.GenerateTextAsync(new GenerationRequest { Prompt = new Prompt(engine.Tokenize("Count to fifty:")), MaxTokens = 32, IgnoreEos = true });
            return s.GenerationTime;
        }
        await Generate();                       // warm up
        var full = await Generate();
        limiter.MaxUtilization = 0.25;          // changes apply to the running engine
        var quarter = await Generate();
        Assert.True(quarter.TotalSeconds > 2.5 * full.TotalSeconds, $"full {full.TotalMilliseconds:F0} ms, at 25%: {quarter.TotalMilliseconds:F0} ms");
    }

    [Fact]
    public void DeviceMemoryCapLimitsOffloadedLayers()
    {
        var path = TestModels.Find(TestModels.Qwen3Q4);
        if (path == null) return;
        using var model = LammaNet.Models.Model.Load(path);
        using var limiter = new ResourceLimiter { MaxDeviceMemoryBytes = 400L << 20 };
        var geometry = new OffloadGeometry(32, 64, 64) { Resources = limiter };
        using var capped = LammaNet.Gpu.GpuLayerOffload.TryCreate(model, geometry, new LammaNet.Gpu.GpuOffloadOptions { Warmup = false });
        if (capped == null) return;   // no CUDA device (or not even one layer fits)
        using var full = LammaNet.Gpu.GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 64), new LammaNet.Gpu.GpuOffloadOptions { Warmup = false });
        Assert.NotNull(full);
        Assert.True(capped.LayerCount < full!.LayerCount, $"{capped.LayerCount} vs {full.LayerCount} layers");
    }

    private static void InterlockedMax(ref int target, int value)
    {
        int cur;
        while ((cur = Volatile.Read(ref target)) < value && Interlocked.CompareExchange(ref target, value, cur) != cur) { }
    }
}
