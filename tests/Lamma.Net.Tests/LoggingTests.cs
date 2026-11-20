using System.Collections.Concurrent;
using LammaNet.Compute;
using LammaNet.Extensions.AI;
using LammaNet.Inference;
using LammaNet.Sampling;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace LammaNet.Tests;

/// <summary>Collects log entries (thread-safe) for assertions on event ids.</summary>
public sealed class LogCapture : ILoggerProvider
{
    public sealed record Entry(string Category, LogLevel Level, EventId Id, string Message, Exception? Exception);

    private readonly ConcurrentQueue<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries.ToArray();

    public ILoggerFactory CreateFactory() => LoggerFactory.Create(b => b.SetMinimumLevel(LogLevel.Trace).AddProvider(this));
    public ILogger CreateLogger(string categoryName) => new Logger(this, categoryName);
    public void Dispose() { }

    private sealed class Logger(LogCapture owner, string category) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
            owner._entries.Enqueue(new Entry(category, logLevel, eventId, formatter(state, exception), exception));
    }
}

public sealed class LoggingTests
{
    private static InferenceEngine? Engine(ILoggerFactory lf, int contextSize = 256)
    {
        var path = TestModels.Find(TestModels.Lfm2Q4);
        return path == null ? null : InferenceEngine.Load(path, new EngineOptions { ContextSize = contextSize, AutoAccelerator = false, Threads = 4, LoggerFactory = lf });
    }

    private static Prompt P(InferenceEngine e, string text) => new(e.Tokenize(text));

    [Fact]
    public void ComputePoolRethrowsTaskExceptionsOnTheCallingThread()
    {
        using var pool = new ComputePool(4);
        var job = new ActionJob { Body = (_, _) => throw new InvalidOperationException("task failed") };
        Assert.Throws<InvalidOperationException>(() => pool.Run(job, 64));

        // an exception on a worker thread (previously fatal for the process) reaches the caller too
        bool fromWorker = false;
        job.Body = (_, tid) =>
        {
            Thread.SpinWait(500);
            if (tid != 0) throw new InvalidOperationException("worker failed");
        };
        for (int round = 0; round < 50 && !fromWorker; round++)
        {
            try { pool.Run(job, 512); }
            catch (InvalidOperationException e) when (e.Message == "worker failed") { fromWorker = true; }
        }
        Assert.True(fromWorker);

        // the pool keeps working
        long sum = 0;
        job.Body = (i, _) => Interlocked.Add(ref sum, i);
        pool.Run(job, 100);
        Assert.Equal(4950, sum);
    }

    [Fact]
    public async Task EngineLogsRejectedAndFailedRequestsAndKeepsServing()
    {
        using var capture = new LogCapture();
        using var lf = capture.CreateFactory();
        using var engine = Engine(lf, contextSize: 64);
        if (engine == null) return;
        Assert.Contains(capture.Entries, e => e.Id.Id == 3000);                                       // model loaded (engine's factory)
        Assert.Contains(capture.Entries, e => e.Id.Id == 1000 && e.Category == typeof(InferenceEngine).FullName);

        // longer than the context: rejected with a warning that names the request
        var tooLong = new Prompt(Enumerable.Repeat(engine.Tokenize("hello", addSpecial: false)[0], 100));
        var rejected = engine.Submit(new GenerationRequest { Prompt = tooLong, MaxTokens = 4 });
        await Assert.ThrowsAsync<ArgumentException>(() => rejected.Completion);
        Assert.Contains(capture.Entries, e => e.Id.Id == 1011 && e.Level == LogLevel.Warning && e.Exception is ArgumentException
            && e.Message.Contains($"request {rejected.RequestId} "));

        foreach (int token in new[] { -1, engine.Config.NVocab })
        {
            var invalid = engine.Submit(new GenerationRequest { Prompt = new Prompt([token]) });
            await Assert.ThrowsAsync<ArgumentException>(() => invalid.Completion);
        }

        // sampling fails for one request: only that request fails, the other one completes
        var good = engine.Submit(new GenerationRequest { Prompt = P(engine, "The capital of France is"), MaxTokens = 6 });
        var bad = engine.Submit(new GenerationRequest { Prompt = P(engine, "One, two, three,"), MaxTokens = 6, Constraint = new ThrowingConstraint() });
        await Assert.ThrowsAsync<InvalidOperationException>(() => bad.Completion);
        var stats = await good.Completion;
        Assert.True(stats.CompletionTokens > 0);
        Assert.Contains(capture.Entries, e => e.Id.Id == 1018 && e.Level == LogLevel.Error && e.Exception is InvalidOperationException
            && e.Message.Contains($"request {bad.RequestId} "));
        Assert.DoesNotContain(capture.Entries, e => e.Id.Id == 1020);

        // and the engine still serves new requests
        var (_, again) = await engine.GenerateTextAsync(new GenerationRequest { Prompt = P(engine, "Hello"), MaxTokens = 3 });
        Assert.True(again.CompletionTokens > 0);
        Assert.Contains(capture.Entries, e => e.Id.Id == 1013 && e.Level == LogLevel.Debug);
    }

    [Fact]
    public async Task ABrokenLogSinkDoesNotBreakTheEngine()
    {
        using var lf = LoggerFactory.Create(b => b.SetMinimumLevel(LogLevel.Trace).AddProvider(new ThrowingProvider()));
        using var engine = Engine(lf);
        if (engine == null) return;
        var (_, stats) = await engine.GenerateTextAsync(new GenerationRequest { Prompt = P(engine, "Hello"), MaxTokens = 4 });
        Assert.True(stats.CompletionTokens > 0);
    }

    [Fact]
    public void DependencyInjectionPassesTheContainerLoggerFactory()
    {
        var path = TestModels.Find(TestModels.Lfm2Q4);
        if (path == null) return;
        using var capture = new LogCapture();
        var services = new ServiceCollection();
        services.AddLogging(b => b.SetMinimumLevel(LogLevel.Debug).AddProvider(capture));
        services.AddLammaEngine(path, o => { o.Threads = 4; o.ContextSize = 256; });
        using var sp = services.BuildServiceProvider();
        var engine = sp.GetRequiredService<InferenceEngine>();
        Assert.Same(sp.GetRequiredService<ILoggerFactory>(), engine.LoggerFactory);
        Assert.Contains(capture.Entries, e => e.Id.Id == 1000 && e.Message.Contains($"Engine {engine.Id} "));
    }

    private sealed class ThrowingConstraint : ITokenConstraint
    {
        public bool IsAllowed(int token) => throw new InvalidOperationException("constraint failed");
        public void Accept(int token) { }
        public bool IsComplete => false;
        public bool AllowsEndOfGeneration => throw new InvalidOperationException("constraint failed");
    }

    private sealed class ThrowingProvider : ILoggerProvider
    {
        public ILogger CreateLogger(string categoryName) => new Throwing();
        public void Dispose() { }

        private sealed class Throwing : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => throw new IOException("log sink down");
            public bool IsEnabled(LogLevel logLevel) => true;
            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
                throw new IOException("log sink down");
        }
    }
}
