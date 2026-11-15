using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace LammaNet.Diagnostics;

/// <summary>
/// Logging configuration of Lamma.Net. Components log through the <see cref="ILoggerFactory"/> given in their options
/// (<see cref="Inference.EngineOptions.LoggerFactory"/>, <see cref="Models.ModelLoadOptions.LoggerFactory"/>,
/// <see cref="Models.ModelCacheOptions.LoggerFactory"/>, <see cref="Multimodal.ProjectorOptions.LoggerFactory"/>); an engine
/// passes its factory on to everything it creates (accelerator, projectors, chat clients). Components created without one,
/// and static code (templates, GGUF metadata), use <see cref="LoggerFactory"/>.
/// </summary>
/// <remarks>
/// Event ids by area: 1xxx engine, 2xxx accelerator, 3xxx model loading, 4xxx model cache, 5xxx multimodal, 6xxx chat and
/// templates, 7xxx Microsoft.Extensions.AI adapters, 8xxx compute. Prompt and output text is never logged; failures carry
/// the exception. A logger that throws never interrupts inference (its exceptions are dropped).
/// </remarks>
public static class LammaLogging
{
    private static volatile ILoggerFactory s_factory = NullLoggerFactory.Instance;

    /// <summary>
    /// Default factory for components created without one (default: none). Set it at startup: components capture the
    /// factory when they are created.
    /// </summary>
    public static ILoggerFactory LoggerFactory
    {
        get => s_factory;
        set => s_factory = value ?? NullLoggerFactory.Instance;
    }

    /// <summary>A logger of <paramref name="factory"/> (or the default factory) that never throws.</summary>
    internal static ILogger Create(ILoggerFactory? factory, string category)
    {
        var f = factory ?? s_factory;
        if (f is NullLoggerFactory) return NullLogger.Instance;
        try { return new SafeLogger(f.CreateLogger(category)); }
        catch (Exception) { return NullLogger.Instance; }
    }

    internal static ILogger Create<T>(ILoggerFactory? factory) => Create(factory, typeof(T).FullName ?? typeof(T).Name);

    /// <summary>Logger of the default factory for static code (resolved per call: the factory may be set later).</summary>
    internal static ILogger Static(string category) => Create(null, category);

    /// <summary>Wraps a logger so a failing provider cannot throw into an inference thread.</summary>
    private sealed class SafeLogger(ILogger inner) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull
        {
            try { return inner.BeginScope(state); }
            catch (Exception) { return null; }
        }

        public bool IsEnabled(LogLevel logLevel)
        {
            try { return inner.IsEnabled(logLevel); }
            catch (Exception) { return false; }
        }

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            try { inner.Log(logLevel, eventId, state, exception, formatter); }
            catch (Exception) { /* a broken log sink must not take an engine thread down; nothing to report it to */ }
        }
    }
}
