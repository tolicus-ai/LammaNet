using LammaNet.Inference;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace LammaNet.Extensions.AI;

/// <summary>Dependency-injection helpers.</summary>
public static class LammaServiceCollectionExtensions
{
    /// <summary>
    /// Registers a singleton <see cref="InferenceEngine"/> for a GGUF model and an <see cref="IChatClient"/> on top of it.
    /// Returns the <see cref="ChatClientBuilder"/> so middleware (function invocation, logging, caching...) can be added.
    /// </summary>
    public static ChatClientBuilder AddLammaChatClient(this IServiceCollection services, string ggufPath,
        Action<EngineOptionsBuilder>? configureEngine = null, Action<LammaChatClientOptions>? configureClient = null)
    {
        services.AddLammaEngine(ggufPath, configureEngine);
        var clientOptions = new LammaChatClientOptions();
        configureClient?.Invoke(clientOptions);
        return services.AddChatClient(sp => new LammaChatClient(sp.GetRequiredService<InferenceEngine>(), clientOptions));
    }

    /// <summary>Registers an <see cref="IEmbeddingGenerator{String, Embedding}"/> backed by the (shared) engine.</summary>
    public static EmbeddingGeneratorBuilder<string, Embedding<float>> AddLammaEmbeddingGenerator(this IServiceCollection services, string? ggufPath = null,
        EmbeddingPooling pooling = EmbeddingPooling.Mean, Action<EngineOptionsBuilder>? configureEngine = null)
    {
        if (ggufPath != null) services.AddLammaEngine(ggufPath, configureEngine);
        return services.AddEmbeddingGenerator<string, Embedding<float>>(sp => new LammaEmbeddingGenerator(sp.GetRequiredService<InferenceEngine>(), pooling));
    }

    /// <summary>
    /// Registers the shared <see cref="InferenceEngine"/> singleton (loaded lazily on first use). It logs through the
    /// container's <see cref="ILoggerFactory"/> unless <see cref="EngineOptionsBuilder.LoggerFactory"/> is set.
    /// </summary>
    public static IServiceCollection AddLammaEngine(this IServiceCollection services, string ggufPath, Action<EngineOptionsBuilder>? configure = null)
    {
        if (services.Any(d => d.ServiceType == typeof(InferenceEngine))) return services;
        var b = new EngineOptionsBuilder();
        configure?.Invoke(b);
        services.AddSingleton(sp => InferenceEngine.Load(ggufPath, b.Build(sp.GetService<ILoggerFactory>())));
        return services;
    }
}

/// <summary>Mutable builder for <see cref="EngineOptions"/> (used by DI configuration callbacks).</summary>
public sealed class EngineOptionsBuilder
{
    public int? ContextSize { get; set; }
    public int? KvCacheTokens { get; set; }
    public int MaxConcurrentSequences { get; set; } = 4;
    public int BatchSize { get; set; } = 512;
    public int? Threads { get; set; }
    public LammaNet.Memory.KvCacheType KvCacheType { get; set; } = LammaNet.Memory.KvCacheType.F16;
    public int PrefixCacheEntries { get; set; } = 8;
    /// <summary>Logging of the engine (default: the container's <see cref="ILoggerFactory"/>).</summary>
    public ILoggerFactory? LoggerFactory { get; set; }

    public EngineOptions Build() => Build(null);

    internal EngineOptions Build(ILoggerFactory? fallbackLoggerFactory) => new()
    {
        ContextSize = ContextSize,
        KvCacheTokens = KvCacheTokens,
        MaxConcurrentSequences = MaxConcurrentSequences,
        BatchSize = BatchSize,
        Threads = Threads,
        KvCacheType = KvCacheType,
        PrefixCacheEntries = PrefixCacheEntries,
        LoggerFactory = LoggerFactory ?? fallbackLoggerFactory,
    };
}
