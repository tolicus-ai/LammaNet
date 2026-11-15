using LammaNet.Inference;
using Microsoft.Extensions.AI;

namespace LammaNet.Extensions.AI;

/// <summary>
/// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> producing pooled, L2-normalized hidden-state embeddings with a Lamma.Net engine.
/// Works with any decoder model; dedicated embedding GGUFs give the best retrieval quality.
/// </summary>
public sealed class LammaEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly InferenceEngine _engine;
    private readonly EmbeddingPooling _pooling;
    private readonly bool _ownsEngine;
    private readonly EmbeddingGeneratorMetadata _metadata;

    /// <param name="engine">Engine over an embedding (or decoder) model.</param>
    /// <param name="pooling">Pooling of token vectors; default: the model's GGUF pooling type (last-token or mean).</param>
    /// <param name="ownsEngine">Dispose the engine with the generator.</param>
    public LammaEmbeddingGenerator(InferenceEngine engine, EmbeddingPooling? pooling = null, bool ownsEngine = false)
    {
        _engine = engine;
        _pooling = pooling ?? (engine.Config.PoolingType == 3 ? EmbeddingPooling.Last : EmbeddingPooling.Mean);
        _ownsEngine = ownsEngine;
        int dims = engine.Config.IsEmbeddingModel && engine.Config.NEmbdOut > 0 ? engine.Config.NEmbdOut : engine.Config.NEmbd;
        _metadata = new EmbeddingGeneratorMetadata("lamma.net", null, engine.Config.Name, dims);
    }

    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        var inputs = values.ToList();
        // submit all at once: the engine batches them together
        var tasks = inputs.Select(v =>
        {
            var tokens = _engine.Tokenize(v, addSpecial: true, parseSpecial: false);
            int max = _engine.Config.Causal ? _engine.ContextSize - 1 : Math.Min(_engine.ContextSize - 1, _engine.Options.BatchSize);
            if (tokens.Count > max) tokens = tokens.Take(max).ToList();
            return (_engine.EmbedAsync(new Prompt(tokens), _pooling, normalize: true, cancellationToken), tokens.Count);
        }).ToList();
        var result = new GeneratedEmbeddings<Embedding<float>>();
        long total = 0;
        foreach (var (task, count) in tasks)
        {
            var v = await task.ConfigureAwait(false);
            if (options?.Dimensions is int dims && dims < v.Length)
            {
                // Matryoshka-style truncation + renormalization
                v = v[..dims];
                double n = 0;
                foreach (var x in v) n += x * x;
                float inv = n > 0 ? (float)(1 / Math.Sqrt(n)) : 0;
                for (int i = 0; i < v.Length; i++) v[i] *= inv;
            }
            result.Add(new Embedding<float>(v) { ModelId = _metadata.DefaultModelId, CreatedAt = DateTimeOffset.UtcNow });
            total += count;
        }
        result.Usage = new UsageDetails { InputTokenCount = total, TotalTokenCount = total };
        return result;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey != null) return null;
        if (serviceType == typeof(EmbeddingGeneratorMetadata)) return _metadata;
        if (serviceType.IsInstanceOfType(this)) return this;
        if (serviceType.IsInstanceOfType(_engine)) return _engine;
        return null;
    }

    public void Dispose()
    {
        if (_ownsEngine) _engine.Dispose();
    }
}
