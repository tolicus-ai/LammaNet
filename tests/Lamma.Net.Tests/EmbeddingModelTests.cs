using System.Numerics.Tensors;
using System.Text.Json;
using LammaNet.Inference;

namespace LammaNet.Tests;

/// <summary>
/// Bidirectional embedding models (EmbeddingGemma 2): embeddings must match llama.cpp's (TestData reference) on the
/// CPU and, when present, the GPU; similar texts must be closer than unrelated ones.
/// </summary>
public class EmbeddingModelTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EmbeddingGemma2MatchesLlamaCpp(bool gpu)
    {
        var path = TestModels.Find(TestModels.EmbeddingGemma2);
        if (path == null) return;
        using var engine = InferenceEngine.Load(path, new EngineOptions { ContextSize = 2048, AutoAccelerator = gpu });
        if (gpu && engine.Offload == null) return; // no GPU
        Assert.True(engine.Config.IsEmbeddingModel);
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "embeddinggemma2-llamacpp.json")));
        foreach (var e in doc.RootElement.GetProperty("embeddings").EnumerateObject())
        {
            var reference = e.Value.EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var ours = await engine.EmbedAsync(new Prompt(engine.Tokenize(e.Name)));
            Assert.Equal(reference.Length, ours.Length);
            float cos = TensorPrimitives.CosineSimilarity(ours, reference);
            Assert.True(cos > 0.999f, $"'{e.Name}' gpu={gpu}: cosine {cos}");
        }
        var cat = await engine.EmbedAsync(new Prompt(engine.Tokenize("The cat sat on the mat.")));
        var feline = await engine.EmbedAsync(new Prompt(engine.Tokenize("A feline rested on the rug.")));
        var stocks = await engine.EmbedAsync(new Prompt(engine.Tokenize("Stock markets fell sharply today.")));
        Assert.True(TensorPrimitives.CosineSimilarity(cat, feline) > TensorPrimitives.CosineSimilarity(cat, stocks) + 0.1f);
    }
}
