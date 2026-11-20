using System.Numerics.Tensors;
using LammaNet.Compute;
using LammaNet.Inference;
using LammaNet.Memory;
using LammaNet.Models;

namespace LammaNet.Tests;

public class Phi2Tests
{
    [Fact]
    public void LastPromptRowProducesFiniteLogitsWithParallelMlp()
    {
        var path = TestModels.Find("phi-2.Q4_K_M.gguf");
        if (path == null) return;
        using var model = Model.Load(path);
        using var pool = new ComputePool(4);
        using var kv = new KvCache(model.Config, 32);
        using var runner = new ModelRunner(model, kv, pool, 32);
        using var seq = new SequenceState(kv, model.Config);
        var tokens = model.Vocab.Tokenize("The capital of France is");
        var batch = new Batch();
        for (int i = 0; i < tokens.Count; i++) batch.Add(seq, tokens[i], i, i == tokens.Count - 1);
        runner.Forward(batch);
        var logits = runner.Logits(0).ToArray();
        Assert.True(logits.All(float.IsFinite));
        Assert.Equal(" Paris", model.Vocab.Decode([TensorPrimitives.IndexOfMax<float>(logits)]));
    }
}
