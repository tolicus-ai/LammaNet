using System.Text.Json;
using LammaNet.Chat;
using LammaNet.Compute;
using LammaNet.Gpu;
using LammaNet.Inference;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Multimodal;

namespace LammaNet.Tests;

/// <summary>
/// Teacher-forced next-token probabilities against llama.cpp references stored in TestData (BF16 weights, so neither
/// side quantizes activations and the comparison is tight): llama.cpp's greedy continuation is fed token by token and
/// the probabilities of its top-5 candidates must agree at every step, on the CPU and (when present) the GPU.
/// References with an "mmproj" render the prompt as a user chat message with the "image" attached; the media encoder
/// runs on the GPU when the model does, and on its own with <c>gpuEncoder</c> (language model on the CPU).
/// </summary>
public class ReferenceProbabilityTests(Xunit.Abstractions.ITestOutputHelper output)
{
    [Theory]
    [InlineData("lfm2.5-bf16-llamacpp-probs.json", false)]
    [InlineData("lfm2.5-bf16-llamacpp-probs.json", true)]
    [InlineData("granite-docling-bf16-llamacpp-probs.json", false)]
    [InlineData("granite-docling-bf16-llamacpp-probs.json", true)]
    [InlineData("granite-docling-bf16-llamacpp-probs.json", false, true)]
    [InlineData("gemma4-e2b-qat-vision-llamacpp-probs.json", false)]
    [InlineData("gemma4-e2b-qat-vision-llamacpp-probs.json", true)]
    [InlineData("gemma4-e2b-qat-vision-llamacpp-probs.json", false, true)]
    [InlineData("qwen3.5-0.8b-vision-llamacpp-probs.json", false)]
    [InlineData("qwen3.5-0.8b-vision-llamacpp-probs.json", true)]
    [InlineData("qwen3.5-0.8b-vision-llamacpp-probs.json", false, true)]
    public void MatchesLlamaCpp(string referenceFile, bool gpu, bool gpuEncoder = false)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", referenceFile)));
        var root = doc.RootElement;
        var path = TestModels.Find(root.GetProperty("model").GetString()!);
        if (path == null) return;
        using var model = Model.Load(path);
        using var dev = gpu || gpuEncoder ? GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 512), new GpuOffloadOptions { Warmup = false }) : null;
        if ((gpu || gpuEncoder) && dev == null) return;
        var off = gpu ? dev : null;
        using var pool = new ComputePool(4);
        using var kv = new KvCache(model.Config, 2048, KvCacheType.F16, 32, off == null ? null : l => !off.OwnsKv(l), off?.RecurrentStore);
        using var runner = new ModelRunner(model, kv, pool, 512, 0, off);
        using var seq = new SequenceState(kv, model.Config);
        var batch = new Batch();
        string prompt = root.GetProperty("prompt").GetString()!;
        if (root.TryGetProperty("mmproj", out var mp))
        {
            var mmproj = TestModels.Find(mp.GetString()!);
            if (mmproj == null) return;
            using var engine = new InferenceEngine(model, new EngineOptions { ContextSize = 64, AutoAccelerator = false });
            var chat = new ChatEngine(engine);
            var enc = MultimodalProjector.Load(mmproj, model, new ProjectorOptions { Accelerator = dev as IVisionAccelerator });
            using var encOwner = enc as IDisposable;
            chat.MediaEncoders.Add(enc);
            var image = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", root.GetProperty("image").GetString()!));
            var msg = new ChatMessage { Role = ChatRole.User, Parts = [new ImagePart { EncodedBytes = image }, new TextPart(prompt)] };
            int pos = 0;
            foreach (var seg in chat.BuildPrompt([msg]).Prompt.Segments)
            {
                if (seg is TokenSegment ts) foreach (var t in ts.Tokens) batch.Add(seq, t, pos++, false);
                else if (seg is EmbeddingSegment es) for (int r = 0; r < es.Rows; r++) batch.Add(seq, 0, pos++, false, es.Row(r));
            }
        }
        else foreach (var t in model.Vocab.Tokenize(prompt)) batch.Add(seq, t, batch.Count, false);
        for (int c0 = 0; c0 < batch.Count; c0 += 512)
        {
            var part = new Batch();
            for (int i = c0; i < Math.Min(batch.Count, c0 + 512); i++)
                part.Add(seq, batch.Tokens[i], batch.Positions[i], i == batch.Count - 1, batch.Embeddings[i]);
            runner.Forward(part);
        }
        // quantized models get a wider tolerance (stored with the reference); top-1 may differ only on near ties
        double tol = root.TryGetProperty("tolerance", out var tv) ? tv.GetDouble() : 0.02;
        double maxD = 0;
        int step = 0;
        foreach (var s in root.GetProperty("steps").EnumerateArray())
        {
            var lg = runner.Logits(0).ToArray();
            float mx = lg.Max();
            double sum = lg.Sum(v => Math.Exp(v - mx));
            var cand = s.GetProperty("cand").EnumerateArray().Select(c => c.GetInt32()).ToArray();
            var refP = s.GetProperty("ref").EnumerateArray().Select(c => c.GetDouble()).ToArray();
            for (int k = 0; k < cand.Length; k++) maxD = Math.Max(maxD, Math.Abs(Math.Exp(lg[cand[k]] - mx) / sum - refP[k]));
            Assert.True(System.Numerics.Tensors.TensorPrimitives.IndexOfMax<float>(lg) == cand[0] || refP[0] - refP[1] < 2 * tol,
                $"step {step}: argmax differs from llama.cpp's top-1 (p {refP[0]:F4} vs {refP[1]:F4})");
            batch.Clear();
            batch.Add(seq, s.GetProperty("next").GetInt32(), seq.Length, true);
            runner.Forward(batch);
            step++;
        }
        output.WriteLine($"{referenceFile} gpu={gpu} gpuEncoder={gpuEncoder}: max |dp| {maxD:F4} (tolerance {tol})");
        Assert.True(maxD < tol, $"max |dp| {maxD} (gpu={gpu}, gpuEncoder={gpuEncoder})");
    }
}
