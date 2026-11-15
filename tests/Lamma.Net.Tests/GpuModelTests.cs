using ILGPU;
using ILGPU.Runtime;
using LammaNet.Gpu;
using LammaNet.Models;

namespace LammaNet.Tests;

/// <summary>
/// Whole-model GPU offload vs the CPU reference (needs a CUDA GPU and the test models): prefill and teacher-forced
/// decode (CUDA-graph replay) for every supported layer kind — attention, Gated DeltaNet and Mamba2 — fully offloaded
/// and split between GPU and CPU.
/// </summary>
public class GpuModelTests
{
    private static readonly bool HasCuda = HasCudaDevice();

    private static bool HasCudaDevice()
    {
        try
        {
            using var ctx = Context.Create(b => b.Default());
            return ctx.Devices.Any(d => d.AcceleratorType == AcceleratorType.Cuda);
        }
        catch { return false; }
    }

    /// <summary>~1.8k-token prompt: several prefill batches, and past Gemma 4's 512-token sliding window.</summary>
    private static readonly string LongPrompt = BuildLongPrompt();

    private static string BuildLongPrompt()
    {
        var rnd = new Random(1);
        var words = "the quick brown fox jumps over a lazy dog while seven robots paint the lighthouse on a rocky coast under silver clouds and quiet rain".Split(' ');
        return string.Join(' ', Enumerable.Range(0, 1500).Select(_ => words[rnd.Next(words.Length)])) + ". Summarize the text above in one sentence.";
    }

    [Theory]
    [InlineData(TestModels.Qwen3Q4, -1, false)]
    [InlineData(TestModels.Qwen3Q4, 10, false)]
    [InlineData(TestModels.Qwen3Q4, -1, true)]
    [InlineData(TestModels.Qwen3Q8, -1, false)]
    [InlineData(TestModels.Llama32, -1, false)]
    [InlineData(TestModels.Granite, -1, false)]
    [InlineData(TestModels.Qwen35, -1, false)]
    [InlineData(TestModels.Qwen35, 9, false)]
    [InlineData(TestModels.Qwen35, -1, true)]
    [InlineData(TestModels.GraniteH, -1, false)]
    [InlineData(TestModels.Lfm2Q4, -1, false)]
    [InlineData(TestModels.Lfm2Q4, 5, false)]
    [InlineData(TestModels.Gemma4, -1, false)]
    [InlineData(TestModels.Gemma4, -1, true)]
    // float activations: prefill held to the tight bound
    [InlineData(TestModels.Qwen3Q4, -1, false, false)]
    [InlineData(TestModels.Qwen35, -1, false, false)]
    [InlineData(TestModels.Gemma4, -1, false, false)]
    public void GpuMatchesCpuReference(string modelFile, int layers, bool longPrompt, bool int8 = true)
    {
        var path = TestModels.Find(modelFile);
        if (path == null || !HasCuda) return;
        using var model = Model.Load(path);
        var options = new GpuOffloadOptions { Layers = layers, Int8 = int8 ? Int8Activations.Normalized : Int8Activations.Off };
        var r = GpuValidation.Compare(model, options, nGen: 12, prompt: longPrompt ? LongPrompt : GpuValidation.DefaultPrompt);
        Assert.Equal(int8, r.Int8);
        Assert.True(r.Passed, $"{modelFile} [{layers}] long={longPrompt} int8={int8}: cpu-dec {r.CpuDecode}; gpu decode {r.GpuDecode}; gpu prefill {r.GpuPrefill}");
        if (layers < 0) Assert.Equal(model.Config.NLayer, r.OffloadedLayers);
    }
}
