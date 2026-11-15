using ILGPU;
using ILGPU.Runtime;
using LammaNet.Compute;
using LammaNet.Gpu;
using LammaNet.Inference;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Tests;

/// <summary>
/// Context shift: dropping KV pages and re-rotating the kept tail must put every key at its new position. Layer 0's
/// keys depend only on the token and its position, so after a shift they must equal those of a fresh prefill of the
/// shortened sequence (deeper layers legitimately keep information from the dropped tokens, as in llama.cpp).
/// </summary>
public class ContextShiftTests
{
    [Theory]
    [InlineData(KvCacheType.F32, false)]
    [InlineData(KvCacheType.F16, false)]
    [InlineData(KvCacheType.Q8_0, false)]
    [InlineData(KvCacheType.F16, true)]
    public void ShiftedKeysMatchFreshPrefill(KvCacheType type, bool gpu)
    {
        var path = TestModels.Find(TestModels.Qwen3Q4);
        if (path == null) return;
        if (gpu && !HasCuda()) return;
        using var model = Model.Load(path);
        var c = model.Config;
        using var pool = new ComputePool(4);
        const int ps = 32, len = 200, keepPages = 1, discardPages = 2;
        var rnd = new Random(3);
        var toks = Enumerable.Range(0, len).Select(_ => rnd.Next(1000, 20000)).ToList();

        // returns layer-0 keys of positions 0.. (after the optional shift)
        List<float[]> Run(IReadOnlyList<int> tokens, bool shift)
        {
            using var off = gpu ? new GpuLayerOffload(model, new OffloadGeometry(ps, 16, 512), new GpuOffloadOptions { Warmup = false }) : null;
            using var kv = new KvCache(c, 16 * ps, type, ps, off == null ? null : l => !off.OwnsKv(l));
            if (off != null) kv.PageCopied += off.CopyPage;
            using var runner = new ModelRunner(model, kv, pool, 512, 0, off);
            using var seq = new SequenceState(kv, c);
            var batch = new Batch();
            for (int i = 0; i < tokens.Count; i++) batch.Add(seq, tokens[i], i, false);
            runner.Forward(batch);
            if (shift)
            {
                int removed = seq.DiscardPages(keepPages, discardPages);
                kv.RotateKeys(seq.PageTable, keepPages * ps, seq.Length, -removed);
                off?.RotateKeys(seq.PageTable, keepPages * ps, seq.Length, -removed);
            }
            var rows = new List<float[]>();
            int n = c.NHeadKv[0] * c.HeadDimK[0];
            for (int pos = 0; pos < seq.Length; pos++)
            {
                int page = seq.PageTable[pos / ps], slot = pos % ps;
                if (off != null) { rows.Add(off.ReadKeyRow(0, page * ps + slot)); continue; }
                rows.Add(KeyRow(kv, page, slot, n));
            }
            return rows;
        }

        var shifted = Run(toks, true);
        var expected = Run(toks.Take(keepPages * ps).Concat(toks.Skip((keepPages + discardPages) * ps)).ToList(), false);
        Assert.Equal(expected.Count, shifted.Count);
        double tol = type switch { KvCacheType.F32 => 1e-3, KvCacheType.F16 => 2e-2, _ => 0.15 };
        double maxD = 0, maxAbs = 0;
        for (int p = 0; p < expected.Count; p++)
            for (int i = 0; i < expected[p].Length; i++)
            {
                maxD = Math.Max(maxD, Math.Abs(expected[p][i] - shifted[p][i]));
                maxAbs = Math.Max(maxAbs, Math.Abs(expected[p][i]));
            }
        Assert.True(maxD <= tol * Math.Max(1, maxAbs / 10), $"{type} gpu={gpu}: max |dK| {maxD} (max |K| {maxAbs})");
    }

    /// <summary>With ContextShift the engine keeps generating past ContextSize (CPU and, when present, GPU).</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EngineGeneratesPastContextSize(bool gpu)
    {
        var path = TestModels.Find(TestModels.Qwen3Q4);
        if (path == null) return;
        if (gpu && !HasCuda()) return;
        using var engine = InferenceEngine.Load(path, new EngineOptions
        {
            ContextSize = 128, ContextShift = true, ContextShiftKeep = 8, AutoAccelerator = gpu, SpeculativeDraftTokens = 0,
        });
        Assert.Equal(gpu, engine.Offload != null);

        var prompt = new Prompt(engine.Tokenize("Count from one to two hundred, separated by commas: one, two, three,"));
        var (text, stats) = await engine.GenerateTextAsync(new GenerationRequest
        {
            Prompt = prompt, Sampling = Sampling.SamplingOptions.Greedy, MaxTokens = 300, IgnoreEos = true, CachePrompt = false,
        });
        Assert.Equal(300, stats.CompletionTokens);
        Assert.Equal(FinishReason.Length, stats.FinishReason);
        Assert.True(engine.TotalContextShifts >= 3, $"shifts: {engine.TotalContextShifts}");
        Assert.Contains(",", text);
    }

    private static unsafe float[] KeyRow(KvCache kv, int page, int slot, int n)
    {
        var r = new float[n];
        fixed (float* rp = r) Dequantize.Row(kv.HeadType(0, false), kv.K(0, page, slot), rp, n);
        return r;
    }

    private static bool HasCuda()
    {
        try
        {
            using var ctx = Context.Create(b => b.Default());
            return ctx.Devices.Any(d => d.AcceleratorType == AcceleratorType.Cuda);
        }
        catch { return false; }
    }
}
