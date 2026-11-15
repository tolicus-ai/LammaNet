using System.Numerics.Tensors;
using LammaNet.Compute;
using LammaNet.Inference;
using LammaNet.Memory;
using LammaNet.Models;

namespace LammaNet.Gpu;

/// <summary>Logit deviation of one run from the reference.</summary>
public readonly record struct LogitDeviation(double Max, double Mean, int Top1Agree, int Count)
{
    public override string ToString() => $"top1 {Top1Agree}/{Count}  max|dlogit| {Max:F4}  mean {Mean:F4}";
}

/// <summary>Result of <see cref="GpuValidation.Compare"/>.</summary>
public sealed record GpuValidationResult(string CpuText, int OffloadedLayers, string Device,
    LogitDeviation CpuDecode, LogitDeviation GpuDecode, LogitDeviation GpuPrefill, bool Int8 = false)
{
    /// <summary>
    /// With float activations GPU prefill must track the reference tightly. With int8 activations
    /// (<see cref="Int8Activations"/>) decode and the prompt GEMM round activations to 8-bit blocks as the CPU's Q8 decode
    /// kernels do, so prefill is held to the decode bound. GPU decode may also pass through the CPU's Q8-activation
    /// kernels (host layers of a split); it must be no worse than CPU decode.
    /// </summary>
    public bool Passed =>
        GpuPrefill.Top1Agree >= GpuPrefill.Count - 1 && (Int8 ? NoWorseThanCpuDecode(GpuPrefill) : GpuPrefill.Max <= 0.1)
        && GpuDecode.Top1Agree >= Math.Min(CpuDecode.Top1Agree, CpuDecode.Count - 1) - 2
        && NoWorseThanCpuDecode(GpuDecode);

    private bool NoWorseThanCpuDecode(LogitDeviation d) => d.Mean <= 1.25 * CpuDecode.Mean + 0.05 && d.Max <= 2 * CpuDecode.Max + 0.1;
}

/// <summary>
/// GPU-vs-CPU regression check: greedy-decodes on the CPU, then replays the same tokens through the GPU split
/// (prefill + teacher-forced decode, and one batched prefill over the whole sequence). The reference is the CPU's
/// batched prefill over the whole sequence (float GEMM); CPU decode (Q8 activations) is reported for scale.
/// </summary>
public static class GpuValidation
{
    public const string DefaultPrompt =
        "Write a short story about a robot who learns to paint. The robot's name is Pixel and it lives in a lighthouse on a rocky coast.";

    public static GpuValidationResult Compare(Model model, GpuOffloadOptions options, int nGen = 24, string prompt = DefaultPrompt, int threads = 0)
    {
        var c = model.Config;
        var pool = new ComputePool(threads > 0 ? threads : Environment.ProcessorCount);
        const int pageSize = 32, maxBatch = 512;
        var promptToks = model.Vocab.Tokenize(prompt);
        int ctx = Math.Max(1024, (promptToks.Count + nGen + 64 + pageSize - 1) / pageSize * pageSize);

        // Feeds tokens in batches of at most maxBatch (long prompts span several prefill batches); returns the logits
        // of positions >= firstLogit in order.
        List<float[]> Feed(ModelRunner runner, SequenceState seq, IReadOnlyList<int> toks, int from, int firstLogit)
        {
            var res = new List<float[]>();
            var batch = new Batch();
            for (int s = from; s < toks.Count; s += maxBatch)
            {
                batch.Clear();
                int e = Math.Min(toks.Count, s + maxBatch);
                for (int i = s; i < e; i++) batch.Add(seq, toks[i], i, i >= firstLogit);
                runner.Forward(batch);
                for (int i = 0; i < batch.LogitCount; i++) res.Add(runner.Logits(i).ToArray());
            }
            return res;
        }

        List<float[]> Run(ILayerOffload? off, IReadOnlyList<int>? forced, bool oneShot, out List<int> toks)
        {
            using var kv = new KvCache(c, ctx, KvCacheType.F16, pageSize, off == null ? null : l => !off.OwnsKv(l), off?.RecurrentStore);
            using var runner = new ModelRunner(model, kv, pool, maxBatch, 0, off);
            using var seq = new SequenceState(kv, c);
            toks = new List<int>(promptToks);
            if (oneShot)
            {
                // the whole sequence as prompt batches; logits of the positions that predict the generated tokens. A few
                // earlier positions are computed too, so the CPU reference's LM head batch is large enough for the float
                // GEMM (smaller batches use int8 dot products, MatMul.GemmMinTokensFor)
                var all = promptToks.Concat(forced!).ToList();
                int first = Math.Max(0, promptToks.Count - 1 - 32);
                var lg = Feed(runner, seq, all, 0, first);
                lg.RemoveRange(0, promptToks.Count - 1 - first);
                lg.RemoveAt(lg.Count - 1);
                return lg;
            }
            var outp = new List<float[]>();
            var last = Feed(runner, seq, promptToks, 0, promptToks.Count - 1);
            var batch = new Batch();
            var logits = last[0];
            for (int g = 0; g < nGen; g++)
            {
                outp.Add(logits);
                int next = forced != null ? forced[g] : TensorPrimitives.IndexOfMax<float>(logits);
                toks.Add(next);
                if (g == nGen - 1) break;
                batch.Clear();
                batch.Add(seq, next, seq.Length, true);
                runner.Forward(batch);
                logits = runner.Logits(0).ToArray();
            }
            return outp;
        }

        var cpu = Run(null, null, false, out var cpuToks);
        var gen = cpuToks.Skip(promptToks.Count).ToList();
        var reference = Run(null, gen, true, out _);
        LogitDeviation Deviation(List<float[]> a)
        {
            double maxD = 0, sumD = 0;
            int agree = 0;
            for (int i = 0; i < a.Count; i++)
            {
                double md = 0;
                for (int v = 0; v < c.NVocab; v++) md = Math.Max(md, Math.Abs(a[i][v] - reference[i][v]));
                maxD = Math.Max(maxD, md);
                sumD += md;
                if (TensorPrimitives.IndexOfMax<float>(a[i]) == TensorPrimitives.IndexOfMax<float>(reference[i])) agree++;
            }
            return new LogitDeviation(maxD, sumD / a.Count, agree, a.Count);
        }
        using var off = new GpuLayerOffload(model, new OffloadGeometry(pageSize, ctx / pageSize, maxBatch), options);
        var dec = Deviation(Run(off, gen, false, out _));
        var pre = Deviation(Run(off, gen, true, out _));
        return new GpuValidationResult(model.Vocab.Decode(gen), off.LayerCount, off.DeviceName, Deviation(cpu), dec, pre, off.UsesInt8Activations);
    }
}
