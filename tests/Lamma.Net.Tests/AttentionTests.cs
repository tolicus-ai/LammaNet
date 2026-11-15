using LammaNet.Compute;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Tests;

/// <summary>The blocked prefill attention path must match the per-token flash-decoding path.</summary>
public unsafe class AttentionTests
{
    [Theory]
    [InlineData(TestModels.Qwen3Q4, 0, 0f, KvCacheType.F16)]
    [InlineData(TestModels.Qwen3Q4, 64, 0f, KvCacheType.F16)]
    [InlineData(TestModels.Qwen3Q4, 0, 30f, KvCacheType.F32)]
    [InlineData(TestModels.Qwen3Q4, 0, 0f, KvCacheType.Q8_0)]
    [InlineData(TestModels.Gemma4, 0, 0f, KvCacheType.F16)]
    [InlineData(TestModels.Gemma4, 100, 0f, KvCacheType.F16)]
    public void BlockedPrefillMatchesPerToken(string modelFile, int window, float softCap, KvCacheType kvType)
    {
        var path = TestModels.Find(modelFile);
        if (path == null) return;
        using var model = Model.Load(path, new ModelLoadOptions { Prefetch = false });
        var c = model.Config;
        using var pool = new ComputePool(4);
        var attn = new AttentionKernel(pool);
        foreach (int layer in Enumerable.Range(0, c.NLayer).Where(l => c.Kinds[l] == LayerKind.Attention && c.KvSourceLayer[l] == l).Take(6).DistinctBy(l => c.HeadDimK[l]))
        {
            const int prefix = 263, n = 37; // queries at positions 263..299, last query block partial
            int ctx = prefix + n;
            using var kv = new KvCache(c, ctx + 64, kvType);
            var seq = new SequenceState(kv, c);
            seq.EnsureCapacity(ctx);
            int kd = c.NHeadKv[layer] * c.HeadDimK[layer], vd = c.NHeadKv[layer] * c.HeadDimV[layer];
            int qd = c.NHead[layer] * c.HeadDimK[layer], od = c.NHead[layer] * c.HeadDimV[layer];
            var rnd = new Random(7 + layer);
            var kr = new float[kd]; var vr = new float[vd];
            fixed (float* kp = kr, vp = vr)
                for (int p = 0; p < ctx; p++)
                {
                    for (int i = 0; i < kd; i++) kr[i] = (float)rnd.NextDouble() * 2 - 1;
                    for (int i = 0; i < vd; i++) vr[i] = (float)rnd.NextDouble() * 2 - 1;
                    kv.Write(layer, seq.Pages[p / kv.PageSize], p % kv.PageSize, kp, vp);
                }
            using var q = new NativeBuffer((long)n * qd * 4);
            using var o1 = new NativeBuffer((long)n * od * 4);
            using var o2 = new NativeBuffer((long)n * od * 4);
            for (long i = 0; i < (long)n * qd; i++) q.F[i] = (float)rnd.NextDouble() * 2 - 1;
            var prm = new AttentionKernel.Params
            {
                Cache = kv, Layer = layer, NHead = c.NHead[layer], NKv = c.NHeadKv[layer],
                HeadDimK = c.HeadDimK[layer], HeadDimV = c.HeadDimV[layer],
                Scale = 1f / MathF.Sqrt(c.HeadDimK[layer]), SoftCap = softCap, Window = window,
                Q = q.F, QStride = qd, Out = o1.F, OutStride = od,
            };
            var pos = Enumerable.Range(prefix, n).ToArray();
            attn.Run(prm, n, Enumerable.Repeat(seq, n).ToArray(), pos); // blocked path

            for (int t = 0; t < n; t++)
            {
                var single = prm;
                single.Q = q.F + (long)t * qd;
                single.Out = o2.F + (long)t * od;
                attn.Run(single, 1, [seq], [pos[t]]); // flash-decoding path
            }
            double maxErr = 0;
            for (long i = 0; i < (long)n * od; i++) maxErr = Math.Max(maxErr, Math.Abs(o1.F[i] - o2.F[i]));
            Assert.True(maxErr < 1e-4, $"layer {layer} hd={c.HeadDimK[layer]}: max abs diff {maxErr}");
        }
    }
}
