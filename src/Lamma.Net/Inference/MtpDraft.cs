using System.Numerics.Tensors;
using LammaNet.Compute;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Inference;

/// <summary>Proposes draft tokens for speculative decoding from the target's state after a forward pass.</summary>
public interface IDraftRunner : IDisposable
{
    /// <summary>
    /// Proposes up to <c>drafts.Length</c> tokens following <paramref name="token"/> (sampled, not yet evaluated: it takes
    /// position <paramref name="position"/> of <paramref name="seq"/>), seeded with the target's LM-head input (normalized
    /// last hidden state) of output row <paramref name="outputRow"/> of the latest forward pass. Stops before a draft whose probability is
    /// below <paramref name="minProbability"/>. Returns the number of drafts.
    /// </summary>
    int Draft(SequenceState seq, int position, int token, int outputRow, float minProbability, Span<int> drafts);

    /// <summary>Where the drafter runs (diagnostics).</summary>
    string Device { get; }
}

/// <summary>
/// <see cref="MtpDrafterModel"/> on the CPU, reading the target's host KV cache and the LM-head inputs the
/// <see cref="ModelRunner"/> keeps for the output rows of its latest forward pass.
/// </summary>
internal sealed unsafe class CpuMtpDrafter : IDraftRunner
{
    private readonly MtpDrafterModel _m;
    private readonly Model _target;
    private readonly KvCache _kv;
    private readonly ModelRunner _runner;
    private readonly MatMul _mm;
    private readonly AttentionKernel _attn;
    private readonly NativeBuffer _in, _x, _xb, _q, _att, _mix, _gate, _up, _logits, _ropeFull, _ropeSwa;
    private readonly int[] _srcLayer;

    public CpuMtpDrafter(MtpDrafterModel drafter, Model target, KvCache kv, ModelRunner runner, ComputePool pool)
    {
        drafter.Validate(target);
        _m = drafter; _target = target; _kv = kv; _runner = runner;
        _mm = new MatMul(pool);
        _attn = new AttentionKernel(pool);
        int d = drafter.NEmbd, dt = drafter.NEmbdTarget, maxQ = drafter.Layers.Max(l => l.Wq.Rows);
        _in = new NativeBuffer(2L * dt * 4); _x = new NativeBuffer(d * 4L); _xb = new NativeBuffer(d * 4L); _mix = new NativeBuffer(d * 4L);
        _q = new NativeBuffer(maxQ * 4L); _att = new NativeBuffer(maxQ * 4L);
        _gate = new NativeBuffer(drafter.NFf * 4L); _up = new NativeBuffer(drafter.NFf * 4L);
        _logits = new NativeBuffer((long)drafter.NVocab * 4);
        _ropeFull = new NativeBuffer(drafter.RopeFull.Dims * 4L); _ropeSwa = new NativeBuffer(drafter.RopeSwa.Dims * 4L);
        _srcLayer = Enumerable.Range(0, drafter.Layers.Length).Select(il => drafter.TargetKvLayer(il, target.Config)).ToArray();
    }

    public string Device => "CPU";

    public int Draft(SequenceState seq, int position, int token, int outputRow, float minProbability, Span<int> drafts)
    {
        if (position <= 0 || drafts.Length == 0) return 0;
        int dt = _m.NEmbdTarget;
        float* inp = _in.F;
        Ops.Copy(_runner.HeadInput(outputRow), inp + dt, dt);
        _m.RopeFull.BuildCache([position], _ropeFull.F);
        _m.RopeSwa.BuildCache([position], _ropeSwa.F);
        float embScale = MathF.Sqrt(dt);
        int count = 0, tok = token;
        while (count < drafts.Length)
        {
            _target.TokenEmbedding.DequantizeRow(tok, inp);
            Ops.Scale(inp, embScale, inp, dt);
            Step(seq, position);   // logits, and the next step's hidden state into inp[dt..]
            var logits = new ReadOnlySpan<float>(_logits.F, _m.NVocab);
            int best = TensorPrimitives.IndexOfMax(logits);
            if (minProbability > 0 && Probability(logits, best) < minProbability) break;
            drafts[count++] = best;
            tok = best;
        }
        return count;
    }

    private static float Probability(ReadOnlySpan<float> logits, int best)
    {
        float mx = logits[best];
        double sum = 0;
        foreach (float l in logits) sum += Math.Exp(l - mx);
        return (float)(1 / sum);
    }

    /// <summary>One drafter pass for the input in <c>_in</c> at <paramref name="position"/>.</summary>
    private void Step(SequenceState seq, int position)
    {
        var m = _m;
        int d = m.NEmbd, dt = m.NEmbdTarget, ff = m.NFf, nh = m.NHead;
        float eps = m.Eps;
        float* x = _x.F, xb = _xb.F, q = _q.F, att = _att.F, mix = _mix.F;
        _mm.Multiply(m.PreProjection, _in.F, 1, x, 2 * dt, d);
        for (int il = 0; il < m.Layers.Length; il++)
        {
            var L = m.Layers[il];
            int hd = L.HeadDim;
            fixed (float* w = L.AttnNorm) Ops.RmsNorm(x, w, xb, d, eps);
            _mm.Multiply(L.Wq, xb, 1, q, d, nh * hd);
            fixed (float* w = L.QNorm) for (int h = 0; h < nh; h++) Ops.RmsNorm(q + h * hd, w, q + h * hd, hd, eps);
            var rope = L.IsSwa ? m.RopeSwa : m.RopeFull;
            rope.Apply(q, nh, hd, L.IsSwa ? _ropeSwa.F : _ropeFull.F);
            // the target's keys before this position (the drafted token itself has no KV)
            int start = L.IsSwa ? Math.Max(0, position - m.SlidingWindow + 1) : 0;
            var prm = new AttentionKernel.Params
            {
                Cache = _kv, Layer = _srcLayer[il], NHead = nh, NKv = m.NHeadKv, HeadDimK = hd, HeadDimV = hd,
                Scale = 1f, SoftCap = 0, Starts = [start], Ends = [position], Q = q, QStride = nh * hd, Out = att, OutStride = nh * hd,
            };
            _attn.Run(prm, 1, [seq], [position]);
            _mm.Multiply(L.Wo, att, 1, mix, nh * hd, d);
            fixed (float* w = L.AttnPostNorm) Ops.RmsNorm(mix, w, mix, d, eps);
            Ops.Add(x, mix, x, d);
            fixed (float* w = L.FfnNorm) Ops.RmsNorm(x, w, xb, d, eps);
            _mm.Multiply(L.Gate, xb, 1, _gate.F, d, ff);
            _mm.Multiply(L.Up, xb, 1, _up.F, d, ff);
            Ops.GeGlu(_gate.F, _up.F, _up.F, ff);
            _mm.Multiply(L.Down, _up.F, 1, mix, ff, d);
            fixed (float* w = L.FfnPostNorm) Ops.RmsNorm(mix, w, mix, d, eps);
            Ops.Add(x, mix, x, d);
            Ops.Scale(x, L.OutScale, x, d);
        }
        fixed (float* w = m.OutputNorm) Ops.RmsNorm(x, w, xb, d, eps);
        _mm.Multiply(m.TokenEmbedding, xb, 1, _logits.F, d, m.NVocab);
        _mm.Multiply(m.PostProjection, xb, 1, _in.F + dt, d, dt);
    }

    public void Dispose()
    {
        foreach (var b in new[] { _in, _x, _xb, _q, _att, _mix, _gate, _up, _logits, _ropeFull, _ropeSwa }) b.Dispose();
    }
}
