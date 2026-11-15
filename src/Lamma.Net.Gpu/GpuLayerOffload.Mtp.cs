using ILGPU;
using ILGPU.Runtime;
using LammaNet.Diagnostics;
using LammaNet.Inference;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Tensors;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

public sealed unsafe partial class GpuLayerOffload
{
    /// <summary>
    /// An MTP drafter on this device: it reads the device KV cache of the target's last layers and the LM-head inputs of
    /// the latest forward pass (<c>_xb</c>), on the same stream. Needs the head on the device.
    /// </summary>
    public IDraftRunner? CreateMtpDrafter(MtpDrafterModel drafter, Model target)
    {
        if (_output == null || _layers < _c.NLayer || _acc.AcceleratorType != AcceleratorType.Cuda) return null;
        try
        {
            _acc.Bind();
            return new MtpRunner(this, drafter, target);
        }
        catch (Exception e) when (e is NotSupportedException or AcceleratorException or OutOfMemoryException)
        {
            Log.EncoderCreateFailed(_log, e, "MTP drafter", DeviceName);
            return null;
        }
    }

    /// <summary>The drafter's forward pass (one token, positions of the target) with this offload's kernels.</summary>
    private sealed unsafe class MtpRunner : IDraftRunner
    {
        private readonly GpuLayerOffload _o;
        private readonly MtpDrafterModel _m;
        private readonly Model _target;
        private readonly DMat _pre, _post, _out;
        private readonly (DMat Wq, DMat Wo, DMat Gate, DMat Up, DMat Down)[] _w;
        private readonly (long AttnNorm, long QNorm, long AttnPost, long FfnNorm, long FfnPost)[] _n;
        private readonly long _outNorm;
        private readonly int[] _src;
        private readonly MemoryBuffer1D<float, Stride1D.Dense> _pf, _in, _x, _xb, _q, _att, _mix, _ff, _logits, _rope, _part, _argPart, _argOut;
        private readonly MemoryBuffer1D<int, Stride1D.Dense> _pages, _info;
        private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, int, ArrayView<float>> _kArgPart;
        private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>> _kArgFinal;
        private readonly float[] _emb, _ropeHost, _argHost = new float[3];
        private const int MaxChunks = 64;

        public string Device => _o.DeviceName;

        public MtpRunner(GpuLayerOffload o, MtpDrafterModel m, Model target)
        {
            m.Validate(target);
            _o = o; _m = m; _target = target;
            var ws = new List<WeightTensor?> { m.PreProjection, m.PostProjection, m.TokenEmbedding };
            foreach (var L in m.Layers) ws.AddRange([L.Wq, L.Wo, L.Gate, L.Up, L.Down]);
            o.UploadMatrices(ws.ToArray(), out var mats);
            _pre = mats[0]!; _post = mats[1]!; _out = mats[2]!;
            _w = new (DMat, DMat, DMat, DMat, DMat)[m.Layers.Length];
            for (int il = 0; il < m.Layers.Length; il++)
                _w[il] = (mats[3 + 5 * il]!, mats[4 + 5 * il]!, mats[5 + 5 * il]!, mats[6 + 5 * il]!, mats[7 + 5 * il]!);

            var fp = new List<float>();
            long F(float[] data)
            {
                while (fp.Count % 4 != 0) fp.Add(0);
                long off = fp.Count;
                fp.AddRange(data);
                return off;
            }
            _n = m.Layers.Select(L => (F(L.AttnNorm), F(L.QNorm), F(L.AttnPostNorm), F(L.FfnNorm), F(L.FfnPostNorm))).ToArray();
            _outNorm = F(m.OutputNorm);
            _pf = o.Alloc<float>(fp.Count);
            _pf.View.SubView(0, fp.Count).CopyFromCPU(fp.ToArray());
            _src = Enumerable.Range(0, m.Layers.Length).Select(il => m.TargetKvLayer(il, o._c)).ToArray();
            foreach (int s in _src)
                if (o._dl[s].KCache == null) throw new NotSupportedException($"target layer {s} keeps no KV cache on the device");

            int d = m.NEmbd, dt = m.NEmbdTarget, maxQ = m.Layers.Max(l => l.Wq.Rows), maxHd = m.Layers.Max(l => l.HeadDim);
            int slack = Kernels.MatVecTokens;   // matvec tiles may read padding rows
            _in = o.Alloc<float>(2L * dt * slack); _x = o.Alloc<float>((long)d * slack); _xb = o.Alloc<float>((long)d * slack);
            _q = o.Alloc<float>((long)maxQ * slack); _att = o.Alloc<float>((long)maxQ * slack); _mix = o.Alloc<float>((long)d * slack);
            _ff = o.Alloc<float>((long)m.NFf * slack);
            _logits = o.Alloc<float>(m.NVocab);
            _rope = o.Alloc<float>(m.RopeFull.Dims + m.RopeSwa.Dims);
            _part = o.Alloc<float>((long)m.NHead * MaxChunks * (2 + maxHd));
            _argPart = o.Alloc<float>(3 * Kernels.ArgmaxBlocks);
            _argOut = o.Alloc<float>(4);
            _pages = o.Alloc<int>(o._pageCount + 64);
            _info = o.Alloc<int>(6);
            _emb = new float[dt];
            _ropeHost = new float[m.RopeFull.Dims + m.RopeSwa.Dims];
            _kArgPart = o._acc.LoadKernel<ArrayView<float>, int, ArrayView<float>>(Kernels.ArgmaxPartial);
            _kArgFinal = o._acc.LoadKernel<ArrayView<float>, ArrayView<float>>(Kernels.ArgmaxFinal);
        }

        public int Draft(SequenceState seq, int position, int token, int outputRow, float minProbability, Span<int> drafts)
        {
            if (position <= 0 || drafts.Length == 0) return 0;
            var o = _o;
            var st = o._stream;
            o._acc.Bind();
            int dt = _m.NEmbdTarget;
            // seed: the target's LM-head input of the row that produced the token
            o._xb.View.SubView((long)outputRow * dt, dt).CopyTo(st, _in.View.SubView(dt, dt));
            fixed (float* r = _ropeHost)
            {
                _m.RopeFull.BuildCache([position], r);
                _m.RopeSwa.BuildCache([position], r + _m.RopeFull.Dims);
            }
            _rope.View.SubView(0, _ropeHost.Length).CopyFromCPU(st, _ropeHost);
            // the page table of the positions before this one; key ranges of the full and the sliding layers
            int nPages = (position + o._pageSize - 1) / o._pageSize;
            var pages = new int[nPages];
            for (int i = 0; i < nPages; i++) pages[i] = seq.PageTable[i];
            _pages.View.SubView(0, nPages).CopyFromCPU(st, pages);
            int swaStart = Math.Max(0, position - _m.SlidingWindow + 1);
            _info.View.SubView(0, 6).CopyFromCPU(st, new[] { 0, 0, position - 1, 0, swaStart, position - 1 });

            float embScale = MathF.Sqrt(dt);
            int count = 0, tok = token;
            while (count < drafts.Length)
            {
                fixed (float* e = _emb) _target.TokenEmbedding.DequantizeRow(tok, e);
                for (int i = 0; i < dt; i++) _emb[i] *= embScale;
                _in.View.SubView(0, dt).CopyFromCPU(st, _emb);
                Step(position);
                _argOut.View.SubView(0, 3).CopyToCPU(st, _argHost);
                st.Synchronize();
                int best = (int)_argHost[0];
                if (minProbability > 0 && 1f / _argHost[2] < minProbability) break;
                drafts[count++] = best;
                tok = best;
            }
            return count;
        }

        /// <summary>One drafter pass: input in _in, logits → arg max in _argOut, next hidden state into _in[dt..].</summary>
        private void Step(int position)
        {
            var o = _o;
            var st = o._stream;
            var m = _m;
            int d = m.NEmbd, dt = m.NEmbdTarget, nh = m.NHead, ff = m.NFf;
            var pf = _pf.View;
            // int8 activations for every drafter projection: its numerics only decide which drafts the target verifies
            void MatVec(ArrayView<float> x, int xStride, ArrayView<float> y, DMat w, int mode = 0, float scale = 1f, DMat? b = null, int act = 0) =>
                o._matVec.Run(st, pf, x, 0, xStride, 1, y, mode, scale, (w, 0, w.Rows), b is { } bb ? (bb, 0, bb.Rows) : null, null, act, 0, normedInput: true);
            void Norm(ArrayView<float> x, ArrayView<float> y, long w, int dim, int mode = 0, float scale = 1f) =>
                o._kRms(st, new KernelConfig(1, Kernels.GroupSize), pf, x, y,
                    new NormParams { XOff = 0, YOff = 0, WOff = w, Rows = 1, D = dim, XStride = dim, YStride = dim, Mode = mode, Eps = m.Eps, Scale = scale, InScale = 1f });

            MatVec(_in.View, 2 * dt, _x.View, _pre);
            for (int il = 0; il < m.Layers.Length; il++)
            {
                var L = m.Layers[il];
                var (wq, wo, gate, up, down) = _w[il];
                var (attnNorm, qNorm, attnPost, ffnNorm, ffnPost) = _n[il];
                int hd = L.HeadDim;
                var S = o._dl[_src[il]];
                Norm(_x.View, _xb.View, attnNorm, d);
                MatVec(_xb.View, d, _q.View, wq);
                var rope = L.IsSwa ? m.RopeSwa : m.RopeFull;
                var qp = new QkvParams
                {
                    QOff = 0, KOff = 0, VOff = 0, QNormOff = qNorm, KNormOff = -1, RopeOff = L.IsSwa ? m.RopeFull.Dims : 0,
                    N = 1, NHead = nh, NKv = m.NHeadKv, Hd = hd, HdV = hd, QStride = nh * hd, KStride = 0, VStride = 0, QHeadPitch = hd,
                    KRow = m.NHeadKv * hd, VRow = m.NHeadKv * hd, OwnKv = 0, VNorm = 0,
                    RopeDims = rope.Dims, RopeNeox = 1, RopeStride = rope.Dims, Eps = m.Eps,
                };
                o._kQkv(st, Warps(nh), pf, _q.View, _q.View, _q.View, _rope.View, S.KCache!.View, S.VCache!.View, o._slots.View, qp);
                int keys = L.IsSwa ? Math.Min(position, m.SlidingWindow - 1) : position;
                int chunks = Math.Clamp((keys + o._minChunk - 1) / o._minChunk, 1, MaxChunks);
                var ap = new AttnParams
                {
                    QOff = 0, OutOff = 0, N = 1, NHead = nh, NKv = m.NHeadKv, Hd = hd, HdV = hd, PageSize = o._pageSize,
                    QStride = nh * hd, QHeadPitch = hd, OutStride = nh * hd, KRow = m.NHeadKv * hd, VRow = m.NHeadKv * hd,
                    NChunks = chunks, Gated = 0, Scale = 1f, SoftCap = 0,
                };
                var info = _info.View.SubView(L.IsSwa ? 3 : 0, 3);
                if (o._gqaDecode)
                    o._kAttnGqa(st, new KernelConfig(m.NHeadKv * chunks, Kernels.GroupSize), _q.View, S.KCache.View, S.VCache.View, _pages.View, info,
                        _att.View, _part.View, ap, SpecializedValue.New(nh / m.NHeadKv), SpecializedValue.New(hd));
                else
                    o._kAttn(st, new KernelConfig((nh + Kernels.WarpsPerGroup - 1) / Kernels.WarpsPerGroup * chunks, Kernels.GroupSize), _q.View,
                        S.KCache.View, S.VCache.View, _pages.View, info, _att.View, _part.View, ap);
                if (chunks > 1)
                    o._kCombine(st, new KernelConfig(nh * ((hd + Kernels.GroupSize - 1) / Kernels.GroupSize), Kernels.GroupSize), _q.View, _part.View, _att.View, ap);
                MatVec(_att.View, nh * hd, _mix.View, wo);
                Norm(_mix.View, _x.View, attnPost, d, mode: 1);                       // x += norm(attn)
                Norm(_x.View, _xb.View, ffnNorm, d);
                MatVec(_xb.View, d, _ff.View, gate, mode: 2, b: up, act: 1);          // gelu(gate) · up
                MatVec(_ff.View, ff, _mix.View, down);
                Norm(_mix.View, _x.View, ffnPost, d, mode: 2, scale: L.OutScale);     // x = (x + norm(ffn)) · out_scale
            }
            Norm(_x.View, _xb.View, _outNorm, d);
            MatVec(_xb.View, d, _logits.View, _out);
            MatVec(_xb.View, d, _in.View.SubView(dt, (long)dt * (Kernels.MatVecTokens * 2 - 1)), _post);   // next step's hidden state
            _kArgPart(st, new KernelConfig(Kernels.ArgmaxBlocks, 256), _logits.View, m.NVocab, _argPart.View);
            _kArgFinal(st, new KernelConfig(1, Kernels.Warp), _argPart.View, _argOut.View);
        }

        public void Dispose() { }   // the buffers belong to the offload
    }
}
