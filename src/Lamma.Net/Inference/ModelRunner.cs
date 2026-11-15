using LammaNet.Compute;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Inference;

/// <summary>
/// Executes forward passes of a <see cref="Model"/> on the CPU for micro-batches that may mix several sequences.
/// Owns all scratch memory; not thread-safe (one runner per execution stream).
/// </summary>
public sealed unsafe class ModelRunner : IDisposable
{
    private readonly Model _model;
    private readonly ModelConfig _c;
    private readonly ComputePool _pool;
    private readonly MatMul _mm;
    private readonly AttentionKernel _attn;
    private readonly KvCache? _kv;
    private readonly int _maxBatch;
    private readonly ActionJob _rowJob = new();

    // scratch buffers (sized for _maxBatch rows)
    private readonly NativeBuffer _x, _xb, _xb2, _q, _k, _v, _att, _gate, _up, _tmp;
    private readonly NativeBuffer _logits;
    private readonly NativeBuffer? _ple, _pleTmp;
    private readonly NativeBuffer? _embOut; // embedding models: per-token output projection (NEmbdOut wide)
    private readonly NativeBuffer _ropeCache;
    private readonly int _maxLogits;
    private readonly List<IDisposable> _pinned = [];

    public Model Model => _model;
    public KvCache? KvCache => _kv;
    public ComputePool Pool => _pool;
    public int MaxBatch => _maxBatch;
    public int VocabSize => _c.NVocab;

    private readonly ILayerOffload? _offload;

    /// <summary>The accelerator executing the first layers, if any.</summary>
    public ILayerOffload? Offload => _offload;

    public ModelRunner(Model model, KvCache? kv, ComputePool pool, int maxBatch = 512, int maxLogits = 0, ILayerOffload? offload = null)
    {
        _offload = offload is { LayerCount: > 0 } ? offload : null;
        _model = model;
        _c = model.Config;
        _pool = pool;
        _mm = new MatMul(pool);
        _attn = new AttentionKernel(pool);
        _kv = kv;
        _maxBatch = maxBatch;
        _maxLogits = maxLogits > 0 ? maxLogits : maxBatch;
        var c = _c;
        int n = maxBatch;
        int maxQ = 0, maxKv = 0, maxAtt = 0, maxFf = 0;
        for (int i = 0; i < c.NLayer; i++)
        {
            maxQ = Math.Max(maxQ, c.NHead[i] * c.HeadDimK[i] * (c.GatedAttention ? 2 : 1));
            maxKv = Math.Max(maxKv, c.NHeadKv[i] * Math.Max(c.HeadDimK[i], c.HeadDimV[i]));
            maxAtt = Math.Max(maxAtt, c.NHead[i] * c.HeadDimV[i]);
            maxFf = Math.Max(maxFf, c.NFf[i]);
        }
        // recurrent layers need extra room
        if (c.HasRecurrentLayers)
        {
            int convDim = c.Kinds.Contains(LayerKind.GatedDeltaNet)
                ? c.SsmStateSize * c.SsmGroupCount * 2 + c.SsmInnerSize
                : c.SsmInnerSize + 2 * c.SsmGroupCount * c.SsmStateSize;
            int inProj = 2 * c.SsmInnerSize + 2 * c.SsmGroupCount * c.SsmStateSize + c.SsmTimeStepRank;
            maxQ = Math.Max(maxQ, Math.Max(convDim, inProj));
            if (c.Kinds.Contains(LayerKind.ShortConv)) maxQ = Math.Max(maxQ, 3 * c.NEmbd);
            maxKv = Math.Max(maxKv, c.SsmInnerSize);
            maxAtt = Math.Max(maxAtt, c.SsmInnerSize);
        }
        maxFf = Math.Max(maxFf, Math.Max(c.NEmbd, Math.Max(maxAtt, c.SsmInnerSize)));
        // buffers the offload transfers every batch are page-aligned so the device can register them for DMA
        bool pin = offload != null;
        _x = new NativeBuffer((long)n * c.NEmbd * 4, pageAligned: pin);
        _xb = new NativeBuffer((long)n * c.NEmbd * 4);
        _xb2 = new NativeBuffer((long)n * c.NEmbd * 4);
        _q = new NativeBuffer((long)n * maxQ * 4);
        _k = new NativeBuffer((long)n * Math.Max(maxKv, 64) * 4);
        _v = new NativeBuffer((long)n * Math.Max(maxKv, 64) * 4);
        _att = new NativeBuffer((long)n * Math.Max(maxAtt, c.NEmbd) * 4);
        _gate = new NativeBuffer((long)n * maxFf * 4);
        _up = new NativeBuffer((long)n * maxFf * 4);
        _tmp = new NativeBuffer((long)n * Math.Max(maxQ, maxFf) * 4);
        _logits = new NativeBuffer((long)_maxLogits * c.NVocab * 4, zero: false, pageAligned: pin);
        if (c.NEmbdPerLayer > 0)
        {
            _ple = new NativeBuffer((long)n * c.NLayer * c.NEmbdPerLayer * 4, pageAligned: pin);
            _pleTmp = new NativeBuffer((long)n * c.NLayer * c.NEmbdPerLayer * 4);
        }
        if (c.IsEmbeddingModel && c.NEmbdOut > 0) _embOut = new NativeBuffer((long)_maxLogits * c.NEmbdOut * 4);
        int maxRot = c.Rope.Where(r => r != null).Select(r => r!.Dims).DefaultIfEmpty(2).Max();
        _ropeCache = new NativeBuffer((long)n * 2 * maxRot * 4, pageAligned: pin);
        if (_offload != null)
            foreach (var b in new[] { _x, _logits, _ple, _ropeCache })
            {
                if (b == null) continue;
                // logits: only the first rows (decode, small batches) — page-locking a whole 512-row x 262k-vocab buffer
                // would lock half a gigabyte of RAM; later rows are still copied, just staged by the driver
                long bytes = b == _logits ? Math.Min(b.Capacity, ((long)64 * c.NVocab * 4 + NativeBuffer.PageSize - 1) / NativeBuffer.PageSize * NativeBuffer.PageSize) : b.Capacity;
                if (_offload.PinHostMemory(b.Pointer, bytes) is { } handle) _pinned.Add(handle);
            }
    }

    /// <summary>Logits of the i-th output row of the last forward pass.</summary>
    public Span<float> Logits(int i) => new(_logits.F + (long)i * _c.NVocab, _c.NVocab);
    public float* LogitsPtr(int i) => _logits.F + (long)i * _c.NVocab;

    /// <summary>
    /// Embedding of output row i after an embeddings-only forward: the final-norm hidden state, or for embedding models
    /// with an output projection its projection (<see cref="EmbeddingSize"/> wide).
    /// </summary>
    public float* HiddenPtr(int i) => _embOut != null ? _embOut.F + (long)i * _c.NEmbdOut : _xb.F + (long)i * _c.NEmbd;

    /// <summary>Width of the vectors returned by <see cref="HiddenPtr"/>.</summary>
    public int EmbeddingSize => _embOut != null ? _c.NEmbdOut : _c.NEmbd;

    /// <summary>
    /// The LM-head input (normalized last hidden state) of output row <paramref name="i"/> of the latest forward pass,
    /// when its head ran on the CPU (MTP drafters start from it). Valid until the next forward pass.
    /// </summary>
    internal float* HeadInput(int i) => _xb.F + (long)i * _c.NEmbd;

    /// <summary>
    /// Runs the model over <paramref name="batch"/>, appending its positions to the KV cache / recurrent states.
    /// Logits are produced for entries with <c>WantLogits</c>, in batch order.
    /// When <paramref name="embeddingsOnly"/> is set the LM head is skipped and normalized hidden states are left in <see cref="HiddenPtr"/>.
    /// </summary>
    public void Forward(Batch batch, bool embeddingsOnly = false)
    {
        using var _prof = Profiler.Section("forward");
        int n = batch.Count;
        if (n == 0) return;
        if (n > _maxBatch) throw new ArgumentException($"Batch of {n} exceeds max batch {_maxBatch}");
        if (batch.LogitCount > _maxLogits) throw new ArgumentException("Too many logit rows requested");
        if (_c.IsEmbeddingModel && !embeddingsOnly && batch.LogitCount > 0) throw new InvalidOperationException("Embedding models produce embeddings, not logits");
        var c = _c;
        int d = c.NEmbd;
        float eps = c.RmsEps;

        var seqs = new SequenceState[n];
        var pos = new int[n];
        for (int t = 0; t < n; t++) { seqs[t] = batch.Sequences[t]; pos[t] = batch.Positions[t]; }

        // make sure the KV cache has room
        if (_kv != null && c.NAttentionLayers > 0)
        {
            var maxPos = new Dictionary<SequenceState, int>();
            for (int t = 0; t < n; t++) maxPos[seqs[t]] = Math.Max(maxPos.GetValueOrDefault(seqs[t], -1), pos[t]);
            foreach (var (s, mp) in maxPos)
                if (!s.EnsureCapacity(mp + 1)) throw new KvCacheFullException("KV cache is full");
        }

        // ---------------- embeddings
        float* x = _x.F;
        for (int t = 0; t < n; t++)
        {
            float* row = x + (long)t * d;
            var emb = batch.Embeddings[t];
            if (emb != null)
            {
                fixed (float* e = emb) Ops.Copy(e, row, d);
            }
            else
            {
                _model.TokenEmbedding.DequantizeRow(batch.Tokens[t], row);
                if (c.EmbeddingScale != 1f) Ops.Scale(row, c.EmbeddingScale, row, d);
            }
        }
        if (_ple != null) ComputePerLayerInputs(batch, n, _offload?.ComputesPerLayerInputs == true);

        // ---------------- rope caches per distinct config
        var ropeCaches = BuildRopeCaches(batch, n);

        // compact output rows for the last layer
        var outRows = new List<int>(batch.LogitCount);
        for (int t = 0; t < n; t++) if (batch.WantLogits[t]) outRows.Add(t);

        int rows = n; // active rows
        int firstHostLayer = 0;
        bool headOnDevice = false;
        if (_offload != null)
        {
            headOnDevice = _offload.ComputesHead && !embeddingsOnly && outRows.Count > 0;
            using (Profiler.Section("offload"))
                _offload.Forward(new OffloadBatch
                {
                    X = x, N = n, Sequences = seqs, Positions = pos,
                    PerLayerInputs = _ple != null ? _ple.F : null, RopeCaches = ropeCaches,
                    OutputRows = headOnDevice ? outRows.ToArray() : null, Logits = headOnDevice ? _logits.F : null,
                });
            firstHostLayer = _offload.LayerCount;
        }
        for (int il = firstHostLayer; il < c.NLayer; il++)
        {
            var L = _model.Layers[il];
            float* xb = _xb.F;
            // attn norm
            RowsRmsNorm(x, L.AttnNorm, xb, rows, d, eps);

            float* mixOut = _att.F; // result of mixer projected back to d
            using (Profiler.Section("mixer"))
            switch (L.Kind)
            {
                case LayerKind.Attention: AttentionLayer(il, L, xb, mixOut, n, seqs, pos, ropeCaches); break;
                case LayerKind.GatedDeltaNet: DeltaNetLayer(il, L, xb, mixOut, n, seqs); break;
                case LayerKind.Mamba2: Mamba2Layer(il, L, xb, mixOut, n, seqs); break;
                case LayerKind.ShortConv: ShortConvLayer(il, L, xb, mixOut, n, seqs); break;
            }

            bool last = il == c.NLayer - 1;
            if (last && outRows.Count < rows)
            {
                // keep only rows that need outputs from here on
                for (int i = 0; i < outRows.Count; i++)
                {
                    int src = outRows[i];
                    if (src == i) continue;
                    Ops.Copy(x + (long)src * d, x + (long)i * d, d);
                    Ops.Copy(mixOut + (long)src * d, mixOut + (long)i * d, d);
                    if (_ple != null)
                        Ops.Copy(_ple.F + (long)src * c.NLayer * c.NEmbdPerLayer, _ple.F + (long)i * c.NLayer * c.NEmbdPerLayer, c.NLayer * c.NEmbdPerLayer);
                }
                rows = outRows.Count;
                if (rows == 0) break;
            }

            if (L.AttnPostNorm != null) RowsRmsNorm(mixOut, L.AttnPostNorm, mixOut, rows, d, eps);
            if (c.ResidualScale != 1f) Ops.Scale(mixOut, c.ResidualScale, mixOut, rows * d);
            Ops.Add(x, mixOut, x, rows * d);

            // ---- FFN
            if (L.FfnDown != null)
            {
                RowsRmsNorm(x, L.FfnNorm, xb, rows, d, eps);
                float* ffo = _att.F;
                using (Profiler.Section("ffn")) Ffn(L, xb, ffo, rows);
                if (L.FfnPostNorm != null) RowsRmsNorm(ffo, L.FfnPostNorm, ffo, rows, d, eps);
                if (c.ResidualScale != 1f) Ops.Scale(ffo, c.ResidualScale, ffo, rows * d);
                Ops.Add(x, ffo, x, rows * d);
            }

            // ---- per-layer embeddings (Gemma 3n/4)
            if (_ple != null && L.PleInpGate != null) PerLayerEmbedding(il, L, x, rows);

            if (L.OutScale != 1f) Ops.Scale(x, L.OutScale, x, rows * d);
            if (s_debug) DebugDump($"l_out-{il}", x, rows, d);
        }

        if (headOnDevice) rows = outRows.Count;
        else if (rows > outRows.Count)
        {
            // every layer ran on the accelerator: compact output rows here
            for (int i = 0; i < outRows.Count; i++)
                if (outRows[i] != i) Ops.Copy(x + (long)outRows[i] * d, x + (long)i * d, d);
            rows = outRows.Count;
        }

        // commit positions to the sequences
        for (int t = 0; t < n; t++)
        {
            var sq = seqs[t];
            if (pos[t] == sq.Length)
            {
                sq.TokenList.Add(batch.Tokens[t]); // token id, or a negative media key for embedding rows
                sq.Length++;
            }
            else if (pos[t] > sq.Length) throw new InvalidOperationException("Batch positions must be contiguous per sequence");
        }

        if (rows == 0 || outRows.Count == 0 || headOnDevice) return;

        // ---------------- head
        float* h = _xb.F;
        RowsRmsNorm(x, _model.OutputNorm, h, rows, d, eps);
        if (embeddingsOnly)
        {
            if (_embOut != null) _mm.Multiply(_model.Output, h, rows, _embOut.F, d, c.NEmbdOut);
            return;
        }
        float* logits = _logits.F;
        if (Environment.GetEnvironmentVariable("LAMMA_X2") == "1") using (Profiler.Section("lm_head_pre")) _mm.Multiply(_model.Output, h, rows, logits, d, c.NVocab);
        using (Profiler.Section("lm_head")) _mm.Multiply(_model.Output, h, rows, logits, d, c.NVocab);
        long total = (long)rows * c.NVocab;
        if (c.LogitScale != 1f) Ops.Scale(logits, 1f / c.LogitScale, logits, (int)total);
        if (c.FinalLogitSoftcap > 0)
            for (int r = 0; r < rows; r++) Ops.SoftCap(logits + (long)r * c.NVocab, logits + (long)r * c.NVocab, c.NVocab, c.FinalLogitSoftcap);
    }

    // ===================================================================== helpers

    private static readonly bool s_debug = Environment.GetEnvironmentVariable("LAMMA_DEBUG") == "1";

    internal static void DebugDump(string name, float* x, int rows, int d)
    {
        double sum = 0;
        for (long i = 0; i < (long)rows * d; i++) sum += x[i];
        Console.Error.WriteLine($"{name} [{d}, {rows}] sum = {sum:F6}");
        for (int r = 0; r < Math.Min(rows, 5); r++)
        {
            float* p = x + (long)r * d;
            Console.Error.WriteLine($"   [{p[0],10:F4}, {p[1],10:F4}, {p[2],10:F4}, ..., {p[d - 3],10:F4}, {p[d - 2],10:F4}, {p[d - 1],10:F4}]");
        }
    }

    private void RowsRmsNorm(float* x, float[]? w, float* y, int rows, int d, float eps)
    {
        if (w == null) { if (x != y) Ops.Copy(x, y, rows * d); return; }
        fixed (float* wp = w)
        {
            float* wpp = wp;
            if (rows >= 16)
            {
                _rowJob.Body = (r, _) => Ops.RmsNorm(x + (long)r * d, wpp, y + (long)r * d, d, eps);
                _pool.Run(_rowJob, rows);
            }
            else for (int r = 0; r < rows; r++) Ops.RmsNorm(x + (long)r * d, wpp, y + (long)r * d, d, eps);
        }
    }

    private static void AddBias(float* y, float[]? b, int rows, int stride, int len)
    {
        if (b == null) return;
        fixed (float* bp = b) for (int r = 0; r < rows; r++) Ops.Add(y + (long)r * stride, bp, y + (long)r * stride, len);
    }

    private Dictionary<RopeConfig, nint> BuildRopeCaches(Batch batch, int n)
    {
        var res = new Dictionary<RopeConfig, nint>();
        float* basePtr = _ropeCache.F;
        long offset = 0;
        Span<int> p4 = stackalloc int[4];
        foreach (var rc in _c.Rope)
        {
            if (rc == null || res.ContainsKey(rc)) continue;
            if (offset + (long)n * rc.Dims > _ropeCache.Length / 4) throw new InvalidOperationException("rope cache too small");
            float* cache = basePtr + offset;
            for (int t = 0; t < n; t++)
            {
                var mp = batch.MultiPositions[t];
                if (mp != null) { p4.Clear(); mp.AsSpan().CopyTo(p4); }
                else { p4[0] = p4[1] = p4[2] = batch.Positions[t]; p4[3] = 0; }
                rc.BuildCache(p4, cache + (long)t * rc.Dims);
            }
            res[rc] = (nint)cache;
            offset += (long)n * rc.Dims;
        }
        return res;
    }

    // ===================================================================== attention

    private void AttentionLayer(int il, LayerWeights L, float* xb, float* outp, int n, SequenceState[] seqs, int[] pos,
        Dictionary<RopeConfig, nint> ropeCaches)
    {
        var c = _c;
        int nh = c.NHead[il], nkv = c.NHeadKv[il], hk = c.HeadDimK[il], hv = c.HeadDimV[il];
        int src = c.KvSourceLayer[il];
        bool ownKv = src == il;
        int qDim = nh * hk * (c.GatedAttention ? 2 : 1);
        float* q = _q.F, k = _k.F, v = _v.F;
        int kDim = nkv * hk, vDim = nkv * hv;

        if (L.Wqkv != null)
        {
            int total = L.Wqkv.Rows;
            float* qkv = _tmp.F;
            _mm.Multiply(L.Wqkv, xb, n, qkv, c.NEmbd, total);
            for (int t = 0; t < n; t++)
            {
                float* r = qkv + (long)t * total;
                Ops.Copy(r, q + (long)t * qDim, qDim);
                if (ownKv)
                {
                    Ops.Copy(r + qDim, k + (long)t * kDim, kDim);
                    Ops.Copy(r + qDim + kDim, v + (long)t * vDim, vDim);
                }
            }
        }
        else
        {
            _mm.Multiply(L.Wq!, xb, n, q, c.NEmbd, qDim);
            if (ownKv)
            {
                _mm.Multiply(L.Wk!, xb, n, k, c.NEmbd, kDim);
                if (L.Wv != null) _mm.Multiply(L.Wv, xb, n, v, c.NEmbd, vDim);
                else Ops.Copy(k, v, n * kDim);
            }
        }
        AddBias(q, L.Bq, n, qDim, qDim);
        if (ownKv) { AddBias(k, L.Bk, n, kDim, kDim); AddBias(v, L.Bv, n, vDim, vDim); }

        // gated attention: q_full per head = [q(hk) | gate(hk)]; split into q (compact) and gate
        float* gate = null;
        if (c.GatedAttention)
        {
            gate = _gate.F;
            float* qc = _tmp.F;
            for (int t = 0; t < n; t++)
                for (int h = 0; h < nh; h++)
                {
                    float* srcp = q + (long)t * qDim + h * 2 * hk;
                    Ops.Copy(srcp, qc + (long)t * nh * hk + h * hk, hk);
                    Ops.Copy(srcp + hk, gate + (long)t * nh * hk + h * hk, hk);
                }
            Ops.Copy(qc, q, n * nh * hk);
            qDim = nh * hk;
        }

        var rope = c.Rope[il];
        float* rc = rope != null ? (float*)ropeCaches[rope] : null;
        float eps = c.RmsEps;
        var kv = _kv!;
        int ps = kv.PageSize;
        float[]? qn = L.QNorm, kn = L.KNorm;
        _rowJob.Body = (t, _) =>
        {
            float* qt = q + (long)t * qDim;
            if (qn != null) fixed (float* w = qn) for (int h = 0; h < nh; h++) Ops.RmsNorm(qt + h * hk, w, qt + h * hk, hk, eps);
            if (rope != null) rope.Apply(qt, nh, hk, rc + (long)t * rope.Dims);
            if (ownKv)
            {
                float* kt = k + (long)t * kDim, vt = v + (long)t * vDim;
                if (kn != null) fixed (float* w = kn) for (int h = 0; h < nkv; h++) Ops.RmsNorm(kt + h * hk, w, kt + h * hk, hk, eps);
                if (c.VNorm) for (int h = 0; h < nkv; h++) Ops.RmsNorm(vt + h * hv, null, vt + h * hv, hv, eps);
                if (rope != null) rope.Apply(kt, nkv, hk, rc + (long)t * rope.Dims);
                var s = seqs[t];
                int p = pos[t];
                kv.Write(il, s.Pages[p / ps], p % ps, kt, vt);
            }
        };
        using (Profiler.Section("attn.rope_kv")) _pool.Run(_rowJob, n);

        float scale = c.AttentionScale != 0 ? c.AttentionScale : 1f / MathF.Sqrt(hk);
        var prm = new AttentionKernel.Params
        {
            Cache = kv, Layer = src, NHead = nh, NKv = nkv, HeadDimK = hk, HeadDimV = hv,
            Scale = scale, SoftCap = c.AttnLogitSoftcap, Window = c.IsSwa[il] ? c.SlidingWindow : 0,
            Q = q, QStride = qDim, Out = _tmp.F, OutStride = nh * hv,
        };
        if (!c.Causal) (prm.Starts, prm.Ends) = NonCausalRanges(c, il, seqs, pos, n);
        using (Profiler.Section("attn.kernel")) _attn.Run(prm, n, seqs, pos);
        float* ao = _tmp.F;
        if (gate != null)
        {
            int m = n * nh * hv;
            Ops.Sigmoid(gate, gate, m);
            Ops.Mul(ao, gate, ao, m);
        }
        _mm.Multiply(L.Wo!, ao, n, outp, nh * hv, c.NEmbd);
        AddBias(outp, L.Bo, n, c.NEmbd, c.NEmbd);
    }

    /// <summary>
    /// Key ranges of bidirectional attention: every token sees its sequence's positions in this batch (the whole input
    /// must be in one batch); sliding-window layers use a window around the token (symmetric or trailing).
    /// </summary>
    internal static (int[] Starts, int[] Ends) NonCausalRanges(ModelConfig c, int il, SequenceState[] seqs, int[] pos, int n)
    {
        var seqEnd = new Dictionary<SequenceState, int>();
        for (int t = 0; t < n; t++) seqEnd[seqs[t]] = Math.Max(seqEnd.GetValueOrDefault(seqs[t]), pos[t] + 1);
        var st = new int[n];
        var en = new int[n];
        bool swa = c.IsSwa[il] && c.SlidingWindow > 0;
        for (int t = 0; t < n; t++)
        {
            int end = seqEnd[seqs[t]];
            if (!swa) { st[t] = 0; en[t] = end; continue; }
            int half = c.SlidingWindow / 2;
            st[t] = c.SymmetricSwa ? Math.Max(0, pos[t] - half) : Math.Max(0, pos[t] - c.SlidingWindow + 1);
            en[t] = c.SymmetricSwa ? Math.Min(end, pos[t] + half + 1) : end;
        }
        return (st, en);
    }

    // ===================================================================== FFN

    private void Ffn(LayerWeights L, float* xb, float* outp, int rows)
    {
        var c = _c;
        int ff = L.FfnDown!.Cols;
        float* g = _gate.F, u = _up.F;
        _mm.Multiply(L.FfnUp!, xb, rows, u, c.NEmbd, ff);
        AddBias(u, L.FfnUpBias, rows, ff, ff);
        if (L.FfnGate != null)
        {
            _mm.Multiply(L.FfnGate, xb, rows, g, c.NEmbd, ff);
            AddBias(g, L.FfnGateBias, rows, ff, ff);
            long total = (long)rows * ff;
            const int chunk = 4096;
            int tasks = (int)((total + chunk - 1) / chunk);
            bool gelu = c.Activation == FfnActivation.GeGlu;
            _rowJob.Body = (i, _) =>
            {
                long o = (long)i * chunk;
                int len = (int)Math.Min(chunk, total - o);
                if (gelu) Ops.GeGlu(g + o, u + o, u + o, len);
                else Ops.SwiGlu(g + o, u + o, u + o, len);
            };
            _pool.Run(_rowJob, tasks);
        }
        _mm.Multiply(L.FfnDown, u, rows, outp, ff, c.NEmbd);
        AddBias(outp, L.FfnDownBias, rows, c.NEmbd, c.NEmbd);
    }

    // ===================================================================== per-layer embeddings

    /// <summary>
    /// Gemma 3n/4 per-layer inputs: (tokenEmbedding·√pd + norm(proj(x)/√d)) / √2 per layer segment. With
    /// <paramref name="tokenPartOnly"/> only the scaled token embeddings are produced (the offload adds the projection).
    /// </summary>
    private void ComputePerLayerInputs(Batch batch, int n, bool tokenPartOnly)
    {
        var c = _c;
        int pd = c.NEmbdPerLayer, nl = c.NLayer, d = c.NEmbd;
        int tot = pd * nl;
        float* ple = _ple!.F;
        float* proj = _pleTmp!.F;
        float tokScale = MathF.Sqrt(pd);
        bool tokenPart = _model.PerLayerTokenEmbedding != null; // EmbeddingGemma 2: projection only
        if (tokenPartOnly)
        {
            if (!tokenPart) return; // the device writes the projection directly
            _rowJob.Body = (t, _) =>
            {
                float* pl = ple + (long)t * tot;
                _model.PerLayerTokenEmbedding!.DequantizeRow(batch.Embeddings[t] != null ? 0 : batch.Tokens[t], pl);
                Ops.Scale(pl, tokScale, pl, tot);
            };
            _pool.Run(_rowJob, n);
            return;
        }
        // projection of the (scaled) input embeddings
        _mm.Multiply(_model.PerLayerModelProj!, _x.F, n, proj, d, tot);
        float projScale = 1f / MathF.Sqrt(d);
        float inScale = tokenPart ? 1f / MathF.Sqrt(2f) : 1f;
        var normW = _model.PerLayerProjNorm!;
        _rowJob.Body = (t, _) =>
        {
            float* pr = proj + (long)t * tot;
            float* pl = ple + (long)t * tot;
            int tok = batch.Embeddings[t] != null ? 0 : batch.Tokens[t];
            if (tokenPart)
            {
                _model.PerLayerTokenEmbedding!.DequantizeRow(tok, pl);
                Ops.Scale(pl, tokScale, pl, tot);
            }
            else new Span<float>(pl, tot).Clear();
            fixed (float* w = normW)
                for (int l = 0; l < nl; l++)
                {
                    float* seg = pr + l * pd;
                    Ops.Scale(seg, projScale, seg, pd);
                    Ops.RmsNorm(seg, w, seg, pd, c.RmsEps);
                }
            Ops.Add(pl, pr, pl, tot);
            Ops.Scale(pl, inScale, pl, tot);
        };
        _pool.Run(_rowJob, n);
    }

    private void PerLayerEmbedding(int il, LayerWeights L, float* x, int rows)
    {
        var c = _c;
        int pd = c.NEmbdPerLayer, d = c.NEmbd, tot = pd * c.NLayer;
        float* g = _gate.F;
        _mm.Multiply(L.PleInpGate!, x, rows, g, d, pd);
        float* ple = _ple!.F;
        for (int r = 0; r < rows; r++)
        {
            float* gr = g + (long)r * pd;
            Ops.Gelu(gr, gr, pd);
            Ops.Mul(gr, ple + (long)r * tot + il * pd, gr, pd);
        }
        float* pout = _xb2.F;
        _mm.Multiply(L.PleProj!, g, rows, pout, pd, d);
        RowsRmsNorm(pout, L.PlePostNorm, pout, rows, d, c.RmsEps);
        Ops.Add(x, pout, x, rows * d);
    }

    // ===================================================================== Gated DeltaNet

    private void DeltaNetLayer(int il, LayerWeights L, float* xb, float* outp, int n, SequenceState[] seqs)
    {
        var c = _c;
        int hkDim = c.SsmStateSize, nK = c.SsmGroupCount, nV = c.SsmTimeStepRank;
        int hvDim = c.SsmInnerSize / nV;
        int keyDim = hkDim * nK, valueDim = hvDim * nV;
        int convDim = keyDim * 2 + valueDim;
        int kSize = c.SsmConvKernel;
        float* qkv = _q.F;            // [n, convDim]
        float* z = _k.F;              // [n, valueDim]
        float* beta = _v.F;           // [n, nV]
        float* alpha = _v.F + (long)n * nV; // [n, nV]
        _mm.Multiply(L.SsmInProjQkv!, xb, n, qkv, c.NEmbd, convDim);
        _mm.Multiply(L.SsmGate!, xb, n, z, c.NEmbd, valueDim);
        _mm.Multiply(L.SsmBeta!, xb, n, beta, c.NEmbd, nV);
        _mm.Multiply(L.SsmAlpha!, xb, n, alpha, c.NEmbd, nV);
        Ops.Sigmoid(beta, beta, n * nV);
        fixed (float* dtb = L.SsmDtBias, a = L.SsmA)
            for (int t = 0; t < n; t++)
                for (int h = 0; h < nV; h++)
                {
                    float* ap = alpha + (long)t * nV + h;
                    *ap = Ops.Softplus(*ap + dtb[h]) * a[h]; // gate (log-decay)
                }

        float* convOut = _tmp.F; // [n, convDim]
        float* core = _gate.F; // [n, valueDim]
        float eps = c.RmsEps;
        float scale = 1f / MathF.Sqrt(hvDim);

        // process sequences: tokens of a sequence are processed in order; different sequences in parallel by heads
        var groups = GroupBySequence(seqs, n);
        foreach (var (seq, idx) in groups)
        {
            var rs = seq.Recurrent!;
            float* convState = rs.Conv(il);   // [convDim][kSize-1]
            float* state = rs.Ssm(il);        // [nV][hv][hk] (row j = v-dim, col i = k-dim)
            // conv + silu, sequential over tokens; channels in parallel
            fixed (float* w = L.SsmConv1d)
            {
                float* wp = w;
                int[] tokIdx = idx;
                _rowJob.Body = (chunkI, _) =>
                {
                    int c0 = chunkI * 256, c1 = Math.Min(convDim, c0 + 256);
                    for (int ch = c0; ch < c1; ch++)
                    {
                        float* st = convState + (long)ch * (kSize - 1);
                        float* wc = wp + (long)ch * kSize;
                        foreach (int t in tokIdx)
                        {
                            float xin = qkv[(long)t * convDim + ch];
                            float acc = xin * wc[kSize - 1];
                            for (int kk = 0; kk < kSize - 1; kk++) acc += st[kk] * wc[kk];
                            for (int kk = 0; kk < kSize - 2; kk++) st[kk] = st[kk + 1];
                            if (kSize > 1) st[kSize - 2] = xin;
                            convOut[(long)t * convDim + ch] = acc / (1f + MathF.Exp(-acc)); // silu
                        }
                    }
                };
                _pool.Run(_rowJob, (convDim + 255) / 256);
            }
            // l2 norm q,k per head
            foreach (int t in idx)
            {
                float* r = convOut + (long)t * convDim;
                for (int h = 0; h < nK; h++)
                {
                    Ops.L2Norm(r + h * hkDim, r + h * hkDim, hkDim, eps);
                    Ops.L2Norm(r + keyDim + h * hkDim, r + keyDim + h * hkDim, hkDim, eps);
                }
            }
            // delta rule, heads in parallel
            int[] tokIdx2 = idx;
            _rowJob.Body = (h, _) =>
            {
                float* S = state + (long)h * hvDim * hkDim;
                float* delta = stackalloc float[hvDim];
                int kh = h % nK;
                foreach (int t in tokIdx2)
                {
                    float* r = convOut + (long)t * convDim;
                    float* qv = r + kh * hkDim;
                    float* kvec = r + keyDim + kh * hkDim;
                    float* vv = r + 2 * keyDim + h * hvDim;
                    float g = MathF.Exp(alpha[(long)t * nV + h]);
                    float b = beta[(long)t * nV + h];
                    Ops.Scale(S, g, S, hvDim * hkDim);
                    for (int j = 0; j < hvDim; j++)
                        delta[j] = (vv[j] - VecDot.F32(S + (long)j * hkDim, kvec, hkDim)) * b;
                    for (int j = 0; j < hvDim; j++)
                        Ops.AddScaled(kvec, delta[j], S + (long)j * hkDim, hkDim);
                    float* outv = core + (long)t * valueDim + h * hvDim;
                    for (int j = 0; j < hvDim; j++)
                        outv[j] = VecDot.F32(S + (long)j * hkDim, qv, hkDim) * scale;
                }
            };
            _pool.Run(_rowJob, nV);
        }

        // gated norm: rmsnorm(core per head, ssm_norm) * silu(z)
        fixed (float* nw = L.SsmNorm)
        {
            float* nwp = nw;
            _rowJob.Body = (t, _) =>
            {
                for (int h = 0; h < nV; h++)
                {
                    float* cv = core + (long)t * valueDim + h * hvDim;
                    float* zv = z + (long)t * valueDim + h * hvDim;
                    Ops.RmsNorm(cv, nwp, cv, hvDim, eps);
                    for (int j = 0; j < hvDim; j++) cv[j] *= zv[j] / (1f + MathF.Exp(-zv[j]));
                }
            };
            _pool.Run(_rowJob, n);
        }
        _mm.Multiply(L.SsmOut!, core, n, outp, valueDim, c.NEmbd);
    }

    // ===================================================================== Mamba2

    private void Mamba2Layer(int il, LayerWeights L, float* xb, float* outp, int n, SequenceState[] seqs)
    {
        var c = _c;
        int dInner = c.SsmInnerSize, dState = c.SsmStateSize, nHead = c.SsmTimeStepRank, nGroup = c.SsmGroupCount;
        int headDim = dInner / nHead;
        int convDim = dInner + 2 * nGroup * dState;
        int dInProj = 2 * dInner + 2 * nGroup * dState + nHead;
        int kSize = c.SsmConvKernel;
        float* zx = _q.F; // [n, dInProj]
        _mm.Multiply(L.SsmIn!, xb, n, zx, c.NEmbd, dInProj);
        float* xbc = _tmp.F; // [n, convDim] after conv
        float* y = _gate.F;  // [n, dInner]
        var groups = GroupBySequence(seqs, n);
        foreach (var (seq, idx) in groups)
        {
            var rs = seq.Recurrent!;
            float* convState = rs.Conv(il);
            float* state = rs.Ssm(il); // [nHead][headDim][dState]
            fixed (float* w = L.SsmConv1d, bias = L.SsmConv1dBias)
            {
                float* wp = w, bp = bias;
                int[] tokIdx = idx;
                _rowJob.Body = (chunkI, _) =>
                {
                    int c0 = chunkI * 256, c1 = Math.Min(convDim, c0 + 256);
                    for (int ch = c0; ch < c1; ch++)
                    {
                        float* st = convState + (long)ch * (kSize - 1);
                        float* wc = wp + (long)ch * kSize;
                        foreach (int t in tokIdx)
                        {
                            float xin = zx[(long)t * dInProj + dInner + ch];
                            float acc = xin * wc[kSize - 1];
                            for (int kk = 0; kk < kSize - 1; kk++) acc += st[kk] * wc[kk];
                            for (int kk = 0; kk < kSize - 2; kk++) st[kk] = st[kk + 1];
                            if (kSize > 1) st[kSize - 2] = xin;
                            if (bp != null) acc += bp[ch];
                            xbc[(long)t * convDim + ch] = acc / (1f + MathF.Exp(-acc));
                        }
                    }
                };
                _pool.Run(_rowJob, (convDim + 255) / 256);
            }
            int[] tokIdx2 = idx;
            fixed (float* A = L.SsmA, D = L.SsmD, dtb = L.SsmDtBias)
            {
                float* Ap = A, Dp = D, dtbp = dtb;
                _rowJob.Body = (h, _) =>
                {
                    int g = h / (nHead / nGroup);
                    float* S = state + (long)h * headDim * dState;
                    foreach (int t in tokIdx2)
                    {
                        float* xr = xbc + (long)t * convDim;
                        float* xh = xr + h * headDim;
                        float* B = xr + dInner + g * dState;
                        float* C = xr + dInner + nGroup * dState + g * dState;
                        float dt = zx[(long)t * dInProj + 2 * dInner + 2 * nGroup * dState + h] + dtbp[h];
                        float dtSp = Ops.Softplus(dt);
                        float dA = MathF.Exp(dtSp * Ap[h]);
                        float* yh = y + (long)t * dInner + h * headDim;
                        for (int i = 0; i < headDim; i++)
                        {
                            float* s = S + (long)i * dState;
                            float xdt = xh[i] * dtSp;
                            Ops.Scale(s, dA, s, dState);
                            Ops.AddScaled(B, xdt, s, dState);
                            yh[i] = VecDot.F32(s, C, dState) + xh[i] * Dp[h];
                        }
                    }
                };
                _pool.Run(_rowJob, nHead);
            }
        }
        // y = y * silu(z); grouped rms norm
        int gsz = dInner / nGroup;
        fixed (float* nw = L.SsmNorm)
        {
            float* nwp = nw;
            _rowJob.Body = (t, _) =>
            {
                float* yt = y + (long)t * dInner;
                float* zt = zx + (long)t * dInProj;
                for (int i = 0; i < dInner; i++) yt[i] *= zt[i] / (1f + MathF.Exp(-zt[i]));
                if (nwp != null) for (int gi = 0; gi < nGroup; gi++) Ops.RmsNorm(yt + gi * gsz, nwp + gi * gsz, yt + gi * gsz, gsz, c.RmsEps);
            };
            _pool.Run(_rowJob, n);
        }
        _mm.Multiply(L.SsmOut!, y, n, outp, dInner, c.NEmbd);
    }

    // ===================================================================== gated short convolution (LFM2)

    private void ShortConvLayer(int il, LayerWeights L, float* xb, float* outp, int n, SequenceState[] seqs)
    {
        var c = _c;
        int d = c.NEmbd, kSize = c.SsmConvKernel;
        float* bcx = _q.F; // [n, 3d]: b | c | x
        _mm.Multiply(L.SsmIn!, xb, n, bcx, d, 3 * d);
        float* y = _gate.F; // [n, d]
        foreach (var (seq, idx) in GroupBySequence(seqs, n))
        {
            float* convState = seq.Recurrent!.Conv(il); // [d][kSize-1]
            fixed (float* w = L.SsmConv1d)
            {
                float* wp = w;
                int[] tokIdx = idx;
                _rowJob.Body = (chunkI, _) =>
                {
                    int c0 = chunkI * 256, c1 = Math.Min(d, c0 + 256);
                    for (int ch = c0; ch < c1; ch++)
                    {
                        float* st = convState + (long)ch * (kSize - 1);
                        float* wc = wp + (long)ch * kSize;
                        foreach (int t in tokIdx)
                        {
                            float* r = bcx + (long)t * 3 * d;
                            float xin = r[ch] * r[2 * d + ch]; // b · x
                            float acc = xin * wc[kSize - 1];
                            for (int kk = 0; kk < kSize - 1; kk++) acc += st[kk] * wc[kk];
                            for (int kk = 0; kk < kSize - 2; kk++) st[kk] = st[kk + 1];
                            if (kSize > 1) st[kSize - 2] = xin;
                            y[(long)t * d + ch] = r[d + ch] * acc; // c · conv
                        }
                    }
                };
                _pool.Run(_rowJob, (d + 255) / 256);
            }
        }
        _mm.Multiply(L.SsmOut!, y, n, outp, d, d);
    }

    private static List<(SequenceState, int[])> GroupBySequence(SequenceState[] seqs, int n)
    {
        var map = new Dictionary<SequenceState, List<int>>();
        var order = new List<SequenceState>();
        for (int t = 0; t < n; t++)
        {
            if (!map.TryGetValue(seqs[t], out var l)) { l = new List<int>(); map[seqs[t]] = l; order.Add(seqs[t]); }
            l.Add(t);
        }
        return order.Select(s => (s, map[s].ToArray())).ToList();
    }

    public void Dispose()
    {
        foreach (var p in _pinned) p.Dispose();   // unregister before the memory is freed
        _pinned.Clear();
        _x.Dispose(); _xb.Dispose(); _xb2.Dispose(); _q.Dispose(); _k.Dispose(); _v.Dispose(); _att.Dispose();
        _gate.Dispose(); _up.Dispose(); _tmp.Dispose(); _logits.Dispose(); _ple?.Dispose(); _pleTmp?.Dispose(); _embOut?.Dispose();
        _ropeCache.Dispose(); _attn.Dispose();
    }
}

/// <summary>Thrown when the KV cache has no free pages for the requested positions.</summary>
public sealed class KvCacheFullException(string message) : Exception(message);
