using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using LammaNet.Memory;
using LammaNet.Tensors;

namespace LammaNet.Compute;

/// <summary>
/// Paged multi-head attention over the KV cache with grouped-query sharing, sliding windows and
/// logit soft-capping. Work is split over (token, kv-head, kv-chunk) — "flash decoding" — so even a
/// single decoded token with one KV head uses every core; partial softmax results are merged afterwards.
/// </summary>
public sealed unsafe class AttentionKernel : IDisposable
{
    private readonly ComputePool _pool;
    private readonly ScoreJob _job;
    private readonly ReduceJob _reduce;
    private NativeBuffer? _partials;

    private readonly BlockJob _block;

    public AttentionKernel(ComputePool pool)
    {
        _pool = pool;
        _job = new ScoreJob(pool.ThreadCount);
        _reduce = new ReduceJob();
        _block = new BlockJob(pool.ThreadCount);
    }

    /// <summary>Query-block size of the prefill path (queries sharing one converted K/V tile).</summary>
    public const int QueryBlock = 16;
    public const int KeyChunk = 128;

    public struct Params
    {
        public KvCache Cache;
        public int Layer;          // KV-owning layer
        public int NHead, NKv, HeadDimK, HeadDimV;
        public float Scale, SoftCap;
        public int Window;         // 0 = full causal
        /// <summary>Optional explicit key ranges per query token: [Starts[t], Ends[t]) (non-causal attention); null = causal from the position and Window.</summary>
        public int[]? Starts, Ends;
        public float* Q; public int QStride;     // per token, heads contiguous (head stride = HeadDimK)
        public float* Out; public int OutStride; // per token, heads contiguous (head stride = HeadDimV)
    }

    /// <summary>Computes attention for <paramref name="n"/> query tokens.</summary>
    public void Run(in Params p, int n, SequenceState[] seqs, int[] positions)
    {
        lock (_block) RunCore(p, n, seqs, positions);   // the jobs and partial buffers are reused per call
    }

    private void RunCore(in Params p, int n, SequenceState[] seqs, int[] positions)
    {
        if (n >= QueryBlock && seqs.All(s => ReferenceEquals(s, seqs[0])))
        {
            RunBlocked(p, n, seqs[0], positions);
            return;
        }
        int threads = _pool.ThreadCount;
        int maxLen = 0;
        for (int t = 0; t < n; t++) maxLen = Math.Max(maxLen, End(p, positions, t) - Start(p, positions, t));
        // choose chunking so that there are enough tasks to keep all threads busy
        int baseTasks = n * p.NKv;
        int nChunks = Math.Max(1, Math.Min((threads * 4 + baseTasks - 1) / baseTasks, (maxLen + 63) / 64));
        int chunk = (maxLen + nChunks - 1) / nChunks;
        int maxChunk = Math.Clamp(65536 / (p.HeadDimK + p.HeadDimV), 16, 512) & ~15;
        chunk = Math.Min(Math.Max(16, (chunk + 15) & ~15), maxChunk);
        nChunks = Math.Max(1, (maxLen + chunk - 1) / chunk);

        long partFloats = (long)n * p.NHead * nChunks * (p.HeadDimV + 2);
        if (_partials == null || _partials.Length < partFloats * 4)
        {
            _partials?.Dispose();
            _partials = new NativeBuffer(Math.Max(partFloats * 4, 1 << 16), zero: false);
        }

        using (Profiler.Section("attn.setup")) _job.Setup(p, seqs, positions, nChunks, chunk, _partials.F);
        using (Profiler.Section("attn.scores")) _pool.Run(_job, n * p.NKv * nChunks);

        _reduce.Setup(p, nChunks, _partials.F);
        using (Profiler.Section("attn.reduce")) _pool.Run(_reduce, n * p.NHead);
    }

    /// <summary>Prefill path: blocks of queries of one sequence share converted K/V tiles per key chunk.</summary>
    private void RunBlocked(in Params p, int n, SequenceState seq, int[] positions)
    {
        int maxEnd = 0;
        for (int t = 0; t < n; t++) maxEnd = Math.Max(maxEnd, End(p, positions, t));
        int chunk = Math.Min(KeyChunk, Math.Max(16, (65536 / (p.HeadDimK + p.HeadDimV)) & ~15));
        int nChunks = (maxEnd + chunk - 1) / chunk;
        int nBlocks = (n + QueryBlock - 1) / QueryBlock;
        long partFloats = (long)n * p.NHead * nChunks * (p.HeadDimV + 2);
        if (_partials == null || _partials.Length < partFloats * 4)
        {
            _partials?.Dispose();
            _partials = new NativeBuffer(Math.Max(partFloats * 4, 1 << 16), zero: false);
        }
        _block.Setup(p, seq, positions, n, nChunks, chunk, _partials.F);
        using (Profiler.Section("attn.block")) _pool.Run(_block, nBlocks * p.NKv * nChunks);
        _reduce.Setup(p, nChunks, _partials.F);
        using (Profiler.Section("attn.reduce")) _pool.Run(_reduce, n * p.NHead);
    }

    private sealed class BlockJob : ParallelJob
    {
        private Params _p;
        private SequenceState _seq = null!;
        private int[] _pos = null!;
        private int _n, _nChunks, _chunk;
        private float* _part;
        private readonly NativeBuffer[] _scratch;
        private GgmlType _kt, _vt;
        private long _kHeadBytes, _vHeadBytes;

        public BlockJob(int threads) => _scratch = new NativeBuffer[threads];

        public void Setup(in Params p, SequenceState seq, int[] pos, int n, int nChunks, int chunk, float* part)
        {
            _p = p; _seq = seq; _pos = pos; _n = n; _nChunks = nChunks; _chunk = chunk; _part = part;
            _kt = p.Cache.HeadType(p.Layer, false);
            _vt = p.Cache.HeadType(p.Layer, true);
            _kHeadBytes = p.Cache.KRowBytes(p.Layer) / p.NKv;
            _vHeadBytes = p.Cache.VRowBytes(p.Layer) / p.NKv;
            int group = p.NHead / p.NKv;
            long need = ((long)chunk * (p.HeadDimK + p.HeadDimV) + (long)QueryBlock * group * chunk + p.HeadDimK + 80) * 4;
            for (int i = 0; i < _scratch.Length; i++)
                if (_scratch[i] == null || _scratch[i].Length < need) { _scratch[i]?.Dispose(); _scratch[i] = new NativeBuffer(need, zero: false); }
        }

        private static void LoadRow(GgmlType t, byte* src, float* dst, int n)
        {
            switch (t)
            {
                case GgmlType.F32: Buffer.MemoryCopy(src, dst, n * 4L, n * 4L); break;
                case GgmlType.F16: HalfHelper.HalfToFloat((ushort*)src, dst, n); break;
                default: Dequantize.Row(t, src, dst, n); break;
            }
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        public override void Execute(int index, int threadId)
        {
            ref readonly var p = ref _p;
            int c = index % _nChunks;
            int rest = index / _nChunks;
            int g = rest % p.NKv;
            int b = rest / p.NKv;
            int group = p.NHead / p.NKv;
            int hdK = p.HeadDimK, hdV = p.HeadDimV, slots = hdV + 2;
            int t0 = b * QueryBlock, t1 = Math.Min(_n, t0 + QueryBlock);
            int j0 = c * _chunk;
            int blockEnd = 0, blockStart = int.MaxValue;
            for (int t = t0; t < t1; t++)
            {
                blockEnd = Math.Max(blockEnd, End(p, _pos, t));
                int st0 = Start(p, _pos, t);
                blockStart = Math.Min(blockStart, st0);
            }
            int j1 = Math.Min((c + 1) * _chunk, blockEnd);
            int lo = Math.Max(j0, blockStart);
            int rows = (t1 - t0) * group;
            if (lo >= j1)
            {
                for (int t = t0; t < t1; t++)
                    for (int hh = 0; hh < group; hh++)
                    {
                        float* part = _part + (((long)t * p.NHead + g * group + hh) * _nChunks + c) * slots;
                        part[0] = float.NegativeInfinity; part[1] = 0;
                    }
                return;
            }
            int m = j1 - lo;
            int mPad = (m + 15) & ~15;

            float* kT = _scratch[threadId].F;               // [hdK][chunk] transposed K tile
            float* vf = kT + (long)_chunk * hdK;              // [chunk][hdV]
            float* krow = vf + (long)_chunk * hdV;            // one converted K row
            float* sc = krow + ((hdK + 15) & ~15);            // [rows][chunk]
            var pages = _seq.Pages;
            int ps = p.Cache.PageSize;
            for (int j = lo; j < j1; j++)
            {
                int page = pages[j / ps], slot = j % ps;
                LoadRow(_kt, p.Cache.K(p.Layer, page, slot) + g * _kHeadBytes, krow, hdK);
                for (int d = 0; d < hdK; d++) kT[(long)d * _chunk + (j - lo)] = krow[d];
                LoadRow(_vt, p.Cache.V(p.Layer, page, slot) + g * _vHeadBytes, vf + (long)(j - lo) * hdV, hdV);
            }
            for (int j = m; j < mPad; j++)
            {
                for (int d = 0; d < hdK; d++) kT[(long)d * _chunk + j] = 0;
                new Span<float>(vf + (long)j * hdV, hdV).Clear();
            }

            // ---- scores for all (token, head) rows of the block
            var qptr = stackalloc nint[rows];
            for (int r = 0; r < rows; r++)
            {
                int t = t0 + r / group, hh = r % group;
                qptr[r] = (nint)(p.Q + (long)t * p.QStride + (long)(g * group + hh) * hdK);
            }
            bool simd = Avx2.IsSupported && Fma.IsSupported;
            for (int r = 0; r < rows; r += 4)
            {
                int nr = Math.Min(4, rows - r);
                if (simd) Scores4(qptr + r, nr, kT, _chunk, hdK, mPad, sc + (long)r * _chunk);
                else
                    for (int k = 0; k < nr; k++)
                        for (int j = 0; j < m; j++)
                        {
                            float acc = 0;
                            float* q = (float*)qptr[r + k];
                            for (int d = 0; d < hdK; d++) acc += q[d] * kT[(long)d * _chunk + j];
                            sc[(long)(r + k) * _chunk + j] = acc;
                        }
            }

            // ---- per-row masking + softmax
            var mx = stackalloc float[rows];
            var sum = stackalloc float[rows];
            for (int r = 0; r < rows; r++)
            {
                int t = t0 + r / group;
                int end = Math.Min(j1, End(p, _pos, t));
                int st = Math.Max(lo, Start(p, _pos, t));
                float* row = sc + (long)r * _chunk;
                if (st >= end) { mx[r] = float.NegativeInfinity; sum[r] = 0; new Span<float>(row, mPad).Clear(); continue; }
                int a = st - lo, e = end - lo;
                if (p.SoftCap > 0)
                {
                    Ops.Scale(row + a, p.Scale / p.SoftCap, row + a, e - a);
                    Ops.Tanh(row + a, row + a, e - a);
                    Ops.Scale(row + a, p.SoftCap, row + a, e - a);
                }
                else Ops.Scale(row + a, p.Scale, row + a, e - a);
                float mmx = Simd.Max(row + a, e - a);
                float ssum = Simd.ExpSubSum(row + a, mmx, row + a, e - a);
                new Span<float>(row, a).Clear();
                new Span<float>(row + e, mPad - e).Clear();
                mx[r] = mmx; sum[r] = ssum;
            }

            // ---- P·V for 4 rows at a time
            var outs = stackalloc nint[4];
            for (int r = 0; r < rows; r += 4)
            {
                int nr = Math.Min(4, rows - r);
                for (int k = 0; k < nr; k++)
                {
                    int t = t0 + (r + k) / group, hh = (r + k) % group;
                    float* part = _part + (((long)t * p.NHead + g * group + hh) * _nChunks + c) * slots;
                    part[0] = mx[r + k]; part[1] = sum[r + k];
                    outs[k] = (nint)(part + 2);
                }
                if (simd && (hdV & 31) == 0) Pv4(sc + (long)r * _chunk, _chunk, nr, vf, hdV, mPad, outs);
                else for (int k = 0; k < nr; k++) AccumulateV(sc + (long)(r + k) * _chunk, vf, mPad, hdV, (float*)outs[k]);
            }
        }

        /// <summary>scores[k][j] = q_k · K[j] for up to 4 query rows and mPad keys (K transposed: kT[d][j]).</summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static void Scores4(nint* q, int nr, float* kT, int stride, int hd, int mPad, float* outp)
        {
            float* q0 = (float*)q[0], q1 = nr > 1 ? (float*)q[1] : q0, q2 = nr > 2 ? (float*)q[2] : q0, q3 = nr > 3 ? (float*)q[3] : q0;
            for (int j = 0; j < mPad; j += 16)
            {
                Vector256<float> a00 = default, a01 = default, a10 = default, a11 = default, a20 = default, a21 = default, a30 = default, a31 = default;
                float* kp = kT + j;
                for (int d = 0; d < hd; d++, kp += stride)
                {
                    var k0 = Avx.LoadVector256(kp);
                    var k1 = Avx.LoadVector256(kp + 8);
                    var b0 = Vector256.Create(q0[d]); a00 = Fma.MultiplyAdd(b0, k0, a00); a01 = Fma.MultiplyAdd(b0, k1, a01);
                    var b1 = Vector256.Create(q1[d]); a10 = Fma.MultiplyAdd(b1, k0, a10); a11 = Fma.MultiplyAdd(b1, k1, a11);
                    var b2 = Vector256.Create(q2[d]); a20 = Fma.MultiplyAdd(b2, k0, a20); a21 = Fma.MultiplyAdd(b2, k1, a21);
                    var b3 = Vector256.Create(q3[d]); a30 = Fma.MultiplyAdd(b3, k0, a30); a31 = Fma.MultiplyAdd(b3, k1, a31);
                }
                Avx.Store(outp + j, a00); Avx.Store(outp + j + 8, a01);
                if (nr > 1) { Avx.Store(outp + stride + j, a10); Avx.Store(outp + stride + j + 8, a11); }
                if (nr > 2) { Avx.Store(outp + 2 * stride + j, a20); Avx.Store(outp + 2 * stride + j + 8, a21); }
                if (nr > 3) { Avx.Store(outp + 3 * stride + j, a30); Avx.Store(outp + 3 * stride + j + 8, a31); }
            }
        }

        /// <summary>out_k[d] = Σ_j p_k[j]·V[j][d] for up to 4 rows; V rows loaded once per key.</summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static void Pv4(float* pr, int stride, int nr, float* v, int hdV, int m, nint* outs)
        {
            float* p0 = pr, p1 = nr > 1 ? pr + stride : pr, p2 = nr > 2 ? pr + 2 * stride : pr, p3 = nr > 3 ? pr + 3 * stride : pr;
            for (int d0 = 0; d0 < hdV; d0 += 16)
            {
                Vector256<float> a00 = default, a01 = default, a10 = default, a11 = default, a20 = default, a21 = default, a30 = default, a31 = default;
                float* vp = v + d0;
                for (int j = 0; j < m; j++, vp += hdV)
                {
                    var v0 = Avx.LoadVector256(vp);
                    var v1 = Avx.LoadVector256(vp + 8);
                    var w0 = Vector256.Create(p0[j]); a00 = Fma.MultiplyAdd(w0, v0, a00); a01 = Fma.MultiplyAdd(w0, v1, a01);
                    var w1 = Vector256.Create(p1[j]); a10 = Fma.MultiplyAdd(w1, v0, a10); a11 = Fma.MultiplyAdd(w1, v1, a11);
                    var w2 = Vector256.Create(p2[j]); a20 = Fma.MultiplyAdd(w2, v0, a20); a21 = Fma.MultiplyAdd(w2, v1, a21);
                    var w3 = Vector256.Create(p3[j]); a30 = Fma.MultiplyAdd(w3, v0, a30); a31 = Fma.MultiplyAdd(w3, v1, a31);
                }
                float* o0 = (float*)outs[0];
                Avx.Store(o0 + d0, a00); Avx.Store(o0 + d0 + 8, a01);
                if (nr > 1) { float* o = (float*)outs[1]; Avx.Store(o + d0, a10); Avx.Store(o + d0 + 8, a11); }
                if (nr > 2) { float* o = (float*)outs[2]; Avx.Store(o + d0, a20); Avx.Store(o + d0 + 8, a21); }
                if (nr > 3) { float* o = (float*)outs[3]; Avx.Store(o + d0, a30); Avx.Store(o + d0 + 8, a31); }
            }
        }
    }

    /// <summary>out[d] = Σ_j w[j] · v[j][d] with register blocking (v rows contiguous, stride hdV).</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    internal static void AccumulateV(float* w, float* v, int m, int hdV, float* outp)
    {
        int d0 = 0;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            for (; d0 + 32 <= hdV; d0 += 32)
            {
                var a0 = Vector256<float>.Zero; var a1 = a0; var a2 = a0; var a3 = a0;
                float* vp = v + d0;
                for (int j = 0; j < m; j++, vp += hdV)
                {
                    var ww = Vector256.Create(w[j]);
                    a0 = Fma.MultiplyAdd(Avx.LoadVector256(vp), ww, a0);
                    a1 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 8), ww, a1);
                    a2 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 16), ww, a2);
                    a3 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 24), ww, a3);
                }
                Avx.Store(outp + d0, a0); Avx.Store(outp + d0 + 8, a1); Avx.Store(outp + d0 + 16, a2); Avx.Store(outp + d0 + 24, a3);
            }
        }
        for (; d0 < hdV; d0++)
        {
            float acc = 0;
            for (int j = 0; j < m; j++) acc += w[j] * v[(long)j * hdV + d0];
            outp[d0] = acc;
        }
    }

    /// <summary>First visible key position of query token t.</summary>
    private static int Start(in Params p, int[] pos, int t) => p.Starts != null ? p.Starts[t] : p.Window > 0 ? Math.Max(0, pos[t] - p.Window + 1) : 0;

    /// <summary>End (exclusive) of the visible keys of query token t.</summary>
    private static int End(in Params p, int[] pos, int t) => p.Ends != null ? p.Ends[t] : pos[t] + 1;

    private sealed class ScoreJob : ParallelJob
    {
        private Params _p;
        private SequenceState[] _seqs = null!;
        private int[] _pos = null!;
        private int _nChunks, _chunk;
        private float* _part;
        private readonly NativeBuffer[] _scratch;
        private GgmlType _kt, _vt;
        private long _kHeadBytes, _vHeadBytes;

        public ScoreJob(int threads)
        {
            _scratch = new NativeBuffer[threads];
        }

        public void Setup(in Params p, SequenceState[] seqs, int[] pos, int nChunks, int chunk, float* part)
        {
            _p = p; _seqs = seqs; _pos = pos; _nChunks = nChunks; _chunk = chunk; _part = part;
            _kt = p.Cache.HeadType(p.Layer, false);
            _vt = p.Cache.HeadType(p.Layer, true);
            _kHeadBytes = p.Cache.KRowBytes(p.Layer) / p.NKv;
            _vHeadBytes = p.Cache.VRowBytes(p.Layer) / p.NKv;
            int group = p.NHead / p.NKv;
            long need = ((long)chunk * (p.HeadDimK + p.HeadDimV) + (long)group * chunk + 4L * chunk + 64) * 4;
            for (int i = 0; i < _scratch.Length; i++)
                if (_scratch[i] == null || _scratch[i].Length < need) { _scratch[i]?.Dispose(); _scratch[i] = new NativeBuffer(need, zero: false); }
        }

        private static void LoadRow(GgmlType t, byte* src, float* dst, int n)
        {
            switch (t)
            {
                case GgmlType.F32: Buffer.MemoryCopy(src, dst, n * 4L, n * 4L); break;
                case GgmlType.F16: HalfHelper.HalfToFloat((ushort*)src, dst, n); break;
                default: Dequantize.Row(t, src, dst, n); break;
            }
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        public override void Execute(int index, int threadId)
        {
            ref readonly var p = ref _p;
            int c = index % _nChunks;
            int rest = index / _nChunks;
            int g = rest % p.NKv;
            int t = rest / p.NKv;
            int group = p.NHead / p.NKv;
            int hdK = p.HeadDimK, hdV = p.HeadDimV;
            int start = Start(p, _pos, t), len = End(p, _pos, t) - start;
            int j0 = start + c * _chunk;
            int j1 = Math.Min(start + len, j0 + _chunk);
            int slots = hdV + 2;
            float* partBase = _part + (((long)t * p.NHead + g * group) * _nChunks + c) * slots;
            long partHeadStride = (long)_nChunks * slots;

            if (j0 >= j1)
            {
                for (int hh = 0; hh < group; hh++)
                {
                    float* part = partBase + hh * partHeadStride;
                    part[0] = float.NegativeInfinity; part[1] = 0;
                }
                return;
            }

            float* scratch = _scratch[threadId].F;
            float* kf = scratch;                           // [chunk][hdK] float tile
            float* vf = kf + (long)_chunk * hdK;           // [chunk][hdV] float tile
            float* scores = vf + (long)_chunk * hdV;       // [group][chunk]
            byte** kptr = (byte**)(scores + group * _chunk); // [chunk]
            byte** vptr = kptr + _chunk;                      // [chunk]

            var pages = _seqs[t].Pages;
            var cache = p.Cache;
            int ps = cache.PageSize;
            int m = j1 - j0;
            for (int j = j0; j < j1;)
            {
                int page = pages[j / ps];
                int slot = j % ps;
                int cnt = Math.Min(ps - slot, j1 - j);
                byte* kb = cache.K(p.Layer, page, slot) + g * _kHeadBytes;
                byte* vb = cache.V(p.Layer, page, slot) + g * _vHeadBytes;
                long krs = cache.KRowBytes(p.Layer), vrs = cache.VRowBytes(p.Layer);
                for (int i = 0; i < cnt; i++)
                {
                    kptr[j - j0 + i] = kb + i * krs;
                    vptr[j - j0 + i] = vb + i * vrs;
                }
                j += cnt;
            }

            float* q0 = p.Q + (long)t * p.QStride + (long)g * group * hdK;

            // ---- pass 1: scores (K converted to a float tile once, reused by all heads of the group)
            float* kt = kf;
            if (_kt == GgmlType.F32 && group == 1)
            {
                for (int jj = 0; jj < m; jj++) scores[jj] = VecDot.F32((float*)kptr[jj], q0, hdK);
            }
            else
            {
                for (int jj = 0; jj < m; jj++) LoadRow(_kt, kptr[jj], kt + (long)jj * hdK, hdK);
                for (int hh = 0; hh < group; hh++)
                {
                    float* qh = q0 + hh * hdK;
                    float* sc = scores + hh * _chunk;
                    for (int jj = 0; jj < m; jj++) sc[jj] = VecDot.F32(kt + (long)jj * hdK, qh, hdK);
                }
            }

            // ---- softmax (unnormalized, per chunk)
            float scale = p.Scale, cap = p.SoftCap;
            for (int hh = 0; hh < group; hh++)
            {
                float* sc = scores + hh * _chunk;
                if (cap > 0)
                {
                    Ops.Scale(sc, scale / cap, sc, m);
                    Ops.Tanh(sc, sc, m);
                    Ops.Scale(sc, cap, sc, m);
                }
                else Ops.Scale(sc, scale, sc, m);
                float mx = Simd.Max(sc, m);
                float sum = Simd.ExpSubSum(sc, mx, sc, m);
                float* part = partBase + hh * partHeadStride;
                part[0] = mx; part[1] = sum;
            }

            // ---- pass 2: weighted values over a float V tile (register-blocked accumulation)
            for (int jj = 0; jj < m; jj++) LoadRow(_vt, vptr[jj], vf + (long)jj * hdV, hdV);
            bool simd = Avx2.IsSupported && Fma.IsSupported && (hdV & 7) == 0;
            for (int hh = 0; hh < group; hh++)
            {
                float* sc = scores + hh * _chunk;
                float* outp = partBase + hh * partHeadStride + 2;
                if (!simd)
                {
                    new Span<float>(outp, hdV).Clear();
                    for (int jj = 0; jj < m; jj++) Ops.AddScaled(vf + (long)jj * hdV, sc[jj], outp, hdV);
                    continue;
                }
                int d0 = 0;
                for (; d0 + 64 <= hdV; d0 += 64)
                {
                    var a0 = Vector256<float>.Zero; var a1 = a0; var a2 = a0; var a3 = a0;
                    var a4 = a0; var a5 = a0; var a6 = a0; var a7 = a0;
                    float* vp = vf + d0;
                    for (int jj = 0; jj < m; jj++, vp += hdV)
                    {
                        var w = Vector256.Create(sc[jj]);
                        a0 = Fma.MultiplyAdd(Avx.LoadVector256(vp), w, a0);
                        a1 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 8), w, a1);
                        a2 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 16), w, a2);
                        a3 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 24), w, a3);
                        a4 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 32), w, a4);
                        a5 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 40), w, a5);
                        a6 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 48), w, a6);
                        a7 = Fma.MultiplyAdd(Avx.LoadVector256(vp + 56), w, a7);
                    }
                    Avx.Store(outp + d0, a0); Avx.Store(outp + d0 + 8, a1);
                    Avx.Store(outp + d0 + 16, a2); Avx.Store(outp + d0 + 24, a3);
                    Avx.Store(outp + d0 + 32, a4); Avx.Store(outp + d0 + 40, a5);
                    Avx.Store(outp + d0 + 48, a6); Avx.Store(outp + d0 + 56, a7);
                }
                for (; d0 < hdV; d0 += 8)
                {
                    var a0 = Vector256<float>.Zero;
                    float* vp = vf + d0;
                    for (int jj = 0; jj < m; jj++, vp += hdV) a0 = Fma.MultiplyAdd(Avx.LoadVector256(vp), Vector256.Create(sc[jj]), a0);
                    Avx.Store(outp + d0, a0);
                }
            }
        }
    }

    private sealed class ReduceJob : ParallelJob
    {
        private Params _p;
        private int _nChunks;
        private float* _part;

        public void Setup(in Params p, int nChunks, float* part) { _p = p; _nChunks = nChunks; _part = part; }

        public override void Execute(int index, int threadId)
        {
            ref readonly var p = ref _p;
            int h = index % p.NHead, t = index / p.NHead;
            int hdV = p.HeadDimV, slots = hdV + 2;
            float* parts = _part + ((long)t * p.NHead + h) * _nChunks * slots;
            float mx = float.NegativeInfinity;
            for (int c = 0; c < _nChunks; c++) mx = MathF.Max(mx, parts[c * slots]);
            float* o = p.Out + (long)t * p.OutStride + (long)h * hdV;
            new Span<float>(o, hdV).Clear();
            float l = 0;
            if (float.IsNegativeInfinity(mx)) return;
            for (int c = 0; c < _nChunks; c++)
            {
                float* pc = parts + c * slots;
                if (pc[1] == 0) continue;
                float w = MathF.Exp(pc[0] - mx);
                l += pc[1] * w;
                Ops.AddScaled(pc + 2, w, o, hdV);
            }
            if (l > 0) Ops.Scale(o, 1f / l, o, hdV);
        }
    }

    public void Dispose() => _partials?.Dispose();
}
