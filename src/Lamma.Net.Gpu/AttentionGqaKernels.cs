using ILGPU;
using ILGPU.Runtime;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

internal static partial class Kernels
{
    /// <summary>Largest query-group size (query heads per KV head) of <see cref="AttnDecodeGqa"/>.</summary>
    public const int GqaMaxGroup = 8;
    /// <summary>Key positions per pass of <see cref="AttnDecodeGqa"/> (one per lane).</summary>
    public const int GqaPass = Warp;

    /// <summary>
    /// Decode attention for grouped queries: a 128-thread block per (token, KV head, chunk of positions) computes all
    /// G = NHead / NKv query heads that share the KV head, so each K and V row is read once per group instead of once per
    /// head; chunks are small (32 positions) so that many blocks share the latency-bound work. Per pass of 32 positions
    /// with an online softmax: scores — lane = position, warp w = a quarter of the head dims (16-byte K loads, four in
    /// flight), q·scale of the group in shared memory (broadcast float4 reads), partial scores of the G heads summed across
    /// the warps in shared memory; softmax — warp w updates heads w and w + 4 (running max / sum, probabilities stored per
    /// position); P·V — warp w takes 64-dim value slices w, w + 4, ... for all heads at once (lane = two dims, coalesced
    /// Half2 rows, eight positions in flight, 2·G accumulators). Partial (max, sum, acc) go to <paramref name="part"/> in
    /// the layout of <see cref="AttnDecode"/> for <see cref="AttnCombine"/>; with one chunk the normalized (gated) output is
    /// written directly. Requires G ≤ 8 and Hd == HdV (JIT constants; shared memory is sized for them), Hd % 64 == 0,
    /// Hd ≤ MaxHeadDim. Grid: N × NKv × NChunks groups of 128.
    /// </summary>
    public static void AttnDecodeGqa(ArrayView<float> q, ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> pages, ArrayView<int> tokInfo,
        ArrayView<float> outp, ArrayView<float> part, AttnParams p, SpecializedValue<int> groupSpec, SpecializedValue<int> hdSpec) =>
        AttnDecodeGqaCore(q, kc, vc, pages, tokInfo, outp, part, p, groupSpec, hdSpec, 0);

    internal static void AttnDecodeGqaQ8(ArrayView<float> q, ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> pages, ArrayView<int> tokInfo,
        ArrayView<float> outp, ArrayView<float> part, AttnParams p, SpecializedValue<int> groupSpec, SpecializedValue<int> hdSpec) =>
        AttnDecodeGqaCore(q, kc, vc, pages, tokInfo, outp, part, p, groupSpec, hdSpec, 1);

    private static void AttnDecodeGqaCore(ArrayView<float> q, ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> pages, ArrayView<int> tokInfo,
        ArrayView<float> outp, ArrayView<float> part, AttnParams p, SpecializedValue<int> groupSpec, SpecializedValue<int> hdSpec, int compact)
    {
        int G = groupSpec.Value;
        var qs = SharedMemory.Allocate<float>(groupSpec.Value * hdSpec.Value);         // scaled q rows of the group
        var acc = SharedMemory.Allocate<float>(groupSpec.Value * hdSpec.Value);        // running P·V per head
        var ps = SharedMemory.Allocate<float>(WarpsPerGroup * GqaPass * GqaMaxGroup);  // [quarter][position][head] partial scores
        var pr = SharedMemory.Allocate<float>(GqaPass * GqaMaxGroup);                 // [position][head] probabilities
        var rows = SharedMemory.Allocate<long>(GqaPass);                              // Half2 index of each position's V row
        var runM = SharedMemory.Allocate<float>(GqaMaxGroup);
        var runL = SharedMemory.Allocate<float>(GqaMaxGroup);
        var corr = SharedMemory.Allocate<float>(GqaMaxGroup);
        var q4 = qs.Cast<Float4>();
        var ps4 = ps.Cast<Float4>();
        var pr4 = pr.Cast<Float4>();
        var qg4 = q.Cast<Float4>();
        int tid = Group.IdxX, warp = tid / Warp, lane = tid & (Warp - 1);
        int blk = Grid.IdxX;
        int c = blk % p.NChunks, rest = blk / p.NChunks;
        int g = rest % p.NKv, t = rest / p.NKv;
        int h0 = g * G;
        int hd = p.Hd, hdv = p.HdV;   // runtime loop bounds (fully unrolled constant loops schedule worse for hd 512)

        // ---- q rows of the group (float4 loads, four in flight), scaled
        long qBase = p.QOff + (long)t * p.QStride + (long)h0 * p.QHeadPitch;
        int rowF4 = hd / 4, n4 = G * rowF4;
        for (int i0 = tid; i0 < n4; i0 += 4 * GroupSize)
        {
            int i1 = i0 + GroupSize, i2 = i0 + 2 * GroupSize, i3 = i0 + 3 * GroupSize;
            Float4 a = qg4[QIndex4(qBase, i0, rowF4, p.QHeadPitch)];
            Float4 b = i1 < n4 ? qg4[QIndex4(qBase, i1, rowF4, p.QHeadPitch)] : a;
            Float4 e = i2 < n4 ? qg4[QIndex4(qBase, i2, rowF4, p.QHeadPitch)] : a;
            Float4 f = i3 < n4 ? qg4[QIndex4(qBase, i3, rowF4, p.QHeadPitch)] : a;
            q4[i0] = Scale4(a, p.Scale);
            if (i1 < n4) q4[i1] = Scale4(b, p.Scale);
            if (i2 < n4) q4[i2] = Scale4(e, p.Scale);
            if (i3 < n4) q4[i3] = Scale4(f, p.Scale);
        }
        for (int i = tid; i < G * hdv; i += GroupSize) acc[i] = 0f;
        if (tid < G) { runM[tid] = float.NegativeInfinity; runL[tid] = 0f; }

        int pt = tokInfo[3 * t], start = tokInfo[3 * t + 1], end = tokInfo[3 * t + 2];
        int chunk = (end - start + p.NChunks) / p.NChunks;
        int p0 = start + c * chunk, p1 = p0 + chunk - 1;
        if (p1 > end) p1 = end;
        int hs = hd / 4;          // float4 stride between the q rows of two heads
        int n8 = hd / 32;         // Half8 loads per quarter of a K row
        int dq = warp * hd / 4;   // first dim of this warp's quarter

        for (int b0 = p0; b0 <= p1; b0 += GqaPass)
        {
            Group.Barrier();   // q loaded / the previous pass is done with ps, pr and rows
            int cnt = p1 - b0 + 1 < GqaPass ? p1 - b0 + 1 : GqaPass;

            // ---- partial scores: lane = position, warp = quarter of the dims
            var s = new Scores8();
            if (lane < cnt)
            {
                int pos = b0 + lane;
                long slot = (long)pages[pt + pos / p.PageSize] * p.PageSize + pos % p.PageSize;
                if (warp == 0) rows[lane] = (slot * p.VRow + (long)g * hdv) >> 1;
                long k8o = (slot * p.KRow + (long)g * hd + dq) >> 3;
                int qi = dq / 4;
                int d8 = 0;
                for (; d8 + 4 <= n8; d8 += 4)
                {
                    Half8 ka = LoadCache8(kc, k8o + d8, compact), kb = LoadCache8(kc, k8o + d8 + 1, compact), kcc = LoadCache8(kc, k8o + d8 + 2, compact), kd = LoadCache8(kc, k8o + d8 + 3, compact);
                    s.Add(ka, q4, qi + 2 * d8, hs, G);
                    s.Add(kb, q4, qi + 2 * d8 + 2, hs, G);
                    s.Add(kcc, q4, qi + 2 * d8 + 4, hs, G);
                    s.Add(kd, q4, qi + 2 * d8 + 6, hs, G);
                }
                for (; d8 < n8; d8++) s.Add(LoadCache8(kc, k8o + d8, compact), q4, qi + 2 * d8, hs, G);
            }
            int po = (warp * GqaPass + lane) * 2;
            ps4[po] = new Float4 { X = s.S0, Y = s.S1, Z = s.S2, W = s.S3 };
            ps4[po + 1] = new Float4 { X = s.S4, Y = s.S5, Z = s.S6, W = s.S7 };
            Group.Barrier();

            // ---- scores and online softmax: warp w takes heads w and w + 4, lane = position
            for (int hh = warp; hh < G; hh += WarpsPerGroup)
            {
                float sc = float.NegativeInfinity;
                if (lane < cnt)
                {
                    sc = ps[lane * GqaMaxGroup + hh] + ps[(GqaPass + lane) * GqaMaxGroup + hh]
                       + ps[(2 * GqaPass + lane) * GqaMaxGroup + hh] + ps[(3 * GqaPass + lane) * GqaMaxGroup + hh];
                    if (p.SoftCap > 0) sc = Tanh(sc / p.SoftCap) * p.SoftCap;
                }
                float mOld = runM[hh];
                float mNew = MathF.Max(mOld, WarpMax(sc));   // finite: a pass holds at least one position
                float pv = MathF.Exp(sc - mNew);
                pr[lane * GqaMaxGroup + hh] = pv;
                float sum = WarpSum(pv);
                if (lane == 0)
                {
                    float cr = MathF.Exp(mOld - mNew);
                    corr[hh] = cr;
                    runL[hh] = runL[hh] * cr + sum;
                    runM[hh] = mNew;
                }
            }
            Group.Barrier();

            // ---- P·V: warp w takes 64-dim slices w, w + 4, ... for all heads; lane owns dims 64·e + 2·lane, +1
            for (int e = warp; e < hdv / 64; e += WarpsPerGroup)
            {
                int d0 = 64 * e + 2 * lane;
                var o = new Acc8x2();
                o.Load(acc, d0, hdv, corr, G);
                long vo = d0 >> 1;
                int j = 0;
                for (; j + 8 <= cnt; j += 8)
                {
                    Half2 v0 = LoadCache2(vc, rows[j] + vo, compact), v1 = LoadCache2(vc, rows[j + 1] + vo, compact), v2_ = LoadCache2(vc, rows[j + 2] + vo, compact), v3 = LoadCache2(vc, rows[j + 3] + vo, compact);
                    Half2 v4 = LoadCache2(vc, rows[j + 4] + vo, compact), v5 = LoadCache2(vc, rows[j + 5] + vo, compact), v6 = LoadCache2(vc, rows[j + 6] + vo, compact), v7 = LoadCache2(vc, rows[j + 7] + vo, compact);
                    o.Add(pr4[2 * j], pr4[2 * j + 1], v0, G);
                    o.Add(pr4[2 * j + 2], pr4[2 * j + 3], v1, G);
                    o.Add(pr4[2 * j + 4], pr4[2 * j + 5], v2_, G);
                    o.Add(pr4[2 * j + 6], pr4[2 * j + 7], v3, G);
                    o.Add(pr4[2 * j + 8], pr4[2 * j + 9], v4, G);
                    o.Add(pr4[2 * j + 10], pr4[2 * j + 11], v5, G);
                    o.Add(pr4[2 * j + 12], pr4[2 * j + 13], v6, G);
                    o.Add(pr4[2 * j + 14], pr4[2 * j + 15], v7, G);
                }
                for (; j < cnt; j++) o.Add(pr4[2 * j], pr4[2 * j + 1], LoadCache2(vc, rows[j] + vo, compact), G);
                o.Store(acc, d0, hdv, G);
            }
        }
        Group.Barrier();

        // ---- results per head of the group
        if (p.NChunks > 1)
        {
            if (tid < G)
            {
                long pIdx = ((long)t * p.NHead + h0 + tid) * p.NChunks + c;
                part[2 * pIdx] = runM[tid];
                part[2 * pIdx + 1] = runL[tid];
            }
            long ab = (long)p.N * p.NHead * p.NChunks * 2;
            for (int i = tid; i < G * hdv; i += GroupSize)
            {
                int hh = i / hdv, dd = i - hh * hdv;
                long pIdx = ((long)t * p.NHead + h0 + hh) * p.NChunks + c;
                part[ab + pIdx * hdv + dd] = acc[i];
            }
            return;
        }
        for (int i = tid; i < G * hdv; i += GroupSize)
        {
            int hh = i / hdv, dd = i - hh * hdv;
            float l = runL[hh];
            float o = l > 0 ? acc[i] / l : 0f;
            if (p.Gated != 0) o *= Sigmoid(q[qBase + (long)hh * p.QHeadPitch + hd + dd]);
            outp[p.OutOff + (long)t * p.OutStride + (long)(h0 + hh) * hdv + dd] = o;
        }
    }

    /// <summary>Largest chunk count <see cref="AttnCombine2"/> merges.</summary>
    public const int CombineMaxChunks = 64;

    /// <summary>
    /// Merges the per-chunk partials of <see cref="AttnDecode"/> / <see cref="AttnDecodeGqa"/>: a 128-thread block per
    /// (token, head, 128-dim slice of the value), one dim per thread. The chunk weights exp(m_c − max) and the total sum
    /// are computed once per block (warp 0) into shared memory; each thread then sums its dim over the chunks with eight
    /// coalesced loads in flight. NChunks ≤ 64. Grid: N × NHead × ceil(HdV / 128) groups of 128.
    /// </summary>
    public static void AttnCombine2(ArrayView<float> q, ArrayView<float> part, ArrayView<float> outp, AttnParams p)
    {
        var wts = SharedMemory.Allocate<float>(CombineMaxChunks);
        var tot = SharedMemory.Allocate<float>(1);
        int tid = Group.IdxX, lane = tid & (Warp - 1);
        int slices = (p.HdV + GroupSize - 1) / GroupSize;
        int blk = Grid.IdxX;
        int sl = blk % slices, w = blk / slices;   // w = token · NHead + head
        int t = w / p.NHead, h = w % p.NHead;
        long pb = (long)w * p.NChunks;
        int nc = p.NChunks;
        if (tid < Warp)
        {
            float m0 = lane < nc ? part[2 * (pb + lane)] : float.NegativeInfinity;
            float m1 = lane + Warp < nc ? part[2 * (pb + lane + Warp)] : float.NegativeInfinity;
            float l0 = lane < nc ? part[2 * (pb + lane) + 1] : 0f;
            float l1 = lane + Warp < nc ? part[2 * (pb + lane + Warp) + 1] : 0f;
            float mx = WarpMax(MathF.Max(m0, m1));
            float w0 = m0 > float.NegativeInfinity ? MathF.Exp(m0 - mx) : 0f;
            float w1 = m1 > float.NegativeInfinity ? MathF.Exp(m1 - mx) : 0f;
            float l = WarpSum(l0 * w0 + l1 * w1);
            wts[lane] = w0;
            wts[lane + Warp] = w1;
            if (lane == 0) tot[0] = l;
        }
        Group.Barrier();
        int i = sl * GroupSize + tid;
        if (i >= p.HdV) return;
        long ab = (long)p.N * p.NHead * p.NChunks * 2 + pb * p.HdV + i;
        float acc = 0;
        int c = 0;
        for (; c + 8 <= nc; c += 8)
        {
            float a0 = part[ab + (long)c * p.HdV], a1 = part[ab + (long)(c + 1) * p.HdV], a2 = part[ab + (long)(c + 2) * p.HdV], a3 = part[ab + (long)(c + 3) * p.HdV];
            float a4 = part[ab + (long)(c + 4) * p.HdV], a5 = part[ab + (long)(c + 5) * p.HdV], a6 = part[ab + (long)(c + 6) * p.HdV], a7 = part[ab + (long)(c + 7) * p.HdV];
            // chunks without positions hold weight 0 (their partial sums are 0 too)
            acc += a0 * wts[c] + a1 * wts[c + 1] + a2 * wts[c + 2] + a3 * wts[c + 3] + a4 * wts[c + 4] + a5 * wts[c + 5] + a6 * wts[c + 6] + a7 * wts[c + 7];
        }
        for (; c < nc; c++) acc += part[ab + (long)c * p.HdV] * wts[c];
        float total = tot[0];
        float o = total > 0 ? acc / total : 0f;
        if (p.Gated != 0) o *= Sigmoid(q[p.QOff + (long)t * p.QStride + (long)h * p.QHeadPitch + p.Hd + i]);
        outp[p.OutOff + (long)t * p.OutStride + (long)h * p.HdV + i] = o;
    }

    /// <summary>Float4 index of float4 <paramref name="i"/> of the group's q rows (rows of <paramref name="rowF4"/> float4s).</summary>
    private static long QIndex4(long qBase, int i, int rowF4, int headPitch)
    {
        int hh = i / rowF4;
        return (qBase + (long)hh * headPitch) / 4 + (i - hh * rowF4);
    }

    private static Float4 Scale4(Float4 v, float s) => new() { X = v.X * s, Y = v.Y * s, Z = v.Z * s, W = v.W * s };

    /// <summary>Scores of up to eight query heads for one key position.</summary>
    private struct Scores8
    {
        public float S0, S1, S2, S3, S4, S5, S6, S7;

        /// <summary>Adds q·k over 8 dims: <paramref name="qi"/> is the float4 index of the dims in head 0's q row.</summary>
        public void Add(Half8 k, ArrayView<Float4> q4, int qi, int hs, int g)
        {
            float k0 = (float)k.H0, k1 = (float)k.H1, k2 = (float)k.H2, k3 = (float)k.H3;
            float k4 = (float)k.H4, k5 = (float)k.H5, k6 = (float)k.H6, k7 = (float)k.H7;
            S0 += Dot8(q4[qi], q4[qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
            if (g > 1) S1 += Dot8(q4[hs + qi], q4[hs + qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
            if (g > 2) S2 += Dot8(q4[2 * hs + qi], q4[2 * hs + qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
            if (g > 3) S3 += Dot8(q4[3 * hs + qi], q4[3 * hs + qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
            if (g > 4) S4 += Dot8(q4[4 * hs + qi], q4[4 * hs + qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
            if (g > 5) S5 += Dot8(q4[5 * hs + qi], q4[5 * hs + qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
            if (g > 6) S6 += Dot8(q4[6 * hs + qi], q4[6 * hs + qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
            if (g > 7) S7 += Dot8(q4[7 * hs + qi], q4[7 * hs + qi + 1], k0, k1, k2, k3, k4, k5, k6, k7);
        }
    }

    /// <summary>Two value dims of up to eight heads (P·V accumulators).</summary>
    private struct Acc8x2
    {
        public float X0, X1, X2, X3, X4, X5, X6, X7, Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7;

        /// <summary>Adds probabilities (heads 0-3 in <paramref name="pa"/>, 4-7 in <paramref name="pb"/>) times one V pair.</summary>
        public void Add(Float4 pa, Float4 pb, Half2 v, int g)
        {
            float vx = (float)v.A, vy = (float)v.B;
            X0 += pa.X * vx; Y0 += pa.X * vy;
            if (g > 1) { X1 += pa.Y * vx; Y1 += pa.Y * vy; }
            if (g > 2) { X2 += pa.Z * vx; Y2 += pa.Z * vy; }
            if (g > 3) { X3 += pa.W * vx; Y3 += pa.W * vy; }
            if (g > 4) { X4 += pb.X * vx; Y4 += pb.X * vy; }
            if (g > 5) { X5 += pb.Y * vx; Y5 += pb.Y * vy; }
            if (g > 6) { X6 += pb.Z * vx; Y6 += pb.Z * vy; }
            if (g > 7) { X7 += pb.W * vx; Y7 += pb.W * vy; }
        }

        /// <summary>Loads dims d0, d0 + 1 of each head's running sum, rescaled by the head's softmax correction.</summary>
        public void Load(ArrayView<float> acc, int d0, int hdv, ArrayView<float> corr, int g)
        {
            float c0 = corr[0]; X0 = acc[d0] * c0; Y0 = acc[d0 + 1] * c0;
            if (g > 1) { float c1 = corr[1]; X1 = acc[hdv + d0] * c1; Y1 = acc[hdv + d0 + 1] * c1; }
            if (g > 2) { float c2 = corr[2]; X2 = acc[2 * hdv + d0] * c2; Y2 = acc[2 * hdv + d0 + 1] * c2; }
            if (g > 3) { float c3 = corr[3]; X3 = acc[3 * hdv + d0] * c3; Y3 = acc[3 * hdv + d0 + 1] * c3; }
            if (g > 4) { float c4 = corr[4]; X4 = acc[4 * hdv + d0] * c4; Y4 = acc[4 * hdv + d0 + 1] * c4; }
            if (g > 5) { float c5 = corr[5]; X5 = acc[5 * hdv + d0] * c5; Y5 = acc[5 * hdv + d0 + 1] * c5; }
            if (g > 6) { float c6 = corr[6]; X6 = acc[6 * hdv + d0] * c6; Y6 = acc[6 * hdv + d0 + 1] * c6; }
            if (g > 7) { float c7 = corr[7]; X7 = acc[7 * hdv + d0] * c7; Y7 = acc[7 * hdv + d0 + 1] * c7; }
        }

        public void Store(ArrayView<float> acc, int d0, int hdv, int g)
        {
            acc[d0] = X0; acc[d0 + 1] = Y0;
            if (g > 1) { acc[hdv + d0] = X1; acc[hdv + d0 + 1] = Y1; }
            if (g > 2) { acc[2 * hdv + d0] = X2; acc[2 * hdv + d0 + 1] = Y2; }
            if (g > 3) { acc[3 * hdv + d0] = X3; acc[3 * hdv + d0 + 1] = Y3; }
            if (g > 4) { acc[4 * hdv + d0] = X4; acc[4 * hdv + d0 + 1] = Y4; }
            if (g > 5) { acc[5 * hdv + d0] = X5; acc[5 * hdv + d0 + 1] = Y5; }
            if (g > 6) { acc[6 * hdv + d0] = X6; acc[6 * hdv + d0 + 1] = Y6; }
            if (g > 7) { acc[7 * hdv + d0] = X7; acc[7 * hdv + d0 + 1] = Y7; }
        }
    }

    private static float Dot8(Float4 a, Float4 b, float k0, float k1, float k2, float k3, float k4, float k5, float k6, float k7) =>
        a.X * k0 + a.Y * k1 + a.Z * k2 + a.W * k3 + b.X * k4 + b.Y * k5 + b.Z * k6 + b.W * k7;
}
