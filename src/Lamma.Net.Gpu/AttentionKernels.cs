using ILGPU;
using ILGPU.Runtime;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

/// <summary>
/// Paged attention over the fp16 KV cache. tokInfo[3t..3t+2] = (page-table offset, first position, last position).
/// </summary>
public struct AttnParams
{
    public long QOff, OutOff;
    public int N, NHead, NKv, Hd, HdV, PageSize, QStride, QHeadPitch, OutStride, KRow, VRow;
    /// <summary>Number of chunks per (token, head) (flash-decoding split; chunk length derived per token).</summary>
    public int NChunks;
    /// <summary>Qwen3.5 gated attention: output *= sigmoid(gate), gate stored after q in each head (q + Hd).</summary>
    public int Gated;
    public float Scale, SoftCap;
}

/// <summary>Eight fp16 values loaded as one 16-byte vector.</summary>
public struct Half8 { public Half H0, H1, H2, H3, H4, H5, H6, H7; }

/// <summary>Four fp16 values loaded as one 8-byte vector.</summary>
public struct Half4 { public Half H0, H1, H2, H3; }

/// <summary>Two fp16 values loaded as one 32-bit word.</summary>
public struct Half2 { public Half A, B; }

internal static partial class Kernels
{
    /// <summary>Largest supported head dimension (shared memory holds q and the accumulator per warp).</summary>
    public const int MaxHeadDim = 512;

    /// <summary>
    /// Flash-decoding attention: one warp per (token, head, chunk of positions) with an online softmax. Lanes take
    /// different positions for Q·K (32 positions per step, each lane reading a whole K row with 16-byte loads), then
    /// split the head dimension for P·V (four positions per step). Requires Hd % 32 == 0 and HdV % 64 == 0. The chunk length is derived per token from its attention range, so the
    /// launch shape depends only on (N, NChunks). With a single chunk the normalized (and gated) output is written
    /// directly; otherwise partial (max, sum, acc) go to <paramref name="part"/> for <see cref="AttnCombine"/>.
    /// Grid: N × ceil(NHead / 4) × NChunks groups of 128.
    /// </summary>
    public static void AttnDecode(ArrayView<float> q, ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> pages, ArrayView<int> tokInfo,
        ArrayView<float> outp, ArrayView<float> part, AttnParams p) =>
        AttnDecodeCore(q, kc, vc, pages, tokInfo, outp, part, p, 0);

    internal static void AttnKvQ8(ArrayView<float> q, ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> pages, ArrayView<int> tokInfo,
        ArrayView<float> outp, ArrayView<float> part, AttnParams p) =>
        AttnDecodeCore(q, kc, vc, pages, tokInfo, outp, part, p, 1);

    private static void AttnDecodeCore(ArrayView<float> q, ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> pages, ArrayView<int> tokInfo,
        ArrayView<float> outp, ArrayView<float> part, AttnParams p, int compact)
    {
        var sh = SharedMemory.Allocate<float>(WarpsPerGroup * 2 * MaxHeadDim);
        var shp = SharedMemory.Allocate<float>(WarpsPerGroup * Warp);
        var shs = SharedMemory.Allocate<long>(WarpsPerGroup * Warp);
        int warp = Group.IdxX / Warp, lane = Lane;
        int headGroups = (p.NHead + WarpsPerGroup - 1) / WarpsPerGroup;
        int blk = Grid.IdxX;
        int c = blk % p.NChunks, rest = blk / p.NChunks;
        int h = (rest % headGroups) * WarpsPerGroup + warp, t = rest / headGroups;
        if (h >= p.NHead || t >= p.N) return;
        int g = h / (p.NHead / p.NKv);
        int qs = warp * 2 * MaxHeadDim, as_ = qs + MaxHeadDim, ws = warp * Warp;
        long qo = p.QOff + (long)t * p.QStride + (long)h * p.QHeadPitch;
        for (int i = lane; i < p.Hd; i += Warp) sh[qs + i] = q[qo + i] * p.Scale;
        for (int i = lane; i < p.HdV; i += Warp) sh[as_ + i] = 0;
        ILGPU.Warp.Barrier();

        int pt = tokInfo[3 * t], start = tokInfo[3 * t + 1], end = tokInfo[3 * t + 2];
        int chunk = (end - start + p.NChunks) / p.NChunks;
        int p0 = start + c * chunk, p1 = p0 + chunk - 1;
        if (p1 > end) p1 = end;
        int vPairs = p.HdV >> 6; // Half2 loads per lane and position (lane owns dims 2·lane + 64·e, +1)
        float m = float.NegativeInfinity, l = 0;
        for (int b0 = p0; b0 <= p1; b0 += Warp)
        {
            int pos = b0 + lane;
            float s = float.NegativeInfinity;
            long slot = 0;
            if (pos <= p1)
            {
                slot = (long)pages[pt + pos / p.PageSize] * p.PageSize + pos % p.PageSize;
                long k8o = (slot * p.KRow + (long)g * p.Hd) >> 3;
                float acc = 0;
                for (int i = 0; i < p.Hd; i += 32)
                {
                    // four independent 16-byte loads in flight per step
                    Half8 ka = LoadCache8(kc, k8o + (i >> 3), compact), kb = LoadCache8(kc, k8o + (i >> 3) + 1, compact), kc8 = LoadCache8(kc, k8o + (i >> 3) + 2, compact), kd = LoadCache8(kc, k8o + (i >> 3) + 3, compact);
                    acc += Dot8(sh, qs + i, ka) + Dot8(sh, qs + i + 8, kb) + Dot8(sh, qs + i + 16, kc8) + Dot8(sh, qs + i + 24, kd);
                }
                s = p.SoftCap > 0 ? Tanh(acc / p.SoftCap) * p.SoftCap : acc;
            }
            float mNew = MathF.Max(m, WarpMax(s));
            float corr = MathF.Exp(m - mNew);
            float pr = pos <= p1 ? MathF.Exp(s - mNew) : 0f;
            l = l * corr + WarpSum(pr);
            m = mNew;
            shp[ws + lane] = pr;
            shs[ws + lane] = (slot * p.VRow + (long)g * p.HdV) >> 1;
            for (int e = 0; e < vPairs; e++)
            {
                int dd = as_ + 2 * lane + 64 * e;
                sh[dd] *= corr;
                sh[dd + 1] *= corr;
            }
            ILGPU.Warp.Barrier();
            int cnt = p1 - b0 + 1 < Warp ? p1 - b0 + 1 : Warp;
            for (int j = 0; j < cnt; j += 4)
            {
                // four positions per step (missing ones get weight 0 and reuse a valid row)
                float q0 = shp[ws + j], q1 = j + 1 < cnt ? shp[ws + j + 1] : 0f, q2 = j + 2 < cnt ? shp[ws + j + 2] : 0f, q3 = j + 3 < cnt ? shp[ws + j + 3] : 0f;
                long r0 = shs[ws + j], r1 = j + 1 < cnt ? shs[ws + j + 1] : r0, r2 = j + 2 < cnt ? shs[ws + j + 2] : r0, r3 = j + 3 < cnt ? shs[ws + j + 3] : r0;
                for (int e = 0; e < vPairs; e++)
                {
                    long o = lane + 32L * e;
                    Half2 va = LoadCache2(vc, r0 + o, compact), vb = LoadCache2(vc, r1 + o, compact), vcc = LoadCache2(vc, r2 + o, compact), vd = LoadCache2(vc, r3 + o, compact);
                    int dd = as_ + 2 * lane + 64 * e;
                    sh[dd] += q0 * (float)va.A + q1 * (float)vb.A + q2 * (float)vcc.A + q3 * (float)vd.A;
                    sh[dd + 1] += q0 * (float)va.B + q1 * (float)vb.B + q2 * (float)vcc.B + q3 * (float)vd.B;
                }
            }
            ILGPU.Warp.Barrier();
        }

        long po = ((long)t * p.NHead + h) * p.NChunks + c;
        if (p.NChunks > 1)
        {
            if (lane == 0) { part[2 * po] = m; part[2 * po + 1] = l; }
            long ao = (long)p.N * p.NHead * p.NChunks * 2 + po * p.HdV;
            for (int i = lane; i < p.HdV; i += Warp) part[ao + i] = sh[as_ + i];
            return;
        }
        float inv = l > 0 ? 1f / l : 0f;
        long oo = p.OutOff + (long)t * p.OutStride + (long)h * p.HdV;
        for (int i = lane; i < p.HdV; i += Warp)
        {
            float o = sh[as_ + i] * inv;
            if (p.Gated != 0) o *= Sigmoid(q[qo + p.Hd + i]);
            outp[oo + i] = o;
        }
    }

    private static float Dot8(ArrayView<float> sh, int o, Half8 k) =>
        sh[o] * (float)k.H0 + sh[o + 1] * (float)k.H1 + sh[o + 2] * (float)k.H2 + sh[o + 3] * (float)k.H3
        + sh[o + 4] * (float)k.H4 + sh[o + 5] * (float)k.H5 + sh[o + 6] * (float)k.H6 + sh[o + 7] * (float)k.H7;

    /// <summary>Merges the per-chunk partial results of <see cref="AttnDecode"/>; one warp per (token, head).</summary>
    public static void AttnCombine(ArrayView<float> q, ArrayView<float> part, ArrayView<float> outp, AttnParams p)
    {
        int w = WarpIndex, lane = Lane;
        if (w >= p.N * p.NHead) return;
        int t = w / p.NHead, h = w % p.NHead;
        long pb = (long)w * p.NChunks;
        float mx = float.NegativeInfinity;
        for (int c = lane; c < p.NChunks; c += Warp) mx = MathF.Max(mx, part[2 * (pb + c)]);
        mx = WarpMax(mx);
        float l = 0;
        for (int c = lane; c < p.NChunks; c += Warp) l += part[2 * (pb + c) + 1] * MathF.Exp(part[2 * (pb + c)] - mx);
        l = WarpSum(l);
        float inv = l > 0 ? 1f / l : 0f;
        long ab = (long)p.N * p.NHead * p.NChunks * 2 + pb * p.HdV;
        long qo = p.QOff + (long)t * p.QStride + (long)h * p.QHeadPitch;
        long oo = p.OutOff + (long)t * p.OutStride + (long)h * p.HdV;
        for (int i = lane; i < p.HdV; i += Warp)
        {
            float acc = 0;
            for (int c = 0; c < p.NChunks; c++)
            {
                float mc = part[2 * (pb + c)];
                if (mc > float.NegativeInfinity) acc += part[ab + (long)c * p.HdV + i] * MathF.Exp(mc - mx);
            }
            float o = acc * inv;
            if (p.Gated != 0) o *= Sigmoid(q[qo + p.Hd + i]);
            outp[oo + i] = o;
        }
    }
}
