using ILGPU;
using ILGPU.Runtime;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

internal static partial class Kernels
{
    /// <summary>Query tokens per block and key positions per chunk of <see cref="AttnPrefill"/>.</summary>
    public const int PrefillBQ = 16, PrefillBKV = 32, PrefillMaxHd = 256;

    /// <summary>
    /// Tiled attention for prompt processing (FlashAttention-style with SIMT math). A 128-thread block handles one head
    /// for a tile of up to 16 consecutive query tokens of one sequence (tiles[2i] = first batch token, tiles[2i+1] =
    /// count); K/V chunks of 32 positions are staged in shared memory (fp16) and reused by every query of the tile.
    /// Thread (q = tid/8, j = tid%8) scores positions 4j..4j+3 of a chunk and accumulates head dims j, j+8, ...
    /// (hd/8 of them, a JIT constant). Grid: tiles × NHead.
    /// </summary>
    public static void AttnPrefill(ArrayView<float> q, ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> pages, ArrayView<int> tokInfo,
        ArrayView<int> tiles, ArrayView<float> outp, AttnParams p, SpecializedValue<int> hdSpec)
    {
        int hd = hdSpec.Value; // == HdV (checked by the host)
        // Q rows padded by one float (conflict-free broadcast reads); K stored transposed [d][pos] so a thread reads its
        // 4 positions with one 8-byte access; V row-major [pos][d]. K and V share one buffer.
        var qs = SharedMemory.Allocate<float>(PrefillBQ * QsLd);
        var kv = SharedMemory.Allocate<Half>(PrefillBKV * PrefillMaxHd);
        var kt4 = kv.Cast<Half4>();
        var kv8 = kv.Cast<Half8>();
        var ps = SharedMemory.Allocate<float>(PrefillBQ * PrefillBKV);
        var slots = SharedMemory.Allocate<long>(PrefillBKV);
        int tid = Group.IdxX;
        int tile = Grid.IdxX / p.NHead, h = Grid.IdxX % p.NHead;
        int t0 = tiles[2 * tile], cnt = tiles[2 * tile + 1];
        int g = h / (p.NHead / p.NKv);
        int qi = tid >> 3, j = tid & 7;
        int dc = hd / 8;

        for (int e = tid; e < PrefillBQ * hd; e += 128)
        {
            int r = e / hd, d = e % hd;
            qs[r * QsLd + d] = r < cnt ? q[p.QOff + (long)(t0 + r) * p.QStride + (long)h * p.QHeadPitch + d] * p.Scale : 0f;
        }
        int pt = tokInfo[3 * t0];
        int kLo = tokInfo[3 * t0 + 1], kHi = tokInfo[3 * (t0 + cnt - 1) + 2];
        bool rowOk = qi < cnt;
        int myStart = rowOk ? tokInfo[3 * (t0 + qi) + 1] : 0, myEnd = rowOk ? tokInfo[3 * (t0 + qi) + 2] : -1;
        float m = float.NegativeInfinity, l = 0;
        var o0 = new Acc16();
        var o1 = new Acc16();

        for (int c0 = kLo; c0 <= kHi; c0 += PrefillBKV)
        {
            Group.Barrier();
            if (tid < PrefillBKV)
            {
                int pos = c0 + tid;
                slots[tid] = pos <= kHi ? (long)pages[pt + pos / p.PageSize] * p.PageSize + pos % p.PageSize : -1;
            }
            Group.Barrier();
            var k8 = kc.Cast<Half8>();
            for (int e = tid; e < PrefillBKV * hd / 8; e += 128)
            {
                int pp = e / (hd / 8), d8 = e % (hd / 8);
                long sl = slots[pp];
                Half8 hv = sl >= 0 ? k8[(sl * p.KRow + (long)g * p.Hd) / 8 + d8] : default;
                int d = d8 * 8;
                kv[(d + 0) * PrefillBKV + pp] = hv.H0; kv[(d + 1) * PrefillBKV + pp] = hv.H1;
                kv[(d + 2) * PrefillBKV + pp] = hv.H2; kv[(d + 3) * PrefillBKV + pp] = hv.H3;
                kv[(d + 4) * PrefillBKV + pp] = hv.H4; kv[(d + 5) * PrefillBKV + pp] = hv.H5;
                kv[(d + 6) * PrefillBKV + pp] = hv.H6; kv[(d + 7) * PrefillBKV + pp] = hv.H7;
            }
            Group.Barrier();
            // scores for positions 4j..4j+3
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int qb = qi * QsLd;
            for (int d = 0; d < hd; d++)
            {
                float qv = qs[qb + d];
                Half4 k4 = kt4[d * (PrefillBKV / 4) + j];
                s0 += qv * (float)k4.H0;
                s1 += qv * (float)k4.H1;
                s2 += qv * (float)k4.H2;
                s3 += qv * (float)k4.H3;
            }
            int pos0 = c0 + 4 * j;
            s0 = Mask(s0, pos0, myStart, myEnd, p.SoftCap);
            s1 = Mask(s1, pos0 + 1, myStart, myEnd, p.SoftCap);
            s2 = Mask(s2, pos0 + 2, myStart, myEnd, p.SoftCap);
            s3 = Mask(s3, pos0 + 3, myStart, myEnd, p.SoftCap);
            float mx = MathF.Max(MathF.Max(s0, s1), MathF.Max(s2, s3));
            mx = MathF.Max(mx, ILGPU.Warp.ShuffleXor(mx, 1));
            mx = MathF.Max(mx, ILGPU.Warp.ShuffleXor(mx, 2));
            mx = MathF.Max(mx, ILGPU.Warp.ShuffleXor(mx, 4));
            float mNew = MathF.Max(m, mx);
            float corr = 1f, p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            if (mNew > float.NegativeInfinity)
            {
                corr = MathF.Exp(m - mNew);
                p0 = MathF.Exp(s0 - mNew); p1 = MathF.Exp(s1 - mNew); p2 = MathF.Exp(s2 - mNew); p3 = MathF.Exp(s3 - mNew);
            }
            float rs = p0 + p1 + p2 + p3;
            rs += ILGPU.Warp.ShuffleXor(rs, 1);
            rs += ILGPU.Warp.ShuffleXor(rs, 2);
            rs += ILGPU.Warp.ShuffleXor(rs, 4);
            l = l * corr + rs;
            m = mNew;
            int pb = qi * PrefillBKV + 4 * j;
            ps[pb] = p0; ps[pb + 1] = p1; ps[pb + 2] = p2; ps[pb + 3] = p3;
            Group.Barrier();
            // V chunk replaces K
            var v8 = vc.Cast<Half8>();
            for (int e = tid; e < PrefillBKV * hd / 8; e += 128)
            {
                int pp = e / (hd / 8), d8 = e % (hd / 8);
                long sl = slots[pp];
                kv8[pp * (PrefillMaxHd / 8) + d8] = sl >= 0 ? v8[(sl * p.VRow + (long)g * p.HdV) / 8 + d8] : default;
            }
            Group.Barrier();
            Scale16(ref o0, corr, dc);
            if (dc > 16) Scale16(ref o1, corr, dc - 16);
            int nPos = kHi - c0 + 1 < PrefillBKV ? kHi - c0 + 1 : PrefillBKV;
            for (int pp = 0; pp < nPos; pp++)
            {
                float pv = ps[qi * PrefillBKV + pp];
                int vb = pp * PrefillMaxHd + j;
                Fma16(ref o0, pv, kv, vb, dc);
                if (dc > 16) Fma16(ref o1, pv, kv, vb + 128, dc - 16);
            }
        }

        if (!rowOk) return;
        float inv = l > 0 ? 1f / l : 0f;
        long oo = p.OutOff + (long)(t0 + qi) * p.OutStride + (long)h * p.HdV;
        long go = p.QOff + (long)(t0 + qi) * p.QStride + (long)h * p.QHeadPitch + p.Hd;
        for (int k = 0; k < dc; k++)
        {
            int d = j + 8 * k;
            float v = (k < 16 ? Get16(ref o0, k) : Get16(ref o1, k - 16)) * inv;
            if (p.Gated != 0) v *= Sigmoid(q[go + d]);
            outp[oo + d] = v;
        }
    }

    private const int QsLd = PrefillMaxHd + 1;

    private static float Mask(float s, int pos, int start, int end, float softCap)
    {
        if (pos < start || pos > end) return float.NegativeInfinity;
        return softCap > 0 ? Tanh(s / softCap) * softCap : s;
    }

    private static void Scale16(ref Acc16 a, float s, int n)
    {
        a.C00 *= s; if (n > 1) a.C01 *= s; if (n > 2) a.C02 *= s; if (n > 3) a.C03 *= s;
        if (n > 4) a.C10 *= s; if (n > 5) a.C11 *= s; if (n > 6) a.C12 *= s; if (n > 7) a.C13 *= s;
        if (n > 8) a.C20 *= s; if (n > 9) a.C21 *= s; if (n > 10) a.C22 *= s; if (n > 11) a.C23 *= s;
        if (n > 12) a.C30 *= s; if (n > 13) a.C31 *= s; if (n > 14) a.C32 *= s; if (n > 15) a.C33 *= s;
    }

    /// <summary>a[k] += p · v[b + 8k] for k &lt; n (n ≤ 16, a JIT constant after specialization).</summary>
    private static void Fma16(ref Acc16 a, float p, ArrayView<Half> v, int b, int n)
    {
        a.C00 += p * (float)v[b];
        if (n > 1) a.C01 += p * (float)v[b + 8];
        if (n > 2) a.C02 += p * (float)v[b + 16];
        if (n > 3) a.C03 += p * (float)v[b + 24];
        if (n > 4) a.C10 += p * (float)v[b + 32];
        if (n > 5) a.C11 += p * (float)v[b + 40];
        if (n > 6) a.C12 += p * (float)v[b + 48];
        if (n > 7) a.C13 += p * (float)v[b + 56];
        if (n > 8) a.C20 += p * (float)v[b + 64];
        if (n > 9) a.C21 += p * (float)v[b + 72];
        if (n > 10) a.C22 += p * (float)v[b + 80];
        if (n > 11) a.C23 += p * (float)v[b + 88];
        if (n > 12) a.C30 += p * (float)v[b + 96];
        if (n > 13) a.C31 += p * (float)v[b + 104];
        if (n > 14) a.C32 += p * (float)v[b + 112];
        if (n > 15) a.C33 += p * (float)v[b + 120];
    }

    private static float Get16(ref Acc16 a, int k) => k switch
    {
        0 => a.C00, 1 => a.C01, 2 => a.C02, 3 => a.C03, 4 => a.C10, 5 => a.C11, 6 => a.C12, 7 => a.C13,
        8 => a.C20, 9 => a.C21, 10 => a.C22, 11 => a.C23, 12 => a.C30, 13 => a.C31, 14 => a.C32, _ => a.C33,
    };
}
