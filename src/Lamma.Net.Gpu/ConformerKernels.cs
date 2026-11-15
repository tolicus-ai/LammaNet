using ILGPU;
using ILGPU.Runtime;

namespace LammaNet.Gpu;

/// <summary>
/// Chunked local self-attention with Transformer-XL relative positions (Gemma 4 audio): queries in chunks of C frames
/// see the S = C + P keys starting P frames before their chunk, masked to causal keys at most P - 1 frames back.
/// Relative keys (P + 1 rows of D) live in the parameter buffer at RelOff.
/// </summary>
public struct ChunkAttnParams
{
    public long RelOff;
    public int N, Heads, Hd, D, C, P, R;
    public float SoftCap;
}

internal static partial class Kernels
{
    /// <summary>q[t, h, i] *= qScale · perDim[i]; k *= kScale.</summary>
    public static void ScaleQK(Index1D i, ArrayView<float> q, ArrayView<float> k, ArrayView<float> f, long perDimOff, int hd, float qScale, float kScale)
    {
        q[i] = q[i] * qScale * f[perDimOff + i % hd];
        k[i] = k[i] * kScale;
    }

    /// <summary>
    /// One warp per (query, head): lane s scores key s of the query's window (S ≤ 32), softmax across the warp, then the
    /// lanes accumulate the head dimensions of the weighted values.
    /// </summary>
    public static void ChunkRelAttn(ArrayView<float> q, ArrayView<float> k, ArrayView<float> v, ArrayView<float> f, ArrayView<float> o, ChunkAttnParams p)
    {
        int w = WarpIndex, lane = Lane;
        if (w >= p.N * p.Heads) return;   // whole warps exit together
        int h = w % p.Heads, gq = w / p.Heads;
        int b = gq / p.C, c = gq - b * p.C, s = lane, S = p.C + p.P;
        long qo = (long)gq * p.D + (long)h * p.Hd;
        int gk = b * p.C - p.P + s;
        bool inWindow = s < S, inRange = inWindow && gk >= 0 && gk < p.N;
        float score = 0;
        if (inRange)
        {
            long ko = (long)gk * p.D + (long)h * p.Hd;
            for (int i = 0; i < p.Hd; i++) score += q[qo + i] * k[ko + i];
        }
        int r = s - c;   // relative-position term (Transformer-XL shift)
        if (inWindow && r >= 0 && r < p.R)
        {
            long ro = p.RelOff + (long)r * p.D + (long)h * p.Hd;
            for (int i = 0; i < p.Hd; i++) score += q[qo + i] * f[ro + i];
        }
        score = Tanh(score / p.SoftCap) * p.SoftCap;
        bool valid = inRange && gk <= gq && gq - gk < p.P;
        if (!valid) score += -1e9f;
        if (!inWindow) score = -3e38f;
        float mx = WarpMax(score);
        float e = inWindow ? MathF.Exp(score - mx) : 0f;
        float prob = e / WarpSum(e);
        for (int i0 = 0; i0 < p.Hd; i0 += Warp)
        {
            int i = i0 + lane;
            float acc = 0;
            for (int s2 = 0; s2 < S; s2++)
            {
                float ps = ILGPU.Warp.Shuffle(prob, s2);
                int gk2 = b * p.C - p.P + s2;
                if (i < p.Hd && ps != 0 && gk2 >= 0 && gk2 < p.N) acc += ps * v[(long)gk2 * p.D + (long)h * p.Hd + i];
            }
            if (i < p.Hd) o[qo + i] = acc;
        }
    }

    /// <summary>GLU: dst[t, c] = src[t, c] · sigmoid(src[t, d + c]) for src [n, 2d].</summary>
    public static void Glu(Index1D i, ArrayView<float> src, ArrayView<float> dst, int d)
    {
        int t = i / d, c = i - t * d;
        long a = (long)t * 2 * d + c;
        dst[i] = src[a] * Sigmoid(src[a + d]);
    }

    /// <summary>Causal depthwise conv over time: dst[t, ch] = Σ_j src[t - (kz - 1) + j, ch] · w[ch · kz + j].</summary>
    public static void DwConvCausal(Index1D i, ArrayView<float> src, ArrayView<float> dst, ArrayView<float> f, long wOff, int d, int kz)
    {
        int t = i / d, ch = i - t * d;
        float acc = 0;
        for (int j = 0; j < kz; j++)
        {
            int ts = t - (kz - 1) + j;
            if (ts >= 0) acc += src[(long)ts * d + ch] * f[wOff + (long)ch * kz + j];
        }
        dst[i] = acc;
    }

    /// <summary>x = silu(x).</summary>
    public static void SiluInPlace(Index1D i, ArrayView<float> x) => x[i] = Silu(x[i]);
}
