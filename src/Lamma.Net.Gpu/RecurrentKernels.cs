using ILGPU;
using ILGPU.Runtime;

namespace LammaNet.Gpu;

// Recurrent-state kernels for hybrid models. Per-sequence data: seqInfo[3s..3s+2] = (first index into tokList,
// token count, state slot), with count 0 for unused entries; tokList holds batch token indices grouped by sequence in
// position order. Launch shapes depend only on the batch size (graph-safe); state lives in per-layer arenas indexed
// by slot, in the host layout.

/// <summary>
/// Causal depthwise conv1d over channels with the rolling (KSize-1)-wide window per channel; input optionally gated
/// (x · inp[MulInOff]), output optionally SiLU and gated (· inp[MulOutOff]) — LFM2 short conv: c · conv(b · x).
/// </summary>
public struct ConvParams
{
    public long InOff, OutOff, WOff, BiasOff, StateStride, MulInOff, MulOutOff;
    public int Channels, KSize, InStride, OutStride, MaxSeqs, Silu;
}

/// <summary>Gated DeltaNet (Qwen3.5) delta rule; one warp per (sequence, value head, state row).</summary>
public struct DeltaParams
{
    public long ConvOff, BaOff, OutOff, DtBiasOff, AOff, StateStride;
    public int MaxSeqs, NV, NK, HK, HV, ConvStride, BaStride, OutStride;
    public float Scale, Eps;
}

/// <summary>Mamba2 selective scan; one warp per (sequence, head, state row).</summary>
public struct ScanParams
{
    public long XbcOff, DtOff, YOff, DtBiasOff, AOff, DOff, StateStride;
    public int MaxSeqs, NHead, HeadDim, DState, NGroup, DInner, XbcStride, DtStride, YStride;
}

/// <summary>
/// Gated RMS norm over groups of D values: NormFirst: y = norm(y)·w · silu(z) (DeltaNet); otherwise
/// y = norm(y · silu(z))·w (Mamba2; no norm when WOff &lt; 0).
/// </summary>
public struct GatedNormParams
{
    public long YOff, ZOff, WOff;
    public int Rows, Groups, D, YStride, ZStride, WStride, NormFirst;
    public float Eps;
}

internal static partial class Kernels
{
    /// <summary>Largest conv kernel / state row length handled on the device.</summary>
    public const int MaxConvKernel = 4, MaxStateRow = 128;

    public static void Conv1dSilu(Index1D i, ArrayView<float> f, ArrayView<float> inp, ArrayView<float> outp, ArrayView<float> state,
        ArrayView<int> seqInfo, ArrayView<int> tokList, ConvParams p)
    {
        int s = i / p.Channels, ch = i % p.Channels;
        if (s >= p.MaxSeqs) return;
        int start = seqInfo[3 * s], cnt = seqInfo[3 * s + 1];
        if (cnt == 0) return;
        int km1 = p.KSize - 1;
        long so = seqInfo[3 * s + 2] * p.StateStride + (long)ch * km1;
        long wo = p.WOff + (long)ch * p.KSize;
        float w0 = f[wo], w1 = km1 > 1 ? f[wo + 1] : 0f, w2 = km1 > 2 ? f[wo + 2] : 0f, wl = f[wo + km1];
        float a0 = km1 > 0 ? state[so] : 0f, a1 = km1 > 1 ? state[so + 1] : 0f, a2 = km1 > 2 ? state[so + 2] : 0f;
        float bias = p.BiasOff >= 0 ? f[p.BiasOff + ch] : 0f;
        for (int j = 0; j < cnt; j++)
        {
            int t = tokList[start + j];
            long ro = (long)t * p.InStride + ch;
            float x = inp[p.InOff + ro];
            if (p.MulInOff >= 0) x *= inp[p.MulInOff + ro];
            float acc = x * wl + bias;
            if (km1 > 0) acc += a0 * w0;
            if (km1 > 1) acc += a1 * w1;
            if (km1 > 2) acc += a2 * w2;
            // shift the window (oldest first) and append x
            if (km1 == 1) a0 = x;
            else if (km1 == 2) { a0 = a1; a1 = x; }
            else { a0 = a1; a1 = a2; a2 = x; }
            float o = p.Silu != 0 ? Silu(acc) : acc;
            if (p.MulOutOff >= 0) o *= inp[p.MulOutOff + ro];
            outp[p.OutOff + (long)t * p.OutStride + ch] = o;
        }
        if (km1 > 0) state[so] = a0;
        if (km1 > 1) state[so + 1] = a1;
        if (km1 > 2) state[so + 2] = a2;
    }

    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));

    /// <summary>
    /// S ← e·S + k δᵀ with δ = (v - e·Sᵀk)·β, out = Sᵀq·scale, per state row j (S stored [head][j][i], i over HK).
    /// q and k are L2-normalized per head here; β = sigmoid, e = exp(softplus(α + dtBias)·A).
    /// </summary>
    public static void DeltaRule(ArrayView<float> f, ArrayView<float> conv, ArrayView<float> ba, ArrayView<float> outp, ArrayView<float> state,
        ArrayView<int> seqInfo, ArrayView<int> tokList, DeltaParams p)
    {
        int w = WarpIndex, lane = Lane;
        int j = w % p.HV, rest = w / p.HV;
        int h = rest % p.NV, s = rest / p.NV;
        if (s >= p.MaxSeqs) return;
        int start = seqInfo[3 * s], cnt = seqInfo[3 * s + 1];
        if (cnt == 0) return;
        int kh = h % p.NK, m = p.HK / Warp; // elements per lane (≤ 4)
        long ro = seqInfo[3 * s + 2] * p.StateStride + ((long)h * p.HV + j) * p.HK + lane;
        float s0 = state[ro], s1 = m > 1 ? state[ro + 32] : 0f, s2 = m > 2 ? state[ro + 64] : 0f, s3 = m > 3 ? state[ro + 96] : 0f;
        float dtb = f[p.DtBiasOff + h], A = f[p.AOff + h];
        int keyDim = p.NK * p.HK;
        for (int jj = 0; jj < cnt; jj++)
        {
            int t = tokList[start + jj];
            long co = p.ConvOff + (long)t * p.ConvStride;
            long qo = co + (long)kh * p.HK + lane, ko = qo + keyDim;
            float q0 = conv[qo], q1 = m > 1 ? conv[qo + 32] : 0f, q2 = m > 2 ? conv[qo + 64] : 0f, q3 = m > 3 ? conv[qo + 96] : 0f;
            float k0 = conv[ko], k1 = m > 1 ? conv[ko + 32] : 0f, k2 = m > 2 ? conv[ko + 64] : 0f, k3 = m > 3 ? conv[ko + 96] : 0f;
            float v = conv[co + 2L * keyDim + (long)h * p.HV + j];
            float iq = 1f / MathF.Sqrt(WarpSum(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) + p.Eps);
            float ik = 1f / MathF.Sqrt(WarpSum(k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3) + p.Eps);
            q0 *= iq; q1 *= iq; q2 *= iq; q3 *= iq;
            k0 *= ik; k1 *= ik; k2 *= ik; k3 *= ik;
            long bo = p.BaOff + (long)t * p.BaStride;
            float beta = Sigmoid(ba[bo + h]);
            float e = MathF.Exp(Softplus(ba[bo + p.NV + h] + dtb) * A);
            float a = WarpSum(s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
            float c = WarpSum(s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3);
            float kq = WarpSum(k0 * q0 + k1 * q1 + k2 * q2 + k3 * q3);
            float delta = (v - e * a) * beta;
            s0 = e * s0 + k0 * delta; s1 = e * s1 + k1 * delta; s2 = e * s2 + k2 * delta; s3 = e * s3 + k3 * delta;
            if (lane == 0) outp[p.OutOff + (long)t * p.OutStride + (long)h * p.HV + j] = (e * c + delta * kq) * p.Scale;
        }
        state[ro] = s0;
        if (m > 1) state[ro + 32] = s1;
        if (m > 2) state[ro + 64] = s2;
        if (m > 3) state[ro + 96] = s3;
    }

    /// <summary>
    /// Mamba2: S[h][i] ← dA·S[h][i] + B·(x_i·dt), y_i = S[h][i]·C + x_i·D[h], with dt = softplus(dt_raw + bias),
    /// dA = exp(dt·A[h]); B and C from the head's group.
    /// </summary>
    public static void SsmScan(ArrayView<float> f, ArrayView<float> xbc, ArrayView<float> dtIn, ArrayView<float> y, ArrayView<float> state,
        ArrayView<int> seqInfo, ArrayView<int> tokList, ScanParams p)
    {
        int w = WarpIndex, lane = Lane;
        int i = w % p.HeadDim, rest = w / p.HeadDim;
        int h = rest % p.NHead, s = rest / p.NHead;
        if (s >= p.MaxSeqs) return;
        int start = seqInfo[3 * s], cnt = seqInfo[3 * s + 1];
        if (cnt == 0) return;
        int m = p.DState / Warp, g = h / (p.NHead / p.NGroup);
        long ro = seqInfo[3 * s + 2] * p.StateStride + ((long)h * p.HeadDim + i) * p.DState + lane;
        float s0 = state[ro], s1 = m > 1 ? state[ro + 32] : 0f, s2 = m > 2 ? state[ro + 64] : 0f, s3 = m > 3 ? state[ro + 96] : 0f;
        float dtb = f[p.DtBiasOff + h], A = f[p.AOff + h], D = p.DOff >= 0 ? f[p.DOff + h] : 0f;
        for (int jj = 0; jj < cnt; jj++)
        {
            int t = tokList[start + jj];
            long xo = p.XbcOff + (long)t * p.XbcStride;
            float x = xbc[xo + (long)h * p.HeadDim + i];
            long bo = xo + p.DInner + (long)g * p.DState + lane, co = bo + (long)p.NGroup * p.DState;
            float b0 = xbc[bo], b1 = m > 1 ? xbc[bo + 32] : 0f, b2 = m > 2 ? xbc[bo + 64] : 0f, b3 = m > 3 ? xbc[bo + 96] : 0f;
            float c0 = xbc[co], c1 = m > 1 ? xbc[co + 32] : 0f, c2 = m > 2 ? xbc[co + 64] : 0f, c3 = m > 3 ? xbc[co + 96] : 0f;
            float dt = Softplus(dtIn[p.DtOff + (long)t * p.DtStride + h] + dtb);
            float dA = MathF.Exp(dt * A), xdt = x * dt;
            s0 = dA * s0 + b0 * xdt; s1 = dA * s1 + b1 * xdt; s2 = dA * s2 + b2 * xdt; s3 = dA * s3 + b3 * xdt;
            float yv = WarpSum(s0 * c0 + s1 * c1 + s2 * c2 + s3 * c3) + x * D;
            if (lane == 0) y[p.YOff + (long)t * p.YStride + (long)h * p.HeadDim + i] = yv;
        }
        state[ro] = s0;
        if (m > 1) state[ro + 32] = s1;
        if (m > 2) state[ro + 64] = s2;
        if (m > 3) state[ro + 96] = s3;
    }

    /// <summary>Gated RMS norm (see <see cref="GatedNormParams"/>); one warp per (token, group).</summary>
    public static void GatedNorm(ArrayView<float> f, ArrayView<float> y, ArrayView<float> z, GatedNormParams p)
    {
        int r = WarpIndex, lane = Lane;
        if (r >= p.Rows) return;
        int t = r / p.Groups, gi = r % p.Groups;
        long yo = p.YOff + (long)t * p.YStride + (long)gi * p.D, zo = p.ZOff + (long)t * p.ZStride + (long)gi * p.D;
        long wo = p.WOff + (long)gi * p.WStride;
        float ss = 0;
        for (int k = lane; k < p.D; k += Warp)
        {
            float v = y[yo + k];
            if (p.NormFirst == 0) v *= Silu(z[zo + k]);
            ss += v * v;
        }
        float scale = p.WOff >= 0 || p.NormFirst != 0 ? 1f / MathF.Sqrt(WarpSum(ss) / p.D + p.Eps) : 1f;
        for (int k = lane; k < p.D; k += Warp)
        {
            float v = y[yo + k];
            float gz = Silu(z[zo + k]);
            float wt = p.WOff >= 0 ? f[wo + k] : 1f;
            y[yo + k] = p.NormFirst != 0 ? v * scale * wt * gz : v * gz * scale * wt;
        }
    }
}
