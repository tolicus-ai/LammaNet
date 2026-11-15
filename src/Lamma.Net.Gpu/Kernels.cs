using ILGPU;
using ILGPU.Runtime;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

/// <summary>Four floats loaded/stored as one 16-byte vector.</summary>
public struct Float4 { public float X, Y, Z, W; }

/// <summary>
/// Row RMS norm of v = norm(x · InScale) * w: y = v (Mode 0), y += Scale·v (Mode 1) or y = (y + v)·Scale (Mode 2).
/// WOff &lt; 0 = no weight.
/// </summary>
public struct NormParams
{
    public long XOff, YOff, WOff;
    public int Rows, D, XStride, YStride, Mode;
    public float Eps, Scale, InScale;
}

/// <summary>Parameters of <see cref="Kernels.RmsNormResidualNorm"/>.</summary>
public struct ResidualNormParams
{
    /// <summary>x: sublayer output; y: residual stream (updated); z: pre-norm of the updated residual; w: post-norm weights, w2: pre-norm weights.</summary>
    public long XOff, YOff, ZOff, WOff, W2Off;
    public int Rows, D, Mode;
    public float Eps, Scale;
}

/// <summary>
/// Post-projection processing of Q/K/V for one layer: per-head RMS norm, rotary embedding, and the K/V write into
/// the fp16 paged cache. One warp per (token, head) unit: [q heads | k heads | v heads].
/// </summary>
public struct QkvParams
{
    public long QOff, KOff, VOff, QNormOff, KNormOff, RopeOff;
    public int N, NHead, NKv, Hd, HdV, QStride, KStride, VStride, QHeadPitch;
    public int KRow, VRow, OwnKv, VNorm, RopeDims, RopeNeox, RopeStride;
    /// <summary>
    /// Neox rotation in segments of this many pairs (0 = one segment of RopeDims/2 pairs): segment s rotates
    /// (s·2P + j, s·2P + j + P). Gemma 4 vision rotates each half of the head separately (P = hd/4).
    /// </summary>
    public int RopeSegPairs;
    public float Eps;
}

/// <summary>
/// ILGPU kernels for transformer layers, written in C# and compiled by ILGPU to CUDA PTX at runtime.
/// Kernels work on groups of 32 lanes ("warps") and reduce with lane shuffles (offsets &lt; 32), which is valid on
/// any device whose hardware warp size is a multiple of 32.
/// </summary>
internal static partial class Kernels
{
    public const int Warp = 32;
    /// <summary>Threads per group for warp-per-item kernels.</summary>
    public const int GroupSize = 128;
    public const int WarpsPerGroup = GroupSize / Warp;

    // ------------------------------------------------------------------ helpers

    internal static float WarpSum(float v)
    {
        v += ILGPU.Warp.ShuffleXor(v, 16);
        v += ILGPU.Warp.ShuffleXor(v, 8);
        v += ILGPU.Warp.ShuffleXor(v, 4);
        v += ILGPU.Warp.ShuffleXor(v, 2);
        v += ILGPU.Warp.ShuffleXor(v, 1);
        return v;
    }

    internal static float WarpMax(float v)
    {
        v = MathF.Max(v, ILGPU.Warp.ShuffleXor(v, 16));
        v = MathF.Max(v, ILGPU.Warp.ShuffleXor(v, 8));
        v = MathF.Max(v, ILGPU.Warp.ShuffleXor(v, 4));
        v = MathF.Max(v, ILGPU.Warp.ShuffleXor(v, 2));
        v = MathF.Max(v, ILGPU.Warp.ShuffleXor(v, 1));
        return v;
    }

    /// <summary>Global warp index and lane for warp-per-item kernels launched with <see cref="GroupSize"/> threads.</summary>
    internal static int WarpIndex => Grid.IdxX * WarpsPerGroup + Group.IdxX / Warp;
    internal static int Lane => Group.IdxX & (Warp - 1);

    /// <summary>
    /// fp16 bits → float without denormal intermediates (GPU fast math flushes denormals, and K-quant super-block
    /// scales are frequently subnormal halves).
    /// </summary>
    internal static float HalfBits(int h)
    {
        int exp = (h >> 10) & 0x1f, man = h & 0x3ff;
        float sub = man * 5.9604644775390625e-8f; // 2^-24
        float norm = Interop.IntAsFloat((uint)(((exp == 31 ? 255 : exp + 112) << 23) | (man << 13)));
        float f = exp == 0 ? sub : norm;
        return (h & 0x8000) != 0 ? -f : f;
    }

    /// <summary>
    /// tanh via exp: ILGPU's fast-math lowering of MathF.Tanh emits <c>tanh.approx.ftz.f32</c>, which ptxas rejects.
    /// </summary>
    internal static float Tanh(float v)
    {
        float a = MathF.Min(MathF.Abs(v), 15f);
        float e = MathF.Exp(-2f * a);
        float t = (1f - e) / (1f + e);
        return v < 0 ? -t : t;
    }

    internal static float GeluTanh(float v) => 0.5f * v * (1f + Tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    internal static float Silu(float v) => v / (1f + MathF.Exp(-v));
    internal static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    // ------------------------------------------------------------------ element-wise

    public static void Scale(Index1D i, ArrayView<float> a, float s) => a[i] = a[i] * s;

    /// <summary>Copies rows: dst[r * dstStride + c] = src[srcRows[r] * srcStride + c] (gathers output rows for the head).</summary>
    public static void GatherRows(Index1D i, ArrayView<float> src, ArrayView<float> dst, ArrayView<int> srcRows, int cols)
    {
        int r = i / cols, c = i % cols;
        dst[(long)r * cols + c] = src[(long)srcRows[r] * cols + c];
    }

    /// <summary>Gemma per-layer inputs: g[t, j] = gelu(g[t, j]) * ple[t * pleStride + pleOff + j].</summary>
    public static void GeluMul(Index1D i, ArrayView<float> g, ArrayView<float> ple, int width, int pleStride, int pleOff)
    {
        int t = i / width, j = i % width;
        g[i] = GeluTanh(g[i]) * ple[(long)t * pleStride + pleOff + j];
    }

    /// <summary>
    /// Context shift: rotates K rows (fp16 cache, slots given per row) by a precomputed rotation table
    /// (interleaved cos/sin per pair). One thread per (row, KV head, pair).
    /// </summary>
    public static void RotateK(Index1D i, ArrayView<Half> kc, ArrayView<int> slots, ArrayView<float> rot, int rows, int nKv, int hd, int dims, int neox)
    {
        int half = dims / 2, pi = i % half, rest = i / half, h = rest % nKv, r = rest / nKv;
        if (r >= rows) return;
        long b = (long)slots[r] * nKv * hd + (long)h * hd;
        int ia = neox != 0 ? pi : 2 * pi, ib = neox != 0 ? pi + half : 2 * pi + 1;
        float x = kc[b + ia], y = kc[b + ib], c = rot[2 * pi], s = rot[2 * pi + 1];
        kc[b + ia] = (Half)(x * c - y * s);
        kc[b + ib] = (Half)(x * s + y * c);
    }

    /// <summary>Copies the first rows of one KV page into another.</summary>
    public static void PageCopy(Index1D i, ArrayView<Half> kv, long srcOffset, long dstOffset) => kv[dstOffset + i] = kv[srcOffset + i];

    // ------------------------------------------------------------------ norms

    /// <summary>
    /// RMS norm, one 128-thread group per row with 16-byte loads (D, offsets and strides multiples of 4).
    /// </summary>
    /// <summary>
    /// A residual post-norm and the next pre-norm in one launch (sandwich-norm models): y = y + scale·norm(x)·w (Mode 1)
    /// or y = (y + norm(x)·w)·scale (Mode 2), then z = norm(y)·w2. One group per row; rows of D (multiple of 4) with
    /// stride D. The updated y is re-read by the thread that wrote it.
    /// </summary>
    public static void RmsNormResidualNorm(ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, ArrayView<float> z, ResidualNormParams p)
    {
        var red = SharedMemory.Allocate<float>(WarpsPerGroup);
        var red2 = SharedMemory.Allocate<float>(WarpsPerGroup);
        int r = Grid.IdxX, tid = Group.IdxX;
        if (r >= p.Rows) return;
        var x4 = x.Cast<Float4>();
        var y4 = y.Cast<Float4>();
        var z4 = z.Cast<Float4>();
        var f4 = f.Cast<Float4>();
        int d4 = p.D >> 2;
        long xo4 = (p.XOff >> 2) + (long)r * d4, yo4 = (p.YOff >> 2) + (long)r * d4, zo4 = (p.ZOff >> 2) + (long)r * d4;
        long wo4 = p.WOff >> 2, w2o4 = p.W2Off >> 2;
        float ss = 0;
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = x4[xo4 + k];
            ss += v.X * v.X + v.Y * v.Y + v.Z * v.Z + v.W * v.W;
        }
        ss = WarpSum(ss);
        if (Lane == 0) red[tid / Warp] = ss;
        Group.Barrier();
        float s1 = 1f / MathF.Sqrt((red[0] + red[1] + red[2] + red[3]) / p.D + p.Eps);
        float ss2 = 0;
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = x4[xo4 + k], w = f4[wo4 + k], old = y4[yo4 + k];
            Float4 o;
            if (p.Mode == 1)
            {
                o.X = old.X + p.Scale * v.X * s1 * w.X; o.Y = old.Y + p.Scale * v.Y * s1 * w.Y;
                o.Z = old.Z + p.Scale * v.Z * s1 * w.Z; o.W = old.W + p.Scale * v.W * s1 * w.W;
            }
            else
            {
                o.X = (old.X + v.X * s1 * w.X) * p.Scale; o.Y = (old.Y + v.Y * s1 * w.Y) * p.Scale;
                o.Z = (old.Z + v.Z * s1 * w.Z) * p.Scale; o.W = (old.W + v.W * s1 * w.W) * p.Scale;
            }
            y4[yo4 + k] = o;
            ss2 += o.X * o.X + o.Y * o.Y + o.Z * o.Z + o.W * o.W;
        }
        ss2 = WarpSum(ss2);
        if (Lane == 0) red2[tid / Warp] = ss2;
        Group.Barrier();
        float s2 = 1f / MathF.Sqrt((red2[0] + red2[1] + red2[2] + red2[3]) / p.D + p.Eps);
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = y4[yo4 + k], w = f4[w2o4 + k];
            z4[zo4 + k] = new Float4 { X = v.X * s2 * w.X, Y = v.Y * s2 * w.Y, Z = v.Z * s2 * w.Z, W = v.W * s2 * w.W };
        }
    }

    public static void RmsNorm(ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, NormParams p)
    {
        var red = SharedMemory.Allocate<float>(WarpsPerGroup);
        int r = Grid.IdxX, tid = Group.IdxX;
        if (r >= p.Rows) return;
        var x4 = x.Cast<Float4>();
        var y4 = y.Cast<Float4>();
        var f4 = f.Cast<Float4>();
        long xo4 = (p.XOff + (long)r * p.XStride) >> 2, yo4 = (p.YOff + (long)r * p.YStride) >> 2, wo4 = p.WOff >> 2;
        int d4 = p.D >> 2;
        float ss = 0;
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = x4[xo4 + k];
            ss += v.X * v.X + v.Y * v.Y + v.Z * v.Z + v.W * v.W;
        }
        ss = WarpSum(ss);
        if (Lane == 0) red[tid / Warp] = ss;
        Group.Barrier();
        float tot = (red[0] + red[1] + red[2] + red[3]) * p.InScale * p.InScale;
        float scale = p.InScale / MathF.Sqrt(tot / p.D + p.Eps);
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = x4[xo4 + k];
            float s0 = scale, s1 = scale, s2 = scale, s3 = scale;
            if (p.WOff >= 0)
            {
                Float4 w = f4[wo4 + k];
                s0 *= w.X; s1 *= w.Y; s2 *= w.Z; s3 *= w.W;
            }
            Float4 o;
            if (p.Mode == 1)
            {
                Float4 old = y4[yo4 + k];
                o.X = old.X + p.Scale * v.X * s0; o.Y = old.Y + p.Scale * v.Y * s1; o.Z = old.Z + p.Scale * v.Z * s2; o.W = old.W + p.Scale * v.W * s3;
            }
            else if (p.Mode == 2)
            {
                Float4 old = y4[yo4 + k];
                o.X = (old.X + v.X * s0) * p.Scale; o.Y = (old.Y + v.Y * s1) * p.Scale; o.Z = (old.Z + v.Z * s2) * p.Scale; o.W = (old.W + v.W * s3) * p.Scale;
            }
            else { o.X = v.X * s0; o.Y = v.Y * s1; o.Z = v.Z * s2; o.W = v.W * s3; }
            y4[yo4 + k] = o;
        }
    }

    // ------------------------------------------------------------------ q/k/v post-processing

    public static void QkvPost(ArrayView<float> f, ArrayView<float> q, ArrayView<float> k, ArrayView<float> v, ArrayView<float> rope,
        ArrayView<Half> kc, ArrayView<Half> vc, ArrayView<int> slots, QkvParams p)
    {
        int unitsPerTok = p.NHead + (p.OwnKv != 0 ? 2 * p.NKv : 0);
        int w = WarpIndex, lane = Lane;
        if (w >= p.N * unitsPerTok) return;
        int t = w / unitsPerTok, u = w % unitsPerTok;
        int kind, h, hd;
        long src, normOff;
        ArrayView<float> buf;
        if (u < p.NHead) { kind = 0; h = u; hd = p.Hd; buf = q; src = p.QOff + (long)t * p.QStride + (long)h * p.QHeadPitch; normOff = p.QNormOff; }
        else if (u < p.NHead + p.NKv) { kind = 1; h = u - p.NHead; hd = p.Hd; buf = k; src = p.KOff + (long)t * p.KStride + (long)h * p.Hd; normOff = p.KNormOff; }
        else { kind = 2; h = u - p.NHead - p.NKv; hd = p.HdV; buf = v; src = p.VOff + (long)t * p.VStride + (long)h * p.HdV; normOff = -1; }
        bool norm = normOff >= 0 || (kind == 2 && p.VNorm != 0);
        float scale = 1f;
        if (norm)
        {
            float ss = 0;
            for (int i = lane; i < hd; i += Warp) { float a = buf[src + i]; ss += a * a; }
            ss = WarpSum(ss);
            scale = 1f / MathF.Sqrt(ss / hd + p.Eps);
        }
        long slot = kind == 0 ? 0 : slots[t];
        long dstK = slot * p.KRow + (long)h * p.Hd, dstV = slot * p.VRow + (long)h * p.HdV;
        int rotDims = kind == 2 ? 0 : p.RopeDims, half = rotDims / 2;
        int seg = p.RopeSegPairs > 0 ? p.RopeSegPairs : half;
        for (int pi = lane; pi < half; pi += Warp)
        {
            int nx = pi + pi / seg * seg;   // pair j of segment s starts at s·2P + j
            int ia = p.RopeNeox != 0 ? nx : 2 * pi, ib = p.RopeNeox != 0 ? nx + seg : 2 * pi + 1;
            float a = buf[src + ia] * scale, b = buf[src + ib] * scale;
            if (normOff >= 0) { a *= f[normOff + ia]; b *= f[normOff + ib]; }
            long ro = p.RopeOff + (long)t * p.RopeStride + 2 * pi;
            float c = rope[ro], s = rope[ro + 1];
            float ra = a * c - b * s, rb = a * s + b * c;
            if (kind == 0) { buf[src + ia] = ra; buf[src + ib] = rb; }
            else { kc[dstK + ia] = (Half)ra; kc[dstK + ib] = (Half)rb; }
        }
        for (int i = rotDims + lane; i < hd; i += Warp)
        {
            float a = buf[src + i] * scale;
            if (normOff >= 0) a *= f[normOff + i];
            if (kind == 0) buf[src + i] = a;
            else if (kind == 1) kc[dstK + i] = (Half)a;
            else vc[dstV + i] = (Half)a;
        }
    }
}
