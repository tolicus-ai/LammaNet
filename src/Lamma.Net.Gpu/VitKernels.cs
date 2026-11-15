using ILGPU;
using ILGPU.Runtime;

namespace LammaNet.Gpu;

/// <summary>LayerNorm over rows of D (multiple of 4): y = (x - mean) / sqrt(var + Eps) · w + b; w/b offsets into the parameter buffer (-1 = none).</summary>
public struct LnParams
{
    public long XOff, YOff, WOff, BOff;
    public int Rows, D, XStride, YStride;
    public float Eps;
}

internal static partial class Kernels
{
    /// <summary>One 128-thread group per row; two-pass (mean, then centered variance) for accuracy.</summary>
    public static void LayerNorm(ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, LnParams p)
    {
        var red = SharedMemory.Allocate<float>(2 * WarpsPerGroup);
        int r = Grid.IdxX, tid = Group.IdxX;
        if (r >= p.Rows) return;
        var x4 = x.Cast<Float4>();
        var y4 = y.Cast<Float4>();
        var f4 = f.Cast<Float4>();
        long xo4 = (p.XOff + (long)r * p.XStride) >> 2, yo4 = (p.YOff + (long)r * p.YStride) >> 2;
        int d4 = p.D >> 2;
        float s = 0;
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = x4[xo4 + k];
            s += v.X + v.Y + v.Z + v.W;
        }
        s = WarpSum(s);
        if (Lane == 0) red[tid / Warp] = s;
        Group.Barrier();
        float mean = (red[0] + red[1] + red[2] + red[3]) / p.D;
        float ss = 0;
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = x4[xo4 + k];
            float a = v.X - mean, b = v.Y - mean, c = v.Z - mean, d = v.W - mean;
            ss += a * a + b * b + c * c + d * d;
        }
        ss = WarpSum(ss);
        if (Lane == 0) red[WarpsPerGroup + tid / Warp] = ss;
        Group.Barrier();
        float inv = 1f / MathF.Sqrt((red[4] + red[5] + red[6] + red[7]) / p.D + p.Eps);
        for (int k = tid; k < d4; k += GroupSize)
        {
            Float4 v = x4[xo4 + k];
            Float4 o;
            o.X = (v.X - mean) * inv; o.Y = (v.Y - mean) * inv; o.Z = (v.Z - mean) * inv; o.W = (v.W - mean) * inv;
            if (p.WOff >= 0)
            {
                Float4 w = f4[(p.WOff >> 2) + k];
                o.X *= w.X; o.Y *= w.Y; o.Z *= w.Z; o.W *= w.W;
            }
            if (p.BOff >= 0)
            {
                Float4 b = f4[(p.BOff >> 2) + k];
                o.X += b.X; o.Y += b.Y; o.Z += b.Z; o.W += b.W;
            }
            y4[yo4 + k] = o;
        }
    }

    /// <summary>x = gelu_tanh(x).</summary>
    public static void GeluInPlace(Index1D i, ArrayView<float> x) => x[i] = GeluTanh(x[i]);

    /// <summary>u = gelu_tanh(g) · u (gated FFN).</summary>
    public static void GeGluInPlace(Index1D i, ArrayView<float> g, ArrayView<float> u) => u[i] = GeluTanh(g[i]) * u[i];

    /// <summary>dst = clamp(src, lo, hi) (src may equal dst).</summary>
    public static void Clamp(Index1D i, ArrayView<float> src, ArrayView<float> dst, float lo, float hi) =>
        dst[i] = MathF.Min(MathF.Max(src[i], lo), hi);

    /// <summary>y += x.</summary>
    public static void AddInPlace(Index1D i, ArrayView<float> x, ArrayView<float> y) => y[i] += x[i];
}
