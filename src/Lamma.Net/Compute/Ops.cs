using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace LammaNet.Compute;

/// <summary>Vectorized element-wise and normalization kernels operating on float rows.</summary>
public static unsafe class Ops
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Span<float> S(float* p, int n) => new(p, n);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ReadOnlySpan<float> R(float* p, int n) => new(p, n);

    private const MethodImplOptions Hot = MethodImplOptions.AggressiveOptimization;

    /// <summary>y = x / rms(x) * w (w optional). x and y may alias.</summary>
    [MethodImpl(Hot)]
    public static void RmsNorm(float* x, float* w, float* y, int n, float eps)
    {
        float ss = Simd.SumSquares(x, n);
        float scale = 1f / MathF.Sqrt(ss / n + eps);
        if (w != null) Simd.ScaleMul(x, scale, w, y, n);
        else Simd.Scale(x, scale, y, n);
    }

    /// <summary>Standard LayerNorm with optional weight and bias.</summary>
    [MethodImpl(Hot)]
    public static void LayerNorm(float* x, float* w, float* b, float* y, int n, float eps)
    {
        float mean = Simd.Sum(x, n) / n;
        Simd.AddScalar(x, -mean, y, n);
        float var = Simd.SumSquares(y, n) / n;
        float scale = 1f / MathF.Sqrt(var + eps);
        if (w != null) Simd.ScaleMul(y, scale, w, y, n);
        else Simd.Scale(y, scale, y, n);
        if (b != null) Simd.Add(y, b, y, n);
    }

    /// <summary>L2 normalization: y = x / sqrt(sum(x^2) + eps) (matches ggml's gated-delta-net l2 norm).</summary>
    [MethodImpl(Hot)]
    public static void L2Norm(float* x, float* y, int n, float eps)
    {
        float ss = Simd.SumSquares(x, n);
        Simd.Scale(x, 1f / MathF.Sqrt(ss + eps), y, n);
    }

    public static void Add(float* a, float* b, float* y, int n) => Simd.Add(a, b, y, n);
    public static void Mul(float* a, float* b, float* y, int n) => Simd.Mul(a, b, y, n);
    public static void Scale(float* a, float s, float* y, int n) => Simd.Scale(a, s, y, n);
    /// <summary>y += a * s</summary>
    public static void AddScaled(float* a, float s, float* y, int n) => Simd.AddScaled(a, s, y, n);
    public static void Copy(float* a, float* y, int n) => Buffer.MemoryCopy(a, y, n * 4L, n * 4L);
    public static void Fill(float* y, float v, int n) => S(y, n).Fill(v);
    public static void AddScalar(float* a, float s, float* y, int n) => Simd.AddScalar(a, s, y, n);

    public static float Dot(float* a, float* b, int n) => Simd.Dot(a, b, n);
    public static float Sum(float* a, int n) => Simd.Sum(a, n);
    public static float Max(float* a, int n) => Simd.Max(a, n);

    public static void Sigmoid(float* x, float* y, int n) => Simd.Sigmoid(x, y, n);
    public static void Tanh(float* x, float* y, int n) => Simd.Tanh(x, y, n);
    public static void Exp(float* x, float* y, int n) => Simd.Exp(x, y, n);
    public static void Relu(float* x, float* y, int n) => TensorPrimitives.Max(R(x, n), 0f, S(y, n));

    /// <summary>y = x * sigmoid(x).</summary>
    public static void Silu(float* x, float* y, int n) => Simd.Silu(x, y, n);

    /// <summary>out = silu(gate) * up.</summary>
    public static void SwiGlu(float* gate, float* up, float* y, int n) => Simd.SwiGlu(gate, up, y, n);

    /// <summary>GELU (tanh approximation, as ggml_gelu).</summary>
    public static void Gelu(float* x, float* y, int n) => Simd.Gelu(x, y, n);

    /// <summary>out = gelu(gate) * up.</summary>
    public static void GeGlu(float* gate, float* up, float* y, int n) => Simd.GeGlu(gate, up, y, n);

    /// <summary>Exact GELU using erf (used by some vision towers).</summary>
    public static void GeluErf(float* x, float* y, int n)
    {
        for (int i = 0; i < n; i++)
        {
            float v = x[i];
            y[i] = 0.5f * v * (1f + Erf(v * 0.70710678118654752f));
        }
    }

    public static float Erf(float x)
    {
        // Abramowitz-Stegun 7.1.26 refined (max err ~1.2e-7 using double)
        double t = 1.0 / (1.0 + 0.5 * Math.Abs(x));
        double y = 1 - t * Math.Exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return (float)(x >= 0 ? y : -y);
    }

    /// <summary>y = cap * tanh(x / cap).</summary>
    public static void SoftCap(float* x, float* y, int n, float cap)
    {
        Simd.Scale(x, 1f / cap, y, n);
        Simd.Tanh(y, y, n);
        Simd.Scale(y, cap, y, n);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));

    /// <summary>In-place numerically stable softmax.</summary>
    public static void Softmax(float* x, int n)
    {
        float max = Simd.Max(x, n);
        float sum = Simd.ExpSubSum(x, max, x, n);
        Simd.Scale(x, 1f / sum, x, n);
    }
}
