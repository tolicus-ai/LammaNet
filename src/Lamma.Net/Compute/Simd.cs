using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace LammaNet.Compute;

/// <summary>
/// Hand-vectorized float kernels (Vector256 on AVX2 hosts, Vector128 on NEON/SSE hosts, scalar otherwise).
/// Every entry point is compiled fully optimized up-front (<see cref="MethodImplOptions.AggressiveOptimization"/>),
/// so performance does not depend on the host's tiered-JIT settings.
/// </summary>
public static unsafe class Simd
{
    private const MethodImplOptions Hot = MethodImplOptions.AggressiveOptimization;
    private static readonly bool V256 = Vector256.IsHardwareAccelerated;
    private static readonly bool V128 = Vector128.IsHardwareAccelerated;

    [MethodImpl(Hot)]
    public static float Dot(float* a, float* b, int n)
    {
        int i = 0;
        float s = 0;
        if (V256)
        {
            var a0 = Vector256<float>.Zero; var a1 = a0; var a2 = a0; var a3 = a0;
            for (; i + 32 <= n; i += 32)
            {
                a0 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i), Vector256.Load(b + i), a0);
                a1 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i + 8), Vector256.Load(b + i + 8), a1);
                a2 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i + 16), Vector256.Load(b + i + 16), a2);
                a3 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i + 24), Vector256.Load(b + i + 24), a3);
            }
            for (; i + 8 <= n; i += 8) a0 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i), Vector256.Load(b + i), a0);
            s = Vector256.Sum((a0 + a1) + (a2 + a3));
        }
        else if (V128)
        {
            var a0 = Vector128<float>.Zero; var a1 = a0;
            for (; i + 8 <= n; i += 8)
            {
                a0 = Vector128.FusedMultiplyAdd(Vector128.Load(a + i), Vector128.Load(b + i), a0);
                a1 = Vector128.FusedMultiplyAdd(Vector128.Load(a + i + 4), Vector128.Load(b + i + 4), a1);
            }
            for (; i + 4 <= n; i += 4) a0 = Vector128.FusedMultiplyAdd(Vector128.Load(a + i), Vector128.Load(b + i), a0);
            s = Vector128.Sum(a0 + a1);
        }
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }

    [MethodImpl(Hot)]
    public static float Sum(float* a, int n)
    {
        int i = 0; float s = 0;
        if (V256)
        {
            var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8) acc += Vector256.Load(a + i);
            s = Vector256.Sum(acc);
        }
        else if (V128)
        {
            var acc = Vector128<float>.Zero;
            for (; i + 4 <= n; i += 4) acc += Vector128.Load(a + i);
            s = Vector128.Sum(acc);
        }
        for (; i < n; i++) s += a[i];
        return s;
    }

    [MethodImpl(Hot)]
    public static float SumSquares(float* a, int n)
    {
        int i = 0; float s = 0;
        if (V256)
        {
            var acc = Vector256<float>.Zero; var acc2 = acc;
            for (; i + 16 <= n; i += 16)
            {
                var v = Vector256.Load(a + i); var w = Vector256.Load(a + i + 8);
                acc = Vector256.FusedMultiplyAdd(v, v, acc);
                acc2 = Vector256.FusedMultiplyAdd(w, w, acc2);
            }
            for (; i + 8 <= n; i += 8) { var v = Vector256.Load(a + i); acc = Vector256.FusedMultiplyAdd(v, v, acc); }
            s = Vector256.Sum(acc + acc2);
        }
        else if (V128)
        {
            var acc = Vector128<float>.Zero;
            for (; i + 4 <= n; i += 4) { var v = Vector128.Load(a + i); acc = Vector128.FusedMultiplyAdd(v, v, acc); }
            s = Vector128.Sum(acc);
        }
        for (; i < n; i++) s += a[i] * a[i];
        return s;
    }

    [MethodImpl(Hot)]
    public static float Max(float* a, int n)
    {
        int i = 0; float m = float.NegativeInfinity;
        if (V256 && n >= 8)
        {
            var acc = Vector256.Load(a); i = 8;
            for (; i + 8 <= n; i += 8) acc = Vector256.Max(acc, Vector256.Load(a + i));
            for (int k = 0; k < 8; k++) m = MathF.Max(m, acc.GetElement(k));
        }
        else if (V128 && n >= 4)
        {
            var acc = Vector128.Load(a); i = 4;
            for (; i + 4 <= n; i += 4) acc = Vector128.Max(acc, Vector128.Load(a + i));
            for (int k = 0; k < 4; k++) m = MathF.Max(m, acc.GetElement(k));
        }
        for (; i < n; i++) m = MathF.Max(m, a[i]);
        return m;
    }

    /// <summary>y = a * s</summary>
    [MethodImpl(Hot)]
    public static void Scale(float* a, float s, float* y, int n)
    {
        int i = 0;
        if (V256) { var vs = Vector256.Create(s); for (; i + 8 <= n; i += 8) (Vector256.Load(a + i) * vs).Store(y + i); }
        else if (V128) { var vs = Vector128.Create(s); for (; i + 4 <= n; i += 4) (Vector128.Load(a + i) * vs).Store(y + i); }
        for (; i < n; i++) y[i] = a[i] * s;
    }

    /// <summary>y = a + s</summary>
    [MethodImpl(Hot)]
    public static void AddScalar(float* a, float s, float* y, int n)
    {
        int i = 0;
        if (V256) { var vs = Vector256.Create(s); for (; i + 8 <= n; i += 8) (Vector256.Load(a + i) + vs).Store(y + i); }
        else if (V128) { var vs = Vector128.Create(s); for (; i + 4 <= n; i += 4) (Vector128.Load(a + i) + vs).Store(y + i); }
        for (; i < n; i++) y[i] = a[i] + s;
    }

    [MethodImpl(Hot)]
    public static void Add(float* a, float* b, float* y, int n)
    {
        int i = 0;
        if (V256) for (; i + 8 <= n; i += 8) (Vector256.Load(a + i) + Vector256.Load(b + i)).Store(y + i);
        else if (V128) for (; i + 4 <= n; i += 4) (Vector128.Load(a + i) + Vector128.Load(b + i)).Store(y + i);
        for (; i < n; i++) y[i] = a[i] + b[i];
    }

    [MethodImpl(Hot)]
    public static void Mul(float* a, float* b, float* y, int n)
    {
        int i = 0;
        if (V256) for (; i + 8 <= n; i += 8) (Vector256.Load(a + i) * Vector256.Load(b + i)).Store(y + i);
        else if (V128) for (; i + 4 <= n; i += 4) (Vector128.Load(a + i) * Vector128.Load(b + i)).Store(y + i);
        for (; i < n; i++) y[i] = a[i] * b[i];
    }

    /// <summary>y += a * s</summary>
    [MethodImpl(Hot)]
    public static void AddScaled(float* a, float s, float* y, int n)
    {
        int i = 0;
        if (V256) { var vs = Vector256.Create(s); for (; i + 8 <= n; i += 8) Vector256.FusedMultiplyAdd(Vector256.Load(a + i), vs, Vector256.Load(y + i)).Store(y + i); }
        else if (V128) { var vs = Vector128.Create(s); for (; i + 4 <= n; i += 4) Vector128.FusedMultiplyAdd(Vector128.Load(a + i), vs, Vector128.Load(y + i)).Store(y + i); }
        for (; i < n; i++) y[i] += a[i] * s;
    }

    /// <summary>y = a * s * w (fused scale + elementwise weight, used by RMS norm)</summary>
    [MethodImpl(Hot)]
    public static void ScaleMul(float* a, float s, float* w, float* y, int n)
    {
        int i = 0;
        if (V256) { var vs = Vector256.Create(s); for (; i + 8 <= n; i += 8) (Vector256.Load(a + i) * vs * Vector256.Load(w + i)).Store(y + i); }
        else if (V128) { var vs = Vector128.Create(s); for (; i + 4 <= n; i += 4) (Vector128.Load(a + i) * vs * Vector128.Load(w + i)).Store(y + i); }
        for (; i < n; i++) y[i] = a[i] * s * w[i];
    }

    // ------------------------------------------------------------------ transcendental

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Exp256(Vector256<float> x)
    {
        var r = Vector256.Create(12582912.0f);
        var z = Vector256.FusedMultiplyAdd(x, Vector256.Create(1.4426950216293335f), r);
        var n = z - r;
        var b = Vector256.FusedMultiplyAdd(-n, Vector256.Create(1.428606765330187e-06f), Vector256.FusedMultiplyAdd(-n, Vector256.Create(0.693145751953125f), x));
        var e = Vector256.ShiftLeft(z.AsInt32(), 23);
        var k = (e + Vector256.Create(1.0f).AsInt32()).AsSingle();
        var c = Vector256.GreaterThan(Vector256.Abs(n), Vector256.Create(126f));
        var u = b * b;
        var j = Vector256.FusedMultiplyAdd(
            Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(Vector256.Create(0.008247390389442444f), b, Vector256.Create(0.04189976677298546f)), u,
                Vector256.FusedMultiplyAdd(Vector256.Create(0.16668395698070526f), b, Vector256.Create(0.4999912679195404f))),
            u, Vector256.Create(0.9999994039535522f) * b);
        if (c == Vector256<float>.Zero) return Vector256.FusedMultiplyAdd(j, k, k);
        var g = Vector256.LessThanOrEqual(n, Vector256<float>.Zero).AsUInt32() & Vector256.Create(0x82000000u);
        var s1 = (g + Vector256.Create(0x7f000000u)).AsSingle();
        var s2 = (e.AsUInt32() - g).AsSingle();
        var d = Vector256.GreaterThan(Vector256.Abs(n), Vector256.Create(192f));
        var big = Vector256.ConditionalSelect(c, Vector256.FusedMultiplyAdd(s2, j, s2) * s1, Vector256.FusedMultiplyAdd(k, j, k));
        return Vector256.ConditionalSelect(d, s1 * s1, big);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Exp128(Vector128<float> x)
    {
        var r = Vector128.Create(12582912.0f);
        var z = Vector128.FusedMultiplyAdd(x, Vector128.Create(1.4426950216293335f), r);
        var n = z - r;
        var b = Vector128.FusedMultiplyAdd(-n, Vector128.Create(1.428606765330187e-06f), Vector128.FusedMultiplyAdd(-n, Vector128.Create(0.693145751953125f), x));
        var e = Vector128.ShiftLeft(z.AsInt32(), 23);
        var k = (e + Vector128.Create(1.0f).AsInt32()).AsSingle();
        var c = Vector128.GreaterThan(Vector128.Abs(n), Vector128.Create(126f));
        var u = b * b;
        var j = Vector128.FusedMultiplyAdd(
            Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(Vector128.Create(0.008247390389442444f), b, Vector128.Create(0.04189976677298546f)), u,
                Vector128.FusedMultiplyAdd(Vector128.Create(0.16668395698070526f), b, Vector128.Create(0.4999912679195404f))),
            u, Vector128.Create(0.9999994039535522f) * b);
        if (c == Vector128<float>.Zero) return Vector128.FusedMultiplyAdd(j, k, k);
        var g = Vector128.LessThanOrEqual(n, Vector128<float>.Zero).AsUInt32() & Vector128.Create(0x82000000u);
        var s1 = (g + Vector128.Create(0x7f000000u)).AsSingle();
        var s2 = (e.AsUInt32() - g).AsSingle();
        var d = Vector128.GreaterThan(Vector128.Abs(n), Vector128.Create(192f));
        var big = Vector128.ConditionalSelect(c, Vector128.FusedMultiplyAdd(s2, j, s2) * s1, Vector128.FusedMultiplyAdd(k, j, k));
        return Vector128.ConditionalSelect(d, s1 * s1, big);
    }

    /// <summary>y = exp(x)</summary>
    [MethodImpl(Hot)]
    public static void Exp(float* x, float* y, int n)
    {
        int i = 0;
        if (V256) for (; i + 8 <= n; i += 8) Exp256(Vector256.Load(x + i)).Store(y + i);
        else if (V128) for (; i + 4 <= n; i += 4) Exp128(Vector128.Load(x + i)).Store(y + i);
        for (; i < n; i++) y[i] = MathF.Exp(x[i]);
    }

    /// <summary>y = exp(x - m); returns sum(y). Fused softmax numerator.</summary>
    [MethodImpl(Hot)]
    public static float ExpSubSum(float* x, float m, float* y, int n)
    {
        int i = 0; float s = 0;
        if (V256)
        {
            var vm = Vector256.Create(m); var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8) { var v = Exp256(Vector256.Load(x + i) - vm); v.Store(y + i); acc += v; }
            s = Vector256.Sum(acc);
        }
        else if (V128)
        {
            var vm = Vector128.Create(m); var acc = Vector128<float>.Zero;
            for (; i + 4 <= n; i += 4) { var v = Exp128(Vector128.Load(x + i) - vm); v.Store(y + i); acc += v; }
            s = Vector128.Sum(acc);
        }
        for (; i < n; i++) { float v = MathF.Exp(x[i] - m); y[i] = v; s += v; }
        return s;
    }

    /// <summary>y = 1 / (1 + exp(-x))</summary>
    [MethodImpl(Hot)]
    public static void Sigmoid(float* x, float* y, int n)
    {
        int i = 0;
        if (V256) { var one = Vector256.Create(1f); for (; i + 8 <= n; i += 8) (one / (one + Exp256(-Vector256.Load(x + i)))).Store(y + i); }
        else if (V128) { var one = Vector128.Create(1f); for (; i + 4 <= n; i += 4) (one / (one + Exp128(-Vector128.Load(x + i)))).Store(y + i); }
        for (; i < n; i++) y[i] = 1f / (1f + MathF.Exp(-x[i]));
    }

    /// <summary>y = silu(g) * u</summary>
    [MethodImpl(Hot)]
    public static void SwiGlu(float* g, float* u, float* y, int n)
    {
        int i = 0;
        if (V256)
        {
            var one = Vector256.Create(1f);
            for (; i + 8 <= n; i += 8) { var v = Vector256.Load(g + i); (v / (one + Exp256(-v)) * Vector256.Load(u + i)).Store(y + i); }
        }
        else if (V128)
        {
            var one = Vector128.Create(1f);
            for (; i + 4 <= n; i += 4) { var v = Vector128.Load(g + i); (v / (one + Exp128(-v)) * Vector128.Load(u + i)).Store(y + i); }
        }
        for (; i < n; i++) { float v = g[i]; y[i] = v / (1f + MathF.Exp(-v)) * u[i]; }
    }

    /// <summary>y = silu(x)</summary>
    [MethodImpl(Hot)]
    public static void Silu(float* x, float* y, int n)
    {
        int i = 0;
        if (V256) { var one = Vector256.Create(1f); for (; i + 8 <= n; i += 8) { var v = Vector256.Load(x + i); (v / (one + Exp256(-v))).Store(y + i); } }
        else if (V128) { var one = Vector128.Create(1f); for (; i + 4 <= n; i += 4) { var v = Vector128.Load(x + i); (v / (one + Exp128(-v))).Store(y + i); } }
        for (; i < n; i++) { float v = x[i]; y[i] = v / (1f + MathF.Exp(-v)); }
    }

    /// <summary>tanh via exp: tanh(x) = 1 - 2 / (exp(2x) + 1), with |x| clamped for stability.</summary>
    [MethodImpl(Hot)]
    public static void Tanh(float* x, float* y, int n)
    {
        int i = 0;
        if (V256)
        {
            var one = Vector256.Create(1f); var two = Vector256.Create(2f); var lim = Vector256.Create(9f);
            for (; i + 8 <= n; i += 8)
            {
                var v = Vector256.Min(Vector256.Max(Vector256.Load(x + i), -lim), lim);
                (one - two / (Exp256(v * two) + one)).Store(y + i);
            }
        }
        else if (V128)
        {
            var one = Vector128.Create(1f); var two = Vector128.Create(2f); var lim = Vector128.Create(9f);
            for (; i + 4 <= n; i += 4)
            {
                var v = Vector128.Min(Vector128.Max(Vector128.Load(x + i), -lim), lim);
                (one - two / (Exp128(v * two) + one)).Store(y + i);
            }
        }
        for (; i < n; i++) y[i] = MathF.Tanh(x[i]);
    }

    private const float GeluA = 0.044715f;
    private const float SqrtTwoOverPi = 0.79788456080286535587989211986876f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Gelu256(Vector256<float> x)
    {
        // 0.5x(1+tanh(k(x+a x^3))) == x * sigmoid(2k(x + a x^3))
        var inner = x * Vector256.FusedMultiplyAdd(x * x, Vector256.Create(GeluA * SqrtTwoOverPi * 2f), Vector256.Create(SqrtTwoOverPi * 2f));
        inner = Vector256.Min(Vector256.Max(inner, Vector256.Create(-80f)), Vector256.Create(80f));
        return x / (Vector256.Create(1f) + Exp256(-inner));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Gelu128(Vector128<float> x)
    {
        var inner = x * Vector128.FusedMultiplyAdd(x * x, Vector128.Create(GeluA * SqrtTwoOverPi * 2f), Vector128.Create(SqrtTwoOverPi * 2f));
        inner = Vector128.Min(Vector128.Max(inner, Vector128.Create(-80f)), Vector128.Create(80f));
        return x / (Vector128.Create(1f) + Exp128(-inner));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float GeluScalar(float x)
    {
        float inner = x * (SqrtTwoOverPi * 2f + GeluA * SqrtTwoOverPi * 2f * x * x);
        inner = Math.Clamp(inner, -80f, 80f);
        return x / (1f + MathF.Exp(-inner));
    }

    /// <summary>GELU, tanh approximation.</summary>
    [MethodImpl(Hot)]
    public static void Gelu(float* x, float* y, int n)
    {
        int i = 0;
        if (EmulateFp16Gelu)
        {
            for (; i < n; i++) y[i] = (float)(Half)GeluScalar((float)(Half)x[i]);
            return;
        }
        if (V256) for (; i + 8 <= n; i += 8) Gelu256(Vector256.Load(x + i)).Store(y + i);
        else if (V128) for (; i + 4 <= n; i += 4) Gelu128(Vector128.Load(x + i)).Store(y + i);
        for (; i < n; i++) y[i] = GeluScalar(x[i]);
    }

    internal static readonly bool EmulateFp16Gelu = Environment.GetEnvironmentVariable("LAMMA_GELU16") == "1";

    /// <summary>y = gelu(g) * u</summary>
    [MethodImpl(Hot)]
    public static void GeGlu(float* g, float* u, float* y, int n)
    {
        int i = 0;
        if (EmulateFp16Gelu)
        {
            for (; i < n; i++) y[i] = (float)(Half)GeluScalar((float)(Half)g[i]) * u[i];
            return;
        }
        if (V256) for (; i + 8 <= n; i += 8) (Gelu256(Vector256.Load(g + i)) * Vector256.Load(u + i)).Store(y + i);
        else if (V128) for (; i + 4 <= n; i += 4) (Gelu128(Vector128.Load(g + i)) * Vector128.Load(u + i)).Store(y + i);
        for (; i < n; i++) y[i] = GeluScalar(g[i]) * u[i];
    }
}
