using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace LammaNet.Tensors;

/// <summary>
/// On-the-fly quantization of float activations into the formats consumed by the integer dot kernels
/// (Q8_0, Q8_1, Q8_K) plus helpers to quantize weights / KV entries.
/// </summary>
public static unsafe class QuantizeActivations
{
    internal static bool ForceScalar;

    /// <summary>The activation format that pairs with a given weight type in <see cref="VecDot"/>.</summary>
    public static GgmlType VecDotType(GgmlType weight) => weight switch
    {
        GgmlType.Q4_0 or GgmlType.Q5_0 or GgmlType.Q8_0 or GgmlType.IQ4_NL => GgmlType.Q8_0,
        GgmlType.Q4_1 or GgmlType.Q5_1 => GgmlType.Q8_1,
        GgmlType.Q2_K or GgmlType.Q3_K or GgmlType.Q4_K or GgmlType.Q5_K or GgmlType.Q6_K or GgmlType.IQ4_XS => GgmlType.Q8_K,
        GgmlType.F32 or GgmlType.F16 or GgmlType.BF16 => GgmlType.F32,
        _ => throw new NotSupportedException($"Weight type {weight} is not supported for matrix multiplication"),
    };

    /// <summary>Quantizes one row of <paramref name="n"/> floats into <paramref name="type"/>.</summary>
    public static void Row(GgmlType type, float* x, byte* y, int n)
    {
        switch (type)
        {
            case GgmlType.F32: Buffer.MemoryCopy(x, y, n * 4L, n * 4L); break;
            case GgmlType.Q8_0: Q8_0(x, y, n); break;
            case GgmlType.Q8_1: Q8_1(x, y, n); break;
            case GgmlType.Q8_K: Q8_K(x, y, n); break;
            case GgmlType.F16: HalfHelper.FloatToHalf(x, (ushort*)y, n); break;
            case GgmlType.BF16: for (int i = 0; i < n; i++) ((ushort*)y)[i] = HalfHelper.FloatToBf16(x[i]); break;
            case GgmlType.Q4_0: Q4_0(x, y, n); break;
            default: throw new NotSupportedException($"Quantization to {type} is not supported");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HMax(Vector256<float> v)
    {
        var m = Sse.Max(v.GetLower(), v.GetUpper());
        m = Sse.Max(m, Sse.MoveHighToLow(m, m));
        m = Sse.MaxScalar(m, Sse.Shuffle(m, m, 0x55));
        return m.ToScalar();
    }

    public static void Q8_0(float* x, byte* y, int n)
    {
        int nb = n / 32;
        for (int i = 0; i < nb; i++, x += 32, y += 34)
        {
            float amax;
            if (Avx2.IsSupported)
            {
                var sgn = Vector256.Create(0x7fffffff).AsSingle();
                var v0 = Avx.LoadVector256(x); var v1 = Avx.LoadVector256(x + 8);
                var v2 = Avx.LoadVector256(x + 16); var v3 = Avx.LoadVector256(x + 24);
                var m = Avx.Max(Avx.Max(Avx.And(v0, sgn), Avx.And(v1, sgn)), Avx.Max(Avx.And(v2, sgn), Avx.And(v3, sgn)));
                amax = HMax(m);
                float d = amax / 127f;
                float id = d != 0 ? 1f / d : 0f;
                *(ushort*)y = HalfHelper.FromFloat(d);
                var vid = Vector256.Create(id);
                var i0 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(v0, vid)));
                var i1 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(v1, vid)));
                var i2 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(v2, vid)));
                var i3 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(v3, vid)));
                Store32(i0, i1, i2, i3, (sbyte*)(y + 2));
            }
            else
            {
                amax = 0;
                for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(x[j]));
                float d = amax / 127f;
                float id = d != 0 ? 1f / d : 0f;
                *(ushort*)y = HalfHelper.FromFloat(d);
                sbyte* q = (sbyte*)(y + 2);
                for (int j = 0; j < 32; j++) q[j] = (sbyte)MathF.Round(x[j] * id, MidpointRounding.ToEven);
            }
        }
    }

    /// <summary>Q8_1: like Q8_0 but also stores s = d * sum(q) (used with Q4_1/Q5_1 weights).</summary>
    public static void Q8_1(float* x, byte* y, int n)
    {
        int nb = n / 32;
        for (int i = 0; i < nb; i++, x += 32, y += 36)
        {
            float amax = 0;
            for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(x[j]));
            float d = amax / 127f;
            float id = d != 0 ? 1f / d : 0f;
            sbyte* q = (sbyte*)(y + 4);
            int sum = 0;
            for (int j = 0; j < 32; j++)
            {
                int v = (int)MathF.Round(x[j] * id, MidpointRounding.ToEven);
                q[j] = (sbyte)v;
                sum += v;
            }
            *(ushort*)y = HalfHelper.FromFloat(d);
            *(ushort*)(y + 2) = HalfHelper.FromFloat(sum * d);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store32(Vector256<int> i0, Vector256<int> i1, Vector256<int> i2, Vector256<int> i3, sbyte* dst)
    {
        // pack int32 -> int16 -> int8 with saturation, then fix lane order
        var s01 = Avx2.PackSignedSaturate(i0, i1);
        var s23 = Avx2.PackSignedSaturate(i2, i3);
        var b = Avx2.PackSignedSaturate(s01, s23);
        var perm = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        b = Avx2.PermuteVar8x32(b.AsInt32(), perm).AsSByte();
        Avx.Store(dst, b);
    }

    /// <summary>Q8_K: float d, int8 qs[256], int16 bsums[16] per 256 elements.</summary>
    public static void Q8_K(float* x, byte* y, int n)
    {
        int nb = n / 256;
        for (int i = 0; i < nb; i++, x += 256, y += 292)
        {
            sbyte* qs = (sbyte*)(y + 4);
            short* bsums = (short*)(y + 260);
            float max = 0, amax = 0;
            if (Avx2.IsSupported && !ForceScalar)
            {
                var sgn = Vector256.Create(0x7fffffff).AsSingle();
                var m = Vector256<float>.Zero;
                for (int j = 0; j < 256; j += 8) m = Avx.Max(m, Avx.And(Avx.LoadVector256(x + j), sgn));
                amax = HMax(m);
                if (amax == 0)
                {
                    *(float*)y = 0;
                    new Span<byte>(y + 4, 288).Clear();
                    continue;
                }
                // find signed value with largest magnitude (first occurrence, as in ggml)
                for (int j = 0; j < 256; j++) if (MathF.Abs(x[j]) == amax) { max = x[j]; break; }
                float iscale = -127f / max;
                var vs = Vector256.Create(iscale);
                var v127 = Vector256.Create(127);
                for (int j = 0; j < 256; j += 32)
                {
                    var i0 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + j), vs))), v127);
                    var i1 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + j + 8), vs))), v127);
                    var i2 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + j + 16), vs))), v127);
                    var i3 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + j + 24), vs))), v127);
                    Store32(i0, i1, i2, i3, qs + j);
                }
                // block sums of 16 via widening
                var ones = Vector256.Create((short)1);
                for (int j = 0; j < 16; j++)
                {
                    var w = Avx2.ConvertToVector256Int16(Sse2.LoadVector128(qs + j * 16)); // 16 x i16
                    var p = Avx2.MultiplyAddAdjacent(w, ones); // 8 x i32
                    var s = Sse2.Add(p.GetLower(), p.GetUpper());
                    s = Ssse3.HorizontalAdd(s, s);
                    s = Ssse3.HorizontalAdd(s, s);
                    bsums[j] = (short)s.ToScalar();
                }
                *(float*)y = 1f / iscale;
            }
            else
            {
                for (int j = 0; j < 256; j++)
                {
                    float ax = MathF.Abs(x[j]);
                    if (ax > amax) { amax = ax; max = x[j]; }
                }
                if (amax == 0)
                {
                    *(float*)y = 0;
                    new Span<byte>(y + 4, 288).Clear();
                    continue;
                }
                float iscale = -127f / max;
                for (int j = 0; j < 256; j++)
                {
                    int v = (int)MathF.Round(iscale * x[j], MidpointRounding.ToEven);
                    qs[j] = (sbyte)Math.Min(127, v);
                }
                for (int j = 0; j < 16; j++)
                {
                    int sum = 0;
                    for (int ii = 0; ii < 16; ii++) sum += qs[j * 16 + ii];
                    bsums[j] = (short)sum;
                }
                *(float*)y = 1f / iscale;
            }
        }
    }

    /// <summary>Q4_0 weight quantization (reference), used for quantized KV caches and tests.</summary>
    public static void Q4_0(float* x, byte* y, int n)
    {
        int nb = n / 32;
        for (int i = 0; i < nb; i++, x += 32, y += 18)
        {
            float amax = 0, max = 0;
            for (int j = 0; j < 32; j++)
            {
                float v = x[j];
                if (MathF.Abs(v) > amax) { amax = MathF.Abs(v); max = v; }
            }
            float d = max / -8f;
            float id = d != 0 ? 1f / d : 0f;
            *(ushort*)y = HalfHelper.FromFloat(d);
            byte* qs = y + 2;
            for (int j = 0; j < 16; j++)
            {
                float x0 = x[j] * id, x1 = x[16 + j] * id;
                int xi0 = Math.Min(15, (int)(x0 + 8.5f));
                int xi1 = Math.Min(15, (int)(x1 + 8.5f));
                qs[j] = (byte)(xi0 | (xi1 << 4));
            }
        }
    }
}
