using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace LammaNet.Tensors;

/// <summary>AVX2 dequantization for the most common weight formats (falls back to <see cref="Dequantize"/>).</summary>
public static unsafe class DequantizeSimd
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Row(GgmlType type, byte* src, float* dst, int n)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) { Dequantize.Row(type, src, dst, n); return; }
        switch (type)
        {
            case GgmlType.Q4_K: Q4_K(src, dst, n); break;
            case GgmlType.Q6_K: Q6_K(src, dst, n); break;
            case GgmlType.Q5_K: Q5_K(src, dst, n); break;
            case GgmlType.Q8_0: Q8_0(src, dst, n); break;
            case GgmlType.Q4_0: Q4_0(src, dst, n); break;
            case GgmlType.F16: HalfHelper.HalfToFloat((ushort*)src, dst, n); break;
            case GgmlType.BF16: HalfHelper.Bf16ToFloat((ushort*)src, dst, n); break;
            case GgmlType.F32: Buffer.MemoryCopy(src, dst, n * 4L, n * 4L); break;
            default: Dequantize.Row(type, src, dst, n); break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float H(byte* p) => HalfHelper.ToFloat(*(ushort*)p);

    /// <summary>Writes 32 bytes (unsigned) as floats: y = d * q - m.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store32U(Vector256<byte> q, Vector256<float> d, Vector256<float> negM, float* y)
    {
        var lo = q.GetLower(); var hi = q.GetUpper();
        Avx.Store(y, Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(lo)), d, negM));
        Avx.Store(y + 8, Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(lo, 8))), d, negM));
        Avx.Store(y + 16, Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(hi)), d, negM));
        Avx.Store(y + 24, Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(hi, 8))), d, negM));
    }

    /// <summary>Writes 16 signed bytes as floats scaled by d.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store16S(Vector128<sbyte> q, Vector256<float> d, float* y)
    {
        Avx.Store(y, Avx.Multiply(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(q)), d));
        Avx.Store(y + 8, Avx.Multiply(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(q, 8))), d));
    }

    private static void Q8_0(byte* x, float* y, int n)
    {
        for (int i = 0; i < n / 32; i++, x += 34, y += 32)
        {
            var d = Vector256.Create(H(x));
            Store16S(Sse2.LoadVector128((sbyte*)(x + 2)), d, y);
            Store16S(Sse2.LoadVector128((sbyte*)(x + 18)), d, y + 16);
        }
    }

    private static void Q4_0(byte* x, float* y, int n)
    {
        var m4 = Vector128.Create((byte)0x0F);
        var off = Vector128.Create((sbyte)8);
        for (int i = 0; i < n / 32; i++, x += 18, y += 32)
        {
            var d = Vector256.Create(H(x));
            var q = Sse2.LoadVector128(x + 2);
            var lo = Sse2.Subtract(Sse2.And(q, m4).AsSByte(), off);
            var hi = Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(q.AsUInt16(), 4).AsByte(), m4).AsSByte(), off);
            Store16S(lo, d, y);
            Store16S(hi, d, y + 16);
        }
    }

    private static void Q4_K(byte* x, float* y, int n)
    {
        var m4 = Vector256.Create((byte)0x0F);
        for (int i = 0; i < n / 256; i++, x += 144)
        {
            float d = H(x), min = H(x + 2);
            byte* scales = x + 4;
            byte* q = x + 16;
            for (int j = 0; j < 4; j++)
            {
                Dequantize.GetScaleMinK4(2 * j, scales, out var sc1, out var mm1);
                Dequantize.GetScaleMinK4(2 * j + 1, scales, out var sc2, out var mm2);
                var bits = Avx.LoadVector256(q);
                Store32U(Avx2.And(bits, m4), Vector256.Create(d * sc1), Vector256.Create(-min * mm1), y);
                Store32U(Avx2.And(Avx2.ShiftRightLogical(bits.AsUInt16(), 4).AsByte(), m4), Vector256.Create(d * sc2), Vector256.Create(-min * mm2), y + 32);
                q += 32; y += 64;
            }
        }
    }

    private static void Q5_K(byte* x, float* y, int n)
    {
        var m4 = Vector256.Create((byte)0x0F);
        var bit16 = Vector256.Create((byte)16);
        for (int i = 0; i < n / 256; i++, x += 176)
        {
            float d = H(x), min = H(x + 2);
            byte* scales = x + 4;
            var hbits = Avx.LoadVector256(x + 16);
            byte* q = x + 48;
            for (int j = 0; j < 4; j++)
            {
                Dequantize.GetScaleMinK4(2 * j, scales, out var sc1, out var mm1);
                Dequantize.GetScaleMinK4(2 * j + 1, scales, out var sc2, out var mm2);
                var bits = Avx.LoadVector256(q);
                var u1 = Vector256.Create((byte)(1 << (2 * j)));
                var u2 = Vector256.Create((byte)(2 << (2 * j)));
                var h1 = Avx2.And(Avx2.CompareEqual(Avx2.And(hbits, u1), u1), bit16);
                var h2 = Avx2.And(Avx2.CompareEqual(Avx2.And(hbits, u2), u2), bit16);
                Store32U(Avx2.Or(Avx2.And(bits, m4), h1), Vector256.Create(d * sc1), Vector256.Create(-min * mm1), y);
                Store32U(Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(bits.AsUInt16(), 4).AsByte(), m4), h2), Vector256.Create(d * sc2), Vector256.Create(-min * mm2), y + 32);
                q += 32; y += 64;
            }
        }
    }

    private static void Q6_K(byte* x, float* y, int n)
    {
        var m4 = Vector256.Create((byte)0x0F);
        var m3 = Vector256.Create((byte)3);
        var off = Vector256.Create((sbyte)32);
        sbyte* tmp = stackalloc sbyte[128];
        for (int i = 0; i < n / 256; i++, x += 210)
        {
            float d = H(x + 208);
            byte* ql = x; byte* qh = x + 128; sbyte* sc = (sbyte*)(x + 192);
            for (int nn = 0; nn < 2; nn++)
            {
                var l0 = Avx.LoadVector256(ql);
                var l1 = Avx.LoadVector256(ql + 32);
                var h = Avx.LoadVector256(qh);
                var q1 = Avx2.Or(Avx2.And(l0, m4), Avx2.ShiftLeftLogical(Avx2.And(h, m3).AsUInt16(), 4).AsByte());
                var q2 = Avx2.Or(Avx2.And(l1, m4), Avx2.ShiftLeftLogical(Avx2.And(Avx2.ShiftRightLogical(h.AsUInt16(), 2).AsByte(), m3).AsUInt16(), 4).AsByte());
                var q3 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(l0.AsUInt16(), 4).AsByte(), m4), Avx2.ShiftLeftLogical(Avx2.And(Avx2.ShiftRightLogical(h.AsUInt16(), 4).AsByte(), m3).AsUInt16(), 4).AsByte());
                var q4 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(l1.AsUInt16(), 4).AsByte(), m4), Avx2.ShiftLeftLogical(Avx2.And(Avx2.ShiftRightLogical(h.AsUInt16(), 6).AsByte(), m3).AsUInt16(), 4).AsByte());
                Avx.Store(tmp, Avx2.Subtract(q1.AsSByte(), off));
                Avx.Store(tmp + 32, Avx2.Subtract(q2.AsSByte(), off));
                Avx.Store(tmp + 64, Avx2.Subtract(q3.AsSByte(), off));
                Avx.Store(tmp + 96, Avx2.Subtract(q4.AsSByte(), off));
                // 8 groups of 16 values, each with its own scale
                for (int g = 0; g < 8; g++)
                {
                    // group g covers tmp[g*16 .. g*16+15]; scale index: (g/2)*2 + (g%2) -> sc[is + 2*(g/2)] with is = g%2
                    int scaleIdx = (g % 2) + 2 * (g / 2);
                    Store16S(Sse2.LoadVector128(tmp + g * 16), Vector256.Create(d * sc[scaleIdx]), y + g * 16);
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
    }
}
