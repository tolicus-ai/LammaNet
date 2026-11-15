using System.Runtime.CompilerServices;

namespace LammaNet.Tensors;

/// <summary>
/// Reference (portable) dequantization for all supported GGUF block formats.
/// Bit-exact ports of ggml's <c>dequantize_row_*</c> functions.
/// </summary>
public static unsafe class Dequantize
{
    internal static readonly sbyte[] KValuesIq4Nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    public static bool IsSupported(GgmlType t) => t switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.BF16 or GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_0
            or GgmlType.Q5_1 or GgmlType.Q8_0 or GgmlType.Q2_K or GgmlType.Q3_K or GgmlType.Q4_K or GgmlType.Q5_K
            or GgmlType.Q6_K or GgmlType.Q8_K or GgmlType.IQ4_NL or GgmlType.IQ4_XS => true,
        _ => false,
    };

    /// <summary>Dequantizes <paramref name="n"/> elements starting at block-aligned <paramref name="src"/>.</summary>
    public static void Row(GgmlType type, byte* src, float* dst, long n)
    {
        switch (type)
        {
            case GgmlType.F32: Buffer.MemoryCopy(src, dst, n * 4, n * 4); break;
            case GgmlType.F16: HalfHelper.HalfToFloat((ushort*)src, dst, n); break;
            case GgmlType.BF16: HalfHelper.Bf16ToFloat((ushort*)src, dst, n); break;
            case GgmlType.Q4_0: Q4_0(src, dst, n); break;
            case GgmlType.Q4_1: Q4_1(src, dst, n); break;
            case GgmlType.Q5_0: Q5_0(src, dst, n); break;
            case GgmlType.Q5_1: Q5_1(src, dst, n); break;
            case GgmlType.Q8_0: Q8_0(src, dst, n); break;
            case GgmlType.Q2_K: Q2_K(src, dst, n); break;
            case GgmlType.Q3_K: Q3_K(src, dst, n); break;
            case GgmlType.Q4_K: Q4_K(src, dst, n); break;
            case GgmlType.Q5_K: Q5_K(src, dst, n); break;
            case GgmlType.Q6_K: Q6_K(src, dst, n); break;
            case GgmlType.Q8_K: Q8_K(src, dst, n); break;
            case GgmlType.IQ4_NL: IQ4_NL(src, dst, n); break;
            case GgmlType.IQ4_XS: IQ4_XS(src, dst, n); break;
            default: throw new NotSupportedException($"Dequantization of {type} is not supported");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float H(byte* p) => HalfHelper.ToFloat(*(ushort*)p);

    private static void Q4_0(byte* x, float* y, long k)
    {
        long nb = k / 32;
        for (long i = 0; i < nb; i++, x += 18, y += 32)
        {
            float d = H(x);
            byte* qs = x + 2;
            for (int j = 0; j < 16; j++)
            {
                y[j] = ((qs[j] & 0x0F) - 8) * d;
                y[j + 16] = ((qs[j] >> 4) - 8) * d;
            }
        }
    }

    private static void Q4_1(byte* x, float* y, long k)
    {
        long nb = k / 32;
        for (long i = 0; i < nb; i++, x += 20, y += 32)
        {
            float d = H(x), m = H(x + 2);
            byte* qs = x + 4;
            for (int j = 0; j < 16; j++)
            {
                y[j] = (qs[j] & 0x0F) * d + m;
                y[j + 16] = (qs[j] >> 4) * d + m;
            }
        }
    }

    private static void Q5_0(byte* x, float* y, long k)
    {
        long nb = k / 32;
        for (long i = 0; i < nb; i++, x += 22, y += 32)
        {
            float d = H(x);
            uint qh = *(uint*)(x + 2);
            byte* qs = x + 6;
            for (int j = 0; j < 16; j++)
            {
                int xh0 = (int)(((qh >> j) << 4) & 0x10);
                int xh1 = (int)((qh >> (j + 12)) & 0x10);
                y[j] = (((qs[j] & 0x0F) | xh0) - 16) * d;
                y[j + 16] = (((qs[j] >> 4) | xh1) - 16) * d;
            }
        }
    }

    private static void Q5_1(byte* x, float* y, long k)
    {
        long nb = k / 32;
        for (long i = 0; i < nb; i++, x += 24, y += 32)
        {
            float d = H(x), m = H(x + 2);
            uint qh = *(uint*)(x + 4);
            byte* qs = x + 8;
            for (int j = 0; j < 16; j++)
            {
                int xh0 = (int)(((qh >> j) << 4) & 0x10);
                int xh1 = (int)((qh >> (j + 12)) & 0x10);
                y[j] = ((qs[j] & 0x0F) | xh0) * d + m;
                y[j + 16] = ((qs[j] >> 4) | xh1) * d + m;
            }
        }
    }

    private static void Q8_0(byte* x, float* y, long k)
    {
        long nb = k / 32;
        for (long i = 0; i < nb; i++, x += 34, y += 32)
        {
            float d = H(x);
            sbyte* qs = (sbyte*)(x + 2);
            for (int j = 0; j < 32; j++) y[j] = qs[j] * d;
        }
    }

    private static void Q8_K(byte* x, float* y, long k)
    {
        long nb = k / 256;
        for (long i = 0; i < nb; i++, x += 292, y += 256)
        {
            float d = *(float*)x;
            sbyte* qs = (sbyte*)(x + 4);
            for (int j = 0; j < 256; j++) y[j] = qs[j] * d;
        }
    }

    // block_q2_K: scales[16], qs[64], d, dmin
    private static void Q2_K(byte* x, float* y, long k)
    {
        long nb = k / 256;
        for (long i = 0; i < nb; i++, x += 84)
        {
            byte* scales = x;
            byte* q = x + 16;
            float d = H(x + 80), min = H(x + 82);
            int isc = 0;
            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte sc = scales[isc++];
                    float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                    sc = scales[isc++];
                    dl = d * (sc & 0xF); ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                q += 32;
            }
        }
    }

    // block_q3_K: hmask[32], qs[64], scales[12], d
    private static void Q3_K(byte* x, float* y, long k)
    {
        const uint kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        long nb = k / 256;
        uint* aux = stackalloc uint[4];
        sbyte* scales = (sbyte*)aux;
        for (long i = 0; i < nb; i++, x += 110)
        {
            float dAll = H(x + 108);
            byte* q = x + 32;
            byte* hm = x;
            byte m = 1;
            Buffer.MemoryCopy(x + 96, aux, 12, 12);
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            int isc = 0;
            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    float dl = dAll * (scales[isc++] - 32);
                    for (int l = 0; l < 16; l++) *y++ = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4));
                    dl = dAll * (scales[isc++] - 32);
                    for (int l = 0; l < 16; l++) *y++ = dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4));
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4) { d = (byte)(q[j] & 63); m = (byte)(q[j + 4] & 63); }
        else
        {
            d = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    // block_q4_K: d, dmin, scales[12], qs[128]
    private static void Q4_K(byte* x, float* y, long k)
    {
        long nb = k / 256;
        for (long i = 0; i < nb; i++, x += 144)
        {
            float d = H(x), min = H(x + 2);
            byte* scales = x + 4;
            byte* q = x + 16;
            int isc = 0;
            for (int j = 0; j < 256; j += 64)
            {
                GetScaleMinK4(isc, scales, out var sc, out var m);
                float d1 = d * sc, m1 = min * m;
                GetScaleMinK4(isc + 1, scales, out sc, out m);
                float d2 = d * sc, m2 = min * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32; isc += 2;
            }
        }
    }

    // block_q5_K: d, dmin, scales[12], qh[32], qs[128]
    private static void Q5_K(byte* x, float* y, long k)
    {
        long nb = k / 256;
        for (long i = 0; i < nb; i++, x += 176)
        {
            float d = H(x), min = H(x + 2);
            byte* scales = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;
            int isc = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64)
            {
                GetScaleMinK4(isc, scales, out var sc, out var m);
                float d1 = d * sc, m1 = min * m;
                GetScaleMinK4(isc + 1, scales, out sc, out m);
                float d2 = d * sc, m2 = min * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
                ql += 32; isc += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
    }

    // block_q6_K: ql[128], qh[64], scales[16], d
    private static void Q6_K(byte* x, float* y, long k)
    {
        long nb = k / 256;
        for (long i = 0; i < nb; i++, x += 210)
        {
            float d = H(x + 208);
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            for (int n = 0; n < 256; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int iss = l / 16;
                    int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[iss] * q1;
                    y[l + 32] = d * sc[iss + 2] * q2;
                    y[l + 64] = d * sc[iss + 4] * q3;
                    y[l + 96] = d * sc[iss + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
    }

    private static void IQ4_NL(byte* x, float* y, long k)
    {
        long nb = k / 32;
        for (long i = 0; i < nb; i++, x += 18, y += 32)
        {
            float d = H(x);
            byte* qs = x + 2;
            for (int j = 0; j < 16; j++)
            {
                y[j] = d * KValuesIq4Nl[qs[j] & 0xF];
                y[j + 16] = d * KValuesIq4Nl[qs[j] >> 4];
            }
        }
    }

    // block_iq4_xs: d, scales_h(u16), scales_l[4], qs[128]
    private static void IQ4_XS(byte* x, float* y, long k)
    {
        long nb = k / 256;
        for (long i = 0; i < nb; i++, x += 136)
        {
            float d = H(x);
            ushort scalesH = *(ushort*)(x + 2);
            byte* scalesL = x + 4;
            byte* qs = x + 8;
            for (int ib = 0; ib < 8; ib++)
            {
                int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++)
                {
                    y[j] = dl * KValuesIq4Nl[qs[j] & 0xF];
                    y[j + 16] = dl * KValuesIq4Nl[qs[j] >> 4];
                }
                y += 32; qs += 16;
            }
        }
    }
}
