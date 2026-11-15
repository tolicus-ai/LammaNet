using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace LammaNet.Tensors;

/// <summary>
/// Dot products between one quantized weight row and one quantized activation row.
/// AVX2 (+FMA, AVX-VNNI when available) fast paths with portable scalar fallbacks that are
/// numerically equivalent to ggml's generic kernels.
/// </summary>
public static unsafe class VecDot
{
    internal static bool UseAvx2 = Avx2.IsSupported && Fma.IsSupported;
    internal static bool UseVnni = AvxVnni.IsSupported;

    /// <summary>Computes dot(weightRow, actRow) for <paramref name="n"/> elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(GgmlType wType, byte* x, byte* y, int n) => wType switch
    {
        GgmlType.Q4_K => Q4_K(x, y, n),
        GgmlType.Q6_K => Q6_K(x, y, n),
        GgmlType.Q5_K => Q5_K(x, y, n),
        GgmlType.Q8_0 => Q8_0(x, y, n),
        GgmlType.Q4_0 => Q4_0(x, y, n),
        GgmlType.F32 => F32((float*)x, (float*)y, n),
        GgmlType.F16 => F16((ushort*)x, (float*)y, n),
        GgmlType.BF16 => BF16((ushort*)x, (float*)y, n),
        GgmlType.Q3_K => Q3_K(x, y, n),
        GgmlType.Q2_K => Q2_K(x, y, n),
        GgmlType.Q5_0 => Q5_0(x, y, n),
        GgmlType.Q4_1 => Q4_1(x, y, n),
        GgmlType.Q5_1 => Q5_1(x, y, n),
        GgmlType.IQ4_NL => IQ4_NL(x, y, n),
        GgmlType.IQ4_XS => IQ4_XS(x, y, n),
        _ => throw new NotSupportedException($"No dot kernel for {wType}"),
    };

    // ------------------------------------------------------------------ helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float H(byte* p) => HalfHelper.ToFloat(*(ushort*)p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float HSum(Vector256<float> v)
    {
        var s = Sse.Add(v.GetLower(), v.GetUpper());
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        s = Sse.AddScalar(s, Sse3.MoveHighAndDuplicate(s));
        return s.ToScalar();
    }

    /// <summary>u8 x s8 multiply, pairwise add to i16, then pairwise add to i32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> MulSumUs8(Vector256<byte> ax, Vector256<sbyte> sy)
    {
        if (UseVnni) return AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, ax, sy);
        var dot = Avx2.MultiplyAddAdjacent(ax, sy);
        return Avx2.MultiplyAddAdjacent(dot, Vector256.Create((short)1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> MulSumI8(Vector256<sbyte> x, Vector256<sbyte> y)
    {
        var ax = Avx2.Sign(x, x).AsByte();
        var sy = Avx2.Sign(y, x);
        return MulSumUs8(ax, sy);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> BytesFromNibbles32(byte* p)
    {
        var tmp = Sse2.LoadVector128(p);
        var bytes = Vector256.Create(tmp, Sse2.ShiftRightLogical(tmp.AsUInt16(), 4).AsByte());
        return Avx2.And(bytes, Vector256.Create((byte)0x0F));
    }

    private static readonly byte* ShuffleK4 = BuildShuffleK4();
    private static readonly byte* ShuffleQ6 = BuildShuffleQ6();

    private static byte* BuildShuffleK4()
    {
        var t = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(256, 32);
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 32; j++) t[i * 32 + j] = (byte)(2 * i + (j & 1));
        return t;
    }

    private static byte* BuildShuffleQ6()
    {
        var t = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(128, 32);
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 16; j++) t[i * 16 + j] = (byte)(2 * i + (j >= 8 ? 1 : 0));
        return t;
    }

    private const uint KMask1 = 0x3f3f3f3f, KMask2 = 0x0f0f0f0f, KMask3 = 0x03030303;

    /// <summary>Unpacks the 12-byte k4 scales into 8 scales (utmp[0..1]) and 8 mins (utmp[2..3]).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackK4Scales(byte* scales, uint* utmp)
    {
        utmp[0] = *(uint*)scales; utmp[1] = *(uint*)(scales + 4); utmp[2] = *(uint*)(scales + 8);
        utmp[3] = ((utmp[2] >> 4) & KMask2) | (((utmp[1] >> 6) & KMask3) << 4);
        uint uaux = utmp[1] & KMask1;
        utmp[1] = (utmp[2] & KMask2) | (((utmp[0] >> 6) & KMask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMask1;
    }

    // ------------------------------------------------------------------ float kernels

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float F32(float* x, float* y, int n)
    {
        int i = 0;
        float sum = 0;
        if (UseAvx2)
        {
            var a0 = Vector256<float>.Zero; var a1 = a0; var a2 = a0; var a3 = a0;
            for (; i + 32 <= n; i += 32)
            {
                a0 = Fma.MultiplyAdd(Avx.LoadVector256(x + i), Avx.LoadVector256(y + i), a0);
                a1 = Fma.MultiplyAdd(Avx.LoadVector256(x + i + 8), Avx.LoadVector256(y + i + 8), a1);
                a2 = Fma.MultiplyAdd(Avx.LoadVector256(x + i + 16), Avx.LoadVector256(y + i + 16), a2);
                a3 = Fma.MultiplyAdd(Avx.LoadVector256(x + i + 24), Avx.LoadVector256(y + i + 24), a3);
            }
            for (; i + 8 <= n; i += 8) a0 = Fma.MultiplyAdd(Avx.LoadVector256(x + i), Avx.LoadVector256(y + i), a0);
            sum = HSum(Avx.Add(Avx.Add(a0, a1), Avx.Add(a2, a3)));
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            var a0 = Vector128<float>.Zero;
            for (; i + 4 <= n; i += 4) a0 += Vector128.Load(x + i) * Vector128.Load(y + i);
            sum = Vector128.Sum(a0);
        }
        for (; i < n; i++) sum += x[i] * y[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float F16(ushort* x, float* y, int n)
    {
        int i = 0;
        float sum = 0;
        if (UseAvx2)
        {
            var a0 = Vector256<float>.Zero; var a1 = a0;
            for (; i + 16 <= n; i += 16)
            {
                a0 = Fma.MultiplyAdd(HalfHelper.Load8Half(x + i), Avx.LoadVector256(y + i), a0);
                a1 = Fma.MultiplyAdd(HalfHelper.Load8Half(x + i + 8), Avx.LoadVector256(y + i + 8), a1);
            }
            for (; i + 8 <= n; i += 8) a0 = Fma.MultiplyAdd(HalfHelper.Load8Half(x + i), Avx.LoadVector256(y + i), a0);
            sum = HSum(Avx.Add(a0, a1));
        }
        for (; i < n; i++) sum += HalfHelper.ToFloat(x[i]) * y[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float BF16(ushort* x, float* y, int n)
    {
        int i = 0;
        float sum = 0;
        if (UseAvx2)
        {
            var a0 = Vector256<float>.Zero; var a1 = a0;
            for (; i + 16 <= n; i += 16)
            {
                a0 = Fma.MultiplyAdd(HalfHelper.Load8Bf16(x + i), Avx.LoadVector256(y + i), a0);
                a1 = Fma.MultiplyAdd(HalfHelper.Load8Bf16(x + i + 8), Avx.LoadVector256(y + i + 8), a1);
            }
            for (; i + 8 <= n; i += 8) a0 = Fma.MultiplyAdd(HalfHelper.Load8Bf16(x + i), Avx.LoadVector256(y + i), a0);
            sum = HSum(Avx.Add(a0, a1));
        }
        for (; i < n; i++) sum += HalfHelper.Bf16ToFloat(x[i]) * y[i];
        return sum;
    }

    // ------------------------------------------------------------------ 32-element block kernels

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q8_0(byte* x, byte* y, int n)
    {
        int nb = n / 32;
        if (UseAvx2)
        {
            var acc = Vector256<float>.Zero;
            for (int i = 0; i < nb; i++, x += 34, y += 34)
            {
                var d = Vector256.Create(H(x) * H(y));
                var qx = Avx.LoadVector256((sbyte*)(x + 2));
                var qy = Avx.LoadVector256((sbyte*)(y + 2));
                acc = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(MulSumI8(qx, qy)), acc);
            }
            return HSum(acc);
        }
        float sum = 0;
        for (int i = 0; i < nb; i++, x += 34, y += 34)
        {
            int s = 0;
            sbyte* qx = (sbyte*)(x + 2); sbyte* qy = (sbyte*)(y + 2);
            for (int j = 0; j < 32; j++) s += qx[j] * qy[j];
            sum += s * H(x) * H(y);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q4_0(byte* x, byte* y, int n)
    {
        int nb = n / 32;
        if (UseAvx2)
        {
            var acc = Vector256<float>.Zero;
            var off = Vector256.Create((sbyte)8);
            for (int i = 0; i < nb; i++, x += 18, y += 34)
            {
                var d = Vector256.Create(H(x) * H(y));
                var qx = Avx2.Subtract(BytesFromNibbles32(x + 2).AsSByte(), off);
                var qy = Avx.LoadVector256((sbyte*)(y + 2));
                acc = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(MulSumI8(qx, qy)), acc);
            }
            return HSum(acc);
        }
        float sum = 0;
        for (int i = 0; i < nb; i++, x += 18, y += 34)
        {
            int s = 0;
            byte* qx = x + 2; sbyte* qy = (sbyte*)(y + 2);
            for (int j = 0; j < 16; j++) s += ((qx[j] & 0xF) - 8) * qy[j] + ((qx[j] >> 4) - 8) * qy[j + 16];
            sum += s * H(x) * H(y);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q4_1(byte* x, byte* y, int n)
    {
        int nb = n / 32;
        float sum = 0;
        for (int i = 0; i < nb; i++, x += 20, y += 36)
        {
            int s = 0;
            byte* qx = x + 4; sbyte* qy = (sbyte*)(y + 4);
            for (int j = 0; j < 16; j++) s += (qx[j] & 0xF) * qy[j] + (qx[j] >> 4) * qy[j + 16];
            sum += H(x) * H(y) * s + H(x + 2) * H(y + 2);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q5_0(byte* x, byte* y, int n)
    {
        int nb = n / 32;
        float sum = 0;
        for (int i = 0; i < nb; i++, x += 22, y += 34)
        {
            uint qh = *(uint*)(x + 2);
            byte* qx = x + 6; sbyte* qy = (sbyte*)(y + 2);
            int s = 0;
            for (int j = 0; j < 16; j++)
            {
                int xh0 = (int)(((qh >> j) << 4) & 0x10);
                int xh1 = (int)((qh >> (j + 12)) & 0x10);
                s += (((qx[j] & 0xF) | xh0) - 16) * qy[j] + (((qx[j] >> 4) | xh1) - 16) * qy[j + 16];
            }
            sum += s * H(x) * H(y);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q5_1(byte* x, byte* y, int n)
    {
        int nb = n / 32;
        float sum = 0;
        for (int i = 0; i < nb; i++, x += 24, y += 36)
        {
            uint qh = *(uint*)(x + 4);
            byte* qx = x + 8; sbyte* qy = (sbyte*)(y + 4);
            int s = 0;
            for (int j = 0; j < 16; j++)
            {
                int xh0 = (int)(((qh >> j) << 4) & 0x10);
                int xh1 = (int)((qh >> (j + 12)) & 0x10);
                s += ((qx[j] & 0xF) | xh0) * qy[j] + ((qx[j] >> 4) | xh1) * qy[j + 16];
            }
            sum += H(x) * H(y) * s + H(x + 2) * H(y + 2);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float IQ4_NL(byte* x, byte* y, int n)
    {
        int nb = n / 32;
        if (UseAvx2)
        {
            var values = Vector128.Create(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113).AsSByte();
            var table = Vector256.Create(values, values);
            var m4 = Vector256.Create((byte)0x0F);
            var acc = Vector256<float>.Zero;
            for (int i = 0; i < nb; i++, x += 18, y += 34)
            {
                var idx = BytesFromNibbles32(x + 2);
                var qx = Avx2.Shuffle(table, Avx2.And(idx, m4).AsSByte());
                var qy = Avx.LoadVector256((sbyte*)(y + 2));
                acc = Fma.MultiplyAdd(Vector256.Create(H(x) * H(y)), Avx.ConvertToVector256Single(MulSumI8(qx, qy)), acc);
            }
            return HSum(acc);
        }
        float sum = 0;
        for (int i = 0; i < nb; i++, x += 18, y += 34)
        {
            int s = 0;
            byte* qx = x + 2; sbyte* qy = (sbyte*)(y + 2);
            for (int j = 0; j < 16; j++) s += Dequantize.KValuesIq4Nl[qx[j] & 0xF] * qy[j] + Dequantize.KValuesIq4Nl[qx[j] >> 4] * qy[j + 16];
            sum += s * H(x) * H(y);
        }
        return sum;
    }

    // ------------------------------------------------------------------ k-quants (Q8_K activations)

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q4_K(byte* x, byte* y, int n)
    {
        int nb = n / 256;
        uint* utmp = stackalloc uint[4];
        if (UseAvx2)
        {
            var m4 = Vector256.Create((byte)0xF);
            var acc = Vector256<float>.Zero;
            var accM = Vector128<float>.Zero;
            for (int i = 0; i < nb; i++, x += 144, y += 292)
            {
                float yd = *(float*)y;
                float d = yd * H(x);
                float dmin = -yd * H(x + 2);
                UnpackK4Scales(x + 4, utmp);
                byte* q4 = x + 16;
                sbyte* q8 = (sbyte*)(y + 4);
                var minsAndScales = Avx2.ConvertToVector256Int16(Vector128.Create(utmp[0], utmp[1], utmp[2], utmp[3]).AsByte());
                var q8sums = Avx.LoadVector256((short*)(y + 260));
                var q8s = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                var prod = Sse2.MultiplyAddAdjacent(minsAndScales.GetUpper(), q8s);
                accM = Fma.MultiplyAdd(Vector128.Create(dmin), Sse2.ConvertToVector128Single(prod), accM);
                var sc128 = minsAndScales.GetLower();
                var scales = Vector256.Create(sc128, sc128).AsByte();
                var sumi = Vector256<int>.Zero;
                for (int j = 0; j < 4; j++)
                {
                    var scaleL = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j) * 32)).AsInt16();
                    var scaleH = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j + 1) * 32)).AsInt16();
                    var q4bits = Avx.LoadVector256(q4); q4 += 32;
                    var q4l = Avx2.And(q4bits, m4);
                    var q4h = Avx2.And(Avx2.ShiftRightLogical(q4bits.AsUInt16(), 4).AsByte(), m4);
                    var q8l = Avx.LoadVector256(q8); q8 += 32;
                    var p16l = Avx2.MultiplyAddAdjacent(scaleL, Avx2.MultiplyAddAdjacent(q4l, q8l));
                    var q8h = Avx.LoadVector256(q8); q8 += 32;
                    var p16h = Avx2.MultiplyAddAdjacent(scaleH, Avx2.MultiplyAddAdjacent(q4h, q8h));
                    sumi = Avx2.Add(sumi, Avx2.Add(p16l, p16h));
                }
                acc = Fma.MultiplyAdd(Vector256.Create(d), Avx.ConvertToVector256Single(sumi), acc);
            }
            accM = Sse.Add(accM, Sse.MoveHighToLow(accM, accM));
            accM = Sse.AddScalar(accM, Sse3.MoveHighAndDuplicate(accM));
            return HSum(acc) + accM.ToScalar();
        }

        float sumf = 0;
        byte* sc = (byte*)utmp;
        for (int i = 0; i < nb; i++, x += 144, y += 292)
        {
            float yd = *(float*)y;
            UnpackK4Scales(x + 4, utmp);
            byte* q4 = x + 16;
            sbyte* q8 = (sbyte*)(y + 4);
            short* bsums = (short*)(y + 260);
            int sumMins = 0;
            for (int j = 0; j < 16; j++) sumMins += bsums[j] * sc[8 + j / 2];
            int isum = 0;
            for (int j = 0; j < 4; j++)
            {
                int s1 = 0, s2 = 0;
                for (int l = 0; l < 32; l++) { s1 += (q4[l] & 0xF) * q8[l]; s2 += (q4[l] >> 4) * q8[l + 32]; }
                isum += s1 * sc[2 * j] + s2 * sc[2 * j + 1];
                q4 += 32; q8 += 64;
            }
            sumf += yd * H(x) * isum - yd * H(x + 2) * sumMins;
        }
        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q5_K(byte* x, byte* y, int n)
    {
        int nb = n / 256;
        uint* utmp = stackalloc uint[4];
        if (UseAvx2)
        {
            var m4 = Vector256.Create((byte)0xF);
            var acc = Vector256<float>.Zero;
            float summs = 0;
            for (int i = 0; i < nb; i++, x += 176, y += 292)
            {
                float yd = *(float*)y;
                float d = yd * H(x);
                float dmin = -yd * H(x + 2);
                UnpackK4Scales(x + 4, utmp);
                byte* q5 = x + 48;
                sbyte* q8 = (sbyte*)(y + 4);
                var minsAndScales = Avx2.ConvertToVector256Int16(Vector128.Create(utmp[0], utmp[1], utmp[2], utmp[3]).AsByte());
                var q8sums = Avx.LoadVector256((short*)(y + 260));
                var q8s = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                var prod = Sse2.MultiplyAddAdjacent(minsAndScales.GetUpper(), q8s);
                var hs = Ssse3.HorizontalAdd(Ssse3.HorizontalAdd(prod, Vector128<int>.Zero), Vector128<int>.Zero);
                summs += dmin * hs.ToScalar();
                var sc128 = minsAndScales.GetLower();
                var scales = Vector256.Create(sc128, sc128).AsByte();
                var hbits = Avx.LoadVector256(x + 16);
                var hmask = Vector256.Create((byte)1);
                var sumi = Vector256<int>.Zero;
                byte bit = 0;
                for (int j = 0; j < 4; j++)
                {
                    var scale0 = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j) * 32)).AsInt16();
                    var scale1 = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j + 1) * 32)).AsInt16();
                    var q5bits = Avx.LoadVector256(q5); q5 += 32;
                    var q5l0 = Avx2.And(q5bits, m4);
                    var q5h0 = Avx2.ShiftLeftLogical(Avx2.ShiftRightLogical(Avx2.And(hbits, hmask).AsUInt16(), bit++), 4).AsByte();
                    var q50 = Avx2.Add(q5l0, q5h0);
                    hmask = Avx2.ShiftLeftLogical(hmask.AsUInt16(), 1).AsByte();
                    var q5l1 = Avx2.And(Avx2.ShiftRightLogical(q5bits.AsUInt16(), 4).AsByte(), m4);
                    var q5h1 = Avx2.ShiftLeftLogical(Avx2.ShiftRightLogical(Avx2.And(hbits, hmask).AsUInt16(), bit++), 4).AsByte();
                    var q51 = Avx2.Add(q5l1, q5h1);
                    hmask = Avx2.ShiftLeftLogical(hmask.AsUInt16(), 1).AsByte();
                    var q80 = Avx.LoadVector256(q8); q8 += 32;
                    var q81 = Avx.LoadVector256(q8); q8 += 32;
                    var p0 = Avx2.MultiplyAddAdjacent(scale0, Avx2.MultiplyAddAdjacent(q50, q80));
                    var p1 = Avx2.MultiplyAddAdjacent(scale1, Avx2.MultiplyAddAdjacent(q51, q81));
                    sumi = Avx2.Add(sumi, Avx2.Add(p0, p1));
                }
                acc = Fma.MultiplyAdd(Vector256.Create(d), Avx.ConvertToVector256Single(sumi), acc);
            }
            return HSum(acc) + summs;
        }

        float sumf = 0;
        byte* sc = (byte*)utmp;
        for (int i = 0; i < nb; i++, x += 176, y += 292)
        {
            float yd = *(float*)y;
            UnpackK4Scales(x + 4, utmp);
            byte* qh = x + 16;
            byte* ql = x + 48;
            sbyte* q8 = (sbyte*)(y + 4);
            short* bsums = (short*)(y + 260);
            int sumMins = 0;
            for (int j = 0; j < 16; j++) sumMins += bsums[j] * sc[8 + j / 2];
            int isum = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < 4; j++)
            {
                int s1 = 0, s2 = 0;
                for (int l = 0; l < 32; l++)
                {
                    s1 += ((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) * q8[l];
                    s2 += ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) * q8[l + 32];
                }
                isum += s1 * sc[2 * j] + s2 * sc[2 * j + 1];
                ql += 32; q8 += 64; u1 <<= 2; u2 <<= 2;
            }
            sumf += yd * H(x) * isum - yd * H(x + 2) * sumMins;
        }
        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q6_K(byte* x, byte* y, int n)
    {
        int nb = n / 256;
        if (UseAvx2)
        {
            var m3 = Vector256.Create((byte)3);
            var m15 = Vector256.Create((byte)15);
            var m12 = Vector256.Create((byte)12);
            var m48 = Vector256.Create((byte)48);
            var m192 = Vector256.Create((byte)192);
            var acc = Vector256<float>.Zero;
            for (int i = 0; i < nb; i++, x += 210, y += 292)
            {
                float d = *(float*)y * H(x + 208);
                byte* q4 = x;
                byte* qh = x + 128;
                sbyte* q8 = (sbyte*)(y + 4);
                var q8sums = Avx.LoadVector256((short*)(y + 260));
                var scales = Sse2.LoadVector128((sbyte*)(x + 192));
                var scales16 = Avx2.ConvertToVector256Int16(scales);
                var q8sclsub = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums, scales16), 5);
                var sumi = Vector256<int>.Zero;
                int isc = 0;
                for (int j = 0; j < 2; j++)
                {
                    var q4bits1 = Avx.LoadVector256(q4); q4 += 32;
                    var q4bits2 = Avx.LoadVector256(q4); q4 += 32;
                    var q4bitsH = Avx.LoadVector256(qh); qh += 32;
                    var q4h0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsUInt16(), 4).AsByte();
                    var q4h1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsUInt16(), 2).AsByte();
                    var q4h2 = Avx2.And(q4bitsH, m48);
                    var q4h3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, m192).AsUInt16(), 2).AsByte();
                    var q40 = Avx2.Or(Avx2.And(q4bits1, m15), q4h0);
                    var q41 = Avx2.Or(Avx2.And(q4bits2, m15), q4h1);
                    var q42 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsUInt16(), 4).AsByte(), m15), q4h2);
                    var q43 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsUInt16(), 4).AsByte(), m15), q4h3);
                    var q80 = Avx.LoadVector256(q8); q8 += 32;
                    var q81 = Avx.LoadVector256(q8); q8 += 32;
                    var q82 = Avx.LoadVector256(q8); q8 += 32;
                    var q83 = Avx.LoadVector256(q8); q8 += 32;
                    var p0 = Avx2.MultiplyAddAdjacent(q40, q80);
                    var p1 = Avx2.MultiplyAddAdjacent(q41, q81);
                    var p2 = Avx2.MultiplyAddAdjacent(q42, q82);
                    var p3 = Avx2.MultiplyAddAdjacent(q43, q83);
                    var s0 = Ssse3.Shuffle(scales, Sse2.LoadVector128((sbyte*)ShuffleQ6 + (isc + 0) * 16));
                    var s1 = Ssse3.Shuffle(scales, Sse2.LoadVector128((sbyte*)ShuffleQ6 + (isc + 1) * 16));
                    var s2 = Ssse3.Shuffle(scales, Sse2.LoadVector128((sbyte*)ShuffleQ6 + (isc + 2) * 16));
                    var s3 = Ssse3.Shuffle(scales, Sse2.LoadVector128((sbyte*)ShuffleQ6 + (isc + 3) * 16));
                    isc += 4;
                    var r0 = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(s0), p0);
                    var r1 = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(s1), p1);
                    var r2 = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(s2), p2);
                    var r3 = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(s3), p3);
                    sumi = Avx2.Add(sumi, Avx2.Add(Avx2.Add(r0, r1), Avx2.Add(r2, r3)));
                }
                sumi = Avx2.Subtract(sumi, q8sclsub);
                acc = Fma.MultiplyAdd(Vector256.Create(d), Avx.ConvertToVector256Single(sumi), acc);
            }
            return HSum(acc);
        }

        float sumf = 0;
        for (int i = 0; i < nb; i++, x += 210, y += 292)
        {
            float d = *(float*)y * H(x + 208);
            byte* ql = x; byte* qh = x + 128; sbyte* sc = (sbyte*)(x + 192);
            sbyte* q8 = (sbyte*)(y + 4);
            int isum = 0;
            for (int nn = 0; nn < 2; nn++)
            {
                for (int l = 0; l < 32; l++)
                {
                    int iss = l / 16;
                    int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    isum += sc[iss] * q1 * q8[l] + sc[iss + 2] * q2 * q8[l + 32] + sc[iss + 4] * q3 * q8[l + 64] + sc[iss + 6] * q4 * q8[l + 96];
                }
                ql += 64; qh += 32; sc += 8; q8 += 128;
            }
            sumf += d * isum;
        }
        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q2_K(byte* x, byte* y, int n)
    {
        int nb = n / 256;
        float sumf = 0;
        for (int i = 0; i < nb; i++, x += 84, y += 292)
        {
            byte* sc = x;
            byte* q2 = x + 16;
            sbyte* q8 = (sbyte*)(y + 4);
            short* bsums = (short*)(y + 260);
            float yd = *(float*)y;
            int summs = 0;
            for (int j = 0; j < 16; j++) summs += bsums[j] * (sc[j] >> 4);
            float dall = yd * H(x + 80), dmin = yd * H(x + 82);
            int isum = 0, isc = 0;
            for (int k = 0; k < 2; k++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int d = sc[isc++] & 0xF;
                    int isuml = 0;
                    for (int l = 0; l < 16; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                    isum += d * isuml;
                    d = sc[isc++] & 0xF;
                    isuml = 0;
                    for (int l = 16; l < 32; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                    isum += d * isuml;
                    shift += 2;
                    q8 += 32;
                }
                q2 += 32;
            }
            sumf += dall * isum - dmin * summs;
        }
        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Q3_K(byte* x, byte* y, int n)
    {
        int nb = n / 256;
        uint* aux = stackalloc uint[4];
        sbyte* scales = (sbyte*)aux;
        float sumf = 0;
        for (int i = 0; i < nb; i++, x += 110, y += 292)
        {
            byte* hm = x; byte* q3 = x + 32;
            sbyte* q8 = (sbyte*)(y + 4);
            Buffer.MemoryCopy(x + 96, aux, 12, 12);
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMask2) | (((tmp >> 4) & KMask3) << 4);
            aux[3] = ((aux[1] >> 4) & KMask2) | (((tmp >> 6) & KMask3) << 4);
            aux[0] = (aux[0] & KMask2) | (((tmp >> 0) & KMask3) << 4);
            aux[1] = (aux[1] & KMask2) | (((tmp >> 2) & KMask3) << 4);
            int isum = 0, isc = 0;
            byte m = 1;
            for (int nn = 0; nn < 2; nn++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int s = 0;
                    for (int l = 0; l < 16; l++) s += q8[l] * (((q3[l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4));
                    isum += (scales[isc++] - 32) * s;
                    s = 0;
                    for (int l = 16; l < 32; l++) s += q8[l] * (((q3[l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4));
                    isum += (scales[isc++] - 32) * s;
                    shift += 2; m <<= 1; q8 += 32;
                }
                q3 += 32;
            }
            sumf += H(x + 108) * *(float*)y * isum;
        }
        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float IQ4_XS(byte* x, byte* y, int n)
    {
        int nb = n / 256;
        float sumf = 0;
        for (int i = 0; i < nb; i++, x += 136, y += 292)
        {
            ushort scalesH = *(ushort*)(x + 2);
            byte* scalesL = x + 4;
            byte* qs = x + 8;
            sbyte* q8 = (sbyte*)(y + 4);
            int isum = 0;
            for (int ib = 0; ib < 8; ib++)
            {
                int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                int s = 0;
                for (int j = 0; j < 16; j++) s += Dequantize.KValuesIq4Nl[qs[j] & 0xF] * q8[j] + Dequantize.KValuesIq4Nl[qs[j] >> 4] * q8[j + 16];
                isum += (ls - 32) * s;
                qs += 16; q8 += 32;
            }
            sumf += H(x) * *(float*)y * isum;
        }
        return sumf;
    }
}
