using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace LammaNet.Tensors;

/// <summary>
/// Multi-token dot kernels: one quantized weight row against up to 4 activation rows. Each weight block is
/// unpacked once and reused for every token, which makes small batches (speculative-decoding verification,
/// concurrent sequences) nearly as cheap as a single token on memory-bound CPUs.
/// </summary>
public static unsafe class VecDotMulti
{
    private const MethodImplOptions Hot = MethodImplOptions.AggressiveOptimization;

    public static bool Supports(GgmlType t) => VecDot.UseAvx2 && t is GgmlType.Q4_K or GgmlType.Q6_K or GgmlType.Q8_0 or GgmlType.Q5_K;

    /// <summary>out[t] = dot(x, y[t]) for t &lt; nt (nt ≤ 4); <paramref name="yStride"/> is the byte stride between activation rows.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Dot(GgmlType type, byte* x, byte* y, long yStride, int nt, int n, float* outp)
    {
        switch (type)
        {
            case GgmlType.Q4_K: Q4_K(x, y, yStride, nt, n, outp); break;
            case GgmlType.Q5_K: Q5_K(x, y, yStride, nt, n, outp); break;
            case GgmlType.Q6_K: Q6_K(x, y, yStride, nt, n, outp); break;
            case GgmlType.Q8_0: Q8_0(x, y, yStride, nt, n, outp); break;
            default: for (int t = 0; t < nt; t++) outp[t] = VecDot.Dot(type, x, y + t * yStride, n); break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float H(byte* p) => HalfHelper.ToFloat(*(ushort*)p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum(Vector256<float> v) => VecDot.HSum(v);

    private static readonly byte* ShuffleK4 = Build(8, 32, (i, j) => (byte)(2 * i + (j & 1)));
    private static readonly byte* ShuffleQ6 = Build(8, 16, (i, j) => (byte)(2 * i + (j >= 8 ? 1 : 0)));

    private static byte* Build(int rows, int cols, Func<int, int, byte> f)
    {
        var t = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(rows * cols), 32);
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) t[i * cols + j] = f(i, j);
        return t;
    }

    private const uint KMask1 = 0x3f3f3f3f, KMask2 = 0x0f0f0f0f, KMask3 = 0x03030303;

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

    [MethodImpl(Hot)]
    private static void Q4_K(byte* x, byte* y, long ys, int nt, int n, float* outp)
    {
        int nb = n / 256;
        uint* utmp = stackalloc uint[4];
        var m4 = Vector256.Create((byte)0xF);
        Vector256<float> acc0 = default, acc1 = default, acc2 = default, acc3 = default;
        float mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* xb = x + i * 144L;
            float xd = H(xb), xdmin = H(xb + 2);
            UnpackK4Scales(xb + 4, utmp);
            var minsAndScales = Avx2.ConvertToVector256Int16(Vector128.Create(utmp[0], utmp[1], utmp[2], utmp[3]).AsByte());
            var mins16 = minsAndScales.GetUpper();
            var sc128 = minsAndScales.GetLower();
            var scales = Vector256.Create(sc128, sc128).AsByte();
            byte* q4 = xb + 16;
            Vector256<int> s0 = default, s1 = default, s2 = default, s3 = default;
            for (int j = 0; j < 4; j++)
            {
                var scaleL = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j) * 32)).AsInt16();
                var scaleH = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j + 1) * 32)).AsInt16();
                var bits = Avx.LoadVector256(q4 + j * 32);
                var lo = Avx2.And(bits, m4);
                var hi = Avx2.And(Avx2.ShiftRightLogical(bits.AsUInt16(), 4).AsByte(), m4);
                int off = 4 + i * 292 + j * 64;
                s0 = Avx2.Add(s0, Avx2.Add(Avx2.MultiplyAddAdjacent(scaleL, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + off)))),
                                           Avx2.MultiplyAddAdjacent(scaleH, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + off + 32))))));
                if (nt > 1) s1 = Avx2.Add(s1, Avx2.Add(Avx2.MultiplyAddAdjacent(scaleL, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + ys + off)))),
                                           Avx2.MultiplyAddAdjacent(scaleH, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + ys + off + 32))))));
                if (nt > 2) s2 = Avx2.Add(s2, Avx2.Add(Avx2.MultiplyAddAdjacent(scaleL, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + 2 * ys + off)))),
                                           Avx2.MultiplyAddAdjacent(scaleH, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + 2 * ys + off + 32))))));
                if (nt > 3) s3 = Avx2.Add(s3, Avx2.Add(Avx2.MultiplyAddAdjacent(scaleL, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + 3 * ys + off)))),
                                           Avx2.MultiplyAddAdjacent(scaleH, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + 3 * ys + off + 32))))));
            }
            for (int t = 0; t < nt; t++)
            {
                byte* yb = y + t * ys + i * 292L;
                float yd = *(float*)yb;
                var q8sums = Avx.LoadVector256((short*)(yb + 260));
                var q8s = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                var prod = Sse2.MultiplyAddAdjacent(mins16, q8s);
                var hs = Ssse3.HorizontalAdd(Ssse3.HorizontalAdd(prod, prod), prod);
                float mins = yd * xdmin * hs.ToScalar();
                var d = Vector256.Create(yd * xd);
                switch (t)
                {
                    case 0: acc0 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s0), acc0); mins0 += mins; break;
                    case 1: acc1 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s1), acc1); mins1 += mins; break;
                    case 2: acc2 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s2), acc2); mins2 += mins; break;
                    default: acc3 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s3), acc3); mins3 += mins; break;
                }
            }
        }
        outp[0] = HSum(acc0) - mins0;
        if (nt > 1) outp[1] = HSum(acc1) - mins1;
        if (nt > 2) outp[2] = HSum(acc2) - mins2;
        if (nt > 3) outp[3] = HSum(acc3) - mins3;
    }

    [MethodImpl(Hot)]
    private static void Q5_K(byte* x, byte* y, long ys, int nt, int n, float* outp)
    {
        int nb = n / 256;
        uint* utmp = stackalloc uint[4];
        var m4 = Vector256.Create((byte)0xF);
        Vector256<float> acc0 = default, acc1 = default, acc2 = default, acc3 = default;
        float mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* xb = x + i * 176L;
            float xd = H(xb), xdmin = H(xb + 2);
            UnpackK4Scales(xb + 4, utmp);
            var minsAndScales = Avx2.ConvertToVector256Int16(Vector128.Create(utmp[0], utmp[1], utmp[2], utmp[3]).AsByte());
            var mins16 = minsAndScales.GetUpper();
            var sc128 = minsAndScales.GetLower();
            var scales = Vector256.Create(sc128, sc128).AsByte();
            var hbits = Avx.LoadVector256(xb + 16);
            byte* q5 = xb + 48;
            Vector256<int> s0 = default, s1 = default, s2 = default, s3 = default;
            var hmask = Vector256.Create((byte)1);
            byte bit = 0;
            for (int j = 0; j < 4; j++)
            {
                var scale0 = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j) * 32)).AsInt16();
                var scale1 = Avx2.Shuffle(scales, Avx.LoadVector256(ShuffleK4 + (2 * j + 1) * 32)).AsInt16();
                var bits = Avx.LoadVector256(q5 + j * 32);
                var lo = Avx2.Add(Avx2.And(bits, m4), Avx2.ShiftLeftLogical(Avx2.ShiftRightLogical(Avx2.And(hbits, hmask).AsUInt16(), bit++), 4).AsByte());
                hmask = Avx2.ShiftLeftLogical(hmask.AsUInt16(), 1).AsByte();
                var hi = Avx2.Add(Avx2.And(Avx2.ShiftRightLogical(bits.AsUInt16(), 4).AsByte(), m4), Avx2.ShiftLeftLogical(Avx2.ShiftRightLogical(Avx2.And(hbits, hmask).AsUInt16(), bit++), 4).AsByte());
                hmask = Avx2.ShiftLeftLogical(hmask.AsUInt16(), 1).AsByte();
                int off = 4 + i * 292 + j * 64;
                s0 = Avx2.Add(s0, Avx2.Add(Avx2.MultiplyAddAdjacent(scale0, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + off)))),
                                           Avx2.MultiplyAddAdjacent(scale1, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + off + 32))))));
                if (nt > 1) s1 = Avx2.Add(s1, Avx2.Add(Avx2.MultiplyAddAdjacent(scale0, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + ys + off)))),
                                           Avx2.MultiplyAddAdjacent(scale1, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + ys + off + 32))))));
                if (nt > 2) s2 = Avx2.Add(s2, Avx2.Add(Avx2.MultiplyAddAdjacent(scale0, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + 2 * ys + off)))),
                                           Avx2.MultiplyAddAdjacent(scale1, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + 2 * ys + off + 32))))));
                if (nt > 3) s3 = Avx2.Add(s3, Avx2.Add(Avx2.MultiplyAddAdjacent(scale0, Avx2.MultiplyAddAdjacent(lo, Avx.LoadVector256((sbyte*)(y + 3 * ys + off)))),
                                           Avx2.MultiplyAddAdjacent(scale1, Avx2.MultiplyAddAdjacent(hi, Avx.LoadVector256((sbyte*)(y + 3 * ys + off + 32))))));
            }
            for (int t = 0; t < nt; t++)
            {
                byte* yb = y + t * ys + i * 292L;
                float yd = *(float*)yb;
                var q8sums = Avx.LoadVector256((short*)(yb + 260));
                var q8s = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                var prod = Sse2.MultiplyAddAdjacent(mins16, q8s);
                var hs = Ssse3.HorizontalAdd(Ssse3.HorizontalAdd(prod, prod), prod);
                float mins = yd * xdmin * hs.ToScalar();
                var d = Vector256.Create(yd * xd);
                switch (t)
                {
                    case 0: acc0 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s0), acc0); mins0 += mins; break;
                    case 1: acc1 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s1), acc1); mins1 += mins; break;
                    case 2: acc2 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s2), acc2); mins2 += mins; break;
                    default: acc3 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(s3), acc3); mins3 += mins; break;
                }
            }
        }
        outp[0] = HSum(acc0) - mins0;
        if (nt > 1) outp[1] = HSum(acc1) - mins1;
        if (nt > 2) outp[2] = HSum(acc2) - mins2;
        if (nt > 3) outp[3] = HSum(acc3) - mins3;
    }

    [MethodImpl(Hot)]
    private static void Q6_K(byte* x, byte* y, long ys, int nt, int n, float* outp)
    {
        int nb = n / 256;
        var m3 = Vector256.Create((byte)3);
        var m15 = Vector256.Create((byte)15);
        var m12 = Vector256.Create((byte)12);
        var m48 = Vector256.Create((byte)48);
        var m192 = Vector256.Create((byte)192);
        Vector256<float> acc0 = default, acc1 = default, acc2 = default, acc3 = default;
        var q = stackalloc Vector256<byte>[4];
        var sc16 = stackalloc Vector256<short>[4];
        for (int i = 0; i < nb; i++)
        {
            byte* xb = x + i * 210L;
            float xd = H(xb + 208);
            var scales = Sse2.LoadVector128((sbyte*)(xb + 192));
            var scales16 = Avx2.ConvertToVector256Int16(scales);
            Vector256<int> s0 = default, s1 = default, s2 = default, s3 = default;
            int isc = 0;
            for (int j = 0; j < 2; j++)
            {
                byte* q4 = xb + j * 64;
                var b1 = Avx.LoadVector256(q4);
                var b2 = Avx.LoadVector256(q4 + 32);
                var bh = Avx.LoadVector256(xb + 128 + j * 32);
                q[0] = Avx2.Or(Avx2.And(b1, m15), Avx2.ShiftLeftLogical(Avx2.And(bh, m3).AsUInt16(), 4).AsByte());
                q[1] = Avx2.Or(Avx2.And(b2, m15), Avx2.ShiftLeftLogical(Avx2.And(bh, m12).AsUInt16(), 2).AsByte());
                q[2] = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(b1.AsUInt16(), 4).AsByte(), m15), Avx2.And(bh, m48));
                q[3] = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(b2.AsUInt16(), 4).AsByte(), m15), Avx2.ShiftRightLogical(Avx2.And(bh, m192).AsUInt16(), 2).AsByte());
                for (int k = 0; k < 4; k++)
                    sc16[k] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Sse2.LoadVector128((sbyte*)ShuffleQ6 + (isc + k) * 16)));
                isc += 4;
                for (int k = 0; k < 4; k++)
                {
                    int off = 4 + i * 292 + j * 128 + k * 32;
                    s0 = Avx2.Add(s0, Avx2.MultiplyAddAdjacent(sc16[k], Avx2.MultiplyAddAdjacent(q[k], Avx.LoadVector256((sbyte*)(y + off)))));
                    if (nt > 1) s1 = Avx2.Add(s1, Avx2.MultiplyAddAdjacent(sc16[k], Avx2.MultiplyAddAdjacent(q[k], Avx.LoadVector256((sbyte*)(y + ys + off)))));
                    if (nt > 2) s2 = Avx2.Add(s2, Avx2.MultiplyAddAdjacent(sc16[k], Avx2.MultiplyAddAdjacent(q[k], Avx.LoadVector256((sbyte*)(y + 2 * ys + off)))));
                    if (nt > 3) s3 = Avx2.Add(s3, Avx2.MultiplyAddAdjacent(sc16[k], Avx2.MultiplyAddAdjacent(q[k], Avx.LoadVector256((sbyte*)(y + 3 * ys + off)))));
                }
            }
            for (int t = 0; t < nt; t++)
            {
                byte* yb = y + t * ys + i * 292L;
                var q8sums = Avx.LoadVector256((short*)(yb + 260));
                var sub = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums, scales16), 5);
                var d = Vector256.Create(*(float*)yb * xd);
                switch (t)
                {
                    case 0: acc0 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(Avx2.Subtract(s0, sub)), acc0); break;
                    case 1: acc1 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(Avx2.Subtract(s1, sub)), acc1); break;
                    case 2: acc2 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(Avx2.Subtract(s2, sub)), acc2); break;
                    default: acc3 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(Avx2.Subtract(s3, sub)), acc3); break;
                }
            }
        }
        outp[0] = HSum(acc0);
        if (nt > 1) outp[1] = HSum(acc1);
        if (nt > 2) outp[2] = HSum(acc2);
        if (nt > 3) outp[3] = HSum(acc3);
    }

    [MethodImpl(Hot)]
    private static void Q8_0(byte* x, byte* y, long ys, int nt, int n, float* outp)
    {
        int nb = n / 32;
        Vector256<float> acc0 = default, acc1 = default, acc2 = default, acc3 = default;
        for (int i = 0; i < nb; i++)
        {
            byte* xb = x + i * 34L;
            float xd = H(xb);
            var qx = Avx.LoadVector256((sbyte*)(xb + 2));
            var ax = Avx2.Sign(qx, qx).AsByte();
            for (int t = 0; t < nt; t++)
            {
                byte* yb = y + t * ys + i * 34L;
                var sy = Avx2.Sign(Avx.LoadVector256((sbyte*)(yb + 2)), qx);
                var dot = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(ax, sy), Vector256.Create((short)1));
                var d = Vector256.Create(xd * H(yb));
                switch (t)
                {
                    case 0: acc0 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(dot), acc0); break;
                    case 1: acc1 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(dot), acc1); break;
                    case 2: acc2 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(dot), acc2); break;
                    default: acc3 = Fma.MultiplyAdd(d, Avx.ConvertToVector256Single(dot), acc3); break;
                }
            }
        }
        outp[0] = HSum(acc0);
        if (nt > 1) outp[1] = HSum(acc1);
        if (nt > 2) outp[2] = HSum(acc2);
        if (nt > 3) outp[3] = HSum(acc3);
    }
}
