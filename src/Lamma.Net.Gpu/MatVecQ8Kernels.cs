using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace LammaNet.Gpu;

/// <summary>Two floats loaded together (q8 activation block: scale, scale·Σq).</summary>
public struct Float2 { public float X, Y; }

/// <summary>
/// Integer dot products of a weight format with int8 activations (q8 blocks of 32 values: int8s plus (d, d·Σq)), as in
/// llama.cpp's MMVQ: per 32 weights a token needs 32 bytes of activations instead of 128 and eight dp4a instead of 32 FMAs.
/// </summary>
public interface IQ8Format
{
    /// <summary>Weights per unit (lane l of a warp takes units l, l + 32, ...).</summary>
    int UnitWeights { get; }

    /// <summary>
    /// Adds unit <paramref name="u"/> of the row at word <paramref name="row"/> times the int8 activations of
    /// <paramref name="nt"/> tokens: token t's values start at UInt4 xq0 / 4 + t·xqs / 4 of <paramref name="xq4"/>, its
    /// block scales at xd0 + t·xds of <paramref name="xd"/>.
    /// </summary>
    void DotQ8(ArrayView<uint> w, long row, int cols, int u, ArrayView<UInt4> xq4, ArrayView<Float2> xd, long xq0, long xd0, int xqs, int xds, int nt, ref Acc8 a);
}

internal static partial class Kernels
{
    // dp4a through inline PTX: ILGPU passes an asm output through local memory, so each helper chains several dp4a
    // and returns one sum (one local round trip per 8 dp4a instead of per instruction)

    private static ulong P(uint lo, uint hi) => lo | ((ulong)hi << 32);

    /// <summary>Σ over 32 values of unsigned bytes q (8 words) × int8 x.</summary>
    internal static int Dot32U(uint q0, uint q1, uint q2, uint q3, uint q4, uint q5, uint q6, uint q7, UInt4 x0, UInt4 x1)
    {
        CudaAsm.Emit("{\n\t.reg .b32 w<8>, x<8>;\n\t.reg .s32 t;\n\t"
            + "mov.b64 {w0, w1}, %1;\n\tmov.b64 {w2, w3}, %2;\n\tmov.b64 {w4, w5}, %3;\n\tmov.b64 {w6, w7}, %4;\n\t"
            + "mov.b64 {x0, x1}, %5;\n\tmov.b64 {x2, x3}, %6;\n\tmov.b64 {x4, x5}, %7;\n\tmov.b64 {x6, x7}, %8;\n\t"
            + "dp4a.u32.s32 t, w0, x0, 0;\n\tdp4a.u32.s32 t, w1, x1, t;\n\tdp4a.u32.s32 t, w2, x2, t;\n\tdp4a.u32.s32 t, w3, x3, t;\n\t"
            + "dp4a.u32.s32 t, w4, x4, t;\n\tdp4a.u32.s32 t, w5, x5, t;\n\tdp4a.u32.s32 t, w6, x6, t;\n\tdp4a.u32.s32 %0, w7, x7, t;\n\t}",
            out int r, P(q0, q1), P(q2, q3), P(q4, q5), P(q6, q7), P(x0.X, x0.Y), P(x0.Z, x0.W), P(x1.X, x1.Y), P(x1.Z, x1.W));
        return r;
    }

    /// <summary>Σ over 32 values of signed bytes q (8 words) × int8 x.</summary>
    internal static int Dot32S(uint q0, uint q1, uint q2, uint q3, uint q4, uint q5, uint q6, uint q7, UInt4 x0, UInt4 x1)
    {
        CudaAsm.Emit("{\n\t.reg .b32 w<8>, x<8>;\n\t.reg .s32 t;\n\t"
            + "mov.b64 {w0, w1}, %1;\n\tmov.b64 {w2, w3}, %2;\n\tmov.b64 {w4, w5}, %3;\n\tmov.b64 {w6, w7}, %4;\n\t"
            + "mov.b64 {x0, x1}, %5;\n\tmov.b64 {x2, x3}, %6;\n\tmov.b64 {x4, x5}, %7;\n\tmov.b64 {x6, x7}, %8;\n\t"
            + "dp4a.s32.s32 t, w0, x0, 0;\n\tdp4a.s32.s32 t, w1, x1, t;\n\tdp4a.s32.s32 t, w2, x2, t;\n\tdp4a.s32.s32 t, w3, x3, t;\n\t"
            + "dp4a.s32.s32 t, w4, x4, t;\n\tdp4a.s32.s32 t, w5, x5, t;\n\tdp4a.s32.s32 t, w6, x6, t;\n\tdp4a.s32.s32 %0, w7, x7, t;\n\t}",
            out int r, P(q0, q1), P(q2, q3), P(q4, q5), P(q6, q7), P(x0.X, x0.Y), P(x0.Z, x0.W), P(x1.X, x1.Y), P(x1.Z, x1.W));
        return r;
    }

    /// <summary>Σ over 16 values of signed bytes q (4 words) × int8 x.</summary>
    internal static int Dot16S(uint q0, uint q1, uint q2, uint q3, UInt4 x)
    {
        CudaAsm.Emit("{\n\t.reg .b32 w<4>, x<4>;\n\t.reg .s32 t;\n\t"
            + "mov.b64 {w0, w1}, %1;\n\tmov.b64 {w2, w3}, %2;\n\tmov.b64 {x0, x1}, %3;\n\tmov.b64 {x2, x3}, %4;\n\t"
            + "dp4a.s32.s32 t, w0, x0, 0;\n\tdp4a.s32.s32 t, w1, x1, t;\n\tdp4a.s32.s32 t, w2, x2, t;\n\tdp4a.s32.s32 %0, w3, x3, t;\n\t}",
            out int r, P(q0, q1), P(q2, q3), P(x.X, x.Y), P(x.Z, x.W));
        return r;
    }

    internal static void AddTok(ref Acc8 a, int t, float v)
    {
        if (t == 0) a.A0 += v;
        else if (t == 1) a.A1 += v;
        else if (t == 2) a.A2 += v;
        else if (t == 3) a.A3 += v;
        else if (t == 4) a.A4 += v;
        else if (t == 5) a.A5 += v;
        else if (t == 6) a.A6 += v;
        else a.A7 += v;
    }

    /// <summary>
    /// Activations to q8 blocks: token t, block k (32 values) → 8 words of int8 at t·cols/4 + 8k of <paramref name="xq"/> and
    /// (d, d·Σq) at t·cols/32 + k of <paramref name="xd"/>, d = max|x| / 127. One thread per 4 values, 8 lanes per block
    /// (cols is a multiple of 128, so warps never straddle tokens and all lanes of a warp are active).
    /// </summary>
    public static void QuantizeQ8(Index1D i, ArrayView<float> x, long xOff, int xStride, int cols, ArrayView<uint> xq, ArrayView<Float2> xd)
    {
        int gpr = cols >> 2;
        int t = i / gpr, g = i - t * gpr;
        Float4 v = x.Cast<Float4>()[((xOff + (long)t * xStride) >> 2) + g];
        float amax = MathF.Max(MathF.Max(MathF.Abs(v.X), MathF.Abs(v.Y)), MathF.Max(MathF.Abs(v.Z), MathF.Abs(v.W)));
        amax = MathF.Max(amax, ILGPU.Warp.ShuffleXor(amax, 1));
        amax = MathF.Max(amax, ILGPU.Warp.ShuffleXor(amax, 2));
        amax = MathF.Max(amax, ILGPU.Warp.ShuffleXor(amax, 4));
        float inv = amax > 0 ? 127f / amax : 0f;
        int q0 = (int)MathF.Round(v.X * inv), q1 = (int)MathF.Round(v.Y * inv), q2 = (int)MathF.Round(v.Z * inv), q3 = (int)MathF.Round(v.W * inv);
        xq[(long)t * gpr + g] = (uint)(q0 & 0xFF) | ((uint)(q1 & 0xFF) << 8) | ((uint)(q2 & 0xFF) << 16) | ((uint)(q3 & 0xFF) << 24);
        int sum = q0 + q1 + q2 + q3;
        sum += ILGPU.Warp.ShuffleXor(sum, 1);
        sum += ILGPU.Warp.ShuffleXor(sum, 2);
        sum += ILGPU.Warp.ShuffleXor(sum, 4);
        if ((g & 7) == 0)
        {
            float d = amax / 127f;
            xd[(long)t * (cols >> 5) + (g >> 3)] = new Float2 { X = d, Y = d * sum };
        }
    }

    /// <summary>
    /// <see cref="MatVec{TA, TB, TC}"/> with q8 activations (formats with <see cref="IQ8Format"/>): same grid, modes and
    /// epilogue; the activations of the batch were quantized by <see cref="QuantizeQ8"/> (compact rows).
    /// </summary>
    public static void MatVecQ8<TA, TB, TC>(ArrayView<uint> w, ArrayView<float> f, ArrayView<uint> xq, ArrayView<Float2> xd, ArrayView<float> y,
        MatVecParams p, SpecializedValue<int> ntSpec)
        where TA : struct, IQ8Format where TB : struct, IQ8Format where TC : struct, IQ8Format
    {
        int nt = ntSpec.Value;
        int wi = WarpIndex, lane = Lane;
        if (wi >= p.TotalRows * p.Tiles) return;
        int r = wi % p.TotalRows, tile = wi / p.TotalRows;
        int t0 = tile * nt;
        var xq4 = xq.Cast<UInt4>();
        int xqs = p.Cols >> 2, xds = p.Cols >> 5;
        long xq0 = (long)t0 * xqs, xd0 = (long)t0 * xds;
        var a = new Acc8();
        var b = new Acc8();
        var m = p.M0;
        int rr = r;
        if (p.Mode == 2)
        {
            RowDotQ8<TA>(w, p.M0.WWord + (long)rr * p.M0.RowWords, p.Cols, xq4, xd, xq0, xd0, xqs, xds, nt, lane, ref a);
            RowDotQ8<TB>(w, p.M1.WWord + (long)rr * p.M1.RowWords, p.Cols, xq4, xd, xq0, xd0, xqs, xds, nt, lane, ref b);
            Reduce(ref b, nt);
        }
        else if (rr < p.M0.Rows || p.NMat == 1)
            RowDotQ8<TA>(w, m.WWord + (long)rr * m.RowWords, p.Cols, xq4, xd, xq0, xd0, xqs, xds, nt, lane, ref a);
        else if (rr < p.M0.Rows + p.M1.Rows || p.NMat == 2)
        {
            rr -= p.M0.Rows; m = p.M1;
            RowDotQ8<TB>(w, m.WWord + (long)rr * m.RowWords, p.Cols, xq4, xd, xq0, xd0, xqs, xds, nt, lane, ref a);
        }
        else
        {
            rr -= p.M0.Rows + p.M1.Rows; m = p.M2;
            RowDotQ8<TC>(w, m.WWord + (long)rr * m.RowWords, p.Cols, xq4, xd, xq0, xd0, xqs, xds, nt, lane, ref a);
        }
        Reduce(ref a, nt);
        if (lane >= nt || t0 + lane >= p.NTok) return;
        StoreRow(ref a, ref b, f, y, p, m, rr, t0, lane);
    }

    private static void RowDotQ8<T>(ArrayView<uint> w, long row, int cols, ArrayView<UInt4> xq4, ArrayView<Float2> xd, long xq0, long xd0, int xqs, int xds,
        int nt, int lane, ref Acc8 a) where T : struct, IQ8Format
    {
        var fmt = default(T);
        int units = cols / fmt.UnitWeights;
        for (int u = lane; u < units; u += Warp) fmt.DotQ8(w, row, cols, u, xq4, xd, xq0, xd0, xqs, xds, nt, ref a);
    }
}

public partial struct FmtQ40 : IQ8Format
{
    public int UnitWeights => 32;

    public void DotQ8(ArrayView<uint> w, long row, int cols, int u, ArrayView<UInt4> xq4, ArrayView<Float2> xd, long xq0, long xd0, int xqs, int xds, int nt, ref Acc8 a)
    {
        UInt4 q = w.Cast<UInt4>()[(row >> 2) + u];
        uint sw = w[row + (long)(cols >> 5) * 4 + (u >> 1)];
        float d = Kernels.HalfBits((int)((u & 1) == 0 ? sw & 0xFFFF : sw >> 16));
        const uint m = 0x0F0F0F0Fu;
        uint l0 = q.X & m, l1 = q.Y & m, l2 = q.Z & m, l3 = q.W & m;
        uint h0 = (q.X >> 4) & m, h1 = (q.Y >> 4) & m, h2 = (q.Z >> 4) & m, h3 = (q.W >> 4) & m;
        for (int t = 0; t < nt; t++)
        {
            long xo = ((xq0 + (long)t * xqs) >> 2) + 2L * u;
            int s = Kernels.Dot32U(l0, l1, l2, l3, h0, h1, h2, h3, xq4[xo], xq4[xo + 1]);
            Float2 ds = xd[xd0 + (long)t * xds + u];
            Kernels.AddTok(ref a, t, d * (ds.X * s - 8f * ds.Y));   // Σ d(q − 8)·x
        }
    }
}

public partial struct FmtQ80 : IQ8Format
{
    public int UnitWeights => 32;

    public void DotQ8(ArrayView<uint> w, long row, int cols, int u, ArrayView<UInt4> xq4, ArrayView<Float2> xd, long xq0, long xd0, int xqs, int xds, int nt, ref Acc8 a)
    {
        var w4 = w.Cast<UInt4>();
        long row4 = row >> 2;
        UInt4 qa = w4[row4 + 2L * u], qb = w4[row4 + 2L * u + 1];
        uint sw = w[row + (long)(cols >> 5) * 8 + (u >> 1)];
        float d = Kernels.HalfBits((int)((u & 1) == 0 ? sw & 0xFFFF : sw >> 16));
        for (int t = 0; t < nt; t++)
        {
            long xo = ((xq0 + (long)t * xqs) >> 2) + 2L * u;
            int s = Kernels.Dot32S(qa.X, qa.Y, qa.Z, qa.W, qb.X, qb.Y, qb.Z, qb.W, xq4[xo], xq4[xo + 1]);
            Kernels.AddTok(ref a, t, d * xd[xd0 + (long)t * xds + u].X * s);
        }
    }
}

public partial struct FmtQ4K : IQ8Format
{
    /// <summary>A 64-weight chunk: low nibbles are one 32-weight sub-block, high nibbles the next.</summary>
    public int UnitWeights => 64;

    public void DotQ8(ArrayView<uint> w, long row, int cols, int u, ArrayView<UInt4> xq4, ArrayView<Float2> xd, long xq0, long xd0, int xqs, int xds, int nt, ref Acc8 a)
    {
        var w4 = w.Cast<UInt4>();
        int b = u >> 2, c = u & 3;
        long b4 = (row >> 2) + b * 9L;
        UInt4 h = w4[b4], q0 = w4[b4 + 1 + 2 * c], q1 = w4[b4 + 2 + 2 * c];
        float dd = Kernels.HalfBits((int)(h.X & 0xFFFF)), dmin = Kernels.HalfBits((int)(h.X >> 16));
        Kernels.ScaleMinK4(2 * c, h.Y, h.Z, h.W, out int sc0, out int m0);
        Kernels.ScaleMinK4(2 * c + 1, h.Y, h.Z, h.W, out int sc1, out int m1);
        const uint m = 0x0F0F0F0Fu;
        uint l0 = q0.X & m, l1 = q0.Y & m, l2 = q0.Z & m, l3 = q0.W & m, l4 = q1.X & m, l5 = q1.Y & m, l6 = q1.Z & m, l7 = q1.W & m;
        uint h0 = (q0.X >> 4) & m, h1 = (q0.Y >> 4) & m, h2 = (q0.Z >> 4) & m, h3 = (q0.W >> 4) & m;
        uint h4 = (q1.X >> 4) & m, h5 = (q1.Y >> 4) & m, h6 = (q1.Z >> 4) & m, h7 = (q1.W >> 4) & m;
        float ds0 = dd * sc0, ds1 = dd * sc1, dm0 = dmin * m0, dm1 = dmin * m1;
        int blk = 8 * b + 2 * c;   // q8 block of the low sub-block
        for (int t = 0; t < nt; t++)
        {
            long xo = ((xq0 + (long)t * xqs) >> 2) + 2L * blk;
            int sLo = Kernels.Dot32U(l0, l1, l2, l3, l4, l5, l6, l7, xq4[xo], xq4[xo + 1]);
            int sHi = Kernels.Dot32U(h0, h1, h2, h3, h4, h5, h6, h7, xq4[xo + 2], xq4[xo + 3]);
            long xb = xd0 + (long)t * xds + blk;
            Float2 xl = xd[xb], xh = xd[xb + 1];
            // Σ (d·sc·q − dmin·m)·x over both sub-blocks
            Kernels.AddTok(ref a, t, ds0 * xl.X * sLo + ds1 * xh.X * sHi - dm0 * xl.Y - dm1 * xh.Y);
        }
    }
}

public partial struct FmtQ6K : IQ8Format
{
    public int UnitWeights => 32;

    public void DotQ8(ArrayView<uint> w, long row, int cols, int u, ArrayView<UInt4> xq4, ArrayView<Float2> xd, long xq0, long xd0, int xqs, int xds, int nt, ref Acc8 a)
    {
        var w4 = w.Cast<UInt4>();
        int b = u >> 3, i = u & 7, hf = i >> 2, r = i & 3;
        long b4 = (row >> 2) + b * 13L;
        UInt4 ql = w4[b4 + hf * 4 + r], qh = w4[b4 + 8 + hf * 2 + (r & 1)], sc = w4[b4 + 12];
        uint sw = w[row + (long)(cols >> 8) * 52 + (b >> 1)];
        float dd = Kernels.HalfBits((int)((b & 1) == 0 ? sw & 0xFFFF : sw >> 16));
        int si = hf * 8 + r;
        uint sw0 = si < 4 ? sc.X : si < 8 ? sc.Y : si < 12 ? sc.Z : sc.W;
        uint sw1 = si + 4 < 8 ? sc.Y : si + 4 < 12 ? sc.Z : sc.W;
        float ds0 = dd * Kernels.SignedByte(sw0, si & 3), ds1 = dd * Kernels.SignedByte(sw1, (si + 4) & 3);
        int hs = 2 * (r >> 1);
        // signed bytes q − 32 of the 16 low (scale ds0) and the 16 high (scale ds1) weights
        uint lo0 = S6((ql.X & Lo) | (((qh.X >> hs) & Hi) << 4)), lo1 = S6((ql.Y & Lo) | (((qh.Y >> hs) & Hi) << 4));
        uint lo2 = S6((ql.Z & Lo) | (((qh.Z >> hs) & Hi) << 4)), lo3 = S6((ql.W & Lo) | (((qh.W >> hs) & Hi) << 4));
        uint hi0 = S6(((ql.X >> 4) & Lo) | (((qh.X >> (hs + 4)) & Hi) << 4)), hi1 = S6(((ql.Y >> 4) & Lo) | (((qh.Y >> (hs + 4)) & Hi) << 4));
        uint hi2 = S6(((ql.Z >> 4) & Lo) | (((qh.Z >> (hs + 4)) & Hi) << 4)), hi3 = S6(((ql.W >> 4) & Lo) | (((qh.W >> (hs + 4)) & Hi) << 4));
        int blk = 8 * b + 4 * hf + (r >> 1), half = r & 1;   // q8 block and half of the low weights; the high ones are 2 blocks on
        for (int t = 0; t < nt; t++)
        {
            long xo = ((xq0 + (long)t * xqs) >> 2) + 2L * blk + half;
            int sLo = Kernels.Dot16S(lo0, lo1, lo2, lo3, xq4[xo]), sHi = Kernels.Dot16S(hi0, hi1, hi2, hi3, xq4[xo + 4]);
            long xb = xd0 + (long)t * xds + blk;
            Kernels.AddTok(ref a, t, ds0 * xd[xb].X * sLo + ds1 * xd[xb + 2].X * sHi);   // Σ d·sc·(q − 32)·x
        }
    }

    /// <summary>Bytes q (0..63) to signed bytes q − 32 without borrows: (q + 128 − 32) per byte, then back by 128.</summary>
    private static uint S6(uint q) => ((q | 0x80808080u) - 0x20202020u) ^ 0x80808080u;

    private const uint Lo = 0x0F0F0F0Fu, Hi = 0x03030303u;
}
