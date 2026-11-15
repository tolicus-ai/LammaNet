using ILGPU;
using ILGPU.Runtime;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

/// <summary>One weight matrix of a (multi-matrix) matvec launch.</summary>
public struct MatDesc
{
    /// <summary>Offset of row 0 in 32-bit words of the weight buffer.</summary>
    public long WWord;
    public int Rows, RowWords, YStride;
    /// <summary>Output offset (floats) of token 0, row 0.</summary>
    public long YOff;
    /// <summary>Bias offset in the float-parameter buffer, or -1.</summary>
    public long BiasOff;
}

/// <summary>
/// Parameters of <see cref="Kernels.MatVec{TA, TB, TC}"/>: up to three matrices sharing the same input (e.g. Q, K, V) in
/// one launch.
/// </summary>
public struct MatVecParams
{
    public MatDesc M0, M1, M2;
    public int NMat, Cols, XStride;
    /// <summary>Token tiles in this launch (each of the specialized tile size), starting at token 0 of x / y.</summary>
    public int Tiles;
    /// <summary>Valid tokens; the last tile may be partial (its extra rows are computed from padding and not stored).</summary>
    public int NTok;
    public long XOff;
    /// <summary>
    /// 0 = store, 1 = accumulate (y += Scale·r), 2 = gated pair: y = act(M0·x) * (M1·x) (rows of M0),
    /// 3 = gated input: y = act(r) * f[MulOff + t·MulStride + row] (the float view passed as f holds the multipliers).
    /// </summary>
    public int Mode;
    public long MulOff;
    public int MulStride;
    /// <summary>Pair activation: 0 = SiLU, 1 = GELU (tanh).</summary>
    public int Act;
    public float Scale, SoftCap;
    /// <summary>Rows covered by the grid: Σ rows (modes 0/1) or M0.Rows (mode 2).</summary>
    public int TotalRows;
}

/// <summary>Accumulators for up to <see cref="Kernels.MatVecTokens"/> tokens.</summary>
public struct Acc8 { public float A0, A1, A2, A3, A4, A5, A6, A7; }

/// <summary>Four 32-bit words loaded as one 16-byte vector.</summary>
public struct UInt4 { public uint X, Y, Z, W; }

/// <summary>Raw memory of one weight unit (up to 64 bytes plus one scale word).</summary>
public struct UnitData { public UInt4 A, B, C, D; public uint S; }

/// <summary>Eight dequantized weights.</summary>
public struct Float8 { public float W0, W1, W2, W3, W4, W5, W6, W7; }

/// <summary>
/// A device weight format. ILGPU specializes the matvec kernel per format (struct type argument), so each
/// instantiation contains only the code of the formats it uses.
/// </summary>
public interface IWeightFormat
{
    /// <summary>Weights per unit.</summary>
    int Weights { get; }
    /// <summary>Reads unit <paramref name="u"/> of the row starting at word <paramref name="row"/>.</summary>
    UnitData Load(ArrayView<uint> w, long row, int cols, int u);
    /// <summary>Adds the dot products of a loaded unit with the token tile to the accumulators.</summary>
    void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a);
    /// <summary>Dequantizes the 8 consecutive weights starting at column <paramref name="k0"/> (a multiple of 8).</summary>
    Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0);
}

internal static partial class Kernels
{
    /// <summary>Largest token tile of the matvec kernel (one accumulator per token in registers).</summary>
    public const int MatVecTokens = 8;

    /// <summary>
    /// y = W·x for a small number of tokens, dequantizing weights on the fly. One warp per output row × token tile;
    /// each lane handles 32 weights per step with 128-bit loads, then the warp reduces with shuffles.
    /// The tile size <paramref name="ntSpec"/> is a JIT-time constant.
    /// </summary>
    public static void MatVec<TA, TB, TC>(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p, SpecializedValue<int> ntSpec)
        where TA : struct, IWeightFormat where TB : struct, IWeightFormat where TC : struct, IWeightFormat =>
        MatVecBody<TA, TB, TC>(w, f, x, y, p, ntSpec.Value);

    /// <summary>Body of <see cref="MatVec{TA, TB, TC}"/> (separate so probes can instantiate it with a constant tile).</summary>
    internal static void MatVecBody<TA, TB, TC>(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p, int nt)
        where TA : struct, IWeightFormat where TB : struct, IWeightFormat where TC : struct, IWeightFormat
    {
        int wi = WarpIndex, lane = Lane;
        if (wi >= p.TotalRows * p.Tiles) return;
        int r = wi % p.TotalRows, tile = wi / p.TotalRows;
        int t0 = tile * nt;
        var x4 = x.Cast<Float4>();
        long xo4 = (p.XOff + (long)t0 * p.XStride) >> 2;
        int xs4 = p.XStride >> 2;
        var a = new Acc8();
        var b = new Acc8();
        var m = p.M0;
        int rr = r;
        if (p.Mode == 2)
        {
            RowDotPair<TA, TB>(w, p.M0.WWord + (long)rr * p.M0.RowWords, p.M1.WWord + (long)rr * p.M1.RowWords, p.Cols, x4, xo4, xs4, nt, lane, ref a, ref b);
            Reduce(ref b, nt);
        }
        else if (rr < p.M0.Rows || p.NMat == 1)
            RowDot<TA>(w, m.WWord + (long)rr * m.RowWords, p.Cols, x4, xo4, xs4, nt, lane, ref a);
        else if (rr < p.M0.Rows + p.M1.Rows || p.NMat == 2)
        {
            rr -= p.M0.Rows; m = p.M1;
            RowDot<TB>(w, m.WWord + (long)rr * m.RowWords, p.Cols, x4, xo4, xs4, nt, lane, ref a);
        }
        else
        {
            rr -= p.M0.Rows + p.M1.Rows; m = p.M2;
            RowDot<TC>(w, m.WWord + (long)rr * m.RowWords, p.Cols, x4, xo4, xs4, nt, lane, ref a);
        }
        Reduce(ref a, nt);
        if (lane >= nt || t0 + lane >= p.NTok) return;
        StoreRow(ref a, ref b, f, y, p, m, rr, t0, lane);
    }

    /// <summary>Epilogue of a matvec row for token t0 + lane: bias, gated pair / multiplier, scale, soft cap, store or accumulate.</summary>
    private static void StoreRow(ref Acc8 a, ref Acc8 b, ArrayView<float> f, ArrayView<float> y, MatVecParams p, MatDesc m, int rr, int t0, int lane)
    {
        float v = Pick(ref a, lane);
        if (m.BiasOff >= 0) v += f[m.BiasOff + rr];
        if (p.Mode == 2)
        {
            float u = Pick(ref b, lane);
            if (p.M1.BiasOff >= 0) u += f[p.M1.BiasOff + rr];
            v = (p.Act == 1 ? GeluTanh(v) : Silu(v)) * u;
        }
        else if (p.Mode == 3) v = (p.Act == 1 ? GeluTanh(v) : Silu(v)) * f[p.MulOff + (long)(t0 + lane) * p.MulStride + rr];
        v *= p.Scale;
        if (p.SoftCap > 0) v = Tanh(v / p.SoftCap) * p.SoftCap;
        long yo = m.YOff + (long)(t0 + lane) * m.YStride + rr;
        if (p.Mode == 1) y[yo] = y[yo] + v;
        else y[yo] = v;
    }

    /// <summary>Lane's share of one row: units lane, lane+32, ...</summary>
    private static void RowDot<T>(ArrayView<uint> w, long row, int cols, ArrayView<Float4> x4, long xo4, int xs4, int nt, int lane, ref Acc8 a)
        where T : struct, IWeightFormat
    {
        var fmt = default(T);
        int units = cols / fmt.Weights;
        for (int u = lane; u < units; u += Warp) fmt.Dot(fmt.Load(w, row, cols, u), u, x4, xo4, xs4, nt, ref a);
    }

    /// <summary>Two rows (gate and up) with the same input, interleaved so both rows' loads are in flight together.</summary>
    private static void RowDotPair<TA, TB>(ArrayView<uint> w, long rowA, long rowB, int cols, ArrayView<Float4> x4, long xo4, int xs4, int nt, int lane, ref Acc8 a, ref Acc8 b)
        where TA : struct, IWeightFormat where TB : struct, IWeightFormat
    {
        var fa = default(TA);
        var fb = default(TB);
        int units = cols / fa.Weights;
        for (int u = lane; u < units; u += Warp)
        {
            var da = fa.Load(w, rowA, cols, u);
            var db = fb.Load(w, rowB, cols, u);
            fa.Dot(da, u, x4, xo4, xs4, nt, ref a);
            fb.Dot(db, u, x4, xo4, xs4, nt, ref b);
        }
    }

    private static void Reduce(ref Acc8 a, int nt)
    {
        a.A0 = WarpSum(a.A0);
        if (nt > 1) a.A1 = WarpSum(a.A1);
        if (nt > 2) a.A2 = WarpSum(a.A2);
        if (nt > 3) a.A3 = WarpSum(a.A3);
        if (nt > 4) a.A4 = WarpSum(a.A4);
        if (nt > 5) a.A5 = WarpSum(a.A5);
        if (nt > 6) a.A6 = WarpSum(a.A6);
        if (nt > 7) a.A7 = WarpSum(a.A7);
    }

    private static float Pick(ref Acc8 a, int i) => i switch
    {
        0 => a.A0, 1 => a.A1, 2 => a.A2, 3 => a.A3, 4 => a.A4, 5 => a.A5, 6 => a.A6, _ => a.A7,
    };

    /// <summary>acc[t] += w0..w3 · x[t, 4 consecutive columns] for each token of the tile.</summary>
    internal static void Fma4(ref Acc8 a, float w0, float w1, float w2, float w3, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        Float4 v = x4[o];
        a.A0 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
        if (nt < 2) return;
        v = x4[o + xs4]; a.A1 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
        if (nt < 3) return;
        v = x4[o + 2L * xs4]; a.A2 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
        if (nt < 4) return;
        v = x4[o + 3L * xs4]; a.A3 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
        if (nt < 5) return;
        v = x4[o + 4L * xs4]; a.A4 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
        if (nt < 6) return;
        v = x4[o + 5L * xs4]; a.A5 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
        if (nt < 7) return;
        v = x4[o + 6L * xs4]; a.A6 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
        if (nt < 8) return;
        v = x4[o + 7L * xs4]; a.A7 += w0 * v.X + w1 * v.Y + w2 * v.Z + w3 * v.W;
    }

    /// <summary>16 weights from 4-bit fields of 4 words (bits [sh, sh+4) of each byte): acc += Σ (ds·q - dm)·x.</summary>
    internal static void Run4Bit(ref Acc8 a, UInt4 q, int sh, float ds, float dm, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        Nib4(ref a, q.X >> sh, ds, dm, x4, o, xs4, nt);
        Nib4(ref a, q.Y >> sh, ds, dm, x4, o + 1, xs4, nt);
        Nib4(ref a, q.Z >> sh, ds, dm, x4, o + 2, xs4, nt);
        Nib4(ref a, q.W >> sh, ds, dm, x4, o + 3, xs4, nt);
    }

    /// <summary>
    /// Exact float of a small unsigned integer (&lt; 2^23) minus <paramref name="bias"/> via the mantissa of 2^23: an OR and a
    /// subtraction at full rate instead of a quarter-rate int→float conversion.
    /// </summary>
    internal static float U2F(uint v, float bias = 8388608f) => Interop.IntAsFloat(0x4B000000u | v) - bias;

    private static void Nib4(ref Acc8 a, uint q, float ds, float dm, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        uint v = q & 0x0F0F0F0Fu;
        Fma4(ref a, ds * U2F(v & 0xFF) - dm, ds * U2F((v >> 8) & 0xFF) - dm, ds * U2F((v >> 16) & 0xFF) - dm, ds * U2F(v >> 24) - dm, x4, o, xs4, nt);
    }

    /// <summary>16 Q5_K weights: low nibbles (shift sh) plus the high bit <paramref name="hb"/> of each qh byte.</summary>
    internal static void Run5Bit(ref Acc8 a, UInt4 q, UInt4 h, int sh, int hb, float ds, float dm, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        Nib5(ref a, q.X >> sh, h.X >> hb, ds, dm, x4, o, xs4, nt);
        Nib5(ref a, q.Y >> sh, h.Y >> hb, ds, dm, x4, o + 1, xs4, nt);
        Nib5(ref a, q.Z >> sh, h.Z >> hb, ds, dm, x4, o + 2, xs4, nt);
        Nib5(ref a, q.W >> sh, h.W >> hb, ds, dm, x4, o + 3, xs4, nt);
    }

    private static void Nib5(ref Acc8 a, uint q, uint h, float ds, float dm, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        uint v = (q & 0x0F0F0F0Fu) | ((h & 0x01010101u) << 4);
        Fma4(ref a, ds * U2F(v & 0xFF) - dm, ds * U2F((v >> 8) & 0xFF) - dm, ds * U2F((v >> 16) & 0xFF) - dm, ds * U2F(v >> 24) - dm, x4, o, xs4, nt);
    }

    /// <summary>16 Q6_K weights: 4 low bits from ql (shift ls), 2 high bits from qh (shift hs), offset -32.</summary>
    internal static void Run6Bit(ref Acc8 a, UInt4 l, UInt4 h, int ls, int hs, float ds, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        Nib6(ref a, l.X >> ls, h.X >> hs, ds, x4, o, xs4, nt);
        Nib6(ref a, l.Y >> ls, h.Y >> hs, ds, x4, o + 1, xs4, nt);
        Nib6(ref a, l.Z >> ls, h.Z >> hs, ds, x4, o + 2, xs4, nt);
        Nib6(ref a, l.W >> ls, h.W >> hs, ds, x4, o + 3, xs4, nt);
    }

    private static void Nib6(ref Acc8 a, uint l, uint h, float ds, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        uint v = (l & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4);
        const float b32 = 8388608f + 32f; // q - 32
        Fma4(ref a, ds * U2F(v & 0xFF, b32), ds * U2F((v >> 8) & 0xFF, b32), ds * U2F((v >> 16) & 0xFF, b32), ds * U2F(v >> 24, b32), x4, o, xs4, nt);
    }

    internal static void Int8x4(ref Acc8 a, uint q, float d, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        uint v = q ^ 0x80808080u; // int8 -> uint8 + 128
        const float b128 = 8388608f + 128f;
        Fma4(ref a, d * U2F(v & 0xFF, b128), d * U2F((v >> 8) & 0xFF, b128), d * U2F((v >> 16) & 0xFF, b128), d * U2F(v >> 24, b128), x4, o, xs4, nt);
    }

    private static int ByteOf(uint w1, uint w2, uint w3, int k)
    {
        uint word = k < 4 ? w1 : k < 8 ? w2 : w3;
        return (int)((word >> ((k & 3) * 8)) & 0xFF);
    }

    /// <summary>6-bit scale and min of sub-block <paramref name="j"/> from the 12 packed K-quant scale bytes.</summary>
    internal static void ScaleMinK4(int j, uint w1, uint w2, uint w3, out int sc, out int mn)
    {
        if (j < 4)
        {
            sc = ByteOf(w1, w2, w3, j) & 63;
            mn = ByteOf(w1, w2, w3, j + 4) & 63;
        }
        else
        {
            sc = (ByteOf(w1, w2, w3, j + 4) & 0xF) | ((ByteOf(w1, w2, w3, j - 4) >> 6) << 4);
            mn = (ByteOf(w1, w2, w3, j + 4) >> 4) | ((ByteOf(w1, w2, w3, j) >> 6) << 4);
        }
    }

    internal static int SignedByte(uint word, int i) => (int)(((word >> (8 * i)) & 0xFF) ^ 0x80) - 128;

    /// <summary>Eight weights ds·q - dm from the bytes of two words (q = the 8-bit fields after masking).</summary>
    internal static Float8 Affine8(uint lo, uint hi, float ds, float dm) => new()
    {
        W0 = ds * U2F(lo & 0xFF) - dm, W1 = ds * U2F((lo >> 8) & 0xFF) - dm, W2 = ds * U2F((lo >> 16) & 0xFF) - dm, W3 = ds * U2F(lo >> 24) - dm,
        W4 = ds * U2F(hi & 0xFF) - dm, W5 = ds * U2F((hi >> 8) & 0xFF) - dm, W6 = ds * U2F((hi >> 16) & 0xFF) - dm, W7 = ds * U2F(hi >> 24) - dm,
    };

    /// <summary>fp16 block scale <paramref name="b"/> of a row whose scale tail starts at word <paramref name="tail"/>.</summary>
    internal static float TailScale(ArrayView<uint> w, long tail, int b)
    {
        uint word = w[tail + (b >> 1)];
        return HalfBits((int)((b & 1) == 0 ? word & 0xFFFF : word >> 16));
    }
}

// ---------------------------------------------------------------------------------------------------- formats
// A unit u of a row is 32 weights (16 for fp32): for K-quants eight lanes share a 256-weight super-block, unit i of
// the block taking the 16-byte quant group i (two runs of 16 weights); for the other formats a unit is consecutive
// weights. Load issues the unit's memory reads; Dot dequantizes and accumulates, so callers can keep several units'
// loads in flight.

/// <summary>Q4_K: 144-byte blocks (d, dmin, 12 scale bytes, 128 bytes of nibbles). A = header, B = quants.</summary>
public partial struct FmtQ4K : IWeightFormat
{
    public int Weights => 32;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u)
    {
        var w4 = w.Cast<UInt4>();
        long b4 = (row >> 2) + (u >> 3) * 9L;
        return new UnitData { A = w4[b4], B = w4[b4 + 1 + (u & 7)] };
    }

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        int b = u >> 3, i = u & 7, c = i >> 1, part = i & 1;
        float dd = Kernels.HalfBits((int)(d.A.X & 0xFFFF)), dmin = Kernels.HalfBits((int)(d.A.X >> 16));
        Kernels.ScaleMinK4(2 * c, d.A.Y, d.A.Z, d.A.W, out int sc0, out int m0);
        Kernels.ScaleMinK4(2 * c + 1, d.A.Y, d.A.Z, d.A.W, out int sc1, out int m1);
        long o = xo4 + ((long)b * 256 + c * 64 + part * 16) / 4;
        Kernels.Run4Bit(ref a, d.B, 0, dd * sc0, dmin * m0, x4, o, xs4, nt);
        Kernels.Run4Bit(ref a, d.B, 4, dd * sc1, dmin * m1, x4, o + 8, xs4, nt);
    }

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        int b = k0 >> 8, within = k0 & 255, c = within >> 6, h = (within >> 5) & 1, l0 = within & 31;
        long bw = row + b * 36L;
        uint dm = w[bw];
        Kernels.ScaleMinK4(2 * c + h, w[bw + 1], w[bw + 2], w[bw + 3], out int sc, out int mn);
        long q = bw + 4 + (c * 32 + l0) / 4;
        int sh = 4 * h;
        return Kernels.Affine8((w[q] >> sh) & 0x0F0F0F0Fu, (w[q + 1] >> sh) & 0x0F0F0F0Fu,
            Kernels.HalfBits((int)(dm & 0xFFFF)) * sc, Kernels.HalfBits((int)(dm >> 16)) * mn);
    }
}

/// <summary>Q5_K: 176-byte blocks (d, dmin, 12 scale bytes, 32 bytes of high bits, 128 bytes of nibbles).</summary>
public struct FmtQ5K : IWeightFormat
{
    public int Weights => 32;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u)
    {
        var w4 = w.Cast<UInt4>();
        int i = u & 7;
        long b4 = (row >> 2) + (u >> 3) * 11L;
        return new UnitData { A = w4[b4], B = w4[b4 + 1 + (i & 1)], C = w4[b4 + 3 + i] };
    }

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        int b = u >> 3, i = u & 7, c = i >> 1, part = i & 1;
        float dd = Kernels.HalfBits((int)(d.A.X & 0xFFFF)), dmin = Kernels.HalfBits((int)(d.A.X >> 16));
        Kernels.ScaleMinK4(2 * c, d.A.Y, d.A.Z, d.A.W, out int sc0, out int m0);
        Kernels.ScaleMinK4(2 * c + 1, d.A.Y, d.A.Z, d.A.W, out int sc1, out int m1);
        long o = xo4 + ((long)b * 256 + c * 64 + part * 16) / 4;
        Kernels.Run5Bit(ref a, d.C, d.B, 0, 2 * c, dd * sc0, dmin * m0, x4, o, xs4, nt);
        Kernels.Run5Bit(ref a, d.C, d.B, 4, 2 * c + 1, dd * sc1, dmin * m1, x4, o + 8, xs4, nt);
    }

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        int b = k0 >> 8, within = k0 & 255, c = within >> 6, h = (within >> 5) & 1, l0 = within & 31, sub = 2 * c + h;
        long bw = row + b * 44L;
        uint dm = w[bw];
        Kernels.ScaleMinK4(sub, w[bw + 1], w[bw + 2], w[bw + 3], out int sc, out int mn);
        long q = bw + 12 + (c * 32 + l0) / 4, qh = bw + 4 + l0 / 4;
        int sh = 4 * h;
        return Kernels.Affine8(((w[q] >> sh) & 0x0F0F0F0Fu) | (((w[qh] >> sub) & 0x01010101u) << 4),
            ((w[q + 1] >> sh) & 0x0F0F0F0Fu) | (((w[qh + 1] >> sub) & 0x01010101u) << 4),
            Kernels.HalfBits((int)(dm & 0xFFFF)) * sc, Kernels.HalfBits((int)(dm >> 16)) * mn);
    }
}

/// <summary>Q6_K: 208-byte payloads (128 bytes low nibbles, 64 bytes high bits, 16 int8 scales) + fp16 scale tail.</summary>
public partial struct FmtQ6K : IWeightFormat
{
    public int Weights => 32;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u)
    {
        var w4 = w.Cast<UInt4>();
        int b = u >> 3, i = u & 7, hf = i >> 2, r = i & 3;
        long b4 = (row >> 2) + b * 13L;
        return new UnitData
        {
            A = w4[b4 + hf * 4 + r], B = w4[b4 + 8 + hf * 2 + (r & 1)], C = w4[b4 + 12],
            S = w[row + (long)(cols >> 8) * 52 + (b >> 1)],
        };
    }

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        int b = u >> 3, i = u & 7, hf = i >> 2, r = i & 3;
        float dd = Kernels.HalfBits((int)((b & 1) == 0 ? d.S & 0xFFFF : d.S >> 16));
        int si = hf * 8 + r;
        uint sw0 = si < 4 ? d.C.X : si < 8 ? d.C.Y : si < 12 ? d.C.Z : d.C.W;
        uint sw1 = si + 4 < 8 ? d.C.Y : si + 4 < 12 ? d.C.Z : d.C.W;
        float ds0 = dd * Kernels.SignedByte(sw0, si & 3), ds1 = dd * Kernels.SignedByte(sw1, (si + 4) & 3);
        long o = xo4 + ((long)b * 256 + hf * 128 + r * 16) / 4;
        int hs = 2 * (r >> 1);
        Kernels.Run6Bit(ref a, d.A, d.B, 0, hs, ds0, x4, o, xs4, nt);
        Kernels.Run6Bit(ref a, d.A, d.B, 4, hs + 4, ds1, x4, o + 16, xs4, nt);
    }

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        int b = k0 >> 8, within = k0 & 255, hf = within >> 7, w2 = within & 127, q4 = w2 >> 5, l0 = w2 & 31;
        long bw = row + b * 52L;
        long ql = bw + (hf * 64 + (q4 & 1) * 32 + l0) / 4, qh = bw + 32 + (hf * 32 + l0) / 4;
        int ls = (q4 >> 1) * 4, hs = 2 * q4, si = hf * 8 + q4 * 2 + l0 / 16;
        uint sw = w[bw + 48 + si / 4];
        float d = Kernels.TailScale(w, row + (long)(cols >> 8) * 52, b) * Kernels.SignedByte(sw, si & 3);
        return Kernels.Affine8(((w[ql] >> ls) & 0x0F0F0F0Fu) | (((w[qh] >> hs) & 0x03030303u) << 4),
            ((w[ql + 1] >> ls) & 0x0F0F0F0Fu) | (((w[qh + 1] >> hs) & 0x03030303u) << 4), d, 32 * d);
    }
}

/// <summary>Q8_0: 32-byte int8 payloads + fp16 scale tail.</summary>
public partial struct FmtQ80 : IWeightFormat
{
    public int Weights => 32;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u)
    {
        var w4 = w.Cast<UInt4>();
        long row4 = row >> 2;
        return new UnitData { A = w4[row4 + 2L * u], B = w4[row4 + 2L * u + 1], S = w[row + (long)(cols >> 5) * 8 + (u >> 1)] };
    }

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        float s = Kernels.HalfBits((int)((u & 1) == 0 ? d.S & 0xFFFF : d.S >> 16));
        long o = xo4 + u * 8L;
        Kernels.Int8x4(ref a, d.A.X, s, x4, o, xs4, nt);
        Kernels.Int8x4(ref a, d.A.Y, s, x4, o + 1, xs4, nt);
        Kernels.Int8x4(ref a, d.A.Z, s, x4, o + 2, xs4, nt);
        Kernels.Int8x4(ref a, d.A.W, s, x4, o + 3, xs4, nt);
        Kernels.Int8x4(ref a, d.B.X, s, x4, o + 4, xs4, nt);
        Kernels.Int8x4(ref a, d.B.Y, s, x4, o + 5, xs4, nt);
        Kernels.Int8x4(ref a, d.B.Z, s, x4, o + 6, xs4, nt);
        Kernels.Int8x4(ref a, d.B.W, s, x4, o + 7, xs4, nt);
    }

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        int b = k0 >> 5;
        long q = row + k0 / 4;
        float d = Kernels.TailScale(w, row + (long)(cols >> 5) * 8, b);
        return Kernels.Affine8(w[q] ^ 0x80808080u, w[q + 1] ^ 0x80808080u, d, 128 * d);
    }
}

/// <summary>Q4_0: 16-byte nibble payloads + fp16 scale tail.</summary>
public partial struct FmtQ40 : IWeightFormat
{
    public int Weights => 32;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u) =>
        new() { A = w.Cast<UInt4>()[(row >> 2) + u], S = w[row + (long)(cols >> 5) * 4 + (u >> 1)] };

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        float s = Kernels.HalfBits((int)((u & 1) == 0 ? d.S & 0xFFFF : d.S >> 16));
        long o = xo4 + u * 8L;
        Kernels.Run4Bit(ref a, d.A, 0, s, 8 * s, x4, o, xs4, nt);
        Kernels.Run4Bit(ref a, d.A, 4, s, 8 * s, x4, o + 4, xs4, nt);
    }

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        int b = k0 >> 5, j0 = k0 & 31, sh = j0 >= 16 ? 4 : 0;
        long q = row + b * 4L + (j0 & 15) / 4;
        float d = Kernels.TailScale(w, row + (long)(cols >> 5) * 4, b);
        return Kernels.Affine8((w[q] >> sh) & 0x0F0F0F0Fu, (w[q + 1] >> sh) & 0x0F0F0F0Fu, d, 8 * d);
    }
}

/// <summary>fp16 weights (32 per unit).</summary>
public struct FmtF16 : IWeightFormat
{
    public int Weights => 32;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u)
    {
        var w4 = w.Cast<UInt4>();
        long b4 = (row >> 2) + 4L * u;
        return new UnitData { A = w4[b4], B = w4[b4 + 1], C = w4[b4 + 2], D = w4[b4 + 3] };
    }

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        long o = xo4 + u * 8L;
        H8(ref a, d.A, x4, o, xs4, nt);
        H8(ref a, d.B, x4, o + 2, xs4, nt);
        H8(ref a, d.C, x4, o + 4, xs4, nt);
        H8(ref a, d.D, x4, o + 6, xs4, nt);
    }

    private static void H8(ref Acc8 a, UInt4 v, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        Kernels.Fma4(ref a, Kernels.HalfBits((int)(v.X & 0xFFFF)), Kernels.HalfBits((int)(v.X >> 16)), Kernels.HalfBits((int)(v.Y & 0xFFFF)), Kernels.HalfBits((int)(v.Y >> 16)), x4, o, xs4, nt);
        Kernels.Fma4(ref a, Kernels.HalfBits((int)(v.Z & 0xFFFF)), Kernels.HalfBits((int)(v.Z >> 16)), Kernels.HalfBits((int)(v.W & 0xFFFF)), Kernels.HalfBits((int)(v.W >> 16)), x4, o + 1, xs4, nt);
    }

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        long q = row + k0 / 2;
        uint a = w[q], b = w[q + 1], c = w[q + 2], d = w[q + 3];
        return new Float8
        {
            W0 = Kernels.HalfBits((int)(a & 0xFFFF)), W1 = Kernels.HalfBits((int)(a >> 16)), W2 = Kernels.HalfBits((int)(b & 0xFFFF)), W3 = Kernels.HalfBits((int)(b >> 16)),
            W4 = Kernels.HalfBits((int)(c & 0xFFFF)), W5 = Kernels.HalfBits((int)(c >> 16)), W6 = Kernels.HalfBits((int)(d & 0xFFFF)), W7 = Kernels.HalfBits((int)(d >> 16)),
        };
    }
}

/// <summary>bf16 weights (32 per unit).</summary>
public struct FmtBF16 : IWeightFormat
{
    public int Weights => 32;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u) => default(FmtF16).Load(w, row, cols, u);

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        long o = xo4 + u * 8L;
        B8(ref a, d.A, x4, o, xs4, nt);
        B8(ref a, d.B, x4, o + 2, xs4, nt);
        B8(ref a, d.C, x4, o + 4, xs4, nt);
        B8(ref a, d.D, x4, o + 6, xs4, nt);
    }

    private static void B8(ref Acc8 a, UInt4 v, ArrayView<Float4> x4, long o, int xs4, int nt)
    {
        Kernels.Fma4(ref a, Interop.IntAsFloat(v.X << 16), Interop.IntAsFloat(v.X & 0xFFFF0000u), Interop.IntAsFloat(v.Y << 16), Interop.IntAsFloat(v.Y & 0xFFFF0000u), x4, o, xs4, nt);
        Kernels.Fma4(ref a, Interop.IntAsFloat(v.Z << 16), Interop.IntAsFloat(v.Z & 0xFFFF0000u), Interop.IntAsFloat(v.W << 16), Interop.IntAsFloat(v.W & 0xFFFF0000u), x4, o + 1, xs4, nt);
    }

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        long q = row + k0 / 2;
        uint a = w[q], b = w[q + 1], c = w[q + 2], d = w[q + 3];
        return new Float8
        {
            W0 = Interop.IntAsFloat(a << 16), W1 = Interop.IntAsFloat(a & 0xFFFF0000u), W2 = Interop.IntAsFloat(b << 16), W3 = Interop.IntAsFloat(b & 0xFFFF0000u),
            W4 = Interop.IntAsFloat(c << 16), W5 = Interop.IntAsFloat(c & 0xFFFF0000u), W6 = Interop.IntAsFloat(d << 16), W7 = Interop.IntAsFloat(d & 0xFFFF0000u),
        };
    }
}

/// <summary>fp32 weights (16 per unit).</summary>
public struct FmtF32 : IWeightFormat
{
    public int Weights => 16;

    public UnitData Load(ArrayView<uint> w, long row, int cols, int u)
    {
        var w4 = w.Cast<UInt4>();
        long b4 = (row >> 2) + 4L * u;
        return new UnitData { A = w4[b4], B = w4[b4 + 1], C = w4[b4 + 2], D = w4[b4 + 3] };
    }

    public void Dot(UnitData d, int u, ArrayView<Float4> x4, long xo4, int xs4, int nt, ref Acc8 a)
    {
        long o = xo4 + u * 4L;
        F4(ref a, d.A, x4, o, xs4, nt);
        F4(ref a, d.B, x4, o + 1, xs4, nt);
        F4(ref a, d.C, x4, o + 2, xs4, nt);
        F4(ref a, d.D, x4, o + 3, xs4, nt);
    }

    private static void F4(ref Acc8 a, UInt4 v, ArrayView<Float4> x4, long o, int xs4, int nt) =>
        Kernels.Fma4(ref a, Interop.IntAsFloat(v.X), Interop.IntAsFloat(v.Y), Interop.IntAsFloat(v.Z), Interop.IntAsFloat(v.W), x4, o, xs4, nt);

    public Float8 Dequant8(ArrayView<uint> w, long row, int cols, int k0)
    {
        long q = row + k0;
        return new Float8
        {
            W0 = Interop.IntAsFloat(w[q]), W1 = Interop.IntAsFloat(w[q + 1]), W2 = Interop.IntAsFloat(w[q + 2]), W3 = Interop.IntAsFloat(w[q + 3]),
            W4 = Interop.IntAsFloat(w[q + 4]), W5 = Interop.IntAsFloat(w[q + 5]), W6 = Interop.IntAsFloat(w[q + 6]), W7 = Interop.IntAsFloat(w[q + 7]),
        };
    }
}
