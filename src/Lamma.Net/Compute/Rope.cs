using System.Runtime.Intrinsics;
namespace LammaNet.Compute;

/// <summary>Rotary embedding variants (ggml rope modes).</summary>
public enum RopeMode
{
    /// <summary>Rotate adjacent pairs (x[2i], x[2i+1]) — Llama / Granite.</summary>
    Normal = 0,
    /// <summary>Rotate (x[i], x[i + n/2]) — GPT-NeoX / Qwen / Gemma.</summary>
    Neox = 2,
    /// <summary>Multi-section RoPE (Qwen2-VL), neox rotation with per-section positions.</summary>
    MRope = 8,
    /// <summary>Interleaved multi-section RoPE (Qwen3-VL / Qwen3.5).</summary>
    IMRope = 40,
    /// <summary>2D vision rope (each half rotated with its own position).</summary>
    Vision = 24,
}

/// <summary>Immutable RoPE configuration for one family of layers.</summary>
public sealed class RopeConfig
{
    public RopeMode Mode { get; init; } = RopeMode.Neox;
    /// <summary>Number of rotated dimensions (n_rot).</summary>
    public int Dims { get; init; }
    public float FreqBase { get; init; } = 10000f;
    public float FreqScale { get; init; } = 1f;
    public float[]? FreqFactors { get; init; }
    public float ExtFactor { get; init; }
    public float AttnFactor { get; init; } = 1f;
    public float BetaFast { get; init; } = 32f;
    public float BetaSlow { get; init; } = 1f;
    public int NCtxOrig { get; init; }
    /// <summary>M-RoPE sections (t, h, w, e).</summary>
    public int[] Sections { get; init; } = [0, 0, 0, 0];

    public bool IsMulti => Mode is RopeMode.MRope or RopeMode.IMRope;

    private float[]? _corr;

    private float[] CorrDims()
    {
        if (_corr != null) return _corr;
        static float CorrDim(int nDims, int nCtxOrig, float nRot, float b) =>
            nDims * MathF.Log(nCtxOrig / (nRot * 2 * MathF.PI)) / (2 * MathF.Log(b));
        float start = MathF.Floor(CorrDim(Dims, NCtxOrig, BetaFast, FreqBase));
        float end = MathF.Ceiling(CorrDim(Dims, NCtxOrig, BetaSlow, FreqBase));
        return _corr = [MathF.Max(0, start), MathF.Min(Dims - 1, end)];
    }

    private void Yarn(float thetaExtrap, int i0, float[] corr, out float cos, out float sin)
    {
        float thetaInterp = FreqScale * thetaExtrap;
        float theta = thetaInterp;
        float mscale = AttnFactor;
        if (ExtFactor != 0f)
        {
            float y = (i0 / 2 - corr[0]) / MathF.Max(0.001f, corr[1] - corr[0]);
            float rampMix = (1 - MathF.Min(1, MathF.Max(0, y))) * ExtFactor;
            theta = thetaInterp * (1 - rampMix) + thetaExtrap * rampMix;
            mscale *= 1.0f + 0.1f * MathF.Log(1.0f / FreqScale);
        }
        cos = MathF.Cos(theta) * mscale;
        sin = MathF.Sin(theta) * mscale;
    }

    /// <summary>
    /// Fills <paramref name="cache"/> (length <see cref="Dims"/>: interleaved cos,sin per pair) for one token.
    /// For multi-section modes <paramref name="pos"/> holds 4 position components (t, h, w, e).
    /// </summary>
    public unsafe void BuildCache(ReadOnlySpan<int> pos, float* cache)
    {
        var corr = ExtFactor != 0f ? CorrDims() : s_zeroCorr;
        float thetaScale = MathF.Pow(FreqBase, -2.0f / Dims);
        if (!IsMulti)
        {
            float theta = pos[0];
            for (int i0 = 0; i0 < Dims; i0 += 2)
            {
                float ff = FreqFactors != null ? FreqFactors[i0 / 2] : 1f;
                Yarn(theta / ff, i0, corr, out cache[i0], out cache[i0 + 1]);
                theta *= thetaScale;
            }
            return;
        }

        float tT = pos[0], tH = pos.Length > 1 ? pos[1] : pos[0], tW = pos.Length > 2 ? pos[2] : pos[0], tE = pos.Length > 3 ? pos[3] : 0;
        int[] s = Sections;
        int sectDims = s[0] + s[1] + s[2] + s[3];
        int secW = s[1] + s[0];
        bool im = Mode == RopeMode.IMRope;
        for (int i0 = 0; i0 < Dims; i0 += 2)
        {
            float ff = FreqFactors != null ? FreqFactors[i0 / 2] : 1f;
            int sector = (i0 / 2) % sectDims;
            float theta;
            if (im)
            {
                if (sector % 3 == 1 && sector < 3 * s[1]) theta = tH;
                else if (sector % 3 == 2 && sector < 3 * s[2]) theta = tW;
                else if (sector % 3 == 0 && sector < 3 * s[0]) theta = tT;
                else theta = tE;
            }
            else
            {
                theta = tT;
                if (sector >= s[0] && sector < secW) theta = tH;
                else if (sector >= secW && sector < secW + s[2]) theta = tW;
                else if (sector >= secW + s[2]) theta = tE;
            }
            Yarn(theta / ff, i0, corr, out cache[i0], out cache[i0 + 1]);
            tT *= thetaScale; tH *= thetaScale; tW *= thetaScale; tE *= thetaScale;
        }
    }

    private static readonly float[] s_zeroCorr = [0, 0];

    /// <summary>
    /// Rotation table (interleaved cos/sin per pair, like <see cref="BuildCache"/>) that moves rotated vectors by
    /// <paramref name="delta"/> positions of text: the per-pair angles of BuildCache with the magnitude scaling (YaRN
    /// attention factor) removed, so it can be applied to vectors that already carry it (KV-cache context shift).
    /// </summary>
    public unsafe float[] ShiftRotation(int delta)
    {
        var cache = new float[Dims];
        Span<int> p4 = [delta, delta, delta, 0];
        fixed (float* c = cache)
        {
            BuildCache(p4, c);
            for (int i = 0; i < Dims; i += 2)
            {
                float m = MathF.Sqrt(c[i] * c[i] + c[i + 1] * c[i + 1]);
                if (m > 0) { c[i] /= m; c[i + 1] /= m; }
                else { c[i] = 1; c[i + 1] = 0; }
            }
        }
        return cache;
    }

    /// <summary>Applies rotation in place to <paramref name="nHeads"/> heads of size <paramref name="headDim"/>.</summary>
    public unsafe void Apply(float* x, int nHeads, int headDim, float* cache)
    {
        int n = Dims;
        bool avx2 = System.Runtime.Intrinsics.X86.Avx2.IsSupported;
        for (int h = 0; h < nHeads; h++)
        {
            float* v = x + h * headDim;
            if (Mode == RopeMode.Normal)
            {
                int j = 0;
                if (avx2)
                    for (; j + 8 <= n; j += 8)
                    {
                        // pairs (a, b) interleaved like the table's (cos, sin): (a·c - b·s, b·c + a·s) = addsub(v·c, swap(v)·s)
                        var vv = System.Runtime.Intrinsics.X86.Avx.LoadVector256(v + j);
                        var cs = System.Runtime.Intrinsics.X86.Avx.LoadVector256(cache + j);
                        var c = System.Runtime.Intrinsics.X86.Avx.DuplicateEvenIndexed(cs);
                        var sn = System.Runtime.Intrinsics.X86.Avx.DuplicateOddIndexed(cs);
                        var sw = System.Runtime.Intrinsics.X86.Avx.Permute(vv, 0b10_11_00_01);
                        System.Runtime.Intrinsics.X86.Avx.Store(v + j, System.Runtime.Intrinsics.X86.Avx.AddSubtract(
                            System.Runtime.Intrinsics.X86.Avx.Multiply(vv, c), System.Runtime.Intrinsics.X86.Avx.Multiply(sw, sn)));
                    }
                for (int i0 = j; i0 < n; i0 += 2)
                {
                    float c = cache[i0], s = cache[i0 + 1];
                    float a = v[i0], b = v[i0 + 1];
                    v[i0] = a * c - b * s;
                    v[i0 + 1] = a * s + b * c;
                }
            }
            else
            {
                int half = n / 2, ic0 = 0;
                if (avx2)
                    for (; ic0 + 8 <= half; ic0 += 8)
                    {
                        // 8 pairs (i, i + half); the table interleaves (cos, sin): split it into 8 cos and 8 sin
                        var lo = System.Runtime.Intrinsics.X86.Avx.LoadVector256(cache + 2 * ic0);
                        var hi = System.Runtime.Intrinsics.X86.Avx.LoadVector256(cache + 2 * ic0 + 8);
                        var c = System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.X86.Avx.Shuffle(lo, hi, 0b10_00_10_00).AsDouble(), 0b11_01_10_00).AsSingle();
                        var sn = System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.X86.Avx.Shuffle(lo, hi, 0b11_01_11_01).AsDouble(), 0b11_01_10_00).AsSingle();
                        var a = System.Runtime.Intrinsics.X86.Avx.LoadVector256(v + ic0);
                        var b = System.Runtime.Intrinsics.X86.Avx.LoadVector256(v + ic0 + half);
                        System.Runtime.Intrinsics.X86.Avx.Store(v + ic0, System.Runtime.Intrinsics.X86.Avx.Subtract(
                            System.Runtime.Intrinsics.X86.Avx.Multiply(a, c), System.Runtime.Intrinsics.X86.Avx.Multiply(b, sn)));
                        System.Runtime.Intrinsics.X86.Avx.Store(v + ic0 + half, System.Runtime.Intrinsics.X86.Avx.Add(
                            System.Runtime.Intrinsics.X86.Avx.Multiply(a, sn), System.Runtime.Intrinsics.X86.Avx.Multiply(b, c)));
                    }
                for (int i0 = 2 * ic0; i0 < n; i0 += 2)
                {
                    int ic = i0 / 2;
                    float c = cache[i0], s = cache[i0 + 1];
                    float a = v[ic], b = v[ic + half];
                    v[ic] = a * c - b * s;
                    v[ic + half] = a * s + b * c;
                }
            }
        }
    }
}
