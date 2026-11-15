using LammaNet.Compute;
using LammaNet.Gguf;
using LammaNet.Tensors;

namespace LammaNet.Multimodal;

/// <summary>A linear layer with optional bias and Gemma-4 style input/output clamping ("ClippableLinear").</summary>
internal sealed class Linear
{
    public required WeightTensor W { get; init; }
    public float[]? Bias { get; init; }
    public float InMin = float.NegativeInfinity, InMax = float.PositiveInfinity;
    public float OutMin = float.NegativeInfinity, OutMax = float.PositiveInfinity;
    public int In => W.Cols;
    public int Out => W.Rows;

    public static Linear? Load(GgufFile f, string baseName)
    {
        if (!f.TryGetTensor(baseName + ".weight", out var wi)) return null;
        float S(string suffix, float def) => f.TryGetTensor(baseName + suffix, out var t) ? WeightTensor.FromGguf(f, t).ToFloatArray()[0] : def;
        return new Linear
        {
            W = WeightTensor.FromGguf(f, wi),
            Bias = f.TryGetTensor(baseName + ".bias", out var bi) ? WeightTensor.FromGguf(f, bi).ToFloatArray() : null,
            InMin = S(".input_min", float.NegativeInfinity),
            InMax = S(".input_max", float.PositiveInfinity),
            OutMin = S(".output_min", float.NegativeInfinity),
            OutMax = S(".output_max", float.PositiveInfinity),
        };
    }

    /// <summary>y[n, Out] = clamp(clamp(x[n, In]) · Wᵀ + b).</summary>
    public unsafe void Forward(MatMul mm, float* x, int n, float* y)
    {
        bool clampIn = !float.IsNegativeInfinity(InMin) || !float.IsPositiveInfinity(InMax);
        if (clampIn)
        {
            // clamp into a scratch copy so the caller's activations are not modified
            using var tmp = new NativeBuffer((long)n * In * 4, zero: false);
            EncoderOps.ClampRows(mm.Pool, x, tmp.F, n, In, InMin, InMax);
            mm.Multiply(W, tmp.F, n, y, In, Out);
        }
        else mm.Multiply(W, x, n, y, In, Out);
        if (Bias != null)
            fixed (float* b = Bias)
            {
                float* bp = b;
                int o = Out;
                EncoderOps.For(mm.Pool, n, r => Ops.Add(y + (long)r * o, bp, y + (long)r * o, o));
            }
        if (!float.IsNegativeInfinity(OutMin) || !float.IsPositiveInfinity(OutMax))
            EncoderOps.ClampRows(mm.Pool, y, y, n, Out, OutMin, OutMax);
    }
}

/// <summary>Row-wise helpers used by the encoders.</summary>
internal static unsafe class EncoderOps
{
    /// <summary>dst = clamp(src, lo, hi) row by row (vectorized, on the pool); src may equal dst.</summary>
    public static void ClampRows(ComputePool? pool, float* src, float* dst, int rows, int cols, float lo, float hi) =>
        For(pool, rows, r => System.Numerics.Tensors.TensorPrimitives.Clamp(
            new ReadOnlySpan<float>(src + (long)r * cols, cols), lo, hi, new Span<float>(dst + (long)r * cols, cols)));

    public static void RmsNormRows(float* x, float[]? w, float* y, int rows, int d, float eps, ComputePool? pool = null)
    {
        fixed (float* wp = w)
        {
            float* wpp = wp;
            For(pool, rows, r => Ops.RmsNorm(x + (long)r * d, wpp, y + (long)r * d, d, eps));
        }
    }

    /// <summary>
    /// Runs <paramref name="body"/> for 0..count-1 on <paramref name="pool"/> (or the TPL without one). Encoders that
    /// call MatMul must keep their row loops on the same pool: mixing in Parallel.For makes the two thread sets compete.
    /// </summary>
    public static void For(ComputePool? pool, int count, Action<int> body)
    {
        if (pool == null) { Parallel.For(0, count, body); return; }
        pool.Run(new ActionJob { Body = (i, _) => body(i) }, count);
    }

    public static void LayerNormRows(float* x, float[] w, float[] b, float* y, int rows, int d, float eps, ComputePool? pool = null)
    {
        fixed (float* wp = w, bp = b)
        {
            float* wpp = wp, bpp = bp;
            For(pool, rows, r => Ops.LayerNorm(x + (long)r * d, wpp, bpp, y + (long)r * d, d, eps));
        }
    }

    public static float[]? F(GgufFile f, string name) =>
        f.TryGetTensor(name, out var t) ? WeightTensor.FromGguf(f, t).ToFloatArray() : null;

    /// <summary>
    /// Full (bidirectional) multi-head attention: q,k,v [n, heads*dh] → out [n, heads*dh]. With a <paramref name="mm"/>
    /// and enough positions, each head runs as two GEMMs (S = Q·Kᵀ, softmax, O = P·V) over gathered K / Vᵀ panels.
    /// </summary>
    public static void Attention(float* q, float* k, float* v, float* o, int n, int heads, int dh, float scale, MatMul? mm = null)
    {
        if (mm != null && n >= 64) { AttentionGemm(mm, q, k, v, o, n, heads, dh, scale); return; }
        int stride = heads * dh;
        Parallel.For(0, n * heads, () => new float[n], (task, _, scores) =>
        {
            int h = task % heads, i = task / heads;
            float* qi = q + (long)i * stride + h * dh;
            fixed (float* s = scores)
            {
                for (int j = 0; j < n; j++) s[j] = Simd.Dot(qi, k + (long)j * stride + h * dh, dh) * scale;
                Ops.Softmax(s, n);
                float* oi = o + (long)i * stride + h * dh;
                new Span<float>(oi, dh).Clear();
                for (int j = 0; j < n; j++) Simd.AddScaled(v + (long)j * stride + h * dh, s[j], oi, dh);
            }
            return scores;
        }, _ => { });
    }

    private static void AttentionGemm(MatMul mm, float* q, float* k, float* v, float* o, int n, int heads, int dh, float scale)
    {
        int stride = heads * dh;
        var pool = mm.Pool;
        using var qs = new NativeBuffer((long)n * stride * 4, zero: false);  // q · scale (folds the score scale into the GEMM)
        using var kh = new NativeBuffer((long)n * dh * 4, zero: false);      // K of one head [n, dh]
        using var vt = new NativeBuffer((long)dh * n * 4, zero: false);      // Vᵀ of one head [dh, n]
        const int QueryBlock = 512;                                          // bounds the score buffer for large images
        int qb = Math.Min(n, QueryBlock);
        using var s = new NativeBuffer((long)qb * n * 4, zero: false);       // scores / probabilities [qb, n]
        var kw = new WeightTensor("attn.k", GgmlType.F32, [dh, n], kh.B);
        var vw = new WeightTensor("attn.vt", GgmlType.F32, [n, dh], vt.B);
        float* qp = qs.F, kp = kh.F, vp = vt.F, sp = s.F;
        For(pool, n, i => Simd.Scale(q + (long)i * stride, scale, qp + (long)i * stride, stride));
        // gather tasks: Vᵀ rows (contiguous writes, no shared cache lines between tasks) and chunks of K rows
        int tasks = Math.Max(dh, 32), kChunk = (n + tasks - 1) / tasks;
        for (int h = 0; h < heads; h++)
        {
            int ho = h * dh;
            using (Profiler.Section("attn.gather"))
            For(pool, tasks, t =>
            {
                if (t < dh)
                {
                    float* dst = vp + (long)t * n, src = v + ho + t;
                    for (int j = 0; j < n; j++) dst[j] = src[(long)j * stride];
                }
                for (int j = t * kChunk, j1 = Math.Min(n, j + kChunk); j < j1; j++)
                    Buffer.MemoryCopy(k + (long)j * stride + ho, kp + (long)j * dh, dh * 4L, dh * 4L);
            });
            for (int q0 = 0; q0 < n; q0 += qb)
            {
                int rows = Math.Min(qb, n - q0);
                mm.Multiply(kw, qp + (long)q0 * stride + ho, rows, sp, stride, n);
                using (Profiler.Section("attn.softmax"))
                For(pool, rows, i => Ops.Softmax(sp + (long)i * n, n));
                mm.Multiply(vw, sp, rows, o + (long)q0 * stride + ho, n, stride);
            }
        }
    }
}
