using LammaNet.Gguf;
using LammaNet.Tensors;

namespace LammaNet.Tests;

public unsafe class KernelTests
{
    private static float RefDot(WeightTensor w, int row, float[] x)
    {
        var tmp = new float[w.Cols];
        fixed (float* t = tmp) w.DequantizeRow(row, t);
        double s = 0;
        for (int i = 0; i < tmp.Length; i++) s += tmp[i] * (double)x[i];
        return (float)s;
    }

    public static IEnumerable<object[]> TensorCases()
    {
        yield return [TestModels.Qwen3Q4, "blk.0.attn_q.weight"];      // Q4_K
        yield return [TestModels.Qwen3Q4, "blk.0.ffn_down.weight"];    // Q6_K
        yield return [TestModels.Qwen3Q8, "blk.0.attn_q.weight"];      // Q8_0
        yield return [TestModels.Qwen35, "blk.0.attn_qkv.weight"];     // Q5_K
        yield return [TestModels.Qwen35, "blk.0.ssm_alpha.weight"];    // Q8_0
        yield return [TestModels.Gemma4, "per_layer_model_proj.weight"]; // BF16
        yield return [TestModels.Qwen3Q4, "token_embd.weight"];        // Q6_K
    }

    [Theory]
    [MemberData(nameof(TensorCases))]
    public void DotMatchesDequantizedReference(string modelFile, string tensor)
    {
        var path = TestModels.Find(modelFile);
        if (path == null) return;
        using var f = new GgufFile(path);
        var w = WeightTensor.FromGguf(f, f.GetTensor(tensor));
        var rnd = new Random(42);
        var x = new float[w.Cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1);
        var vt = QuantizeActivations.VecDotType(w.Type);
        var q = new byte[GgmlTypeInfo.RowSize(vt, w.Cols)];
        fixed (float* xp = x) fixed (byte* qp = q)
        {
            QuantizeActivations.Row(vt, xp, qp, w.Cols);
            for (int r = 0; r < Math.Min(w.Rows, 64); r++)
            {
                float expected = RefDot(w, r, x);
                bool saved = VecDot.UseAvx2;
                VecDot.UseAvx2 = false;
                float scalar = VecDot.Dot(w.Type, w.Row(r), qp, w.Cols);
                VecDot.UseAvx2 = saved;
                float simd = VecDot.Dot(w.Type, w.Row(r), qp, w.Cols);
                float tol = 0.02f * MathF.Max(1f, MathF.Abs(expected)) + 0.05f;
                Assert.True(MathF.Abs(simd - scalar) <= 1e-3f * MathF.Max(1, MathF.Abs(scalar)), $"{w.Type} row {r}: simd {simd} vs scalar {scalar}");
                Assert.True(MathF.Abs(scalar - expected) <= tol, $"{w.Type} row {r}: scalar {scalar} vs ref {expected}");
            }
        }
    }

    [Fact]
    public void HalfConversionMatchesSystemHalf()
    {
        var src = new ushort[65536];
        for (int i = 0; i < src.Length; i++) src[i] = (ushort)i;
        var dst = new float[src.Length];
        fixed (ushort* s = src) fixed (float* d = dst) HalfHelper.HalfToFloat(s, d, src.Length);
        for (int i = 0; i < src.Length; i++)
        {
            var h = BitConverter.UInt16BitsToHalf((ushort)i);
            if (Half.IsNaN(h) || Half.IsInfinity(h)) continue;
            Assert.Equal((float)h, dst[i]);
        }
    }
}

public unsafe class QuantizerTests
{
    [Theory]
    [InlineData(1.0, false)]
    [InlineData(1.0, true)]
    [InlineData(300.0, true)]
    public void Q8KSimdMatchesScalar(double scale, bool outliers)
    {
        var rnd = new Random(7);
        int n = 1024;
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)((rnd.NextDouble() * 2 - 1) * scale);
        if (outliers) for (int i = 0; i < n; i += 97) x[i] *= 50;
        var a = new byte[GgmlTypeInfo.RowSize(GgmlType.Q8_K, n)];
        var b = new byte[a.Length];
        fixed (float* xp = x) fixed (byte* ap = a) fixed (byte* bp = b)
        {
            QuantizeActivations.Q8_K(xp, ap, n);
            bool saved = System.Runtime.Intrinsics.X86.Avx2.IsSupported;
            QuantizeActivations.ForceScalar = true;
            QuantizeActivations.Q8_K(xp, bp, n);
            QuantizeActivations.ForceScalar = false;
        }
        Assert.Equal(b, a);
    }
}

public unsafe class ExactDotTests
{
    [Theory]
    [MemberData(nameof(KernelTests.TensorCases), MemberType = typeof(KernelTests))]
    public void DotEqualsExactDotOfDequantizedOperands(string modelFile, string tensor)
    {
        var path = TestModels.Find(modelFile);
        if (path == null) return;
        using var f = new GgufFile(path);
        var w = WeightTensor.FromGguf(f, f.GetTensor(tensor));
        var vt = QuantizeActivations.VecDotType(w.Type);
        if (vt == GgmlType.F32) return;
        var rnd = new Random(11);
        var x = new float[w.Cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1) * (i % 37 == 0 ? 20 : 1);
        var q = new byte[GgmlTypeInfo.RowSize(vt, w.Cols)];
        var xq = new float[w.Cols];
        var wr = new float[w.Cols];
        double worst = 0;
        fixed (float* xp = x) fixed (byte* qp = q) fixed (float* xqp = xq) fixed (float* wrp = wr)
        {
            QuantizeActivations.Row(vt, xp, qp, w.Cols);
            if (vt == GgmlType.Q8_1)
            {
                // Q8_1 dequant: d * q
                for (int b = 0; b < w.Cols / 32; b++)
                {
                    float d = HalfHelper.ToFloat(*(ushort*)(qp + b * 36));
                    for (int j = 0; j < 32; j++) xqp[b * 32 + j] = d * ((sbyte*)(qp + b * 36 + 4))[j];
                }
            }
            else Dequantize.Row(vt, qp, xqp, w.Cols);
            for (int r = 0; r < Math.Min(w.Rows, 256); r++)
            {
                w.DequantizeRow(r, wrp);
                double exact = 0, mag = 0;
                for (int i = 0; i < w.Cols; i++) { exact += (double)wrp[i] * xqp[i]; mag += Math.Abs((double)wrp[i] * xqp[i]); }
                float got = VecDot.Dot(w.Type, w.Row(r), qp, w.Cols);
                double rel = Math.Abs(got - exact) / Math.Max(1e-6, mag);
                worst = Math.Max(worst, rel);
                Assert.True(rel < 2e-4, $"{w.Type} row {r}: got {got} exact {exact} (rel {rel:E2} of magnitude {mag:F3})");
            }
        }
    }
}

public unsafe class MultiDotTests
{
    [Theory]
    [MemberData(nameof(KernelTests.TensorCases), MemberType = typeof(KernelTests))]
    public void MultiTokenKernelsMatchSingleToken(string modelFile, string tensor)
    {
        var path = TestModels.Find(modelFile);
        if (path == null) return;
        using var f = new GgufFile(path);
        var w = WeightTensor.FromGguf(f, f.GetTensor(tensor));
        if (!VecDotMulti.Supports(w.Type)) return;
        var vt = QuantizeActivations.VecDotType(w.Type);
        long qrow = GgmlTypeInfo.RowSize(vt, w.Cols);
        var rnd = new Random(3);
        var x = new float[4 * w.Cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1);
        var q = new byte[4 * qrow];
        var res = new float[4];
        fixed (float* xp = x) fixed (byte* qp = q) fixed (float* rp = res)
        {
            for (int t = 0; t < 4; t++) QuantizeActivations.Row(vt, xp + t * w.Cols, qp + t * qrow, w.Cols);
            for (int r = 0; r < 16; r++)
                for (int nt = 1; nt <= 4; nt++)
                {
                    VecDotMulti.Dot(w.Type, w.Row(r), qp, qrow, nt, w.Cols, rp);
                    for (int t = 0; t < nt; t++)
                    {
                        float single = VecDot.Dot(w.Type, w.Row(r), qp + t * qrow, w.Cols);
                        Assert.True(MathF.Abs(single - res[t]) <= 1e-4f * MathF.Max(1, MathF.Abs(single)), $"{w.Type} r{r} nt{nt} t{t}: {res[t]} vs {single}");
                    }
                }
        }
    }
}
