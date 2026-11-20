using System.Reflection;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.PTX;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using LammaNet.Gguf;
using LammaNet.Gpu;
using LammaNet.Tensors;
using Half = ILGPU.Half;

namespace LammaNet.Tests;

/// <summary>
/// GPU backend tests. Kernels run on a real CUDA GPU when one is present (LAMMA_TEST_GPU=cpu forces emulation),
/// otherwise on ILGPU's CPU emulation (32-lane warps), and are compared with the CPU reference; additionally every
/// kernel is compiled to CUDA PTX (and, with a GPU present, through the driver's PTX JIT).
/// </summary>
public unsafe class GpuKernelTests : IDisposable
{
    private readonly Context _ctx = Context.Create(b => b.Default().Math(MathMode.Fast32BitOnly).EnableAlgorithms());
    private readonly Accelerator _acc;

    public GpuKernelTests()
    {
        var cuda = Environment.GetEnvironmentVariable("LAMMA_TEST_GPU") == "cpu" ? null
            : _ctx.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        _acc = (cuda ?? new CPUDevice(Kernels.Warp, Kernels.WarpsPerGroup, 2)).CreateAccelerator(_ctx);
    }

    public void Dispose()
    {
        _acc.Dispose();
        _ctx.Dispose();
    }

    // ------------------------------------------------------------------ weights

    /// <summary>A synthetic weight tensor of the given type (random values, quantized on the host).</summary>
    private static WeightTensor Synthetic(GgmlType type, int rows, int cols, int seed)
    {
        var rnd = new Random(seed);
        long rowBytes = GgmlTypeInfo.RowSize(type, cols);
        var buf = new NativeBuffer(rowBytes * rows);
        var row = new float[cols];
        fixed (float* rp = row)
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++) row[c] = (float)(rnd.NextDouble() * 2 - 1) * (1 + r % 3);
                byte* dst = buf.B + r * rowBytes;
                if (type == GgmlType.Q4_0) QuantizeQ4_0(rp, dst, cols);
                else QuantizeActivations.Row(type, rp, dst, cols);
            }
        return new WeightTensor($"synthetic-{type}", type, [cols, rows], buf.B, buf);
    }

    private static void QuantizeQ4_0(float* x, byte* y, int n)
    {
        for (int b = 0; b < n / 32; b++)
        {
            float amax = 0, max = 0;
            for (int i = 0; i < 32; i++) { float v = x[b * 32 + i]; if (MathF.Abs(v) > amax) { amax = MathF.Abs(v); max = v; } }
            float d = max / -8, id = d != 0 ? 1 / d : 0;
            *(ushort*)(y + b * 18) = HalfHelper.FromFloat(d);
            for (int i = 0; i < 16; i++)
            {
                int q0 = Math.Min(15, (int)(x[b * 32 + i] * id + 8.5f)), q1 = Math.Min(15, (int)(x[b * 32 + 16 + i] * id + 8.5f));
                y[b * 18 + 2 + i] = (byte)(q0 | (q1 << 4));
            }
        }
    }

    private static WeightTensor ModelTensor(string modelFile, string tensor, out GgufFile file)
    {
        file = new GgufFile(TestModels.Find(modelFile)!);
        return WeightTensor.FromGguf(file, file.GetTensor(tensor));
    }

    private DMat Upload(WeightTensor w, out MemoryBuffer1D<uint, Stride1D.Dense> buf, int rows)
    {
        int type = DevFormat.For(w.Type, w.Cols);
        int rw = DevFormat.RowWords(type, w.Cols);
        var host = new uint[(long)rw * w.Rows];
        fixed (uint* hp = host) DevFormat.Pack(w, type, hp);
        buf = _acc.Allocate1D(host.AsSpan(0, rw * rows).ToArray());
        return new DMat { Buf = buf, Word = 0, Rows = rows, Cols = w.Cols, RowWords = rw, Type = type };
    }

    private static double[,] Reference(WeightTensor w, int rows, float[] x, int nTok)
    {
        var y = new double[nTok, rows];
        var row = new float[w.Cols];
        for (int r = 0; r < rows; r++)
        {
            fixed (float* rp = row) w.DequantizeRow(r, rp);
            for (int t = 0; t < nTok; t++)
            {
                double s = 0;
                for (int c = 0; c < w.Cols; c++) s += row[c] * (double)x[t * w.Cols + c];
                y[t, r] = s;
            }
        }
        return y;
    }

    private static float[] RandomX(int n, int seed)
    {
        var rnd = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(rnd.NextDouble() * 2 - 1)).ToArray();
    }

    /// <summary>x rounded as the int8 kernels round activations: blocks of 32, d = max|x| / 127, q = round-half-even(x / d).</summary>
    private static float[] RoundQ8(float[] x)
    {
        var xq = (float[])x.Clone();
        for (int b0 = 0; b0 < xq.Length; b0 += 32)
        {
            float amax = 0;
            for (int j = 0; j < 32; j++) amax = Math.Max(amax, Math.Abs(xq[b0 + j]));
            float d = amax / 127f, inv = amax > 0 ? 127f / amax : 0f;
            for (int j = 0; j < 32; j++) xq[b0 + j] = MathF.Round(xq[b0 + j] * inv) * d;
        }
        return xq;
    }

    private static void AssertClose(double expect, float actual, string what) =>
        Assert.True(Math.Abs(actual - expect) <= 1e-3 * Math.Max(1, Math.Abs(expect)), $"{what}: {actual} vs {expect}");

    public static IEnumerable<object[]> MatVecCases()
    {
        foreach (var n in new[] { 1, 3, 8, 11 })
        {
            yield return [TestModels.Qwen3Q4, "blk.0.attn_q.weight", n];   // Q4_K
            yield return [TestModels.Qwen3Q4, "blk.0.ffn_down.weight", n]; // Q6_K
            yield return [TestModels.Qwen3Q8, "blk.0.attn_q.weight", n];   // Q8_0
            yield return [TestModels.Qwen35, "blk.0.attn_qkv.weight", n];  // Q5_K
        }
        foreach (var t in new[] { "F32", "F16", "BF16", "Q8_0", "Q4_0" })
            yield return ["synthetic", t, 5];
    }

    /// <summary>Every device weight format, with token counts covering full and partial tiles.</summary>
    [Theory]
    [MemberData(nameof(MatVecCases))]
    public void MatVecMatchesDequantizedReference(string modelFile, string tensor, int nTok)
    {
        WeightTensor w;
        GgufFile? file = null;
        if (modelFile == "synthetic")
            w = Synthetic(Enum.Parse<GgmlType>(tensor), 24, 256, 7);
        else
        {
            if (TestModels.Find(modelFile) == null) return;
            w = ModelTensor(modelFile, tensor, out file);
        }
        using var _ = file;
        int rows = Math.Min(40, w.Rows);
        var m = Upload(w, out var wb, rows);
        using var __ = wb;
        var x = RandomX(nTok * w.Cols, 5);
        using var dx = _acc.Allocate1D(x);
        using var dy = _acc.Allocate1D<float>(nTok * rows);
        using var df = _acc.Allocate1D<float>(4);
        new MatVecLauncher(_acc).Run(_acc.DefaultStream, df.View, dx.View, 0, w.Cols, nTok, dy.View, 0, 1f, (m, 0, rows), null, null, 0, 0);
        _acc.Synchronize();
        var y = dy.GetAsArray1D();
        var expect = Reference(w, rows, x, nTok);
        for (int t = 0; t < nTok; t++)
            for (int r = 0; r < rows; r++) AssertClose(expect[t, r], y[t * rows + r], $"{w.Type} t{t} r{r}");
    }

    public static IEnumerable<object[]> Int8MatVecCases()
    {
        foreach (var n in new[] { 1, 3, 8, 11 })
        {
            yield return [TestModels.Qwen3Q4, "blk.0.attn_q.weight", n];   // Q4_K
            yield return [TestModels.Qwen3Q4, "blk.0.ffn_down.weight", n]; // Q6_K
            yield return [TestModels.Qwen3Q8, "blk.0.attn_q.weight", n];   // Q8_0
            foreach (var t in new[] { "Q8_0", "Q4_0" }) yield return ["synthetic", t, n];
        }
    }

    /// <summary>
    /// int8-activation (dp4a) matvec: exact (to float summation order) on activations that lie on the int8 grid (q·2^-7
    /// with a ±127 in every block, so the rounding step is unambiguous; with arbitrary values the device's fast division
    /// can round a value at an exact half step the other way), and batch-invariant: each token alone gives bit-identical
    /// rows, whatever tile or partial tile it was in. Needs CUDA (dp4a).
    /// </summary>
    [Theory]
    [MemberData(nameof(Int8MatVecCases))]
    public void Int8MatVecIsExactAndBatchInvariant(string modelFile, string tensor, int nTok)
    {
        if (_acc is not CudaAccelerator) return;
        WeightTensor w;
        GgufFile? file = null;
        if (modelFile == "synthetic")
            w = Synthetic(Enum.Parse<GgmlType>(tensor), 24, 512, 7);
        else
        {
            if (TestModels.Find(modelFile) == null) return;
            w = ModelTensor(modelFile, tensor, out file);
        }
        using var _ = file;
        int rows = Math.Min(40, w.Rows);
        var m = Upload(w, out var wb, rows);
        using var __ = wb;
        var rnd = new Random(5);
        var grid = Enumerable.Range(0, nTok * w.Cols).Select(i => (i % 32 == 0 ? (i & 32) == 0 ? 127 : -127 : rnd.Next(-127, 128)) / 128f).ToArray();
        using var dx = _acc.Allocate1D(grid);
        using var dy = _acc.Allocate1D<float>(nTok * rows);
        using var dy1 = _acc.Allocate1D<float>(rows);
        using var df = _acc.Allocate1D<float>(4);
        var mv = new MatVecLauncher(_acc) { Int8 = Int8Activations.All };
        mv.Reserve(nTok + Kernels.MatVecTokens, w.Cols);
        Assert.True(mv.UsesQ8(m.Type, m.Type, m.Type, w.Cols, 0, w.Cols, normedInput: false));
        mv.Run(_acc.DefaultStream, df.View, dx.View, 0, w.Cols, nTok, dy.View, 0, 1f, (m, 0, rows), null, null, 0, 0);
        _acc.Synchronize();
        var y = dy.GetAsArray1D();
        var expect = Reference(w, rows, grid, nTok);
        for (int t = 0; t < nTok; t++)
            for (int r = 0; r < rows; r++)
                Assert.True(Math.Abs(y[t * rows + r] - expect[t, r]) <= 1e-5 * Math.Max(1, Math.Abs(expect[t, r])), $"{w.Type} t{t} r{r}: {y[t * rows + r]} vs {expect[t, r]}");

        // batch invariance on arbitrary activations
        dx.View.CopyFromCPU(RandomX(nTok * w.Cols, 6));
        mv.Run(_acc.DefaultStream, df.View, dx.View, 0, w.Cols, nTok, dy.View, 0, 1f, (m, 0, rows), null, null, 0, 0);
        _acc.Synchronize();
        y = dy.GetAsArray1D();
        for (int t = 0; t < nTok; t++)
        {
            mv.Run(_acc.DefaultStream, df.View, dx.View, (long)t * w.Cols, w.Cols, 1, dy1.View, 0, 1f, (m, 0, rows), null, null, 0, 0);
            _acc.Synchronize();
            var y1 = dy1.GetAsArray1D();
            for (int r = 0; r < rows; r++)
                Assert.True(BitConverter.SingleToInt32Bits(y1[r]) == BitConverter.SingleToInt32Bits(y[t * rows + r]), $"{w.Type} t{t} r{r}: alone {y1[r]} vs in batch {y[t * rows + r]}");
        }
    }

    public static IEnumerable<object[]> GemmCases()
    {
        foreach (bool tc in new[] { false, true })
        {
            yield return [TestModels.Qwen3Q4, "blk.0.attn_q.weight", tc];   // Q4_K
            yield return [TestModels.Qwen3Q4, "blk.0.ffn_down.weight", tc]; // Q6_K
            yield return [TestModels.Qwen3Q8, "blk.0.attn_q.weight", tc];   // Q8_0
            yield return [TestModels.Qwen35, "blk.0.attn_qkv.weight", tc];  // Q5_K
            foreach (var t in new[] { "F32", "F16", "BF16", "Q4_0" }) yield return ["synthetic", t, tc];
        }
    }

    /// <summary>
    /// Prompt-processing GEMM (SIMT, tensor-core and pipelined tensor-core variants) for every weight format; 37 tokens
    /// x 70 rows exercise partial tiles, 300 x 260 several tiles in both directions. The tensor-core kernels round both
    /// operands to fp16, so the bound scales with Σ|w·x|.
    /// </summary>
    [Theory]
    [MemberData(nameof(GemmCases))]
    public void GemmMatchesDequantizedReference(string modelFile, string tensor, bool tensorCores)
    {
        foreach (bool pipelined in tensorCores ? [false, true] : new[] { false })
            foreach (var (n, rows) in new[] { (37, 70), (300, 260) })
                GemmCase(modelFile, tensor, tensorCores, pipelined, n, rows);
        // int8 activations (prompt numerics matching the int8 decode kernels): rounding applied (outliers), and the int8
        // tensor-core GEMM of Q4_K / Q4_0 / Q8_0 exact on activations on the int8 grid (partial and several tiles)
        if (tensorCores)
        {
            GemmCase(modelFile, tensor, true, true, 37, 70, Q8Input.Outliers);
            GemmCase(modelFile, tensor, true, true, 300, 260, Q8Input.Grid);
        }
    }

    private enum Q8Input { None, Outliers, Grid }

    private void GemmCase(string modelFile, string tensor, bool tensorCores, bool pipelined, int n, int rows, Q8Input q8 = Q8Input.None)
    {
        using var gemm = new GemmLauncher(_acc, tensorCores, pipelined);
        if (tensorCores && !gemm.TensorCores) return; // no sm_80+ device
        WeightTensor w;
        GgufFile? file = null;
        if (modelFile == "synthetic") w = Synthetic(Enum.Parse<GgmlType>(tensor), rows, 512, 11);
        else
        {
            if (TestModels.Find(modelFile) == null) return;
            w = ModelTensor(modelFile, tensor, out file);
        }
        using var _ = file;
        var m = Upload(w, out var wb, rows);
        using var __ = wb;
        if (q8 != Q8Input.None && (!MatVecLauncher.HasQ8(m.Type) || w.Cols % 256 != 0)) return;
        var x = RandomX(n * w.Cols, 3);
        // one outlier per block of 32: int8 rounding zeroes the rest, so only the rounded reference matches
        if (q8 == Q8Input.Outliers) for (int i = 0; i < x.Length; i += 32) x[i] = (i & 32) == 0 ? 100f : -100f;
        // q·2^-7 with a ±127 in every block: the rounding is the identity, so an int8 GEMM must match to float order
        var rnd = new Random(9);
        if (q8 == Q8Input.Grid) for (int i = 0; i < x.Length; i++) x[i] = (i % 32 == 0 ? (i & 32) == 0 ? 127 : -127 : rnd.Next(-127, 128)) / 128f;
        using var dx = _acc.Allocate1D(x);
        using var dy = _acc.Allocate1D<float>(n * rows);
        using var df = _acc.Allocate1D<float>(4);
        gemm.Run(_acc.DefaultStream, df.View, dx.View, 0, w.Cols, n, dy.View, 0, 1f, m, 0, rows, q8 != Q8Input.None);
        _acc.Synchronize();
        var y = dy.GetAsArray1D();
        if (q8 == Q8Input.Outliers) x = RoundQ8(x);
        bool exact = q8 == Q8Input.Grid && GemmLauncher.HasInt8Gemm(m.Type);
        var row = new float[w.Cols];
        for (int r = 0; r < rows; r++)
        {
            fixed (float* rp = row) w.DequantizeRow(r, rp);
            for (int t = 0; t < n; t++)
            {
                double e = 0, l1 = 0;
                for (int c = 0; c < w.Cols; c++) { double v = row[c] * (double)x[t * w.Cols + c]; e += v; l1 += Math.Abs(v); }
                double tol = (exact ? 1e-5 : tensorCores ? 2e-3 : 1e-5) * l1 + 1e-4;
                Assert.True(Math.Abs(y[t * rows + r] - e) <= tol, $"{w.Type} tc={tensorCores} pipelined={pipelined} q8={q8} n={n} r{r} t{t}: {y[t * rows + r]} vs {e}");
            }
        }
    }

    /// <summary>Three matrices of different formats in one launch (Q/K/V with a bias), and the gated pair mode (gate/up).</summary>
    [Fact]
    public void MultiMatrixAndGatedPair()
    {
        const int cols = 512, n = 3;
        var ws = new[] { Synthetic(GgmlType.Q8_0, 12, cols, 2), Synthetic(GgmlType.F16, 8, cols, 4), Synthetic(GgmlType.Q4_0, 16, cols, 3) };
        var types = ws.Select(w => DevFormat.For(w.Type, cols)).ToArray();
        var words = new long[4];
        for (int i = 0; i < 3; i++) words[i + 1] = words[i] + (long)DevFormat.RowWords(types[i], cols) * ws[i].Rows;
        var host = new uint[words[3]];
        fixed (uint* hp = host) for (int i = 0; i < 3; i++) DevFormat.Pack(ws[i], types[i], hp + words[i]);
        using var wb = _acc.Allocate1D(host);
        var mats = Enumerable.Range(0, 3).Select(i => new DMat { Buf = wb, Word = words[i], Rows = ws[i].Rows, Cols = cols, RowWords = DevFormat.RowWords(types[i], cols), Type = types[i] }).ToArray();
        var x = RandomX(n * cols, 9);
        using var dx = _acc.Allocate1D(x);
        var bias = Enumerable.Range(0, 16).Select(i => 0.25f * i).ToArray();
        using var df = _acc.Allocate1D(bias);
        mats[2].BiasOff = 0;
        var mv = new MatVecLauncher(_acc);
        // one output buffer: Q at 0 (stride 12), K at 100 (stride 8), V at 200 (stride 16)
        using var dy = _acc.Allocate1D<float>(400);
        mv.Run(_acc.DefaultStream, df.View, dx.View, 0, cols, n, dy.View, 0, 1f, (mats[0], 0, 12), (mats[1], 100, 8), (mats[2], 200, 16), 0, 0);
        _acc.Synchronize();
        var y = dy.GetAsArray1D();
        long[] offs = [0, 100, 200];
        for (int i = 0; i < 3; i++)
        {
            var e = Reference(ws[i], ws[i].Rows, x, n);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < ws[i].Rows; r++)
                    AssertClose(e[t, r] + (i == 2 ? bias[r] : 0), y[offs[i] + t * ws[i].Rows + r], $"mat{i} t{t} r{r}");
        }

        // gated pair: silu(gate·x) * (up·x + bias), gate = first 8 rows of F16, up = first 8 rows of Q4_0
        mv.Run(_acc.DefaultStream, df.View, dx.View, 0, cols, n, dy.View, 2, 1f, (mats[1], 0, 8), (mats[2], 0, 8), null, 0, 0);
        _acc.Synchronize();
        var yp = dy.GetAsArray1D();
        var eg = Reference(ws[1], 8, x, n);
        var eu = Reference(ws[2], 8, x, n);
        for (int t = 0; t < n; t++)
            for (int r = 0; r < 8; r++)
                AssertClose(eg[t, r] / (1 + Math.Exp(-eg[t, r])) * (eu[t, r] + bias[r]), yp[t * 8 + r], $"pair t{t} r{r}");
    }

    [Fact]
    public void RmsNormModes()
    {
        var rnd = new Random(9);
        int rows = 3, d = 96;
        var x = Enumerable.Range(0, rows * d).Select(_ => (float)rnd.NextDouble() - 0.5f).ToArray();
        var wt = Enumerable.Range(0, d).Select(_ => (float)rnd.NextDouble() + 0.5f).ToArray();
        var y0 = Enumerable.Range(0, rows * d).Select(_ => (float)rnd.NextDouble()).ToArray();
        using var dx = _acc.Allocate1D(x);
        using var dwt = _acc.Allocate1D(wt);
        using var dy = _acc.Allocate1D(y0);
        var k = _acc.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, NormParams>(Kernels.RmsNorm);
        foreach (int mode in new[] { 1, 0, 2 })
        {
            dy.CopyFromCPU(y0);
            float inScale = mode == 2 ? 0.25f : 1f;
            k(new KernelConfig(rows, Kernels.GroupSize), dwt.View, dx.View, dy.View, new NormParams { Rows = rows, D = d, XStride = d, YStride = d, WOff = 0, Eps = 1e-6f, Mode = mode, Scale = 0.5f, InScale = inScale });
            _acc.Synchronize();
            var y = dy.GetAsArray1D();
            for (int r = 0; r < rows; r++)
            {
                double ss = 0;
                for (int i = 0; i < d; i++) ss += x[r * d + i] * inScale * x[r * d + i] * inScale;
                double sc = 1 / Math.Sqrt(ss / d + 1e-6);
                for (int i = 0; i < d; i++)
                {
                    double v = x[r * d + i] * inScale * sc * wt[i];
                    double expect = mode switch { 1 => y0[r * d + i] + 0.5 * v, 2 => (y0[r * d + i] + v) * 0.5, _ => v };
                    Assert.Equal(expect, y[r * d + i], 4);
                }
            }
        }
    }

    /// <summary>
    /// Paged fp16 attention: GQA, a sliding-window token, several chunks (flash-decoding + combine), gating and soft-cap.
    /// </summary>
    [Theory]
    [InlineData(64, 1, false, 0f)]
    [InlineData(128, 3, true, 0f)]
    [InlineData(256, 5, false, 20f)]
    [InlineData(64, 1, false, 0f, true)]
    [InlineData(128, 3, true, 20f, true)]
    public void AttentionMatchesReference(int hd, int chunks, bool gated, float softCap, bool compact = false)
    {
        var rnd = new Random(hd + chunks);
        int nh = 4, nkv = 2, ctx = 70, ps = 16, n = 2;
        int[] pages = [3, 0, 4, 1, 2];
        int nPages = 6;
        var kc = new Half[nPages * ps * nkv * hd];
        var vc = new Half[nPages * ps * nkv * hd];
        var K = new float[ctx, nkv, hd];
        var V = new float[ctx, nkv, hd];
        for (int p = 0; p < ctx; p++)
            for (int g = 0; g < nkv; g++)
                for (int i = 0; i < hd; i++)
                {
                    K[p, g, i] = (float)(Half)(float)(rnd.NextDouble() - 0.5);
                    V[p, g, i] = (float)(Half)(float)(rnd.NextDouble() - 0.5);
                    if (compact)
                    {
                        // Exactly representable Q8_0 blocks isolate indexing, packing and attention math.
                        K[p, g, i] = (i % 32 == 0 ? 127 : rnd.Next(-127, 128)) / 256f;
                        V[p, g, i] = (i % 32 == 0 ? -127 : rnd.Next(-127, 128)) / 512f;
                    }
                    long slot = pages[p / ps] * ps + p % ps;
                    kc[(slot * nkv + g) * hd + i] = (Half)K[p, g, i];
                    vc[(slot * nkv + g) * hd + i] = (Half)V[p, g, i];
                }
        int qPitch = gated ? 2 * hd : hd, qStride = nh * qPitch;
        var q = Enumerable.Range(0, n * qStride).Select(_ => (float)(rnd.NextDouble() - 0.5)).ToArray();
        int[] info = [0, 0, ctx - 1, 0, 30, ctx - 2]; // token 0: full context; token 1: positions 30..68 (window)
        float scale = 0.2f;
        var prm = new AttnParams
        {
            N = n, NHead = nh, NKv = nkv, Hd = hd, HdV = hd, PageSize = ps, QStride = qStride, QHeadPitch = qPitch, OutStride = nh * hd,
            KRow = nkv * hd, VRow = nkv * hd, NChunks = chunks, Gated = gated ? 1 : 0, Scale = scale, SoftCap = softCap,
        };
        using var dq = _acc.Allocate1D(q); using var dk = _acc.Allocate1D(kc); using var dv = _acc.Allocate1D(vc);
        using var dp = _acc.Allocate1D(pages); using var di = _acc.Allocate1D(info);
        using var dout = _acc.Allocate1D<float>(n * nh * hd);
        using var dpart = _acc.Allocate1D<float>(n * nh * chunks * (2 + hd));
        using var packedK = compact ? _acc.Allocate1D<Half>(kc.Length / 32 * 17) : null;
        using var packedV = compact ? _acc.Allocate1D<Half>(vc.Length / 32 * 17) : null;
        if (compact)
        {
            using var slots = _acc.Allocate1D(Enumerable.Range(0, nPages * ps).ToArray());
            var pack = _acc.LoadStreamKernel<ArrayView<Half>, ArrayView<Half>, ArrayView<int>, int, int>(Kernels.PackKvQ8);
            var shape = new KernelConfig((kc.Length / 32 + 3) / 4, Kernels.GroupSize);
            pack(shape, dk.View, packedK!.View, slots.View, nkv * hd, nPages * ps);
            pack(shape, dv.View, packedV!.View, slots.View, nkv * hd, nPages * ps);
            _acc.Synchronize();
        }
        var attn = _acc.LoadStreamKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams>(compact ? Kernels.AttnKvQ8 : Kernels.AttnDecode);
        var comb = _acc.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, AttnParams>(Kernels.AttnCombine);
        attn(new KernelConfig(n * ((nh + 3) / 4) * chunks, Kernels.GroupSize), dq.View, (packedK ?? dk).View, (packedV ?? dv).View, dp.View, di.View, dout.View, dpart.View, prm);
        if (chunks > 1) comb(new KernelConfig((n * nh + 3) / 4, Kernels.GroupSize), dq.View, dpart.View, dout.View, prm);
        _acc.Synchronize();
        var o = dout.GetAsArray1D();
        for (int t = 0; t < n; t++)
            for (int h = 0; h < nh; h++)
            {
                int g = h / (nh / nkv), start = info[3 * t + 1], end = info[3 * t + 2];
                var s = new double[end - start + 1];
                for (int p = start; p <= end; p++)
                {
                    double dot = 0;
                    for (int i = 0; i < hd; i++) dot += q[t * qStride + h * qPitch + i] * scale * K[p, g, i];
                    s[p - start] = softCap > 0 ? Math.Tanh(dot / softCap) * softCap : dot;
                }
                double mx = s.Max(), sum = s.Sum(v => Math.Exp(v - mx));
                for (int i = 0; i < hd; i++)
                {
                    double acc = 0;
                    for (int p = start; p <= end; p++) acc += Math.Exp(s[p - start] - mx) / sum * V[p, g, i];
                    if (gated) acc *= 1 / (1 + Math.Exp(-q[t * qStride + h * qPitch + hd + i]));
                    Assert.True(Math.Abs(acc - o[(t * nh + h) * hd + i]) < 2e-4, $"t{t} h{h} i{i}: {o[(t * nh + h) * hd + i]} vs {acc}");
                }
            }
    }

    /// <summary>Prompt-batch attention kernels: SIMT, tensor-core 16-query tiles, tensor-core 64-query tiles (FlashAttention-2 split).</summary>
    public enum PrefillKernel { Simt, TensorCore, TensorCore64, TensorCore64Small }

    /// <summary>
    /// Prompt-batch attention (<c>AttnPrefill</c>, <c>TcAttnPrefill</c>, <c>TcAttnPrefill64</c>): sequence A contributes 41
    /// query tokens at positions 100..140 (tiles of 16/16/9, or one 41-row tile of 64; many 32-position chunks, a partial
    /// last chunk), sequence B a single token at position 70 with its own page table; optional sliding window, GQA,
    /// gating and soft-cap.
    /// </summary>
    [Theory]
    [InlineData(64, PrefillKernel.Simt, 0, false, 0f)]
    [InlineData(128, PrefillKernel.Simt, 48, true, 0f)]
    [InlineData(256, PrefillKernel.Simt, 0, false, 30f)]
    [InlineData(64, PrefillKernel.TensorCore, 0, false, 0f)]
    [InlineData(128, PrefillKernel.TensorCore, 48, true, 0f)]
    [InlineData(256, PrefillKernel.TensorCore, 0, false, 30f)]
    [InlineData(64, PrefillKernel.TensorCore64, 0, false, 0f)]
    [InlineData(64, PrefillKernel.TensorCore64, 0, false, 30f)]
    [InlineData(128, PrefillKernel.TensorCore64, 48, true, 0f)]
    [InlineData(128, PrefillKernel.TensorCore64, 0, false, 0f)]
    [InlineData(64, PrefillKernel.TensorCore64Small, 48, true, 0f)]
    [InlineData(32, PrefillKernel.TensorCore64Small, 0, false, 30f)]
    public void PrefillAttentionMatchesReference(int hd, PrefillKernel kind, int window, bool gated, float softCap)
    {
        bool tensorCores = kind != PrefillKernel.Simt;
        if (tensorCores && !new GemmLauncher(_acc, true).TensorCores) return;
        var rnd = new Random(hd + window);
        int nh = 4, nkv = 2, ps = 16;
        // page tables: A uses pages 0..8 (shuffled), B uses pages 9..13
        int[] pagesA = [7, 2, 0, 8, 5, 1, 3, 6, 4], pagesB = [11, 9, 13, 10, 12];
        int nPages = 14, lenA = 141, lenB = 71;
        var kc = new Half[nPages * ps * nkv * hd];
        var vc = new Half[nPages * ps * nkv * hd];
        float[,,] K = new float[2, Math.Max(lenA, lenB) * nkv, hd], V = new float[2, Math.Max(lenA, lenB) * nkv, hd];
        for (int sq = 0; sq < 2; sq++)
        {
            var pt = sq == 0 ? pagesA : pagesB;
            int len = sq == 0 ? lenA : lenB;
            for (int pos = 0; pos < len; pos++)
                for (int gh = 0; gh < nkv; gh++)
                    for (int i = 0; i < hd; i++)
                    {
                        float kv = (float)(Half)(float)(rnd.NextDouble() - 0.5), vv = (float)(Half)(float)(rnd.NextDouble() - 0.5);
                        K[sq, pos * nkv + gh, i] = kv; V[sq, pos * nkv + gh, i] = vv;
                        long slot = (long)pt[pos / ps] * ps + pos % ps;
                        kc[(slot * nkv + gh) * hd + i] = (Half)kv;
                        vc[(slot * nkv + gh) * hd + i] = (Half)vv;
                    }
        }
        int[] pageList = [.. pagesA, .. pagesB];
        // batch: 41 tokens of A (positions 100..140) then one token of B (position 70)
        int n = 42;
        var info = new int[3 * n];
        for (int t = 0; t < 41; t++) { int pos = 100 + t; info[3 * t] = 0; info[3 * t + 1] = window > 0 ? Math.Max(0, pos - window + 1) : 0; info[3 * t + 2] = pos; }
        info[3 * 41] = pagesA.Length; info[3 * 41 + 1] = window > 0 ? 70 - window + 1 : 0; info[3 * 41 + 2] = 70;
        int[] tiles = kind is PrefillKernel.TensorCore64 or PrefillKernel.TensorCore64Small ? [0, 41, 41, 1] : [0, 16, 16, 16, 32, 9, 41, 1];
        int qPitch = gated ? 2 * hd : hd, qStride = nh * qPitch;
        var q = Enumerable.Range(0, n * qStride).Select(_ => (float)(rnd.NextDouble() - 0.5)).ToArray();
        float scale = 0.15f;
        var prm = new AttnParams
        {
            N = n, NHead = nh, NKv = nkv, Hd = hd, HdV = hd, PageSize = ps, QStride = qStride, QHeadPitch = qPitch, OutStride = nh * hd,
            KRow = nkv * hd, VRow = nkv * hd, NChunks = 1, Gated = gated ? 1 : 0, Scale = scale, SoftCap = softCap,
        };
        using var dq = _acc.Allocate1D(q); using var dk = _acc.Allocate1D(kc); using var dv = _acc.Allocate1D(vc);
        using var dp = _acc.Allocate1D(pageList); using var di = _acc.Allocate1D(info); using var dt = _acc.Allocate1D(tiles);
        using var dout = _acc.Allocate1D<float>(n * nh * hd);
        var kernel = _acc.LoadStreamKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>>(
            kind switch { PrefillKernel.TensorCore64 => Kernels.TcAttnPrefill64, PrefillKernel.TensorCore64Small => Kernels.TcAttnPrefill64Small, PrefillKernel.TensorCore => Kernels.TcAttnPrefill, _ => Kernels.AttnPrefill });
        kernel(new KernelConfig(tiles.Length / 2 * nh, 128), dq.View, dk.View, dv.View, dp.View, di.View, dt.View, dout.View, prm, SpecializedValue.New(hd));
        _acc.Synchronize();
        var o = dout.GetAsArray1D();
        double tol = tensorCores ? 3e-3 : 2e-4; // fp16 Q and P on tensor cores
        for (int t = 0; t < n; t++)
        {
            int sq = t < 41 ? 0 : 1, start = info[3 * t + 1], end = info[3 * t + 2];
            for (int h = 0; h < nh; h++)
            {
                int g = h / (nh / nkv);
                var sc = new double[end - start + 1];
                for (int pos = start; pos <= end; pos++)
                {
                    double dot = 0;
                    for (int i = 0; i < hd; i++) dot += q[t * qStride + h * qPitch + i] * scale * K[sq, pos * nkv + g, i];
                    sc[pos - start] = softCap > 0 ? Math.Tanh(dot / softCap) * softCap : dot;
                }
                double mx = sc.Max(), sum = sc.Sum(v => Math.Exp(v - mx));
                for (int i = 0; i < hd; i++)
                {
                    double acc = 0;
                    for (int pos = start; pos <= end; pos++) acc += Math.Exp(sc[pos - start] - mx) / sum * V[sq, pos * nkv + g, i];
                    if (gated) acc *= 1 / (1 + Math.Exp(-q[t * qStride + h * qPitch + hd + i]));
                    float got = o[(t * nh + h) * hd + i];
                    Assert.True(Math.Abs(acc - got) < tol, $"tc={tensorCores} t{t} h{h} i{i}: {got} vs {acc}");
                }
            }
        }
    }

    /// <summary>
    /// Vision-style bidirectional attention with the 64-query kernel: three images (groups) of 100, 64 and 37 patches —
    /// several and partial 64-row tiles per group, every row seeing its whole group — through a shuffled page table.
    /// </summary>
    [Theory]
    [InlineData(64, false)]
    [InlineData(128, false)]
    [InlineData(64, true)]
    public void BidirectionalGroupsMatchReference(int hd, bool small)
    {
        if (!new GemmLauncher(_acc, true).TensorCores) return;
        var rnd = new Random(hd);
        int nh = 3, ps = 128;
        int[] lens = [100, 64, 37];
        int n = lens.Sum();
        int[] pageOf = [2, 0, 1];   // group g lives on page pageOf[g]
        var kc = new Half[3 * ps * nh * hd];
        var vc = new Half[3 * ps * nh * hd];
        var K = new float[n, nh, hd];
        var V = new float[n, nh, hd];
        var info = new int[3 * n];
        var tiles = new List<int>();
        int r0 = 0;
        for (int g = 0; g < lens.Length; g++)
        {
            for (int t = 0; t < lens[g]; t++)
            {
                int row = r0 + t;
                info[3 * row] = g; info[3 * row + 1] = 0; info[3 * row + 2] = lens[g] - 1;
                for (int h = 0; h < nh; h++)
                    for (int i = 0; i < hd; i++)
                    {
                        float kv = (float)(Half)(float)(rnd.NextDouble() - 0.5), vv = (float)(Half)(float)(rnd.NextDouble() - 0.5);
                        K[row, h, i] = kv; V[row, h, i] = vv;
                        long slot = (long)pageOf[g] * ps + t;
                        kc[(slot * nh + h) * hd + i] = (Half)kv;
                        vc[(slot * nh + h) * hd + i] = (Half)vv;
                    }
            }
            for (int q0 = 0; q0 < lens[g]; q0 += Kernels.Fa2Q) { tiles.Add(r0 + q0); tiles.Add(Math.Min(Kernels.Fa2Q, lens[g] - q0)); }
            r0 += lens[g];
        }
        var q = Enumerable.Range(0, n * nh * hd).Select(_ => (float)(rnd.NextDouble() - 0.5)).ToArray();
        float scale = 1f / MathF.Sqrt(hd);
        var prm = new AttnParams
        {
            N = n, NHead = nh, NKv = nh, Hd = hd, HdV = hd, PageSize = ps, QStride = nh * hd, QHeadPitch = hd, OutStride = nh * hd,
            KRow = nh * hd, VRow = nh * hd, NChunks = 1, Scale = scale,
        };
        using var dq = _acc.Allocate1D(q); using var dk = _acc.Allocate1D(kc); using var dv = _acc.Allocate1D(vc);
        using var dp = _acc.Allocate1D(pageOf); using var di = _acc.Allocate1D(info); using var dt = _acc.Allocate1D(tiles.ToArray());
        using var dout = _acc.Allocate1D<float>(n * nh * hd);
        var kernel = _acc.LoadStreamKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, AttnParams, SpecializedValue<int>>(small ? Kernels.TcAttnPrefill64Small : Kernels.TcAttnPrefill64);
        kernel(new KernelConfig(tiles.Count / 2 * nh, 128), dq.View, dk.View, dv.View, dp.View, di.View, dt.View, dout.View, prm, SpecializedValue.New(hd));
        _acc.Synchronize();
        var o = dout.GetAsArray1D();
        r0 = 0;
        for (int g = 0; g < lens.Length; r0 += lens[g], g++)
            for (int t = r0; t < r0 + lens[g]; t++)
                for (int h = 0; h < nh; h++)
                {
                    var sc = new double[lens[g]];
                    for (int k = 0; k < lens[g]; k++)
                    {
                        double dot = 0;
                        for (int i = 0; i < hd; i++) dot += q[(t * nh + h) * hd + i] * scale * K[r0 + k, h, i];
                        sc[k] = dot;
                    }
                    double mx = sc.Max(), sum = sc.Sum(v => Math.Exp(v - mx));
                    for (int i = 0; i < hd; i++)
                    {
                        double acc = 0;
                        for (int k = 0; k < lens[g]; k++) acc += Math.Exp(sc[k] - mx) / sum * V[r0 + k, h, i];
                        float got = o[(t * nh + h) * hd + i];
                        Assert.True(Math.Abs(acc - got) < 3e-3, $"group {g} row {t} h{h} i{i}: {got} vs {acc}");
                    }
                }
    }

    /// <summary>
    /// Grouped-query decode attention (+ chunk combine) against a double-precision reference on the same fp16 K/V:
    /// group sizes 1-8, head dims 64-512, contexts from one position to several 32-position passes per chunk, scrambled
    /// KV pages, one or many chunks, gated output.
    /// </summary>
    [Theory]
    [InlineData(64, 4, 2, 1, 1, false)]
    [InlineData(64, 4, 2, 37, 1, false)]
    [InlineData(128, 2, 3, 100, 4, false)]
    [InlineData(128, 1, 2, 700, 23, true)]
    [InlineData(256, 8, 1, 513, 16, false)]
    [InlineData(512, 8, 1, 300, 1, false)]
    [InlineData(512, 8, 1, 700, 7, true)]
    public void GqaDecodeAttentionMatchesReference(int hd, int group, int nkv, int ctx, int chunks, bool gated)
    {
        int nh = group * nkv, pageSize = 32, pagesN = (ctx + pageSize - 1) / pageSize, qPitch = gated ? 2 * hd : hd;
        var rnd = new Random(hd * 31 + ctx);
        var pages = Enumerable.Range(0, pagesN).OrderBy(_ => rnd.Next()).ToArray();
        long slots = (long)pagesN * pageSize;
        var kh = new Half[slots * nkv * hd];
        var vh = new Half[slots * nkv * hd];
        for (long i = 0; i < kh.LongLength; i++) { kh[i] = (Half)(float)(rnd.NextDouble() - 0.5); vh[i] = (Half)(float)(rnd.NextDouble() - 0.5); }
        var q = Enumerable.Range(0, nh * qPitch).Select(_ => (float)(rnd.NextDouble() - 0.5) * 2).ToArray();
        float scale = 1f / MathF.Sqrt(hd);
        using var dq = _acc.Allocate1D(q);
        using var dk = _acc.Allocate1D(kh);
        using var dv = _acc.Allocate1D(vh);
        using var dp = _acc.Allocate1D(pages);
        using var dinfo = _acc.Allocate1D(new[] { 0, 0, ctx - 1 });
        using var dpart = _acc.Allocate1D<float>((long)nh * chunks * (2 + hd));
        using var dout = _acc.Allocate1D<float>(nh * hd);
        var p = new AttnParams
        {
            QOff = 0, OutOff = 0, N = 1, NHead = nh, NKv = nkv, Hd = hd, HdV = hd, PageSize = pageSize, QStride = nh * qPitch, QHeadPitch = qPitch,
            OutStride = nh * hd, KRow = nkv * hd, VRow = nkv * hd, NChunks = chunks, Gated = gated ? 1 : 0, Scale = scale, SoftCap = 0,
        };
        var attn = _acc.LoadStreamKernel<ArrayView<float>, ArrayView<Half>, ArrayView<Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams, SpecializedValue<int>, SpecializedValue<int>>(Kernels.AttnDecodeGqa);
        var combine = _acc.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, AttnParams>(Kernels.AttnCombine2);
        attn(new KernelConfig(nkv * chunks, Kernels.GroupSize), dq.View, dk.View, dv.View, dp.View, dinfo.View, dout.View, dpart.View, p,
            SpecializedValue.New(group), SpecializedValue.New(hd));
        if (chunks > 1) combine(new KernelConfig(nh * ((hd + Kernels.GroupSize - 1) / Kernels.GroupSize), Kernels.GroupSize), dq.View, dpart.View, dout.View, p);
        _acc.Synchronize();
        var o = dout.GetAsArray1D();

        for (int h = 0; h < nh; h++)
        {
            int g = h / group;
            var sc = new double[ctx];
            for (int pos = 0; pos < ctx; pos++)
            {
                long slot = (long)pages[pos / pageSize] * pageSize + pos % pageSize;
                double dot = 0;
                for (int i = 0; i < hd; i++) dot += q[h * qPitch + i] * scale * (float)kh[slot * nkv * hd + g * hd + i];
                sc[pos] = dot;
            }
            double mx = sc.Max(), sum = sc.Sum(v => Math.Exp(v - mx));
            for (int i = 0; i < hd; i++)
            {
                double acc = 0;
                for (int pos = 0; pos < ctx; pos++)
                {
                    long slot = (long)pages[pos / pageSize] * pageSize + pos % pageSize;
                    acc += Math.Exp(sc[pos] - mx) / sum * (float)vh[slot * nkv * hd + g * hd + i];
                }
                if (gated) acc *= 1 / (1 + Math.Exp(-q[h * qPitch + hd + i]));
                Assert.True(Math.Abs(acc - o[h * hd + i]) < 1e-4, $"h{h} i{i}: {o[h * hd + i]} vs {acc}");
            }
        }
    }

    // AttnDecodeGqa sizes its shared memory from its JIT constants: it is compiled through fixed instantiations instead
    public static IEnumerable<object[]> KernelMethods() =>
        typeof(Kernels).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.ReturnType == typeof(void) && !m.IsGenericMethodDefinition && m.Name != nameof(Kernels.AttnDecodeGqa))
            .Select(m => new object[] { m.Name })
            .Concat(new[] { "AttnGqa8", "AttnGqa2" }.Select(n => new object[] { "probe:" + n }));

    /// <summary>Compiles every kernel to CUDA PTX (sm_75) without needing a GPU.</summary>
    [Theory]
    [MemberData(nameof(KernelMethods))]
    public void KernelsCompileForCuda(string name)
    {
        var m = name.StartsWith("probe:")
            ? typeof(ProbeKernels).GetMethod(name[6..], BindingFlags.Public | BindingFlags.Static)!
            : typeof(Kernels).GetMethod(name, BindingFlags.Public | BindingFlags.Static)!;
        bool implicitGroups = m.GetParameters()[0].ParameterType == typeof(Index1D);
        var entry = implicitGroups ? EntryPointDescription.FromImplicitlyGroupedKernel(m) : EntryPointDescription.FromExplicitlyGroupedKernel(m);
        var arch = new CudaArchitecture(7, 5);
        using var ptx = new PTXBackend(_ctx, new CudaCapabilityContext(arch), arch, new CudaInstructionSet(7, 0), null!);
        var compiled = (PTXCompiledKernel)ptx.Compile(entry, new KernelSpecialization());
        Assert.Contains(".entry", compiled.PTXAssembly);

        // with a real GPU, also run the driver's PTX JIT (ptxas) on code generated for the device itself
        if (_acc is CudaAccelerator cuda)
        {
            var dev = (PTXCompiledKernel)cuda.Backend.Compile(entry, new KernelSpecialization());
            var err = CudaAPI.CurrentAPI.LoadModule(out IntPtr module, dev.PTXAssembly, out string log);
            Assert.True(err == CudaError.CUDA_SUCCESS, $"{name}: {err} {log}");
            CudaAPI.CurrentAPI.DestroyModule(module);
        }
    }

    /// <summary>Every fp16 bit pattern (incl. subnormals, which GPU fast math would flush) decodes exactly on the device.</summary>
    [Fact]
    public void HalfDecodingIsExactOnDevice()
    {
        using var dy = _acc.Allocate1D<float>(1 << 16);
        var k = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(static (i, y) => y[i] = Kernels.HalfBits(i));
        k(1 << 16, dy.View);
        _acc.Synchronize();
        var y = dy.GetAsArray1D();
        for (int h = 0; h < 1 << 16; h++)
        {
            float expect = (float)BitConverter.UInt16BitsToHalf((ushort)h);
            if (float.IsNaN(expect)) continue;
            Assert.True(expect.Equals(y[h]), $"0x{h:x4}: {y[h]} vs {expect}");
        }
    }
}
