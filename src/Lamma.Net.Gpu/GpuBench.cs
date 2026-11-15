using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using LammaNet.Gguf;
using LammaNet.Tensors;

namespace LammaNet.Gpu;

/// <summary>Kernel micro-benchmarks (device time measured with stream markers over back-to-back launches).</summary>
public static unsafe class GpuBench
{
    /// <summary>
    /// Times the matvec kernel on one tensor of a GGUF file for <paramref name="nTok"/> tokens; the rows can be tiled
    /// <paramref name="repeat"/> times to measure large matrices of the same format.
    /// </summary>
    public static string MatVec(string ggufPath, string tensor, int nTok, int repeat = 1, int iters = 200)
    {
        using var f = new GgufFile(ggufPath);
        var w = WeightTensor.FromGguf(f, f.GetTensor(tensor));
        using var ctx = Context.Create(b => b.Default().Math(MathMode.Fast32BitOnly).EnableAlgorithms().Profiling());
        using var acc = ctx.GetCudaDevice(0).CreateAccelerator(ctx);
        int type = DevFormat.For(w.Type, w.Cols);
        int rowWords = DevFormat.RowWords(type, w.Cols);
        var one = new uint[(long)rowWords * w.Rows];
        fixed (uint* hp = one) DevFormat.Pack(w, type, hp);
        var host = new uint[one.LongLength * repeat];
        for (int r = 0; r < repeat; r++) Array.Copy(one, 0, host, one.LongLength * r, one.LongLength);
        using var dw = acc.Allocate1D(host);
        int rows = w.Rows * repeat;
        using var df = acc.Allocate1D<float>(4);
        var rnd = new Random(1);
        using var dx = acc.Allocate1D(Enumerable.Range(0, nTok * w.Cols).Select(_ => (float)rnd.NextDouble() - 0.5f).ToArray());
        using var dy = acc.Allocate1D<float>((long)nTok * rows);
        var mv = new MatVecLauncher(acc);
        mv.Reserve(nTok + Kernels.MatVecTokens, w.Cols);
        var m = new DMat { Buf = dw, Word = 0, Rows = rows, Cols = w.Cols, RowWords = rowWords, Type = type };
        var stream = acc.CreateStream();
        void Run() => mv.Run(stream, df.View, dx.View, 0, w.Cols, nTok, dy.View, 0, 1f, (m, 0, rows), null, null, 0, 0, normedInput: true);
        Run();
        acc.Synchronize();
        // launches replayed from a CUDA graph (as in decoding): the host-side launch cost is not measured
        var cs = ((CudaStream)stream).StreamPtr;
        using var graph = CudaGraph.Capture(cs, () => { for (int i = 0; i < iters; i++) Run(); });
        graph.Launch(cs);
        acc.Synchronize();
        using var m0 = stream.AddProfilingMarker();
        graph.Launch(cs);
        using var m1 = stream.AddProfilingMarker();
        acc.Synchronize();
        double us = m1.MeasureFrom(m0).TotalMicroseconds / iters;
        double gb = host.Length * 4.0 / 1e9;
        // spot-check rows against the CPU dequantization
        var y = dy.GetAsArray1D();
        var xh = dx.GetAsArray1D();
        var rowF = new float[w.Cols];
        // the same activations rounded to int8 blocks as the q8 kernels round them (error of the int8 path beyond that
        // rounding is a kernel bug)
        var xq = (float[])xh.Clone();
        for (int b0 = 0; b0 < xq.Length; b0 += 32)
        {
            float amax = 0;
            for (int j = 0; j < 32; j++) amax = Math.Max(amax, Math.Abs(xq[b0 + j]));
            float d = amax / 127f, inv = amax > 0 ? 127f / amax : 0f;
            for (int j = 0; j < 32; j++) xq[b0 + j] = MathF.Round(xq[b0 + j] * inv) * d;
        }
        double maxErr = 0, maxErrQ = 0, maxRef = 0;
        for (int r = 0; r < Math.Min(64, w.Rows); r++)
        {
            fixed (float* rp = rowF) w.DequantizeRow(r, rp);
            for (int t = 0; t < nTok; t++)
            {
                double e = 0, eq = 0;
                for (int c = 0; c < w.Cols; c++) { e += rowF[c] * (double)xh[(long)t * w.Cols + c]; eq += rowF[c] * (double)xq[(long)t * w.Cols + c]; }
                maxErr = Math.Max(maxErr, Math.Abs(e - y[(long)t * rows + r]));
                maxErrQ = Math.Max(maxErrQ, Math.Abs(eq - y[(long)t * rows + r]));
                maxRef = Math.Max(maxRef, Math.Abs(e));
            }
        }
        // batch invariance: each token alone must give bit-identical rows (speculative verification relies on it)
        int differ = 0;
        if (nTok > 1)
        {
            using var dy1 = acc.Allocate1D<float>(rows);
            for (int t = 0; t < nTok; t++)
            {
                mv.Run(stream, df.View, dx.View, (long)t * w.Cols, w.Cols, 1, dy1.View, 0, 1f, (m, 0, rows), null, null, 0, 0, normedInput: true);
                acc.Synchronize();
                var y1 = dy1.GetAsArray1D();
                for (int r = 0; r < rows; r++)
                    if (BitConverter.SingleToInt32Bits(y1[r]) != BitConverter.SingleToInt32Bits(y[(long)t * rows + r])) differ++;
            }
        }
        return $"{w.Name} {w.Type}->{type} [{w.Cols}x{rows}] n={nTok}: {us:F1} us  {gb / (us / 1e6):F0} GB/s  (max err {maxErr:E1}, vs int8-rounded x {maxErrQ:E1}, |ref| {maxRef:E1})"
               + (nTok > 1 ? differ == 0 ? "  batch-invariant" : $"  {differ} values differ from single-token calls" : "");
    }
    /// <summary>
    /// Times one decode-attention launch (one token, all heads, + the chunk combine) on a paged synthetic KV cache:
    /// <see cref="Kernels.AttnDecode"/> (old) against <see cref="Kernels.AttnDecodeGqa"/> (chunk counts as the offload
    /// chooses them, or <paramref name="gqaChunk"/> positions per chunk), and the largest output difference.
    /// </summary>
    public static string DecodeAttention(int hd, int group, int nkv, int ctx, int gqaChunk = Kernels.GqaPass, int iters = 200)
    {
        using var context = Context.Create(b => b.Default().Math(MathMode.Fast32BitOnly).EnableAlgorithms().Profiling());
        using var acc = context.GetCudaDevice(0).CreateAccelerator(context);
        var stream = acc.CreateStream();
        int nh = group * nkv, pageSize = 32, pagesN = (ctx + pageSize - 1) / pageSize;
        var rnd = new Random(1);
        // pages in a scrambled order (as after prefix sharing / eviction)
        var pages = Enumerable.Range(0, pagesN).OrderBy(_ => rnd.Next()).ToArray();
        long slots = (long)pagesN * pageSize;
        var kh = new ILGPU.Half[slots * nkv * hd];
        var vh = new ILGPU.Half[slots * nkv * hd];
        for (long i = 0; i < kh.LongLength; i++) { kh[i] = (ILGPU.Half)(float)(rnd.NextDouble() - 0.5); vh[i] = (ILGPU.Half)(float)(rnd.NextDouble() - 0.5); }
        var qh = Enumerable.Range(0, nh * hd).Select(_ => (float)(rnd.NextDouble() - 0.5) * 2).ToArray();
        using var dq = acc.Allocate1D(qh);
        using var dk = acc.Allocate1D(kh);
        using var dv = acc.Allocate1D(vh);
        using var dp = acc.Allocate1D(pages);
        using var dinfo = acc.Allocate1D(new[] { 0, 0, ctx - 1 });
        int maxChunks = 64;
        using var dpart = acc.Allocate1D<float>((long)nh * maxChunks * (2 + hd));
        using var dOld = acc.Allocate1D<float>(nh * hd);
        using var dNew = acc.Allocate1D<float>(nh * hd);
        var kOld = acc.LoadKernel<ArrayView<float>, ArrayView<ILGPU.Half>, ArrayView<ILGPU.Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams>(Kernels.AttnDecode);
        var kNew = acc.LoadKernel<ArrayView<float>, ArrayView<ILGPU.Half>, ArrayView<ILGPU.Half>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, AttnParams, SpecializedValue<int>, SpecializedValue<int>>(Kernels.AttnDecodeGqa);
        var kComb = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, AttnParams>(Kernels.AttnCombine);
        var kComb2 = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, AttnParams>(Kernels.AttnCombine2);
        int slices = (hd + Kernels.GroupSize - 1) / Kernels.GroupSize;
        AttnParams P(int chunks) => new()
        {
            QOff = 0, OutOff = 0, N = 1, NHead = nh, NKv = nkv, Hd = hd, HdV = hd, PageSize = pageSize, QStride = nh * hd, QHeadPitch = hd,
            OutStride = nh * hd, KRow = nkv * hd, VRow = nkv * hd, NChunks = chunks, Gated = 0, Scale = 1f / MathF.Sqrt(hd), SoftCap = 0,
        };
        int cOld = Math.Clamp((ctx + 31) / 32, 1, maxChunks), cNew = Math.Clamp((ctx + gqaChunk - 1) / gqaChunk, 1, maxChunks);
        var pOld = P(cOld); var pNew = P(cNew);
        void RunOld()
        {
            kOld(stream, new KernelConfig(((nh + Kernels.WarpsPerGroup - 1) / Kernels.WarpsPerGroup) * cOld, Kernels.GroupSize), dq.View, dk.View, dv.View, dp.View, dinfo.View, dOld.View, dpart.View, pOld);
            if (cOld > 1) kComb2(stream, new KernelConfig(nh * slices, Kernels.GroupSize), dq.View, dpart.View, dOld.View, pOld);
        }
        void RunNew()
        {
            kNew(stream, new KernelConfig(nkv * cNew, Kernels.GroupSize), dq.View, dk.View, dv.View, dp.View, dinfo.View, dNew.View, dpart.View, pNew, SpecializedValue.New(group), SpecializedValue.New(hd));
            if (cNew > 1) kComb2(stream, new KernelConfig(nh * slices, Kernels.GroupSize), dq.View, dpart.View, dNew.View, pNew);
        }
        // launches replayed from a CUDA graph (as in decoding): the host-side launch cost is not measured
        double Time(Action run)
        {
            run();
            acc.Synchronize();
            var cs = ((CudaStream)stream).StreamPtr;
            using var graph = CudaGraph.Capture(cs, () => { for (int i = 0; i < iters; i++) run(); });
            graph.Launch(cs);
            acc.Synchronize();
            using var m0 = stream.AddProfilingMarker();
            graph.Launch(cs);
            using var m1 = stream.AddProfilingMarker();
            acc.Synchronize();
            return m1.MeasureFrom(m0).TotalMicroseconds / iters;
        }
        double tOld = Time(RunOld), tNew = Time(RunNew);
        double tComb = cNew > 1 ? Time(() => kComb2(stream, new KernelConfig(nh * slices, Kernels.GroupSize), dq.View, dpart.View, dNew.View, pNew)) : 0;
        var a = dOld.GetAsArray1D(); var b = dNew.GetAsArray1D();
        double diff = a.Zip(b, (x, y) => Math.Abs(x - y)).Max();
        double mb = 2.0 * ctx * nkv * hd * 2 / 1e6;
        return $"hd {hd,3} G {group} kv {nkv} ctx {ctx,5} ({mb,6:F2} MB K+V): old+combine2 {tOld,7:F1} us ({cOld,2} chunks)  new {tNew,7:F1} us ({cNew,2} chunks, combine {tComb:F1})  max|diff| {diff:E1}";
    }

    /// <summary>
    /// Times the prompt GEMM (<see cref="GemmLauncher"/>) on one tensor of a GGUF file for <paramref name="nTok"/> tokens
    /// and reports TFLOPS plus the largest difference from a CPU dot product (dequantized weights) on a few rows.
    /// </summary>
    public static string Gemm(string ggufPath, string tensor, int nTok, int iters = 50)
    {
        using var f = new GgufFile(ggufPath);
        var w = WeightTensor.FromGguf(f, f.GetTensor(tensor));
        using var ctx = Context.Create(b => b.Default().Math(MathMode.Fast32BitOnly).EnableAlgorithms().Profiling());
        using var acc = ctx.GetCudaDevice(0).CreateAccelerator(ctx);
        int type = DevFormat.For(w.Type, w.Cols);
        int rowWords = DevFormat.RowWords(type, w.Cols);
        var host = new uint[(long)rowWords * w.Rows];
        fixed (uint* hp = host) DevFormat.Pack(w, type, hp);
        using var dw = acc.Allocate1D(host);
        using var df = acc.Allocate1D<float>(4);
        var rnd = new Random(1);
        var xh = Enumerable.Range(0, nTok * w.Cols).Select(_ => (float)rnd.NextDouble() - 0.5f).ToArray();
        using var dx = acc.Allocate1D(xh);
        using var dy = acc.Allocate1D<float>((long)nTok * w.Rows);
        var gemm = new GemmLauncher(acc);
        var m = new DMat { Buf = dw, Word = 0, Rows = w.Rows, Cols = w.Cols, RowWords = rowWords, Type = type };
        var stream = acc.DefaultStream;
        bool q8 = Environment.GetEnvironmentVariable("LAMMA_BENCH_Q8") == "1";   // int8-activation path (int8 GEMM for Q4_K)
        void Run() => gemm.Run(stream, df.View, dx.View, 0, w.Cols, nTok, dy.View, 0, 1f, m, 0, w.Rows, q8);
        for (int i = 0; i < 5; i++) Run();
        acc.Synchronize();
        using var m0 = stream.AddProfilingMarker();
        for (int i = 0; i < iters; i++) Run();
        using var m1 = stream.AddProfilingMarker();
        acc.Synchronize();
        double us = m1.MeasureFrom(m0).TotalMicroseconds / iters;
        var y = dy.GetAsArray1D();
        var rowF = new float[w.Cols];
        double maxErr = 0, maxRef = 0;
        foreach (int r in new[] { 0, 1, w.Rows / 2, w.Rows - 1 })
        {
            fixed (float* rp = rowF) w.DequantizeRow(r, rp);
            foreach (int t in new[] { 0, nTok / 3, nTok - 1 })
            {
                double e = 0;
                for (int c = 0; c < w.Cols; c++) e += rowF[c] * (double)xh[(long)t * w.Cols + c];
                maxErr = Math.Max(maxErr, Math.Abs(e - y[(long)t * w.Rows + r]));
                maxRef = Math.Max(maxRef, Math.Abs(e));
            }
        }
        double tflops = 2.0 * w.Rows * w.Cols * nTok / (us * 1e-6) / 1e12;
        return $"{tensor} {w.Type} [{w.Cols}x{w.Rows}] n={nTok}: {us,8:F1} us  {tflops,5:F1} TFLOPS  max|err| {maxErr:G3} (|ref| {maxRef:G3}){(gemm.TensorCores ? "" : " [no tensor cores]")}";
    }

    public static IEnumerable<string> KernelStats()
    {
        using var ctx = Context.Create(b => b.Default().Math(MathMode.Fast32BitOnly).EnableAlgorithms());
        using var acc = (CudaAccelerator)ctx.GetCudaDevice(0).CreateAccelerator(ctx);
        foreach (var m in typeof(ProbeKernels).GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)
                     .Concat(new[] { "AttnDecode", "AttnCombine", "RmsNorm", "QkvPost", "TcAttnPrefill", "TcAttnPrefill64", "TcAttnPrefill64Small", "TcGemmQ4K", "TcGemmQ40", "TcGemmQ80" }.Select(n => typeof(Kernels).GetMethod(n)!))
                     .Concat(typeof(Kernels).GetMethods().Where(g => g.Name.StartsWith("TcGemm") && g.IsGenericMethodDefinition).Select(g => g.MakeGenericMethod(typeof(FmtQ4K)))))
        {
            var entry = ILGPU.Backends.EntryPoints.EntryPointDescription.FromExplicitlyGroupedKernel(m);
            var ck = (ILGPU.Backends.PTX.PTXCompiledKernel)acc.Backend.Compile(entry, new KernelSpecialization());
            if (Environment.GetEnvironmentVariable("LAMMA_PTX_DUMP") is { Length: > 0 } dump && m.Name.Contains(dump))
                File.WriteAllText(Path.Combine(Path.GetTempPath(), m.Name + ".ptx"), ck.PTXAssembly);
            CudaAPI.CurrentAPI.LoadModule(out IntPtr mod, ck.PTXAssembly, out _);
            string name = ck.PTXAssembly.Split('\n').First(l => l.Contains(".entry")).Split(' ', '(')[^2].Trim();
            CudaAPI.CurrentAPI.GetModuleFunction(out IntPtr fn, mod, name);
            Native.cuFuncGetAttribute(out int regs, 4, fn);
            Native.cuFuncGetAttribute(out int local, 3, fn);
            Native.cuFuncGetAttribute(out int smem, 1, fn);
            int threads = m.Name.StartsWith("TcGemm2") ? Kernels.Tc2Threads : m.Name.StartsWith("TcGemmQ") ? Kernels.TcQ8Threads : Kernels.GroupSize;
            Native.cuOccupancyMaxActiveBlocksPerMultiprocessor(out int blocks, fn, threads, IntPtr.Zero);
            yield return $"{m.Name,-20} regs {regs,4}  local {local,5} B  smem {smem,6} B  warps/SM {blocks * threads / 32,3}";
            CudaAPI.CurrentAPI.DestroyModule(mod);
        }
    }

    private static class Native
    {
        [System.Runtime.InteropServices.DllImport("nvcuda")] public static extern int cuFuncGetAttribute(out int value, int attribute, IntPtr function);
        [System.Runtime.InteropServices.DllImport("nvcuda")] public static extern int cuOccupancyMaxActiveBlocksPerMultiprocessor(out int blocks, IntPtr function, int blockSize, IntPtr dynamicSmem);
    }
}

/// <summary>Fixed instantiations of the matvec body for <see cref="GpuBench.KernelStats"/>.</summary>
internal static class ProbeKernels
{
    public static void Q4K_n1(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p) => Kernels.MatVecBody<FmtQ4K, FmtQ4K, FmtQ4K>(w, f, x, y, p, 1);
    public static void Q6K_n1(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p) => Kernels.MatVecBody<FmtQ6K, FmtQ6K, FmtQ6K>(w, f, x, y, p, 1);
    public static void Q4K_Q6K_n1(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p) => Kernels.MatVecBody<FmtQ4K, FmtQ4K, FmtQ6K>(w, f, x, y, p, 1);
    public static void Q80_n1(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p) => Kernels.MatVecBody<FmtQ80, FmtQ80, FmtQ80>(w, f, x, y, p, 1);
    public static void Q4K_n4(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p) => Kernels.MatVecBody<FmtQ4K, FmtQ4K, FmtQ4K>(w, f, x, y, p, 4);
    public static void Q4K_n8(ArrayView<uint> w, ArrayView<float> f, ArrayView<float> x, ArrayView<float> y, MatVecParams p) => Kernels.MatVecBody<FmtQ4K, FmtQ4K, FmtQ4K>(w, f, x, y, p, 8);
    public static void Q6K_q8_n1(ArrayView<uint> w, ArrayView<float> f, ArrayView<uint> xq, ArrayView<Float2> xd, ArrayView<float> y, MatVecParams p) =>
        Kernels.MatVecQ8<FmtQ6K, FmtQ6K, FmtQ6K>(w, f, xq, xd, y, p, SpecializedValue.New(1));
    public static void Q6K_q8_n4(ArrayView<uint> w, ArrayView<float> f, ArrayView<uint> xq, ArrayView<Float2> xd, ArrayView<float> y, MatVecParams p) =>
        Kernels.MatVecQ8<FmtQ6K, FmtQ6K, FmtQ6K>(w, f, xq, xd, y, p, SpecializedValue.New(4));
    public static void Q4K_q8_n4(ArrayView<uint> w, ArrayView<float> f, ArrayView<uint> xq, ArrayView<Float2> xd, ArrayView<float> y, MatVecParams p) =>
        Kernels.MatVecQ8<FmtQ4K, FmtQ4K, FmtQ4K>(w, f, xq, xd, y, p, SpecializedValue.New(4));
    public static void Q40_q8_n4(ArrayView<uint> w, ArrayView<float> f, ArrayView<uint> xq, ArrayView<Float2> xd, ArrayView<float> y, MatVecParams p) =>
        Kernels.MatVecQ8<FmtQ40, FmtQ40, FmtQ40>(w, f, xq, xd, y, p, SpecializedValue.New(4));
    public static void AttnGqa8(ArrayView<float> q, ArrayView<ILGPU.Half> k, ArrayView<ILGPU.Half> v, ArrayView<int> pg, ArrayView<int> ti, ArrayView<float> o, ArrayView<float> pt, AttnParams p) =>
        Kernels.AttnDecodeGqa(q, k, v, pg, ti, o, pt, p, SpecializedValue.New(8), SpecializedValue.New(512));
    public static void AttnGqa2(ArrayView<float> q, ArrayView<ILGPU.Half> k, ArrayView<ILGPU.Half> v, ArrayView<int> pg, ArrayView<int> ti, ArrayView<float> o, ArrayView<float> pt, AttnParams p) =>
        Kernels.AttnDecodeGqa(q, k, v, pg, ti, o, pt, p, SpecializedValue.New(2), SpecializedValue.New(128));
}
