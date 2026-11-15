using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace LammaNet.Gpu;

/// <summary>
/// Tensor-core throughput probes: back-to-back independent mma instructions on registers (no memory traffic), to
/// measure the device's real ceilings for the GEMM formats (fp16 with fp32 or fp16 accumulation, int8).
/// </summary>
internal static class MmaPeakKernels
{
    /// <summary>fp16 inputs, fp32 accumulation, m16n8k16: 8 independent accumulator tiles per warp.</summary>
    public static void F16F32(ArrayView<float> outp, int iters)
    {
        uint a = (uint)Group.IdxX * 0x3C003C00u, b = (uint)Grid.IdxX * 0x00010001u + 0x3C003C00u; // runtime values: mma takes no immediates
        CudaAsm.Emit(".reg .f32 zf<32>;\n\tmov.f32 zf0, 0f00000000;\n\tmov.f32 zf1, 0f00000000;\n\tmov.f32 zf2, 0f00000000;\n\tmov.f32 zf3, 0f00000000;\n\tmov.f32 zf4, 0f00000000;\n\tmov.f32 zf5, 0f00000000;\n\tmov.f32 zf6, 0f00000000;\n\tmov.f32 zf7, 0f00000000;\n\tmov.f32 zf8, 0f00000000;\n\tmov.f32 zf9, 0f00000000;\n\tmov.f32 zf10, 0f00000000;\n\tmov.f32 zf11, 0f00000000;\n\tmov.f32 zf12, 0f00000000;\n\tmov.f32 zf13, 0f00000000;\n\tmov.f32 zf14, 0f00000000;\n\tmov.f32 zf15, 0f00000000;\n\tmov.f32 zf16, 0f00000000;\n\tmov.f32 zf17, 0f00000000;\n\tmov.f32 zf18, 0f00000000;\n\tmov.f32 zf19, 0f00000000;\n\tmov.f32 zf20, 0f00000000;\n\tmov.f32 zf21, 0f00000000;\n\tmov.f32 zf22, 0f00000000;\n\tmov.f32 zf23, 0f00000000;\n\tmov.f32 zf24, 0f00000000;\n\tmov.f32 zf25, 0f00000000;\n\tmov.f32 zf26, 0f00000000;\n\tmov.f32 zf27, 0f00000000;\n\tmov.f32 zf28, 0f00000000;\n\tmov.f32 zf29, 0f00000000;\n\tmov.f32 zf30, 0f00000000;\n\tmov.f32 zf31, 0f00000000;");
        for (int i = 0; i < iters; i++)
        {
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf0,zf1,zf2,zf3}, {%0,%0,%0,%0}, {%1,%1}, {zf0,zf1,zf2,zf3};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf4,zf5,zf6,zf7}, {%0,%0,%0,%0}, {%1,%1}, {zf4,zf5,zf6,zf7};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf8,zf9,zf10,zf11}, {%0,%0,%0,%0}, {%1,%1}, {zf8,zf9,zf10,zf11};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf12,zf13,zf14,zf15}, {%0,%0,%0,%0}, {%1,%1}, {zf12,zf13,zf14,zf15};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf16,zf17,zf18,zf19}, {%0,%0,%0,%0}, {%1,%1}, {zf16,zf17,zf18,zf19};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf20,zf21,zf22,zf23}, {%0,%0,%0,%0}, {%1,%1}, {zf20,zf21,zf22,zf23};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf24,zf25,zf26,zf27}, {%0,%0,%0,%0}, {%1,%1}, {zf24,zf25,zf26,zf27};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {zf28,zf29,zf30,zf31}, {%0,%0,%0,%0}, {%1,%1}, {zf28,zf29,zf30,zf31};", a, b);
        }
        CudaAsm.Emit("add.f32 zf0, zf0, zf4;\n\tadd.f32 zf0, zf0, zf8;\n\tadd.f32 zf0, zf0, zf12;\n\tadd.f32 zf0, zf0, zf16;\n\tadd.f32 zf0, zf0, zf20;\n\tadd.f32 zf0, zf0, zf24;\n\tadd.f32 zf0, zf0, zf28;");
        CudaAsm.Emit("mov.f32 %0, zf0;", out float s);
        outp[Grid.IdxX * Group.DimX + Group.IdxX] = s;
    }

    /// <summary>int8 inputs, int32 accumulation, m16n8k32: 8 independent accumulator tiles per warp.</summary>
    public static void S8S32(ArrayView<float> outp, int iters)
    {
        uint a = (uint)Group.IdxX * 0x01010101u, b = (uint)Grid.IdxX * 0x01010101u + 0x01010101u;
        CudaAsm.Emit(".reg .s32 zi<32>;\n\tmov.s32 zi0, 0;\n\tmov.s32 zi1, 0;\n\tmov.s32 zi2, 0;\n\tmov.s32 zi3, 0;\n\tmov.s32 zi4, 0;\n\tmov.s32 zi5, 0;\n\tmov.s32 zi6, 0;\n\tmov.s32 zi7, 0;\n\tmov.s32 zi8, 0;\n\tmov.s32 zi9, 0;\n\tmov.s32 zi10, 0;\n\tmov.s32 zi11, 0;\n\tmov.s32 zi12, 0;\n\tmov.s32 zi13, 0;\n\tmov.s32 zi14, 0;\n\tmov.s32 zi15, 0;\n\tmov.s32 zi16, 0;\n\tmov.s32 zi17, 0;\n\tmov.s32 zi18, 0;\n\tmov.s32 zi19, 0;\n\tmov.s32 zi20, 0;\n\tmov.s32 zi21, 0;\n\tmov.s32 zi22, 0;\n\tmov.s32 zi23, 0;\n\tmov.s32 zi24, 0;\n\tmov.s32 zi25, 0;\n\tmov.s32 zi26, 0;\n\tmov.s32 zi27, 0;\n\tmov.s32 zi28, 0;\n\tmov.s32 zi29, 0;\n\tmov.s32 zi30, 0;\n\tmov.s32 zi31, 0;");
        for (int i = 0; i < iters; i++)
        {
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi0,zi1,zi2,zi3}, {%0,%0,%0,%0}, {%1,%1}, {zi0,zi1,zi2,zi3};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi4,zi5,zi6,zi7}, {%0,%0,%0,%0}, {%1,%1}, {zi4,zi5,zi6,zi7};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi8,zi9,zi10,zi11}, {%0,%0,%0,%0}, {%1,%1}, {zi8,zi9,zi10,zi11};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi12,zi13,zi14,zi15}, {%0,%0,%0,%0}, {%1,%1}, {zi12,zi13,zi14,zi15};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi16,zi17,zi18,zi19}, {%0,%0,%0,%0}, {%1,%1}, {zi16,zi17,zi18,zi19};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi20,zi21,zi22,zi23}, {%0,%0,%0,%0}, {%1,%1}, {zi20,zi21,zi22,zi23};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi24,zi25,zi26,zi27}, {%0,%0,%0,%0}, {%1,%1}, {zi24,zi25,zi26,zi27};", a, b);
            CudaAsm.Emit("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {zi28,zi29,zi30,zi31}, {%0,%0,%0,%0}, {%1,%1}, {zi28,zi29,zi30,zi31};", a, b);
        }
        CudaAsm.Emit("add.s32 zi0, zi0, zi4;\n\tadd.s32 zi0, zi0, zi8;\n\tadd.s32 zi0, zi0, zi12;\n\tadd.s32 zi0, zi0, zi16;\n\tadd.s32 zi0, zi0, zi20;\n\tadd.s32 zi0, zi0, zi24;\n\tadd.s32 zi0, zi0, zi28;");
        CudaAsm.Emit("mov.s32 %0, zi0;", out int s);
        outp[Grid.IdxX * Group.DimX + Group.IdxX] = s;
    }
}

public static partial class GpuBenchPeaks
{
    /// <summary>Measured tensor-core throughput (TFLOPS / TOPS) of fp16→fp32 and int8→int32 mma on device 0.</summary>
    public static IEnumerable<string> Run()
    {
        using var ctx = Context.Create(b => b.Default().Profiling());
        using var acc = ctx.GetCudaDevice(0).CreateAccelerator(ctx);
        int sms = acc.MaxNumThreads / acc.MaxNumThreadsPerMultiprocessor;
        int blocks = Math.Max(1, sms) * 16, threads = 128, iters = 2048;
        using var outp = acc.Allocate1D<float>(blocks * threads);
        foreach (var (name, flopsPerMma, kernel) in new (string, double, Action<ArrayView<float>, int>)[]
                 { ("fp16 x fp16 -> fp32 (m16n8k16)", 16 * 8 * 16 * 2, MmaPeakKernels.F16F32), ("int8 x int8 -> int32 (m16n8k32)", 16 * 8 * 32 * 2, MmaPeakKernels.S8S32) })
        {
            var cuda = (CudaAccelerator)acc;
            var entry = ILGPU.Backends.EntryPoints.EntryPointDescription.FromExplicitlyGroupedKernel(kernel.Method);
            var ptx = (ILGPU.Backends.PTX.PTXCompiledKernel)cuda.Backend.Compile(entry, new KernelSpecialization());
            if (CudaAPI.CurrentAPI.LoadModule(out IntPtr module, ptx.PTXAssembly, out string log) != CudaError.CUDA_SUCCESS)
            {
                File.WriteAllText(Path.Combine(Path.GetTempPath(), "mma-peak.ptx"), ptx.PTXAssembly);
                yield return $"{name}: PTX JIT failed: {log}";
                continue;
            }
            CudaAPI.CurrentAPI.DestroyModule(module);
            var k = acc.LoadStreamKernel(kernel);
            k(new KernelConfig(blocks, threads), outp.View, 16);
            acc.Synchronize();
            using var m0 = acc.DefaultStream.AddProfilingMarker();
            k(new KernelConfig(blocks, threads), outp.View, iters);
            using var m1 = acc.DefaultStream.AddProfilingMarker();
            acc.Synchronize();
            double s = m1.MeasureFrom(m0).TotalSeconds;
            double total = (double)blocks * (threads / 32) * iters * 8 * flopsPerMma;
            yield return $"{name,-34} {total / s / 1e12,7:F1} T/s  ({acc.Name}, {sms} SMs)";
        }
    }
}
