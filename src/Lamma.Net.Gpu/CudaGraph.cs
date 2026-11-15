using System.Runtime.InteropServices;

namespace LammaNet.Gpu;

/// <summary>
/// Minimal CUDA driver bindings for stream capture into CUDA graphs. A graph replays a fixed sequence of kernel
/// launches with one driver call, removing the per-launch CPU cost (several µs per ILGPU launch).
/// </summary>
internal static class CudaGraphApi
{
    private const string Lib = "nvcuda";
    /// <summary>CU_STREAM_CAPTURE_MODE_RELAXED: other API calls on this thread are not restricted during capture.</summary>
    public const int CaptureModeRelaxed = 2;

    [DllImport(Lib, EntryPoint = "cuStreamBeginCapture_v2")] public static extern int StreamBeginCapture(IntPtr stream, int mode);
    [DllImport(Lib, EntryPoint = "cuStreamEndCapture")] public static extern int StreamEndCapture(IntPtr stream, out IntPtr graph);
    [DllImport(Lib, EntryPoint = "cuGraphInstantiateWithFlags")] public static extern int GraphInstantiate(out IntPtr exec, IntPtr graph, ulong flags);
    [DllImport(Lib, EntryPoint = "cuGraphLaunch")] public static extern int GraphLaunch(IntPtr exec, IntPtr stream);
    [DllImport(Lib, EntryPoint = "cuGraphExecDestroy")] public static extern int GraphExecDestroy(IntPtr exec);
    [DllImport(Lib, EntryPoint = "cuGraphDestroy")] public static extern int GraphDestroy(IntPtr graph);

    public static void Check(int result, string what)
    {
        if (result != 0) throw new InvalidOperationException($"{what} failed with CUDA error {result}");
    }
}

/// <summary>An instantiated CUDA graph captured from a stream.</summary>
internal sealed class CudaGraph : IDisposable
{
    private IntPtr _graph, _exec;

    private CudaGraph(IntPtr graph, IntPtr exec) { _graph = graph; _exec = exec; }

    /// <summary>Captures the work <paramref name="record"/> enqueues on <paramref name="stream"/> (nothing executes).</summary>
    public static CudaGraph Capture(IntPtr stream, Action record)
    {
        CudaGraphApi.Check(CudaGraphApi.StreamBeginCapture(stream, CudaGraphApi.CaptureModeRelaxed), "cuStreamBeginCapture");
        try
        {
            record();
        }
        catch
        {
            CudaGraphApi.StreamEndCapture(stream, out var g);
            if (g != IntPtr.Zero) CudaGraphApi.GraphDestroy(g);
            throw;
        }
        CudaGraphApi.Check(CudaGraphApi.StreamEndCapture(stream, out var graph), "cuStreamEndCapture");
        int r = CudaGraphApi.GraphInstantiate(out var exec, graph, 0);
        if (r != 0) { CudaGraphApi.GraphDestroy(graph); CudaGraphApi.Check(r, "cuGraphInstantiate"); }
        return new CudaGraph(graph, exec);
    }

    public void Launch(IntPtr stream) => CudaGraphApi.Check(CudaGraphApi.GraphLaunch(_exec, stream), "cuGraphLaunch");

    public void Dispose()
    {
        if (_exec != IntPtr.Zero) { CudaGraphApi.GraphExecDestroy(_exec); _exec = IntPtr.Zero; }
        if (_graph != IntPtr.Zero) { CudaGraphApi.GraphDestroy(_graph); _graph = IntPtr.Zero; }
    }
}
