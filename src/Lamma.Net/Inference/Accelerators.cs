using System.Reflection;
using LammaNet.Models;

namespace LammaNet.Inference;

/// <summary>
/// Discovers an installed accelerator package (Lamma.Net.Gpu) so engines use a supported GPU automatically: when the
/// package's assembly can be loaded, its <c>GpuLayerOffload.TryCreate(Model, OffloadGeometry)</c> is called, which
/// returns null (CPU only) when no usable device is present. Set LAMMA_DEVICE=cpu to disable.
/// </summary>
public static class Accelerators
{
    private static readonly Lazy<Func<Model, OffloadGeometry, ILayerOffload?>?> s_factory = new(Discover);

    /// <summary>True if an accelerator package is installed (a device may still be missing).</summary>
    public static bool PackageAvailable => s_factory.Value != null;

    /// <summary>Creates the default accelerator offload for <paramref name="model"/>, or null to run on the CPU.</summary>
    public static ILayerOffload? TryCreate(Model model, OffloadGeometry geometry)
    {
        var log = Diagnostics.LammaLogging.Create(geometry.LoggerFactory, "LammaNet.Inference.Accelerators");
        if (string.Equals(Environment.GetEnvironmentVariable("LAMMA_DEVICE"), "cpu", StringComparison.OrdinalIgnoreCase))
        {
            Diagnostics.Log.AcceleratorUnavailable(log, "LAMMA_DEVICE=cpu");
            return null;
        }
        if (s_factory.Value is not { } factory)
        {
            Diagnostics.Log.AcceleratorUnavailable(log, s_discoveryError != null ? "the accelerator package failed to load" : "no accelerator package installed");
            return null;
        }
        try
        {
            return factory(model, geometry);   // logs its own outcome
        }
        catch (Exception e)
        {
            Diagnostics.Log.AcceleratorFailed(log, e);
            return null;
        }
    }

    private static Exception? s_discoveryError;

    private static Func<Model, OffloadGeometry, ILayerOffload?>? Discover()
    {
        try
        {
            var type = Type.GetType("LammaNet.Gpu.GpuLayerOffload, Lamma.Net.Gpu", throwOnError: false);
            var m = type?.GetMethod("TryCreate", BindingFlags.Public | BindingFlags.Static, [typeof(Model), typeof(OffloadGeometry)]);
            return m == null ? null : (Func<Model, OffloadGeometry, ILayerOffload?>)Delegate.CreateDelegate(typeof(Func<Model, OffloadGeometry, ILayerOffload?>), m);
        }
        catch (Exception e)
        {
            s_discoveryError = e;
            Diagnostics.Log.AcceleratorDiscoveryFailed(Diagnostics.LammaLogging.Static("LammaNet.Inference.Accelerators"), e);
            return null;
        }
    }
}
