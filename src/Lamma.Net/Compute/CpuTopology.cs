using System.Runtime.InteropServices;

namespace LammaNet.Compute;

/// <summary>
/// CPU topology queries used to pick defaults: compute threads beyond one per physical core (SMT siblings) do not
/// help the bandwidth-bound decode and hurt prefill through contention.
/// </summary>
public static unsafe class CpuTopology
{
    private static readonly Lazy<int> s_physical = new(DetectPhysicalCores);

    /// <summary>Physical cores available to the process (falls back to a heuristic when unknown).</summary>
    public static int PhysicalCores => s_physical.Value;

    private static int DetectPhysicalCores()
    {
        int logical = Environment.ProcessorCount;
        int? n = null;
        try
        {
            if (OperatingSystem.IsWindows()) n = WindowsCores();
            else if (OperatingSystem.IsLinux()) n = LinuxCores();
            else if (OperatingSystem.IsMacOS()) n = MacCores();
        }
        catch (Exception e)
        {
            Diagnostics.Log.CpuTopologyFallback(Diagnostics.LammaLogging.Static("LammaNet.Compute.CpuTopology"), e, logical > 4 ? Math.Max(1, logical / 2) : logical);
        }
        if (n is > 0) return Math.Min(n.Value, logical);
        return logical > 4 ? Math.Max(1, logical / 2) : logical;
    }

    // ---- Windows: one RelationProcessorCore record per physical core

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(int relationship, byte* buffer, ref uint length);

    private static int? WindowsCores()
    {
        const int RelationProcessorCore = 0;
        uint len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, null, ref len);
        if (len == 0) return null;
        var buf = new byte[len];
        fixed (byte* p = buf)
        {
            if (!GetLogicalProcessorInformationEx(RelationProcessorCore, p, ref len)) return null;
            int cores = 0;
            for (uint off = 0; off < len;)
            {
                uint size = *(uint*)(p + off + 4); // { int Relationship; uint Size; ... }
                if (size == 0) break;
                cores++;
                off += size;
            }
            return cores;
        }
    }

    // ---- Linux: distinct (package, core) pairs of online CPUs

    private static int? LinuxCores()
    {
        var set = new HashSet<(string, string)>();
        foreach (var dir in Directory.GetDirectories("/sys/devices/system/cpu", "cpu*"))
        {
            if (!int.TryParse(Path.GetFileName(dir)[3..], out _)) continue;
            string core = Path.Combine(dir, "topology", "core_id"), pkg = Path.Combine(dir, "topology", "physical_package_id");
            if (!File.Exists(core)) continue;
            string online = Path.Combine(dir, "online");
            if (File.Exists(online) && File.ReadAllText(online).Trim() == "0") continue;
            set.Add((File.Exists(pkg) ? File.ReadAllText(pkg).Trim() : "0", File.ReadAllText(core).Trim()));
        }
        return set.Count > 0 ? set.Count : null;
    }

    // ---- macOS

    [DllImport("libc")]
    private static extern int sysctlbyname(string name, int* oldp, nint* oldlenp, void* newp, nint newlen);

    private static int? MacCores()
    {
        int v = 0;
        nint len = sizeof(int);
        return sysctlbyname("hw.physicalcpu", &v, &len, null, 0) == 0 && v > 0 ? v : null;
    }
}
