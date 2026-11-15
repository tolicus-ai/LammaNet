using LammaNet.Compute;
using LammaNet.Memory;

namespace LammaNet.Inference;

/// <summary>Inputs for running a range of layers on an accelerator.</summary>
public sealed unsafe class OffloadBatch
{
    /// <summary>Hidden states [N, NEmbd] on the host; updated in place with the device result.</summary>
    public float* X;
    public int N;
    public required SequenceState[] Sequences;
    public required int[] Positions;
    /// <summary>
    /// Per-layer embedding inputs [N, NLayer * NEmbdPerLayer] (Gemma 3n/4) or null. When the offload
    /// <see cref="ILayerOffload.ComputesPerLayerInputs"/>, this holds only the scaled per-layer token embeddings and the
    /// offload adds the projection of the input embeddings itself.
    /// </summary>
    public float* PerLayerInputs;
    /// <summary>Host-computed rotary caches per rope configuration ([N][dims] interleaved cos/sin).</summary>
    public required IReadOnlyDictionary<RopeConfig, nint> RopeCaches;
    /// <summary>
    /// When non-null (only passed if <see cref="ILayerOffload.ComputesHead"/>), the offload also applies the final norm
    /// and LM head to these batch rows and writes their logits, in order, to <see cref="Logits"/>; X is then not updated.
    /// </summary>
    public int[]? OutputRows;
    public float* Logits;
}

/// <summary>
/// Executes the first <see cref="LayerCount"/> decoder layers on an accelerator (GPU), enabling CPU/GPU layer splitting.
/// The KV cache of those layers lives in device memory, addressed with the same page/slot indices as the host <see cref="KvCache"/>.
/// </summary>
public interface ILayerOffload : IDisposable
{
    /// <summary>Layers [0, LayerCount) are executed by this offload.</summary>
    int LayerCount { get; }
    /// <summary>Human-readable device description.</summary>
    string DeviceName { get; }
    /// <summary>True when every layer is offloaded and the device can also produce logits (see <see cref="OffloadBatch.OutputRows"/>).</summary>
    bool ComputesHead => false;
    /// <summary>True when the device computes the per-layer input projection (see <see cref="OffloadBatch.PerLayerInputs"/>).</summary>
    bool ComputesPerLayerInputs => false;
    /// <summary>Device storage of offloaded recurrent layers' states (pass it to the <see cref="KvCache"/>), or null.</summary>
    IRecurrentStateStore? RecurrentStore => null;
    /// <summary>Runs the offloaded layers over the batch (host → device → host).</summary>
    void Forward(OffloadBatch batch);
    /// <summary>True if the KV cache of <paramref name="layer"/> lives on the device (the host cache then skips it).</summary>
    bool OwnsKv(int layer);
    /// <summary>Copies the first <paramref name="slots"/> token rows of KV page <paramref name="src"/> to <paramref name="dst"/> (prefix sharing).</summary>
    void CopyPage(int src, int dst, int slots);
    /// <summary>Rotates device-owned K rows of positions [from, to) of a page table by <paramref name="delta"/> rotary positions (context shift).</summary>
    void RotateKeys(IReadOnlyList<int> pageTable, int from, int to, int delta) => throw new NotSupportedException("Context shift is not supported by this offload");
    /// <summary>
    /// Registers a page-aligned host range the batches use (activations, logits) so transfers to and from it are direct
    /// DMA instead of being staged through a driver bounce buffer; dispose the handle before freeing the memory.
    /// Null when the device does not need or support it.
    /// </summary>
    unsafe IDisposable? PinHostMemory(void* pointer, long bytes) => null;

    /// <summary>
    /// A drafter that runs on the device against the device KV cache and the hidden states of the head's output rows
    /// (only when <see cref="ComputesHead"/>); null when the device cannot run it.
    /// </summary>
    IDraftRunner? CreateMtpDrafter(Models.MtpDrafterModel drafter, Models.Model target) => null;
}

/// <summary>KV-cache geometry handed to an offload factory.</summary>
public sealed record OffloadGeometry(int PageSize, int PageCount, int MaxBatch)
{
    /// <summary>Logging of the engine creating the offload (null: <see cref="Diagnostics.LammaLogging.LoggerFactory"/>).</summary>
    public Microsoft.Extensions.Logging.ILoggerFactory? LoggerFactory { get; init; }
    /// <summary>Resource limits of the engine (device memory cap); null: <see cref="Compute.ResourceLimiter.Default"/>.</summary>
    public Compute.ResourceLimiter? Resources { get; init; }
}
