using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Memory;

/// <summary>
/// Device-resident storage for the recurrent (DeltaNet / Mamba) layer states of live sequences, provided by an
/// accelerator offload. A slot holds every owned layer's conv window and state; host layouts are used for transfers.
/// </summary>
public unsafe interface IRecurrentStateStore
{
    /// <summary>True if <paramref name="layer"/>'s state lives in the store (the host keeps no buffer for it).</summary>
    bool OwnsLayer(int layer);
    /// <summary>Allocates a zeroed slot.</summary>
    int Allocate();
    void Free(int slot);
    void Reset(int slot);
    /// <summary>Copies all owned layers of one slot to another.</summary>
    void Copy(int srcSlot, int dstSlot);
    /// <summary>Reads one layer of a slot into host buffers (conv window, state).</summary>
    void Download(int slot, int layer, float* conv, float* ssm);
    /// <summary>Writes one layer of a slot from host buffers.</summary>
    void Upload(int slot, int layer, float* conv, float* ssm);
}

/// <summary>
/// Recurrent state of the linear-attention / SSM layers of a hybrid model for one sequence:
/// a short convolution window and the per-head state matrices. With an <see cref="IRecurrentStateStore"/>, a live
/// state keeps the store's layers in a device slot; snapshots (<see cref="Clone"/>) are always host copies.
/// </summary>
public sealed unsafe class RecurrentState : IDisposable
{
    private readonly NativeBuffer?[] _conv;
    private readonly NativeBuffer?[] _ssm;
    private readonly IRecurrentStateStore? _store;
    private readonly bool[] _onDevice;
    public int[] ConvSize { get; }
    public int[] SsmSize { get; }
    /// <summary>Device slot of a live state with store-owned layers, or -1.</summary>
    public int Slot { get; private set; } = -1;

    /// <param name="c">Model configuration.</param>
    /// <param name="store">Device store for the layers it owns (null: everything on the host).</param>
    /// <param name="hostOnly">Keep every layer on the host (snapshots), even with a store.</param>
    public RecurrentState(ModelConfig c, IRecurrentStateStore? store = null, bool hostOnly = false)
    {
        int n = c.NLayer;
        _store = store;
        _conv = new NativeBuffer?[n];
        _ssm = new NativeBuffer?[n];
        _onDevice = new bool[n];
        ConvSize = new int[n];
        SsmSize = new int[n];
        for (int il = 0; il < n; il++)
        {
            switch (c.Kinds[il])
            {
                case LayerKind.GatedDeltaNet:
                {
                    int headK = c.SsmStateSize, nK = c.SsmGroupCount, nV = c.SsmTimeStepRank;
                    int headV = c.SsmInnerSize / nV;
                    int convDim = headK * nK * 2 + headV * nV;
                    ConvSize[il] = (c.SsmConvKernel - 1) * convDim;
                    SsmSize[il] = headV * headV * nV;
                    break;
                }
                case LayerKind.Mamba2:
                {
                    int convDim = c.SsmInnerSize + 2 * c.SsmGroupCount * c.SsmStateSize;
                    ConvSize[il] = (c.SsmConvKernel - 1) * convDim;
                    SsmSize[il] = c.SsmStateSize * c.SsmInnerSize;
                    break;
                }
                case LayerKind.ShortConv:
                    ConvSize[il] = (c.SsmConvKernel - 1) * c.NEmbd;
                    SsmSize[il] = 0;
                    break;
                default: continue;
            }
            if (store != null && !hostOnly && store.OwnsLayer(il)) { _onDevice[il] = true; continue; }
            _conv[il] = new NativeBuffer(Math.Max(1, ConvSize[il]) * 4L);
            _ssm[il] = new NativeBuffer(Math.Max(1, SsmSize[il]) * 4L);
        }
        if (_onDevice.Any(d => d)) Slot = store!.Allocate();
    }

    public float* Conv(int layer) => (_conv[layer] ?? throw new InvalidOperationException($"Recurrent state of layer {layer} lives on the device")).F;
    public float* Ssm(int layer) => (_ssm[layer] ?? throw new InvalidOperationException($"Recurrent state of layer {layer} lives on the device")).F;

    /// <summary>Host memory in use (device slots not included).</summary>
    public long Bytes => _conv.Sum(b => b?.Length ?? 0) + _ssm.Sum(b => b?.Length ?? 0);

    public void Reset()
    {
        foreach (var b in _conv) if (b != null) System.Runtime.InteropServices.NativeMemory.Clear(b.Pointer, (nuint)b.Length);
        foreach (var b in _ssm) if (b != null) System.Runtime.InteropServices.NativeMemory.Clear(b.Pointer, (nuint)b.Length);
        if (Slot >= 0) _store!.Reset(Slot);
    }

    public void CopyFrom(RecurrentState other)
    {
        if (Slot >= 0 && other.Slot >= 0) _store!.Copy(other.Slot, Slot);
        for (int i = 0; i < _conv.Length; i++)
        {
            if (ConvSize[i] == 0 && SsmSize[i] == 0) continue;
            bool dst = _onDevice[i], src = other._onDevice[i];
            if (dst && src) continue; // copied above
            if (dst) _store!.Upload(Slot, i, other._conv[i]!.F, other._ssm[i]!.F);
            else if (src) other._store!.Download(other.Slot, i, _conv[i]!.F, _ssm[i]!.F);
            else
            {
                Buffer.MemoryCopy(other._conv[i]!.Pointer, _conv[i]!.Pointer, _conv[i]!.Length, _conv[i]!.Length);
                Buffer.MemoryCopy(other._ssm[i]!.Pointer, _ssm[i]!.Pointer, _ssm[i]!.Length, _ssm[i]!.Length);
            }
        }
    }

    /// <summary>A host-resident copy (snapshot) of this state.</summary>
    public RecurrentState Clone(ModelConfig c)
    {
        var r = new RecurrentState(c, _store, hostOnly: true);
        r.CopyFrom(this);
        return r;
    }

    public void Dispose()
    {
        foreach (var b in _conv) b?.Dispose();
        foreach (var b in _ssm) b?.Dispose();
        if (Slot >= 0) { _store!.Free(Slot); Slot = -1; }
    }
}

/// <summary>
/// The memory of one sequence: its token history, the KV pages backing it, and (for hybrid models)
/// its recurrent state plus snapshots used to resume from a shared prefix.
/// </summary>
public sealed class SequenceState : IDisposable
{
    private static int s_nextId;
    private readonly KvCache? _kv;
    internal readonly List<int> Pages = new();
    internal readonly List<int> TokenList = new();

    public int Id { get; } = Interlocked.Increment(ref s_nextId);
    /// <summary>KV page ids backing positions [0, Length) (page i holds positions i*PageSize ...).</summary>
    public IReadOnlyList<int> PageTable => Pages;
    /// <summary>Number of positions stored (tokens + embedded multimodal positions).</summary>
    public int Length { get; internal set; }
    /// <summary>Token ids per position (-1 for positions filled from raw embeddings, e.g. image patches).</summary>
    public IReadOnlyList<int> Tokens => TokenList;
    public RecurrentState? Recurrent { get; internal set; }
    /// <summary>Snapshots of the recurrent state at given lengths (for prefix reuse on hybrid models).</summary>
    internal readonly SortedDictionary<int, RecurrentState> Snapshots = new();
    internal long LastUsedTicks = Environment.TickCount64;

    internal SequenceState(KvCache? kv, ModelConfig config)
    {
        _kv = kv;
        if (config.HasRecurrentLayers) Recurrent = new RecurrentState(config, kv?.RecurrentStore);
    }

    internal int PageSize => _kv?.PageSize ?? int.MaxValue;

    /// <summary>Makes sure KV pages exist for positions [0, <paramref name="length"/>).</summary>
    internal bool EnsureCapacity(int length)
    {
        if (_kv == null) return true;
        int need = (length + _kv.PageSize - 1) / _kv.PageSize;
        while (Pages.Count < need)
        {
            if (!_kv.TryAllocatePage(out int p)) return false;
            Pages.Add(p);
        }
        return true;
    }

    /// <summary>Truncates the sequence to <paramref name="length"/> positions, releasing unused pages.</summary>
    internal void Truncate(int length)
    {
        if (length >= Length) return;
        Length = length;
        if (TokenList.Count > length) TokenList.RemoveRange(length, TokenList.Count - length);
        if (_kv != null)
        {
            int need = (length + _kv.PageSize - 1) / _kv.PageSize;
            for (int i = Pages.Count - 1; i >= need; i--) { _kv.Release(Pages[i]); Pages.RemoveAt(i); }
        }
        foreach (var k in Snapshots.Keys.Where(k => k > length).ToList())
        {
            Snapshots[k].Dispose();
            Snapshots.Remove(k);
        }
    }

    /// <summary>
    /// Initializes this (empty) sequence with the first <paramref name="length"/> positions of <paramref name="src"/>,
    /// sharing full KV pages and copying the trailing partial page.
    /// </summary>
    internal bool CopyPrefixFrom(SequenceState src, int length, ModelConfig config)
    {
        if (Length != 0) throw new InvalidOperationException("Sequence must be empty");
        if (_kv != null)
        {
            int full = length / _kv.PageSize, rem = length % _kv.PageSize;
            for (int i = 0; i < full; i++) { _kv.Retain(src.Pages[i]); Pages.Add(src.Pages[i]); }
            if (rem > 0)
            {
                if (!_kv.TryAllocatePage(out int p)) { Truncate(0); return false; }
                _kv.CopyPage(src.Pages[full], p, rem);
                Pages.Add(p);
            }
        }
        TokenList.AddRange(src.TokenList.Take(length));
        Length = length;
        if (Recurrent != null)
        {
            if (length == src.Length && src.Recurrent != null) Recurrent.CopyFrom(src.Recurrent);
            else if (src.Snapshots.TryGetValue(length, out var snap)) Recurrent.CopyFrom(snap);
            else if (length == 0) Recurrent.Reset();
            else throw new InvalidOperationException("No recurrent state available at the requested prefix length");
        }
        return true;
    }

    /// <summary>
    /// Context shift: drops the <paramref name="discardPages"/> pages after the first <paramref name="keepPages"/>, so
    /// later positions move down by discardPages·PageSize; pages after the kept ones that other sequences share are
    /// copied first (their keys are about to be re-rotated). Returns the number of positions removed.
    /// </summary>
    internal int DiscardPages(int keepPages, int discardPages)
    {
        if (_kv == null || Recurrent != null) throw new InvalidOperationException("Context shift needs an attention-only KV cache");
        int ps = _kv.PageSize, removed = discardPages * ps;
        for (int i = 0; i < discardPages; i++) _kv.Release(Pages[keepPages + i]);
        Pages.RemoveRange(keepPages, discardPages);
        TokenList.RemoveRange(keepPages * ps, Math.Min(removed, TokenList.Count - keepPages * ps));
        Length -= removed;
        for (int i = keepPages; i < Pages.Count; i++)
        {
            if (_kv.RefCount(Pages[i]) <= 1) continue;
            if (!_kv.TryAllocatePage(out int p)) throw new KvCacheFullException("KV cache is full");
            _kv.CopyPage(Pages[i], p, ps);
            _kv.Release(Pages[i]);
            Pages[i] = p;
        }
        return removed;
    }

    /// <summary>Stores a snapshot of the recurrent state at the current length.</summary>
    internal void Snapshot(ModelConfig config, int maxSnapshots = 4)
    {
        if (Recurrent == null) return;
        if (Snapshots.TryGetValue(Length, out var existing)) { existing.CopyFrom(Recurrent); return; }
        while (Snapshots.Count >= maxSnapshots)
        {
            var first = Snapshots.Keys.First();
            Snapshots[first].Dispose();
            Snapshots.Remove(first);
        }
        Snapshots[Length] = Recurrent.Clone(config);
    }

    public void Dispose()
    {
        if (_kv != null) foreach (var p in Pages) _kv.Release(p);
        Pages.Clear();
        Recurrent?.Dispose();
        foreach (var s in Snapshots.Values) s.Dispose();
        Snapshots.Clear();
    }
}
