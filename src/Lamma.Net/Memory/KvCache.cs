using LammaNet.Compute;
using LammaNet.Models;
using LammaNet.Tensors;

namespace LammaNet.Memory;

/// <summary>Storage precision of the KV cache.</summary>
public enum KvCacheType
{
    F32,
    /// <summary>IEEE half precision (default; same as llama.cpp).</summary>
    F16,
    /// <summary>8-bit block quantization (Q8_0), using 34 bytes per 32 values on compatible heads.</summary>
    Q8_0,
}

/// <summary>
/// Paged KV cache (vLLM-style). Memory is split into fixed-size pages of <see cref="PageSize"/> tokens that are
/// reference counted, so sequences can share common prefixes (prompt caching, forks for parallel sampling)
/// without copies. Each attention layer that owns KV state gets its own K and V page pools.
/// </summary>
public sealed unsafe class KvCache : IDisposable
{
    public int PageSize { get; }
    public int PageCount { get; }
    public KvCacheType Type { get; }
    public ModelConfig Config { get; }
    /// <summary>Device store for recurrent-layer states of sequences on this cache (accelerator offload), or null.</summary>
    public IRecurrentStateStore? RecurrentStore { get; }

    private readonly NativeBuffer?[] _k;
    private readonly NativeBuffer?[] _v;
    private readonly long[] _kRowBytes;
    private readonly long[] _vRowBytes;
    private readonly int[] _kDim;
    private readonly int[] _vDim;
    private readonly int[] _refCount;
    private readonly Stack<int> _free = new();
    private readonly object _lock = new();

    /// <summary>Invoked when a page is copied (so device-resident KV layers can mirror the copy).</summary>
    public Action<int, int, int>? PageCopied { get; set; }

    /// <param name="hostLayer">Optional predicate: layers for which it returns false keep their KV on an accelerator (no host storage).</param>
    public KvCache(ModelConfig config, int maxTokens, KvCacheType type = KvCacheType.F16, int pageSize = 32, Func<int, bool>? hostLayer = null,
        IRecurrentStateStore? recurrentStore = null)
    {
        RecurrentStore = recurrentStore;
        Config = config;
        Type = type;
        PageSize = pageSize;
        PageCount = PagesFor(maxTokens, pageSize);
        int n = config.NLayer;
        _k = new NativeBuffer?[n];
        _v = new NativeBuffer?[n];
        _kRowBytes = new long[n];
        _vRowBytes = new long[n];
        _kDim = new int[n];
        _vDim = new int[n];
        var gt = ElementType;
        for (int il = 0; il < n; il++)
        {
            if (config.Kinds[il] != LayerKind.Attention || config.KvSourceLayer[il] != il) continue;
            if (hostLayer != null && !hostLayer(il)) continue;
            _kDim[il] = config.NHeadKv[il] * config.HeadDimK[il];
            _vDim[il] = config.NHeadKv[il] * config.HeadDimV[il];
            _kRowBytes[il] = RowBytes(gt, config.HeadDimK[il]) * config.NHeadKv[il];
            _vRowBytes[il] = RowBytes(gt, config.HeadDimV[il]) * config.NHeadKv[il];
            _k[il] = new NativeBuffer(_kRowBytes[il] * PageCount * pageSize, zero: true);
            _v[il] = new NativeBuffer(_vRowBytes[il] * PageCount * pageSize, zero: true);
        }
        _refCount = new int[PageCount];
        for (int p = PageCount - 1; p >= 0; p--) _free.Push(p);
    }

    internal static int PagesFor(int tokens, int pageSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pageSize);
        return tokens == 0 ? 1 : 1 + (tokens - 1) / pageSize;
    }

    private static long RowBytes(GgmlType t, int headDim) =>
        t == GgmlType.Q8_0 && headDim % 32 != 0 ? headDim * 2L : GgmlTypeInfo.RowSize(t, headDim);

    /// <summary>Element type used per head row.</summary>
    public GgmlType ElementType => Type switch
    {
        KvCacheType.F32 => GgmlType.F32,
        KvCacheType.Q8_0 => GgmlType.Q8_0,
        _ => GgmlType.F16,
    };

    /// <summary>Per-head element type for a given layer (Q8_0 falls back to F16 when head dim is not a multiple of 32).</summary>
    public GgmlType HeadType(int layer, bool isV)
    {
        if (Type != KvCacheType.Q8_0) return ElementType;
        int hd = isV ? Config.HeadDimV[layer] : Config.HeadDimK[layer];
        return hd % 32 == 0 ? GgmlType.Q8_0 : GgmlType.F16;
    }

    public long TotalBytes
    {
        get
        {
            long s = 0;
            for (int i = 0; i < _k.Length; i++) s += (_k[i]?.Length ?? 0) + (_v[i]?.Length ?? 0);
            return s;
        }
    }

    public int FreePages { get { lock (_lock) return _free.Count; } }

    public long KRowBytes(int layer) => _kRowBytes[layer];
    public long VRowBytes(int layer) => _vRowBytes[layer];

    /// <summary>Pointer to the K row (all KV heads) for a token slot.</summary>
    public byte* K(int layer, int page, int slot) => _k[layer]!.B + ((long)page * PageSize + slot) * _kRowBytes[layer];
    public byte* V(int layer, int page, int slot) => _v[layer]!.B + ((long)page * PageSize + slot) * _vRowBytes[layer];

    public bool TryAllocatePage(out int page)
    {
        lock (_lock)
        {
            if (_free.Count == 0) { page = -1; return false; }
            page = _free.Pop();
            _refCount[page] = 1;
            return true;
        }
    }

    public void Retain(int page)
    {
        lock (_lock) _refCount[page]++;
    }

    public void Release(int page)
    {
        lock (_lock)
        {
            if (--_refCount[page] == 0) _free.Push(page);
        }
    }

    public int RefCount(int page) { lock (_lock) return _refCount[page]; }

    /// <summary>Copies the first <paramref name="slots"/> token rows of page <paramref name="src"/> into <paramref name="dst"/> for every layer.</summary>
    public void CopyPage(int src, int dst, int slots)
    {
        for (int il = 0; il < _k.Length; il++)
        {
            if (_k[il] == null) continue;
            long kb = _kRowBytes[il] * slots, vb = _vRowBytes[il] * slots;
            Buffer.MemoryCopy(K(il, src, 0), K(il, dst, 0), kb, kb);
            Buffer.MemoryCopy(V(il, src, 0), V(il, dst, 0), vb, vb);
        }
        PageCopied?.Invoke(src, dst, slots);
    }

    /// <summary>
    /// Rotates the K rows of positions [<paramref name="from"/>, <paramref name="to"/>) of a page table by
    /// <paramref name="delta"/> rotary positions on every host layer with its own KV (context shift).
    /// </summary>
    public void RotateKeys(IReadOnlyList<int> pageTable, int from, int to, int delta)
    {
        var rots = new Dictionary<RopeConfig, float[]>();
        for (int il = 0; il < _k.Length; il++)
        {
            var rope = Config.Rope[il];
            if (_k[il] == null || rope == null) continue;
            if (!rots.TryGetValue(rope, out var rot)) rots[rope] = rot = rope.ShiftRotation(delta);
            var t = HeadType(il, false);
            int n = _kDim[il], hd = Config.HeadDimK[il];
            var row = new float[n];
            fixed (float* r = row, rc = rot)
                for (int pos = from; pos < to; pos++)
                {
                    byte* k = K(il, pageTable[pos / PageSize], pos % PageSize);
                    Dequantize.Row(t, k, r, n);
                    rope.Apply(r, Config.NHeadKv[il], hd, rc);
                    WriteRow(t, r, k, n, hd);
                }
        }
    }

    /// <summary>Writes one token's K/V (float, [nKv*headDim]) into the cache, converting to the storage type.</summary>
    public void Write(int layer, int page, int slot, float* k, float* v)
    {
        WriteRow(HeadType(layer, false), k, K(layer, page, slot), _kDim[layer], Config.HeadDimK[layer]);
        WriteRow(HeadType(layer, true), v, V(layer, page, slot), _vDim[layer], Config.HeadDimV[layer]);
    }

    private static void WriteRow(GgmlType t, float* src, byte* dst, int n, int headDim)
    {
        switch (t)
        {
            case GgmlType.F32: Buffer.MemoryCopy(src, dst, n * 4L, n * 4L); break;
            case GgmlType.F16: HalfHelper.FloatToHalf(src, (ushort*)dst, n); break;
            case GgmlType.Q8_0: QuantizeActivations.Q8_0(src, dst, n); break;
        }
    }

    public void Dispose()
    {
        foreach (var b in _k) b?.Dispose();
        foreach (var b in _v) b?.Dispose();
    }
}
