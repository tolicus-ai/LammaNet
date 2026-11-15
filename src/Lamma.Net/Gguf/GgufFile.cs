using System.IO.MemoryMappedFiles;
using System.Text;
using LammaNet.Tensors;

namespace LammaNet.Gguf;

/// <summary>GGUF metadata value types.</summary>
public enum GgufValueType : uint
{
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3, UInt32 = 4, Int32 = 5, Float32 = 6,
    Bool = 7, String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12,
}

/// <summary>Description of one tensor stored in a GGUF file.</summary>
public sealed class GgufTensorInfo
{
    public required string Name { get; init; }
    public required GgmlType Type { get; init; }
    /// <summary>Dimensions in ggml order (ne[0] is the contiguous/innermost dimension).</summary>
    public required long[] Dims { get; init; }
    /// <summary>Offset relative to the start of the tensor data section.</summary>
    public required ulong Offset { get; init; }

    public long ElementCount { get { long n = 1; foreach (var d in Dims) n *= d; return n; } }
    public long ByteSize => GgmlTypeInfo.RowSize(Type, Dims[0]) * (ElementCount / Dims[0]);
    /// <summary>False when the tensor type is unknown to this library (its size cannot be computed).</summary>
    public bool HasKnownType => GgmlTypeInfo.IsKnown(Type);
    public override string ToString() => $"{Name} {Type} [{string.Join(", ", Dims)}]";
}

/// <summary>
/// A memory-mapped GGUF model file. Tensor data is never copied: weights are used directly
/// from the mapping (zero-copy, page-cache backed, shared between processes).
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    private const uint Magic = 0x46554747; // "GGUF"

    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private byte* _base;
    private readonly long _length;
    private readonly long _dataOffset;
    private readonly GgufMetadata _metadata;
    private readonly Dictionary<string, GgufTensorInfo> _tensors;
    private readonly List<GgufTensorInfo> _tensorList;

    public string Path { get; }
    public uint Version { get; }
    public int Alignment { get; }
    public GgufMetadata Metadata => _metadata;
    public IReadOnlyList<GgufTensorInfo> Tensors => _tensorList;
    public long FileSize => _length;
    /// <summary>Offset of the tensor data section (= size of the header, including alignment padding).</summary>
    public long DataOffset => _dataOffset;

    public GgufFile(string path) : this(path, long.MaxValue) { }

    /// <summary>Opens a file, skipping metadata arrays longer than <paramref name="maxArrayLength"/> (see <see cref="GgufArraySummary"/>).</summary>
    internal GgufFile(string path, long maxArrayLength)
    {
        Path = path;
        var fi = new FileInfo(path);
        if (!fi.Exists) throw new FileNotFoundException("GGUF file not found", path);
        _length = fi.Length;
        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref _base);
        _base += _view.PointerOffset;
        try
        {
            var h = GgufHeader.Parse(_base, _length, maxArrayLength, path);
            Version = h.Version;
            Alignment = h.Alignment;
            _metadata = h.Metadata;
            _tensorList = h.Tensors;
            _tensors = h.TensorsByName();
            _dataOffset = h.DataOffset;
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public bool TryGetTensor(string name, out GgufTensorInfo info) => _tensors.TryGetValue(name, out info!);
    public GgufTensorInfo GetTensor(string name) =>
        _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"Tensor '{name}' not found in {System.IO.Path.GetFileName(Path)}");

    /// <summary>Pointer to the raw (possibly quantized) data of the tensor inside the mapping.</summary>
    public byte* GetTensorData(GgufTensorInfo info)
    {
        long off = _dataOffset + (long)info.Offset;
        if (off + info.ByteSize > _length) throw new InvalidDataException($"Tensor {info.Name} exceeds file bounds");
        return _base + off;
    }

    // ---------------- metadata helpers ----------------

    public bool Has(string key) => _metadata.Has(key);
    public T Get<T>(string key, T defaultValue) => _metadata.Get(key, defaultValue);
    public T Get<T>(string key) => _metadata.Get<T>(key);
    public bool TryGet<T>(string key, out T value) => _metadata.TryGet(key, out value);
    /// <summary>Gets a key that may be either a scalar or a per-layer array; result has <paramref name="n"/> entries.</summary>
    public int[]? GetIntOrArray(string key, int n) => _metadata.GetIntOrArray(key, n);

    public void Dispose()
    {
        if (_base != null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }
        _view?.Dispose();
        _mmf?.Dispose();
    }
}
