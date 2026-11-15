using LammaNet.Gguf;

namespace LammaNet.Tensors;

/// <summary>
/// A model weight living in host memory (usually directly inside the memory-mapped GGUF file).
/// Rows are the outer dimension: a 2D weight [ne0 = Cols, ne1 = Rows] multiplies activation vectors of length Cols.
/// </summary>
public sealed unsafe class WeightTensor
{
    public string Name { get; }
    public GgmlType Type { get; }
    public long[] Dims { get; }
    public byte* Data { get; private set; }
    /// <summary>Contiguous (innermost) dimension, ne[0].</summary>
    public int Cols => (int)Dims[0];
    /// <summary>Number of rows (product of all outer dimensions).</summary>
    public int Rows { get; }
    public long RowBytes { get; }
    public long ByteSize => RowBytes * Rows;
    /// <summary>Optional owner of the memory when the tensor was materialized (e.g. converted or repacked).</summary>
    internal NativeBuffer? Owned { get; }

    /// <summary>Opaque per-backend cache slot (e.g. a GPU-resident copy).</summary>
    public object? DeviceCopy { get; set; }

    public WeightTensor(string name, GgmlType type, long[] dims, byte* data, NativeBuffer? owned = null)
    {
        Name = name;
        Type = type;
        Dims = dims;
        Data = data;
        long rows = 1;
        for (int i = 1; i < dims.Length; i++) rows *= dims[i];
        Rows = checked((int)rows);
        RowBytes = GgmlTypeInfo.RowSize(type, dims[0]);
        Owned = owned;
    }

    public static WeightTensor FromGguf(GgufFile file, GgufTensorInfo info) =>
        new(info.Name, info.Type, info.Dims, file.GetTensorData(info));

    public byte* Row(long r) => Data + r * RowBytes;

    /// <summary>Dequantizes row <paramref name="r"/> into <paramref name="dst"/> (length <see cref="Cols"/>).</summary>
    public void DequantizeRow(long r, float* dst) => Dequantize.Row(Type, Row(r), dst, Cols);

    /// <summary>Reads a small F32 tensor (norm weights, biases, scalars) into a managed array.</summary>
    public float[] ToFloatArray()
    {
        long n = (long)Cols * Rows;
        var arr = new float[n];
        fixed (float* p = arr)
            for (int r = 0; r < Rows; r++) DequantizeRow(r, p + (long)r * Cols);
        return arr;
    }

    /// <summary>Creates an F32 copy of this tensor (owned memory).</summary>
    public WeightTensor ToF32()
    {
        if (Type == GgmlType.F32) return this;
        var buf = new NativeBuffer((long)Cols * Rows * 4);
        float* p = (float*)buf.Pointer;
        for (int r = 0; r < Rows; r++) DequantizeRow(r, p + (long)r * Cols);
        return new WeightTensor(Name, GgmlType.F32, Dims, (byte*)p, buf);
    }

    public override string ToString() => $"{Name} {Type} [{string.Join(", ", Dims)}]";
}

/// <summary>Aligned native memory block (64 bytes by default; whole pages for memory a device registers for DMA).</summary>
public sealed unsafe class NativeBuffer : IDisposable
{
    public const int PageSize = 4096;

    public void* Pointer { get; private set; }
    public long Length { get; }
    /// <summary>Allocated bytes (at least <see cref="Length"/>; whole pages when page-aligned).</summary>
    public long Capacity { get; }

    public NativeBuffer(long bytes, bool zero = true, bool pageAligned = false)
    {
        Length = bytes;
        int align = pageAligned ? PageSize : 64;
        Capacity = Math.Max(bytes, 64);
        if (pageAligned) Capacity = (Capacity + PageSize - 1) / PageSize * PageSize;
        Pointer = System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)Capacity, (nuint)align);
        if (zero) System.Runtime.InteropServices.NativeMemory.Clear(Pointer, (nuint)Capacity);
        if (bytes > 1 << 20) GC.AddMemoryPressure(bytes);
    }

    public float* F => (float*)Pointer;
    public byte* B => (byte*)Pointer;

    public Span<float> AsFloatSpan() => new(Pointer, checked((int)(Length / 4)));

    public void Dispose()
    {
        if (Pointer != null)
        {
            System.Runtime.InteropServices.NativeMemory.AlignedFree(Pointer);
            if (Length > 1 << 20) GC.RemoveMemoryPressure(Length);
            Pointer = null;
        }
        GC.SuppressFinalize(this);
    }

    ~NativeBuffer() => Dispose();
}
