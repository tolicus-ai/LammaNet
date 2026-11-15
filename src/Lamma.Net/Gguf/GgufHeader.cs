using System.Text;
using LammaNet.Tensors;

namespace LammaNet.Gguf;

/// <summary>The header ran past the end of the bytes available (more input may complete it).</summary>
internal sealed class GgufTruncatedException() : Exception(TruncatedMessage)
{
    public const string TruncatedMessage = "Unexpected end of GGUF header";
}

/// <summary>
/// The parsed header of a GGUF file: version, key/value metadata and tensor descriptors, plus the offset at which the
/// tensor data starts. Parsed from memory (the mapping of <see cref="GgufFile"/>) or from a stream, reading only the header bytes.
/// </summary>
internal sealed class GgufHeader
{
    private const uint Magic = 0x46554747; // "GGUF"
    /// <summary>Upper bound on the header bytes buffered from a stream (guards against corrupt lengths).</summary>
    private const long MaxStreamHeaderBytes = 512L << 20;

    public required uint Version { get; init; }
    public required int Alignment { get; init; }
    public required GgufMetadata Metadata { get; init; }
    public required List<GgufTensorInfo> Tensors { get; init; }
    public required long DataOffset { get; init; }

    public Dictionary<string, GgufTensorInfo> TensorsByName()
    {
        var d = new Dictionary<string, GgufTensorInfo>(Tensors.Count, StringComparer.Ordinal);
        foreach (var t in Tensors) d[t.Name] = t;
        return d;
    }

    /// <summary>
    /// Parses the header from <paramref name="len"/> bytes at <paramref name="p"/>. Arrays with more than
    /// <paramref name="maxArrayLength"/> elements are skipped and recorded as <see cref="GgufArraySummary"/>.
    /// Throws <see cref="InvalidDataException"/> when the header extends past <paramref name="len"/>.
    /// </summary>
    public static unsafe GgufHeader Parse(byte* p, long len, long maxArrayLength, string? source)
    {
        try { return ParseCore(p, len, maxArrayLength, source); }
        catch (GgufTruncatedException) { throw new InvalidDataException(GgufTruncatedException.TruncatedMessage); }
    }

    /// <summary>As <see cref="Parse"/>, but signals a truncated header with <see cref="GgufTruncatedException"/>.</summary>
    private static unsafe GgufHeader ParseCore(byte* p, long len, long maxArrayLength, string? source)
    {
        var r = new Reader(p, len, maxArrayLength);
        if (r.U32() != Magic) throw new InvalidDataException($"'{source}' is not a GGUF file");
        uint version = r.U32();
        if (version < 2 || version > 3) throw new InvalidDataException($"Unsupported GGUF version {version}");
        long nTensors = (long)r.U64();
        long nKv = (long)r.U64();
        if (nTensors < 0 || nKv < 0 || nTensors > int.MaxValue) throw new InvalidDataException($"Corrupt GGUF header ({nTensors} tensors, {nKv} keys)");
        var metadata = new GgufMetadata();
        for (long i = 0; i < nKv; i++)
        {
            string key = r.Str();
            var type = (GgufValueType)r.U32();
            metadata.Set(key, r.Value(type));
        }
        int alignment = metadata.TryGetValue("general.alignment", out var a) ? Convert.ToInt32(a) : 32;
        if (alignment <= 0) throw new InvalidDataException($"Invalid GGUF alignment {alignment}");
        // every tensor descriptor takes at least 8 (name length) + 4 (dims) + 4 (type) + 8 (offset) bytes
        r.Need(checked(nTensors * 24));
        var tensors = new List<GgufTensorInfo>((int)Math.Min(nTensors, 1 << 20));
        for (long i = 0; i < nTensors; i++)
        {
            string name = r.Str();
            uint nd = r.U32();
            if (nd > 8) throw new InvalidDataException($"Tensor {name} has {nd} dimensions");
            var dims = new long[nd];
            for (int d = 0; d < nd; d++) dims[d] = (long)r.U64();
            var t = (GgmlType)r.U32();
            ulong off = r.U64();
            tensors.Add(new GgufTensorInfo { Name = name, Type = t, Dims = dims, Offset = off });
        }
        long pos = r.Position;
        return new GgufHeader
        {
            Version = version,
            Alignment = alignment,
            Metadata = metadata,
            Tensors = tensors,
            DataOffset = (pos + alignment - 1) / alignment * alignment,
        };
    }

    /// <summary>
    /// Reads the header from a stream, consuming only (roughly) the header bytes: the stream is read in growing chunks
    /// until the header parses. <paramref name="bytesRead"/> is the number of bytes consumed from the stream.
    /// </summary>
    public static GgufHeader Read(Stream stream, long maxArrayLength, string? source, out long bytesRead, int initialChunk = 1 << 20)
    {
        var buf = new byte[initialChunk];
        int filled = 0;
        bool eof = false;
        while (true)
        {
            while (filled < buf.Length && !eof)
            {
                int n = stream.Read(buf, filled, buf.Length - filled);
                if (n == 0) eof = true;
                filled += n;
            }
            if (TryParse(buf, filled, eof, maxArrayLength, source, out var h)) { bytesRead = filled; return h!; }
            buf = Grow(buf);
        }
    }

    /// <inheritdoc cref="Read"/>
    public static async Task<(GgufHeader Header, long BytesRead)> ReadAsync(Stream stream, long maxArrayLength, string? source,
        CancellationToken cancellationToken, int initialChunk = 1 << 20)
    {
        var buf = new byte[initialChunk];
        int filled = 0;
        bool eof = false;
        while (true)
        {
            while (filled < buf.Length && !eof)
            {
                int n = await stream.ReadAsync(buf.AsMemory(filled), cancellationToken).ConfigureAwait(false);
                if (n == 0) eof = true;
                filled += n;
            }
            if (TryParse(buf, filled, eof, maxArrayLength, source, out var h)) return (h!, filled);
            buf = Grow(buf);
        }
    }

    private static unsafe bool TryParse(byte[] buf, int filled, bool eof, long maxArrayLength, string? source, out GgufHeader? header)
    {
        try
        {
            fixed (byte* p = buf) header = ParseCore(p, filled, maxArrayLength, source);
            return true;
        }
        catch (GgufTruncatedException)
        {
            if (eof) throw new InvalidDataException(GgufTruncatedException.TruncatedMessage);
            if (buf.Length >= MaxStreamHeaderBytes)
                throw new InvalidDataException($"GGUF header of '{source}' exceeds {MaxStreamHeaderBytes >> 20} MB (corrupt file?)");
            header = null;
            return false;
        }
    }

    private static byte[] Grow(byte[] buf)
    {
        var bigger = new byte[Math.Min((long)buf.Length * 4, MaxStreamHeaderBytes)];
        Buffer.BlockCopy(buf, 0, bigger, 0, buf.Length);
        return bigger;
    }

    private unsafe struct Reader(byte* p, long len, long maxArrayLength)
    {
        private long _pos;
        public readonly long Position => _pos;

        public void Need(long n)
        {
            if (n < 0 || _pos + n > len) throw new GgufTruncatedException();
        }

        public byte U8() { Need(1); return p[_pos++]; }
        public uint U32() { Need(4); uint v = *(uint*)(p + _pos); _pos += 4; return v; }
        public ulong U64() { Need(8); ulong v = *(ulong*)(p + _pos); _pos += 8; return v; }
        public string Str()
        {
            long n = (long)U64();
            if (n < 0 || n > int.MaxValue) throw new InvalidDataException($"Corrupt GGUF string length {n}");
            Need(n);
            var s = Encoding.UTF8.GetString(p + _pos, (int)n);
            _pos += n;
            return s;
        }

        private void SkipStr()
        {
            long n = (long)U64();
            if (n < 0 || n > int.MaxValue) throw new InvalidDataException($"Corrupt GGUF string length {n}");
            Need(n);
            _pos += n;
        }

        public object Value(GgufValueType t)
        {
            switch (t)
            {
                case GgufValueType.UInt8: return U8();
                case GgufValueType.Int8: return (sbyte)U8();
                case GgufValueType.UInt16: { Need(2); var v = *(ushort*)(p + _pos); _pos += 2; return v; }
                case GgufValueType.Int16: { Need(2); var v = *(short*)(p + _pos); _pos += 2; return v; }
                case GgufValueType.UInt32: return U32();
                case GgufValueType.Int32: return (int)U32();
                case GgufValueType.Float32: { Need(4); var v = *(float*)(p + _pos); _pos += 4; return v; }
                case GgufValueType.Bool: return U8() != 0;
                case GgufValueType.String: return Str();
                case GgufValueType.UInt64: return U64();
                case GgufValueType.Int64: return (long)U64();
                case GgufValueType.Float64: { Need(8); var v = *(double*)(p + _pos); _pos += 8; return v; }
                case GgufValueType.Array:
                {
                    var et = (GgufValueType)U32();
                    long n = (long)U64();
                    if (n < 0 || n > long.MaxValue / 16) throw new InvalidDataException($"Corrupt GGUF array length {n}");
                    // every element takes at least MinSize bytes: an impossible count is a truncation (or corruption)
                    // and must not be allocated
                    Need(checked(n * MinSize(et)));
                    if (n > maxArrayLength) { Skip(et, n); return new GgufArraySummary(et, n); }
                    return ReadArray(et, n);
                }
                default: throw new InvalidDataException($"Unknown GGUF value type {t}");
            }
        }

        private static int MinSize(GgufValueType t) => t switch
        {
            GgufValueType.UInt8 or GgufValueType.Int8 or GgufValueType.Bool => 1,
            GgufValueType.UInt16 or GgufValueType.Int16 => 2,
            GgufValueType.UInt32 or GgufValueType.Int32 or GgufValueType.Float32 => 4,
            GgufValueType.UInt64 or GgufValueType.Int64 or GgufValueType.Float64 or GgufValueType.String => 8,
            GgufValueType.Array => 12,
            _ => throw new InvalidDataException($"Unknown GGUF value type {t}"),
        };

        private void Skip(GgufValueType et, long n)
        {
            switch (et)
            {
                case GgufValueType.String:
                    for (long i = 0; i < n; i++) SkipStr();
                    break;
                case GgufValueType.Array:
                    for (long i = 0; i < n; i++) Value(GgufValueType.Array);
                    break;
                default:
                {
                    long bytes = n * MinSize(et);
                    Need(bytes);
                    _pos += bytes;
                    break;
                }
            }
        }

        private Array ReadArray(GgufValueType et, long n)
        {
            switch (et)
            {
                case GgufValueType.String:
                {
                    var a = new string[n];
                    for (long i = 0; i < n; i++) a[i] = Str();
                    return a;
                }
                case GgufValueType.Float32: return Copy<float>(n);
                case GgufValueType.Int32: return Copy<int>(n);
                case GgufValueType.UInt32: return Copy<uint>(n);
                case GgufValueType.UInt8: return Copy<byte>(n);
                case GgufValueType.Int8: return Copy<sbyte>(n);
                case GgufValueType.Int16: return Copy<short>(n);
                case GgufValueType.UInt16: return Copy<ushort>(n);
                case GgufValueType.Int64: return Copy<long>(n);
                case GgufValueType.UInt64: return Copy<ulong>(n);
                case GgufValueType.Float64: return Copy<double>(n);
                case GgufValueType.Bool:
                {
                    var a = new bool[n];
                    for (long i = 0; i < n; i++) a[i] = U8() != 0;
                    return a;
                }
                default:
                {
                    var a = new object[n];
                    for (long i = 0; i < n; i++) a[i] = Value(et);
                    return a;
                }
            }
        }

        private T[] Copy<T>(long n) where T : unmanaged
        {
            long bytes = n * sizeof(T);
            Need(bytes);
            var a = new T[n];
            fixed (T* dst = a) Buffer.MemoryCopy(p + _pos, dst, bytes, bytes);
            _pos += bytes;
            return a;
        }
    }
}
