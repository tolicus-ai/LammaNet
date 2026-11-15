using LammaNet.Tensors;

namespace LammaNet.Gpu;

/// <summary>
/// Weight formats as stored on the device. Rows start on 16-byte boundaries and every 16-byte group of quants is
/// 16-byte aligned so kernels can use 128-bit loads:
/// <list type="bullet">
/// <item>Q4_K (144 B) and Q5_K (176 B) blocks are kept as is (multiples of 16 bytes).</item>
/// <item>Q8_0, Q4_0 and Q6_K are stored per row as structure-of-arrays: all block payloads first
/// (32 / 16 / 208 bytes each), then the fp16 block scales, padded to 16 bytes.</item>
/// </list>
/// Formats without a device kernel are re-quantized to Q8_0 at upload.
/// </summary>
internal static class DevFormat
{
    public const int F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, BF16 = 30;

    /// <summary>Payload bytes per block stored in the main region of a row, and elements per block.</summary>
    public static (int Bytes, int Elems) Block(int type) => type switch
    {
        F32 => (4, 1),
        F16 or BF16 => (2, 1),
        Q4_0 => (16, 32),
        Q8_0 => (32, 32),
        Q4_K => (144, 256),
        Q5_K => (176, 256),
        Q6_K => (208, 256),
        _ => throw new NotSupportedException($"device format {type}"),
    };

    /// <summary>True for formats whose fp16 block scales live after the block payloads of each row.</summary>
    public static bool HasScaleTail(int type) => type is Q4_0 or Q8_0 or Q6_K;

    /// <summary>32-bit words of the payload region of a row (start of the scale tail).</summary>
    public static int PayloadWords(int type, int cols)
    {
        var (bytes, elems) = Block(type);
        return checked((int)(((long)cols / elems * bytes + 3) / 4));
    }

    /// <summary>32-bit words per device row of <paramref name="cols"/> elements (a multiple of 4).</summary>
    public static int RowWords(int type, int cols)
    {
        long w = PayloadWords(type, cols);
        if (HasScaleTail(type)) w += (cols / Block(type).Elems + 1) / 2;
        return checked((int)((w + 3) & ~3L));
    }

    /// <summary>Device format used for a host tensor type.</summary>
    public static int For(GgmlType t, int cols) => t switch
    {
        GgmlType.F32 => F32,
        GgmlType.F16 => F16,
        GgmlType.BF16 => BF16,
        GgmlType.Q4_0 => Q4_0,
        GgmlType.Q8_0 => Q8_0,
        GgmlType.Q4_K => Q4_K,
        GgmlType.Q5_K => Q5_K,
        GgmlType.Q6_K => Q6_K,
        _ => cols % 32 == 0 ? Q8_0 : F32,
    };

    /// <summary>Writes the device representation of <paramref name="w"/> (rows × RowWords) into <paramref name="dst"/>.</summary>
    public static unsafe void Pack(WeightTensor w, int type, uint* dst)
    {
        int rowWords = RowWords(type, w.Cols);
        long rowBytes = rowWords * 4L;
        byte* d = (byte*)dst;
        new Span<byte>(d, checked((int)(rowBytes * w.Rows))).Clear();
        bool native = (int)w.Type == type;
        float[]? tmp = native ? null : new float[w.Cols];
        byte[]? q8 = native ? null : new byte[GgmlTypeInfo.RowSize(GgmlType.Q8_0, w.Cols)];
        long payload = PayloadWords(type, w.Cols) * 4L;
        for (int r = 0; r < w.Rows; r++)
        {
            byte* src = w.Row(r);
            byte* o = d + r * rowBytes;
            if (!native)
            {
                fixed (float* tp = tmp)
                fixed (byte* qp = q8)
                {
                    w.DequantizeRow(r, tp);
                    if (type == F32) { Buffer.MemoryCopy(tp, o, rowBytes, w.Cols * 4L); continue; }
                    QuantizeActivations.Row(GgmlType.Q8_0, tp, qp, w.Cols);
                    SplitScales(qp, o, payload, w.Cols / 32, 34, 0, 2);
                }
                continue;
            }
            switch (type)
            {
                case Q8_0: SplitScales(src, o, payload, w.Cols / 32, 34, 0, 2); break;
                case Q4_0: SplitScales(src, o, payload, w.Cols / 32, 18, 0, 2); break;
                case Q6_K: SplitScales(src, o, payload, w.Cols / 256, 210, 208, 0); break;
                default: Buffer.MemoryCopy(src, o, rowBytes, w.RowBytes); break;
            }
        }
    }

    /// <summary>
    /// Splits blocks of <paramref name="blockBytes"/> (with a 2-byte fp16 scale at <paramref name="scaleAt"/>) into the
    /// payload region (block bytes without the scale, starting at <paramref name="payloadFrom"/>) and the scale tail.
    /// </summary>
    private static unsafe void SplitScales(byte* src, byte* dst, long tailOffset, int blocks, int blockBytes, int scaleAt, int payloadFrom)
    {
        int pay = blockBytes - 2;
        for (int b = 0; b < blocks; b++)
        {
            byte* s = src + (long)b * blockBytes;
            byte* o = dst + (long)b * pay;
            if (scaleAt == 0) Buffer.MemoryCopy(s + payloadFrom, o, pay, pay);
            else Buffer.MemoryCopy(s, o, pay, pay); // scale at the end
            *(ushort*)(dst + tailOffset + 2L * b) = *(ushort*)(s + scaleAt);
        }
    }
}
