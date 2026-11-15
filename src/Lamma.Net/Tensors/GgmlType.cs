namespace LammaNet.Tensors;

/// <summary>Tensor element / block types, numerically identical to ggml's <c>ggml_type</c>.</summary>
public enum GgmlType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
}

/// <summary>Static layout information for every supported <see cref="GgmlType"/>.</summary>
public static class GgmlTypeInfo
{
    /// <summary>Super-block size used by k-quants.</summary>
    public const int QK_K = 256;

    /// <summary>Number of elements stored in one block.</summary>
    public static int BlockSize(GgmlType t) => t switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.BF16 or GgmlType.I8 or GgmlType.I16
            or GgmlType.I32 or GgmlType.I64 or GgmlType.F64 => 1,
        GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_0 or GgmlType.Q5_1 or GgmlType.Q8_0
            or GgmlType.Q8_1 or GgmlType.IQ4_NL or GgmlType.MXFP4 => 32,
        _ => QK_K,
    };

    /// <summary>Number of bytes used by one block.</summary>
    public static int TypeSize(GgmlType t) => t switch
    {
        GgmlType.F32 => 4,
        GgmlType.F16 => 2,
        GgmlType.BF16 => 2,
        GgmlType.F64 => 8,
        GgmlType.I8 => 1,
        GgmlType.I16 => 2,
        GgmlType.I32 => 4,
        GgmlType.I64 => 8,
        GgmlType.Q4_0 => 18,
        GgmlType.Q4_1 => 20,
        GgmlType.Q5_0 => 22,
        GgmlType.Q5_1 => 24,
        GgmlType.Q8_0 => 34,
        GgmlType.Q8_1 => 36,
        GgmlType.Q2_K => 84,
        GgmlType.Q3_K => 110,
        GgmlType.Q4_K => 144,
        GgmlType.Q5_K => 176,
        GgmlType.Q6_K => 210,
        GgmlType.Q8_K => 292,
        GgmlType.IQ2_XXS => 66,
        GgmlType.IQ2_XS => 74,
        GgmlType.IQ3_XXS => 98,
        GgmlType.IQ1_S => 50,
        GgmlType.IQ4_NL => 18,
        GgmlType.IQ3_S => 110,
        GgmlType.IQ2_S => 82,
        GgmlType.IQ4_XS => 136,
        GgmlType.IQ1_M => 56,
        GgmlType.TQ1_0 => 54,
        GgmlType.TQ2_0 => 66,
        GgmlType.MXFP4 => 17,
        _ => throw new NotSupportedException($"Unknown ggml type {t}"),
    };

    /// <summary>Bytes needed to store a row of <paramref name="n"/> elements.</summary>
    public static long RowSize(GgmlType t, long n)
    {
        int bs = BlockSize(t);
        if (n % bs != 0) throw new ArgumentException($"Row length {n} is not a multiple of block size {bs} for {t}");
        return n / bs * TypeSize(t);
    }

    public static bool IsQuantized(GgmlType t) => BlockSize(t) > 1;

    /// <summary>True for the tensor types this library knows the layout of (files may contain newer ones).</summary>
    public static bool IsKnown(GgmlType t) => Enum.IsDefined(t);
}
