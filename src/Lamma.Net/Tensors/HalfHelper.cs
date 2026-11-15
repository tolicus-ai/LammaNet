using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace LammaNet.Tensors;

/// <summary>Fast IEEE half / bfloat16 conversions (scalar table lookup and SIMD bit tricks).</summary>
public static unsafe class HalfHelper
{
    private static readonly float* Table = BuildTable();

    private static float* BuildTable()
    {
        var t = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(65536 * sizeof(float), 64);
        for (int i = 0; i < 65536; i++) t[i] = (float)BitConverter.UInt16BitsToHalf((ushort)i);
        return t;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToFloat(ushort h) => Table[h];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FromFloat(float f) => BitConverter.HalfToUInt16Bits((Half)f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Bf16ToFloat(ushort b) => BitConverter.Int32BitsToSingle(b << 16);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FloatToBf16(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        if ((u & 0x7fffffff) > 0x7f800000) return (ushort)((u >> 16) | 64); // NaN -> quiet NaN
        return (ushort)((u + (0x7fff + ((u >> 16) & 1))) >> 16); // round to nearest even
    }

    /// <summary>Converts 8 halves to 8 floats. Handles normals, subnormals and zero (Inf/NaN are not preserved).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Load8Half(ushort* p)
    {
        var h = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p)); // zero-extend u16 -> i32
        var sign = Avx2.ShiftLeftLogical(Avx2.And(h, Vector256.Create(0x8000)), 16);
        var mag = Avx2.ShiftLeftLogical(Avx2.And(h, Vector256.Create(0x7fff)), 13);
        // reinterpret magnitude as float and rescale exponent by 2^112 (handles subnormals exactly)
        var f = Avx.Multiply(mag.AsSingle(), Vector256.Create(5.192296858534828e+33f));
        return Avx.Or(f, sign.AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Load8Bf16(ushort* p)
    {
        var h = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p));
        return Avx2.ShiftLeftLogical(h, 16).AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void HalfToFloat(ushort* src, float* dst, long n)
    {
        long i = 0;
        if (Avx2.IsSupported)
            for (; i + 8 <= n; i += 8) Avx.Store(dst + i, Load8Half(src + i));
        for (; i < n; i++) dst[i] = Table[src[i]];
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Bf16ToFloat(ushort* src, float* dst, long n)
    {
        long i = 0;
        if (Avx2.IsSupported)
            for (; i + 8 <= n; i += 8) Avx.Store(dst + i, Load8Bf16(src + i));
        for (; i < n; i++) dst[i] = Bf16ToFloat(src[i]);
    }

    /// <summary>float → IEEE half (round to nearest even), vectorized; identical to <c>(Half)f</c> per element.</summary>
    public static void FloatToHalf(float* src, ushort* dst, long n)
    {
        while (n > 0)
        {
            int chunk = (int)Math.Min(n, int.MaxValue);
            System.Numerics.Tensors.TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(src, chunk), new Span<Half>(dst, chunk));
            src += chunk; dst += chunk; n -= chunk;
        }
    }
}
