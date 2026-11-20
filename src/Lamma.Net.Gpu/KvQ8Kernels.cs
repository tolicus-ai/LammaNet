using ILGPU;
using ILGPU.Runtime;
using Half = ILGPU.Half;

namespace LammaNet.Gpu;

internal static partial class Kernels
{
    // Each 32-element block occupies 17 half words: a half scale followed by 32 signed bytes.
    private static float ReadKvQ8(ArrayView<Half> cache, long index)
    {
        long block = index / Warp;
        return (float)cache[17 * block] * cache.Cast<sbyte>()[34 * block + 2 + index % Warp];
    }

    // A tile load always lies inside one block. Expand it while staging K/V for tensor-core attention.
    private static Half8 LoadCache8(ArrayView<Half> cache, long vectorIndex, int compact)
    {
        if (compact == 0) return cache.Cast<Half8>()[vectorIndex];
        long block = vectorIndex / 4;
        long start = 34 * block + 2 + vectorIndex % 4 * 8;
        float d = cache[17 * block];
        var b = cache.Cast<sbyte>();
        return new Half8 {
            H0 = (Half)(d * b[start]), H1 = (Half)(d * b[start + 1]),
            H2 = (Half)(d * b[start + 2]), H3 = (Half)(d * b[start + 3]),
            H4 = (Half)(d * b[start + 4]), H5 = (Half)(d * b[start + 5]),
            H6 = (Half)(d * b[start + 6]), H7 = (Half)(d * b[start + 7]),
        };
    }

    private static Half2 LoadCache2(ArrayView<Half> cache, long pair, int compact)
    {
        if (compact == 0) return cache.Cast<Half2>()[pair];
        long block = pair / 16, start = 34 * block + 2 + pair % 16 * 2;
        float d = cache[17 * block];
        var bytes = cache.Cast<sbyte>();
        return new Half2 { A = (Half)(d * bytes[start]), B = (Half)(d * bytes[start + 1]) };
    }

    private static void WriteKvQ8(ArrayView<Half> cache, long block, float value)
    {
        float max = WarpMax(MathF.Abs(value));
        float scale = max / 127f;
        if (Lane == 0) cache[17 * block] = (Half)scale;
        cache.Cast<sbyte>()[34 * block + 2 + Lane] = scale > 0 ? (sbyte)MathF.Round(value / scale) : (sbyte)0;
    }

    public static void PackKvQ8(ArrayView<Half> rows, ArrayView<Half> cache, ArrayView<int> slots, int width, int count)
    {
        int block = WarpIndex, blocksPerRow = width / Warp, row = block / blocksPerRow;
        if (row >= count) return;
        WriteKvQ8(cache, (long)slots[row] * blocksPerRow + block % blocksPerRow, rows[(long)block * Warp + Lane]);
    }

    public static void RotateKvQ8(ArrayView<Half> cache, ArrayView<int> slots, ArrayView<float> rot,
        int count, int heads, int hd, int dims, int neox)
    {
        int head = WarpIndex, row = head / heads;
        if (row >= count) return;
        var sh = SharedMemory.Allocate<float>(WarpsPerGroup * MaxHeadDim);
        int start = Group.IdxX / Warp * MaxHeadDim;
        long offset = ((long)slots[row] * heads + head % heads) * hd;
        for (int i = Lane; i < hd; i += Warp) sh[start + i] = ReadKvQ8(cache, offset + i);
        ILGPU.Warp.Barrier();
        for (int j = Lane; j < dims / 2; j += Warp)
        {
            int a = start + (neox != 0 ? j : 2 * j), b = neox != 0 ? a + dims / 2 : a + 1;
            float x = sh[a], y = sh[b], c = rot[2 * j], s = rot[2 * j + 1];
            sh[a] = x * c - y * s;
            sh[b] = x * s + y * c;
        }
        ILGPU.Warp.Barrier();
        for (int i = Lane; i < (dims + Warp - 1) / Warp * Warp; i += Warp) WriteKvQ8(cache, (offset + i) / Warp, sh[start + i]);
    }

}
