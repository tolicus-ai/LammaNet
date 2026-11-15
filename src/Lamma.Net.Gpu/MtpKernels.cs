using ILGPU;
using ILGPU.Runtime;

namespace LammaNet.Gpu;

internal static partial class Kernels
{
    /// <summary>Blocks of <see cref="ArgmaxPartial"/> (each reduces a strided share of the logits).</summary>
    public const int ArgmaxBlocks = 64;

    /// <summary>
    /// First stage of the arg-max with softmax mass: block b reduces elements b·256 + i·(ArgmaxBlocks·256) of x[0, n) to
    /// part[3b] = its max, part[3b + 1] = the index of the max (as float, exact below 2^24), part[3b + 2] = Σ exp(x − max).
    /// Grid: ArgmaxBlocks groups of 256.
    /// </summary>
    public static void ArgmaxPartial(ArrayView<float> x, int n, ArrayView<float> part)
    {
        const int threads = 256;
        var sv = SharedMemory.Allocate<float>(threads / Warp);
        var si = SharedMemory.Allocate<int>(threads / Warp);
        int tid = Group.IdxX, lane = tid & (Warp - 1), w = tid / Warp, b = Grid.IdxX;
        int stride = ArgmaxBlocks * threads;
        float best = float.NegativeInfinity;
        int bi = 0;
        for (int i = b * threads + tid; i < n; i += stride)
        {
            float v = x[i];
            if (v > best) { best = v; bi = i; }
        }
        for (int o = 16; o > 0; o >>= 1)
        {
            float ov = ILGPU.Warp.ShuffleXor(best, o);
            int oi = ILGPU.Warp.ShuffleXor(bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) { sv[w] = best; si[w] = bi; }
        Group.Barrier();
        float mx = sv[0];
        int mi = si[0];
        for (int k = 1; k < threads / Warp; k++)
            if (sv[k] > mx || (sv[k] == mx && si[k] < mi)) { mx = sv[k]; mi = si[k]; }
        float sum = 0;
        if (mx > float.NegativeInfinity)
            for (int i = b * threads + tid; i < n; i += stride) sum += MathF.Exp(x[i] - mx);
        sum = WarpSum(sum);
        Group.Barrier();
        if (lane == 0) sv[w] = sum;
        Group.Barrier();
        if (tid == 0)
        {
            float tot = 0;
            for (int k = 0; k < threads / Warp; k++) tot += sv[k];
            part[3 * b] = mx;
            part[3 * b + 1] = mi;
            part[3 * b + 2] = tot;
        }
    }

    /// <summary>Second stage: out[0] = arg max (as float), out[1] = max, out[2] = Σ exp(x − max) over all blocks. One group of 32.</summary>
    public static void ArgmaxFinal(ArrayView<float> part, ArrayView<float> outp)
    {
        int lane = Group.IdxX;
        float best = float.NegativeInfinity;
        int bi = 0;
        for (int k = lane; k < ArgmaxBlocks; k += Warp)
        {
            float v = part[3 * k];
            int i = (int)part[3 * k + 1];
            if (v > best || (v == best && i < bi)) { best = v; bi = i; }
        }
        for (int o = 16; o > 0; o >>= 1)
        {
            float ov = ILGPU.Warp.ShuffleXor(best, o);
            int oi = ILGPU.Warp.ShuffleXor(bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        float sum = 0;
        for (int k = lane; k < ArgmaxBlocks; k += Warp)
        {
            float m = part[3 * k];
            if (m > float.NegativeInfinity) sum += part[3 * k + 2] * MathF.Exp(m - best);
        }
        sum = WarpSum(sum);
        if (lane == 0) { outp[0] = bi; outp[1] = best; outp[2] = sum; }
    }
}
