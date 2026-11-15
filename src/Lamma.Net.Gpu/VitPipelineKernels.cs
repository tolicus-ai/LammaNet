using ILGPU;
using ILGPU.Runtime;

namespace LammaNet.Gpu;

/// <summary>Geometry of a batch of square tiles cut into patches (see <see cref="Kernels.Patchify"/>).</summary>
public struct PatchParams
{
    /// <summary>Patch rows per tile (Side²), patches per side, patch size, tile size in pixels, row length of a patch (3·ps²).</summary>
    public int N, Side, Ps, ImageSize, Pp;
    /// <summary>Offsets of the per-channel mean and std in the parameter buffer.</summary>
    public long MeanOff, StdOff;
}

/// <summary>Patch grid of one image for <see cref="Kernels.PatchifyGrid"/> and friends (merge-block order).</summary>
public struct GridPatchParams
{
    /// <summary>Patches per row and column, patch size, merge factor, row length of a patch (3·Ps²), image width in pixels.</summary>
    public int Pw, Ph, Ps, Merge, Pp, Width;
    public long MeanOff, StdOff;
}

internal static partial class Kernels
{
    /// <summary>Grid position (py, px) of patch row <paramref name="r"/> in merge-block order.</summary>
    private static void GridPatch(int r, GridPatchParams p, out int py, out int px)
    {
        int m2 = p.Merge * p.Merge, blk = r / m2, inner = r - blk * m2, bpr = p.Pw / p.Merge;
        int by = blk / bpr, bx = blk - by * bpr;
        py = by * p.Merge + inner / p.Merge;
        px = bx * p.Merge + inner % p.Merge;
    }

    /// <summary>Patches of one image (any size multiple of Ps·Merge): rows in merge-block order, (pixel/255 − mean[c]) / std[c].</summary>
    public static void PatchifyGrid(Index1D i, ArrayView<byte> pixels, ArrayView<float> patches, ArrayView<float> f, GridPatchParams p)
    {
        int r = (int)(i / p.Pp), e = (int)(i - (long)r * p.Pp);
        GridPatch(r, p, out int py, out int px);
        int pp2 = p.Ps * p.Ps, c = e / pp2, rem = e - c * pp2, ky = rem / p.Ps, kx = rem - ky * p.Ps;
        long pix = ((long)(py * p.Ps + ky) * p.Width + px * p.Ps + kx) * 3 + c;
        patches[i] = (pixels[pix] / 255f - f[p.MeanOff + c]) / f[p.StdOff + c];
    }

    /// <summary>x[r, :] += the side×side position table resized to the patch grid (bilinear, align corners) at row r's patch.</summary>
    public static void AddGridPositions(Index1D i, ArrayView<float> x, ArrayView<float> f, GridPatchParams p, long posOff, int side, int d)
    {
        int r = (int)(i / d), c = (int)(i - (long)r * d);
        GridPatch(r, p, out int py, out int px);
        float fy = p.Ph > 1 ? (float)py * (side - 1) / (p.Ph - 1) : 0f, fx = p.Pw > 1 ? (float)px * (side - 1) / (p.Pw - 1) : 0f;
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + 1 < side ? y0 + 1 : side - 1, x1 = x0 + 1 < side ? x0 + 1 : side - 1;
        float wy = fy - y0, wx = fx - x0;
        float v = (1 - wy) * ((1 - wx) * f[posOff + ((long)y0 * side + x0) * d + c] + wx * f[posOff + ((long)y0 * side + x1) * d + c])
                + wy * ((1 - wx) * f[posOff + ((long)y1 * side + x0) * d + c] + wx * f[posOff + ((long)y1 * side + x1) * d + c]);
        x[i] += v;
    }

    /// <summary>
    /// Qwen-VL 2D rotary angles: rope[r, 2s], rope[r, 2s + 1] = cos, sin of θ, θ = (s &lt; half/2 ? py : px) · 10000^(−2·sec/half),
    /// sec = s mod half/2, for s &lt; half = hd/2.
    /// </summary>
    public static void GridRope(Index1D i, ArrayView<float> rope, GridPatchParams p, int hd)
    {
        int half = hd / 2, r = (int)(i / half), s = (int)(i - (long)r * half);
        GridPatch(r, p, out int py, out int px);
        int q = half / 2, sec = s < q ? s : s - q;
        // double precision (as the CPU encoder): fast 32-bit cos/sin are inexact for angles of tens of radians
        double theta = (s < q ? py : px) * Math.Pow(10000.0, -2.0 * sec / half);
        rope[(long)r * hd + 2 * s] = (float)Math.Cos(theta);
        rope[(long)r * hd + 2 * s + 1] = (float)Math.Sin(theta);
    }

    /// <summary>
    /// Patches of RGB tiles: patches[r, (c, ky, kx)] = (pixel / 255 - mean[c]) / std[c] for patch row r = tile · N + j,
    /// where j is the patch's position in merge-block order and order[j] its index in the tile's row-major patch grid.
    /// </summary>
    public static void Patchify(Index1D i, ArrayView<byte> pixels, ArrayView<float> patches, ArrayView<int> order, ArrayView<float> f, PatchParams p)
    {
        int r = (int)(i / p.Pp), e = (int)(i - (long)r * p.Pp);
        int tile = r / p.N, pi = order[r - tile * p.N];
        int py = pi / p.Side, px = pi - py * p.Side;
        int pp2 = p.Ps * p.Ps, c = e / pp2, rem = e - c * pp2, ky = rem / p.Ps, kx = rem - ky * p.Ps;
        long pix = (((long)tile * p.ImageSize + py * p.Ps + ky) * p.ImageSize + px * p.Ps + kx) * 3 + c;
        patches[i] = (pixels[pix] / 255f - f[p.MeanOff + c]) / f[p.StdOff + c];
    }

    /// <summary>x[r, :] += positions[order[r mod n], :] (learned absolute position of the row's patch).</summary>
    public static void AddPositions(Index1D i, ArrayView<float> x, ArrayView<float> f, ArrayView<int> order, long posOff, int n, int d)
    {
        int r = (int)(i / d), c = (int)(i - (long)r * d);
        x[i] += f[posOff + (long)order[r % n] * d + c];
    }
}
