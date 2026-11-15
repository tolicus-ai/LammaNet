using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using LammaNet.Chat;
using LammaNet.Gpu;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Multimodal;

namespace LammaNet.Tests;

/// <summary>Media encoders on the GPU (vision transformers, the Gemma 4 audio Conformer) against the CPU encoders, on real inputs.</summary>
public class VisionEncoderGpuTests
{
    private static (double WorstCos, double MaxAbs) Compare(float[] a, float[] b, int dim)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 1, maxAbs = 0;
        for (int t = 0; t < a.Length / dim; t++)
        {
            var ra = a.AsSpan(t * dim, dim); var rb = b.AsSpan(t * dim, dim);
            worst = Math.Min(worst, TensorPrimitives.CosineSimilarity(ra, rb));
            for (int i = 0; i < dim; i++) maxAbs = Math.Max(maxAbs, Math.Abs(ra[i] - rb[i]));
        }
        return (worst, maxAbs);
    }

    private static byte[] TestImage(string name) => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", name));

    [Fact]
    public void Idefics3TileEmbeddingsMatchCpu()
    {
        var path = TestModels.Find("granite-docling-258M-BF16.gguf");
        var mmproj = TestModels.Find("mmproj-granite-docling-258M-f16.gguf");
        if (path == null || mmproj == null) return;
        using var model = Model.Load(path);
        using var off = GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 512), new GpuOffloadOptions { Warmup = false });
        if (off == null) return;
        using var cpu = new Idefics3Projector(mmproj, model);
        using var gpu = new Idefics3Projector(mmproj, model, new ProjectorOptions { Accelerator = off as IVisionAccelerator });
        Assert.StartsWith("Cuda", gpu.EncoderDevice);

        var page = RgbImage.Decode(TestImage("docpage.png"));
        var (tiles, _, _) = cpu.Preprocess(page);
        // the whole page (16 tiles + overview): three device passes (8, 8, 1 tiles), text, table and blank tiles
        Assert.True(tiles.Count > 16, $"{tiles.Count} tiles");
        var a = cpu.EncodeTiles(tiles);
        var b = gpu.EncodeTiles(tiles);
        Assert.Equal(tiles.Count * cpu.TokensPerTile, a.Length / model.Config.NEmbd);
        var (worst, maxAbs) = Compare(a, b, model.Config.NEmbd);
        // fp16 attention operands and tensor-core GEMMs over 12 layers: rows agree to cosine ~0.9999
        Assert.True(worst > 0.999, $"worst token cosine {worst:F5}, max |diff| {maxAbs:F4}");
    }

    /// <summary>Qwen-VL whole-image path: the device patch embedding (patches, linear, resized positions, 2D rope) equals the CPU one.</summary>
    [Fact]
    public unsafe void QwenImagePatchEmbeddingOnDeviceMatchesCpu()
    {
        var path = TestModels.Find(TestModels.Qwen35);
        var mmproj = TestModels.Find("mmproj-Qwen3.5-0.8B-F16.gguf");
        if (path == null || mmproj == null) return;
        using var model = Model.Load(path);
        using var off = GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 512), new GpuOffloadOptions { Warmup = false });
        if (off == null) return;
        using var cpu = new Qwen3VLProjector(mmproj, model);
        var spec = cpu.DeviceSpec();
        using var vit = (GpuVit)((IVisionAccelerator)off).CreateVit(spec)!;
        var img = cpu.PrepareImage(RgbImage.Decode(TestImage("nyt-moon.jpeg")));
        int ps = spec.PatchEmbedding!.Patch, n = img.Width / ps * (img.Height / ps), d = spec.Dim, hd = d / spec.Heads;
        var xc = new float[n * d];
        float[] rc;
        fixed (float* p = xc) rc = cpu.EmbedOnCpu(img, p);
        var xg = new float[n * d];
        var rg = new float[n * hd];
        fixed (byte* px = img.Pixels) fixed (float* a = xg, b = rg) vit.EmbedImage(px, img.Width, img.Height, a, b);
        double ropeDiff = rc.Zip(rg, (u, v) => Math.Abs(u - v)).Max();
        var (worst, maxAbs) = Compare(xc, xg, d);
        Assert.True(ropeDiff < 1e-5, $"rope max |diff| {ropeDiff:E2}");
        Assert.True(worst > 0.999999 && maxAbs < 1e-4, $"embedding worst row cosine {worst:F7}, max |diff| {maxAbs:E2}");

        // the whole image path equals the blocks + merger run on the device embedding
        int outDim = spec.Merger!.OutDim, tokens = n / (spec.Merger.Merge * spec.Merger.Merge);
        var viaImage = new float[tokens * outDim];
        var viaDeviceX = new float[tokens * outDim];
        var viaCpuX = new float[tokens * outDim];
        fixed (byte* px = img.Pixels) fixed (float* o = viaImage) vit.RunImage(px, img.Width, img.Height, o);
        fixed (float* x = xg) fixed (float* r = rg) fixed (float* o = viaDeviceX) vit.RunProjected(x, n, n, r, o);
        fixed (float* x = xc) fixed (float* r = rc) fixed (float* o = viaCpuX) vit.RunProjected(x, n, n, r, o);
        var (w1, m1) = Compare(viaImage, viaDeviceX, outDim);
        var (w2, m2) = Compare(viaDeviceX, viaCpuX, outDim);
        Assert.True(w1 > 0.99999, $"RunImage vs RunProjected(device x): worst {w1:F6} max {m1:F4}; device x vs CPU x through the blocks: worst {w2:F6} max {m2:F4}");
        // the blocks are deterministic but sensitive on this image (padded border patches: near-constant rows whose layer
        // norms amplify differences of ~1e-6 to a worst-row cosine of ~0.999), so the routes agree only that closely
        Assert.True(w2 > 0.995, $"device x vs CPU x through the blocks: worst {w2:F6} max {m2:F4} (embedding max |diff| {maxAbs:E2})");
    }

    [Fact]
    public void Gemma4ImageEmbeddingsMatchCpu()
    {
        var path = TestModels.Find("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
        var mmproj = TestModels.Find("mmproj-gemma-4-E2B-it-qat-F16.gguf");
        if (path == null || mmproj == null) return;
        using var model = Model.Load(path, new ModelLoadOptions { Prefetch = false });
        using var off = GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 512), new GpuOffloadOptions { Warmup = false, Layers = 1 });
        if (off == null) return;
        using var cpu = new Gemma4Projector(mmproj, model);
        using var gpu = new Gemma4Projector(mmproj, model, new ProjectorOptions { Accelerator = off as IVisionAccelerator });
        Assert.StartsWith("Cuda", gpu.VisionDevice);
        var img = RgbImage.Decode(TestImage("nyt-moon.jpeg"));
        var (a, ta) = cpu.EncodeImage(img);
        var (b, tb) = gpu.EncodeImage(img);
        Assert.Equal(ta, tb);
        var (worst, maxAbs) = Compare(a, b, model.Config.NEmbd);
        Assert.True(worst > 0.9995, $"worst token cosine {worst:F5}, max |diff| {maxAbs:F4}");
    }

    [Fact]
    public void QwenImageEmbeddingsMatchCpu()
    {
        var path = TestModels.Find(TestModels.Qwen35);
        var mmproj = TestModels.Find("mmproj-Qwen3.5-0.8B-F16.gguf");
        if (path == null || mmproj == null) return;
        using var model = Model.Load(path, new ModelLoadOptions { Prefetch = false });
        using var off = GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 512), new GpuOffloadOptions { Warmup = false, Layers = 1 });
        if (off == null) return;
        using var cpu = new Qwen3VLProjector(mmproj, model);
        using var gpu = new Qwen3VLProjector(mmproj, model, new ProjectorOptions { Accelerator = off as IVisionAccelerator });
        Assert.StartsWith("Cuda", gpu.EncoderDevice);
        var img = RgbImage.Decode(TestImage("nyt-moon.jpeg"));
        var (a, wa, ha) = cpu.EncodeImage(img);
        var (b, wb, hb) = gpu.EncodeImage(img);
        Assert.Equal((wa, ha), (wb, hb));
        var (worst, maxAbs) = Compare(a, b, model.Config.NEmbd);
        // the Qwen ViT has no q/k norms (larger activations): the llama.cpp reference test agrees as well with either encoder
        Assert.True(worst > 0.997, $"worst token cosine {worst:F5}, max |diff| {maxAbs:F4}");
    }

    // ------------------------------------------------------------------ audio

    /// <summary>A deterministic 6 s test signal at 16 kHz: a gliding tone with harmonics, a tremolo and a little noise.</summary>
    internal static float[] SyntheticAudio(int seconds = 6)
    {
        int n = 16000 * seconds;
        var x = new float[n];
        uint lcg = 12345;
        double phase = 0;
        for (int i = 0; i < n; i++)
        {
            double t = i / 16000.0;
            double f = 180 + 1400 * (0.5 - 0.5 * Math.Cos(2 * Math.PI * t / seconds));
            phase += 2 * Math.PI * f / 16000.0;
            double v = Math.Sin(phase) + 0.4 * Math.Sin(2 * phase) + 0.2 * Math.Sin(3.01 * phase);
            v *= 0.25 * (1 + 0.6 * Math.Sin(2 * Math.PI * 3 * t));
            lcg = lcg * 1664525 + 1013904223;
            v += 0.01 * ((lcg >> 8) / (double)(1 << 24) - 0.5);
            x[i] = (float)v;
        }
        return x;
    }

    /// <summary>Rows of the audio embeddings kept in the golden file.</summary>
    private static int[] GoldenRows(int rows) => [0, 1, 2, 7, rows / 2, rows - 2, rows - 1];

    private static string GoldenPath([CallerFilePath] string here = "") => Path.Combine(Path.GetDirectoryName(here)!, "TestData", "gemma4-audio-golden.f32");

    /// <summary>
    /// The audio encoder against embeddings recorded from the original (naive) CPU implementation, on the CPU and on the GPU.
    /// Set LAMMA_WRITE_GOLDEN=1 to re-record (only when the encoder is meant to change its output).
    /// </summary>
    [Fact]
    public void Gemma4AudioMatchesGoldenAndCpu()
    {
        var path = TestModels.Find("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
        var mmproj = TestModels.Find("mmproj-gemma-4-E2B-it-qat-F16.gguf");
        if (path == null || mmproj == null) return;
        using var model = Model.Load(path, new ModelLoadOptions { Prefetch = false });
        var pcm = SyntheticAudio();
        int dim = model.Config.NEmbd;
        float[] a;
        int rows;
        using (var cpu = new Gemma4Projector(mmproj, model)) (a, rows) = cpu.EncodeAudio(pcm);
        var keep = GoldenRows(rows);
        var picked = keep.SelectMany(r => a.AsSpan(r * dim, dim).ToArray()).ToArray();

        if (Environment.GetEnvironmentVariable("LAMMA_WRITE_GOLDEN") == "1")
        {
            File.WriteAllBytes(GoldenPath(), MemoryMarshal.AsBytes(new[] { (float)rows }.Concat(picked).ToArray().AsSpan()).ToArray());
            return;
        }
        var golden = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", "gemma4-audio-golden.f32"))).ToArray();
        Assert.Equal(rows, (int)golden[0]);
        var (gw, gm) = Compare(golden[1..], picked, dim);
        Assert.True(gw > 0.99999, $"CPU vs golden: worst row cosine {gw:F6}, max |diff| {gm:F5}");

        using var off = GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 512), new GpuOffloadOptions { Warmup = false, Layers = 1 });
        if (off == null) return;
        using var gpu = new Gemma4Projector(mmproj, model, new ProjectorOptions { Accelerator = off as IVisionAccelerator });
        var (b, rowsB) = gpu.EncodeAudio(pcm);
        Assert.StartsWith("Cuda", gpu.AudioDevice);
        Assert.Equal(rows, rowsB);
        var (worst, maxAbs) = Compare(a, b, dim);
        Assert.True(worst > 0.999, $"GPU vs CPU: worst row cosine {worst:F5}, max |diff| {maxAbs:F4}");
    }
}
