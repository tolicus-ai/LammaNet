using LammaNet.Multimodal;

namespace LammaNet.Tests;

/// <summary><see cref="RgbImage.ResizePil"/> against outputs of Pillow 12 (Image.resize) stored in TestData.</summary>
public class ImageResizeTests
{
    private static RgbImage Page() => RgbImage.Decode(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", "docpage.png")));
    private static byte[] Expected(string name) => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", name));

    [Fact]
    public void LanczosDownscaleMatchesPillow() =>
        Assert.Equal(Expected("docpage-pil-lanczos-61x79.raw"), Page().ResizePil(61, 79, ResampleFilter.Lanczos).Pixels);

    [Fact]
    public void BicubicUpscaleOfCropMatchesPillow() =>
        Assert.Equal(Expected("docpage-pil-bicubic-crop-97x71.raw"), Page().Crop(60, 350, 40, 30).ResizePil(97, 71, ResampleFilter.Bicubic).Pixels);
}
