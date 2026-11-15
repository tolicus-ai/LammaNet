using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using StbImageSharp;

namespace LammaNet.Multimodal;

/// <summary>Resampling filters of <see cref="RgbImage.ResizePil"/> (Pillow semantics; bicubic uses a = -0.5).</summary>
public enum ResampleFilter { Bilinear, Bicubic, Lanczos }

/// <summary>An 8-bit RGB image.</summary>
public sealed class RgbImage(int width, int height, byte[] pixels)
{
    public int Width { get; } = width;
    public int Height { get; } = height;
    /// <summary>Interleaved RGB, row-major, 3 bytes per pixel.</summary>
    public byte[] Pixels { get; } = pixels;

    /// <summary>Decodes PNG, JPEG, BMP, GIF (first frame), TGA, PSD or HDR bytes.</summary>
    public static RgbImage Decode(byte[] encoded)
    {
        var img = ImageResult.FromMemory(encoded, ColorComponents.RedGreenBlue);
        return new RgbImage(img.Width, img.Height, img.Data);
    }

    /// <summary>Copies the <paramref name="w"/>×<paramref name="h"/> region at (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public RgbImage Crop(int x, int y, int w, int h)
    {
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > Width || y + h > Height) throw new ArgumentOutOfRangeException(nameof(w), "crop outside the image");
        var px = new byte[w * h * 3];
        for (int r = 0; r < h; r++) Array.Copy(Pixels, ((y + r) * Width + x) * 3, px, r * w * 3, w * 3);
        return new RgbImage(w, h, px);
    }

    /// <summary>
    /// Pillow-compatible resampling: separable horizontal then vertical pass with fixed-point (22-bit) weights and
    /// 8-bit rounding/clipping after each pass — bit-exact with PIL's Image.resize and llama.cpp's mtmd preprocessing.
    /// </summary>
    public RgbImage ResizePil(int w, int h, ResampleFilter filter)
    {
        if (w <= 0 || h <= 0) throw new ArgumentOutOfRangeException(nameof(w), "resize target must be positive");
        byte[] px = Pixels;
        if (w != Width) px = PilPass(px, Width, Height, w, filter, horizontal: true);
        if (h != Height) px = PilPass(px, w, Height, h, filter, horizontal: false);
        return ReferenceEquals(px, Pixels) ? this : new RgbImage(w, h, px);
    }

    /// <summary>
    /// Aspect-preserving <see cref="ResizePil"/> into a <paramref name="w"/>×<paramref name="h"/> canvas: scaled by
    /// min(w/W, h/H) with the size rounded up, centered, the rest filled with black (llama.cpp mtmd PAD_CEIL).
    /// </summary>
    public RgbImage ResizePilPadded(int w, int h, ResampleFilter filter)
    {
        if (w == Width && h == Height) return this;
        float scale = Math.Min((float)w / Width, (float)h / Height);
        int nw = Math.Min((int)MathF.Ceiling(Width * scale), w), nh = Math.Min((int)MathF.Ceiling(Height * scale), h);
        var inner = ResizePil(nw, nh, filter);
        if (nw == w && nh == h) return inner;
        var px = new byte[w * h * 3];
        int ox = (w - nw) / 2, oy = (h - nh) / 2;
        for (int r = 0; r < nh; r++) Array.Copy(inner.Pixels, r * nw * 3, px, ((oy + r) * w + ox) * 3, nw * 3);
        return new RgbImage(w, h, px);
    }

    private const int PilPrecision = 22;

    private static double PilFilter(ResampleFilter f, double x)
    {
        switch (f)
        {
            case ResampleFilter.Lanczos:
                if (x < -3.0 || x >= 3.0) return 0;
                static double Sinc(double v) => v == 0 ? 1 : Math.Sin(v * Math.PI) / (v * Math.PI);
                return Sinc(x) * Sinc(x / 3.0);
            case ResampleFilter.Bilinear:
                x = Math.Abs(x);
                return x < 1 ? 1 - x : 0;
            default:
                const double a = -0.5;
                x = Math.Abs(x);
                if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
                if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
                return 0;
        }
    }

    /// <summary>One Pillow pass along an axis: per output index, the first input index, the tap count and fixed-point taps.</summary>
    private static byte[] PilPass(byte[] src, int sw, int sh, int outN, ResampleFilter filter, bool horizontal)
    {
        int inN = horizontal ? sw : sh;
        double scale = (double)inN / outN, filterScale = Math.Max(1.0, scale);
        double support = filter switch { ResampleFilter.Bilinear => 1.0, ResampleFilter.Bicubic => 2.0, _ => 3.0 } * filterScale;
        int ksize = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outN * 2];
        var k = new int[outN * ksize];
        var pre = new double[ksize];
        for (int o = 0; o < outN; o++)
        {
            double center = (o + 0.5) * scale, ww = 0, ss = 1.0 / filterScale;
            int xmin = Math.Max(0, (int)(center - support + 0.5));
            int xmax = Math.Min(inN, (int)(center + support + 0.5)) - xmin;
            for (int i = 0; i < xmax; i++) { pre[i] = PilFilter(filter, (i + xmin - center + 0.5) * ss); ww += pre[i]; }
            for (int i = 0; i < ksize; i++)
            {
                double v = i < xmax ? (ww != 0 ? pre[i] / ww : pre[i]) : 0;
                k[o * ksize + i] = (int)(v * (1 << PilPrecision) + (v < 0 ? -0.5 : 0.5));
            }
            bounds[o * 2] = xmin; bounds[o * 2 + 1] = xmax;
        }
        int dw = horizontal ? outN : sw, dh = horizontal ? sh : outN;
        var dst = new byte[dw * dh * 3];
        Parallel.For(0, horizontal ? sh : outN, line =>
        {
            if (horizontal) PilHorizontal(src, dst, line, sw, dw, outN, bounds, k, ksize);
            else PilVertical(src, dst, line, sw * 3, bounds[line * 2], bounds[line * 2 + 1], k, line * ksize);
        });
        return dst;
    }

    private static byte Clip8(int v) => (byte)Math.Clamp(v >> PilPrecision, 0, 255);

    // The SIMD paths below do the same int32 multiply-adds as the scalar loops (two's-complement sums do not depend on
    // the order), so the output stays byte-identical to Pillow.

    /// <summary>One output row of the horizontal pass: per output pixel, the RGB taps (SSE4.1: r, g, b and a spare lane at once).</summary>
    private static unsafe void PilHorizontal(byte[] src, byte[] dst, int line, int sw, int dw, int outN, int[] bounds, int[] k, int ksize)
    {
        const int Half = 1 << (PilPrecision - 1);
        fixed (byte* sp = src, dp = dst)
        fixed (int* kp = k)
        {
            for (int o = 0; o < outN; o++)
            {
                int xmin = bounds[o * 2], cnt = bounds[o * 2 + 1];
                long p = ((long)line * sw + xmin) * 3;
                int* w = kp + (long)o * ksize;
                byte* d = dp + ((long)line * dw + o) * 3;
                // 4-byte loads may read one byte past the last tap: only within the image (not past the final pixel)
                if (Sse41.IsSupported && p + 3L * cnt + 1 <= src.Length)
                {
                    var acc = Vector128.Create(Half);
                    for (int i = 0; i < cnt; i++, p += 3)
                        acc = Sse2.Add(acc, Sse41.MultiplyLow(
                            Sse41.ConvertToVector128Int32(sp + p), Vector128.Create(w[i])));
                    d[0] = Clip8(acc.GetElement(0)); d[1] = Clip8(acc.GetElement(1)); d[2] = Clip8(acc.GetElement(2));
                }
                else
                {
                    int s0 = Half, s1 = Half, s2 = Half;
                    for (int i = 0; i < cnt; i++, p += 3)
                    {
                        int wi = w[i];
                        s0 += sp[p] * wi; s1 += sp[p + 1] * wi; s2 += sp[p + 2] * wi;
                    }
                    d[0] = Clip8(s0); d[1] = Clip8(s1); d[2] = Clip8(s2);
                }
            }
        }
    }

    /// <summary>One output row of the vertical pass: every element of the row from <paramref name="cnt"/> input rows (AVX2: 8 at a time).</summary>
    private static unsafe void PilVertical(byte[] src, byte[] dst, int line, int row, int ymin, int cnt, int[] k, int kOff)
    {
        const int Half = 1 << (PilPrecision - 1);
        fixed (byte* sp = src, dp = dst)
        fixed (int* kp = k)
        {
            byte* outRow = dp + (long)line * row;
            byte* inRow = sp + (long)ymin * row;
            int* w = kp + kOff;
            int e = 0;
            if (Avx2.IsSupported)
            {
                var zero = Vector256<int>.Zero;
                var max = Vector256.Create(255);
                for (; e + 8 <= row; e += 8)
                {
                    var acc = Vector256.Create(Half);
                    byte* c = inRow + e;
                    for (int i = 0; i < cnt; i++, c += row)
                        acc = Avx2.Add(acc, Avx2.MultiplyLow(
                            Avx2.ConvertToVector256Int32(c), Vector256.Create(w[i])));
                    var v = Vector256.Min(Vector256.Max(
                        Avx2.ShiftRightArithmetic(acc, PilPrecision), zero), max);
                    for (int j = 0; j < 8; j++) outRow[e + j] = (byte)v.GetElement(j);
                }
            }
            for (; e < row; e++)
            {
                int acc = Half;
                for (int i = 0; i < cnt; i++) acc += inRow[(long)i * row + e] * w[i];
                outRow[e] = Clip8(acc);
            }
        }
    }

    /// <summary>Pillow bicubic resize (see <see cref="ResizePil"/>).</summary>
    public RgbImage Resize(int w, int h) => ResizePil(w, h, ResampleFilter.Bicubic);
}

/// <summary>Audio decoding, resampling and log-mel feature extraction.</summary>
public static class AudioProcessing
{
    /// <summary>Decodes a RIFF/WAVE file (PCM 8/16/24/32-bit or IEEE float) to mono float samples.</summary>
    public static (float[] Samples, int SampleRate) DecodeWav(byte[] wav)
    {
        if (wav.Length < 12 || wav[0] != 'R' || wav[1] != 'I' || wav[2] != 'F' || wav[3] != 'F')
            throw new NotSupportedException("Only WAV (RIFF) audio is supported; provide PCM samples for other formats");
        int pos = 12, channels = 1, rate = 16000, bits = 16, fmt = 1;
        while (pos + 8 <= wav.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(wav, pos, 4);
            int size = BitConverter.ToInt32(wav, pos + 4);
            int body = pos + 8;
            if (id == "fmt ")
            {
                fmt = BitConverter.ToUInt16(wav, body);
                channels = BitConverter.ToUInt16(wav, body + 2);
                rate = BitConverter.ToInt32(wav, body + 4);
                bits = BitConverter.ToUInt16(wav, body + 14);
                if (fmt == 0xFFFE && size >= 26) fmt = BitConverter.ToUInt16(wav, body + 24); // WAVE_FORMAT_EXTENSIBLE
            }
            else if (id == "data")
            {
                int bytesPer = bits / 8;
                int len = Math.Min(size, wav.Length - body);
                int frames = len / (bytesPer * channels);
                var outS = new float[frames];
                for (int f = 0; f < frames; f++)
                {
                    double acc = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        int o = body + (f * channels + c) * bytesPer;
                        acc += fmt == 3
                            ? (bits == 64 ? BitConverter.ToDouble(wav, o) : BitConverter.ToSingle(wav, o))
                            : bits switch
                            {
                                8 => (wav[o] - 128) / 128.0,
                                16 => BitConverter.ToInt16(wav, o) / 32768.0,
                                24 => ((wav[o] | (wav[o + 1] << 8) | ((sbyte)wav[o + 2] << 16))) / 8388608.0,
                                32 => BitConverter.ToInt32(wav, o) / 2147483648.0,
                                _ => throw new NotSupportedException($"{bits}-bit PCM not supported"),
                            };
                    }
                    outS[f] = (float)(acc / channels);
                }
                return (outS, rate);
            }
            pos = body + size + (size & 1);
        }
        throw new InvalidDataException("WAV file has no data chunk");
    }

    /// <summary>Band-limited resampling (windowed sinc).</summary>
    public static float[] Resample(float[] x, int from, int to)
    {
        if (from == to) return x;
        double ratio = (double)to / from;
        int n = (int)Math.Ceiling(x.Length * ratio);
        var y = new float[n];
        double cutoff = Math.Min(1.0, ratio);
        const int Half = 16;
        Parallel.For(0, n, i =>
        {
            double t = i / ratio;
            int c = (int)Math.Floor(t);
            double acc = 0, wsum = 0;
            for (int k = c - Half + 1; k <= c + Half; k++)
            {
                if (k < 0 || k >= x.Length) continue;
                double d = (t - k) * cutoff;
                double sinc = d == 0 ? 1 : Math.Sin(Math.PI * d) / (Math.PI * d);
                double win = 0.5 + 0.5 * Math.Cos(Math.PI * (t - k) / Half);
                double w = sinc * win;
                acc += x[k] * w; wsum += w;
            }
            y[i] = (float)(wsum != 0 ? acc / wsum : 0);
        });
        return y;
    }

    /// <summary>In-place iterative radix-2 complex FFT (n must be a power of two).</summary>
    internal static void Fft(double[] re, double[] im)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { (re[i], re[j]) = (re[j], re[i]); (im[i], im[j]) = (im[j], im[i]); }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = -2 * Math.PI / len;
            double wr = Math.Cos(ang), wi = Math.Sin(ang);
            for (int i = 0; i < n; i += len)
            {
                double cr = 1, ci = 0;
                for (int k = 0; k < len / 2; k++)
                {
                    int a = i + k, b = i + k + len / 2;
                    double tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr; im[b] = im[a] - ti;
                    re[a] += tr; im[a] += ti;
                    double ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr; cr = ncr;
                }
            }
        }
    }

    /// <summary>HTK-scale triangular mel filterbank [nMel][nFft/2+1] (no area normalization).</summary>
    internal static float[] MelFilterbankHtk(int nMel, int nFft, int sampleRate)
    {
        static double HzToMel(double f) => 2595.0 * Math.Log10(1.0 + f / 700.0);
        static double MelToHz(double m) => 700.0 * (Math.Pow(10.0, m / 2595.0) - 1.0);
        int bins = nFft / 2 + 1;
        double mlo = HzToMel(0), mhi = HzToMel(sampleRate / 2.0);
        var hz = new double[nMel + 2];
        for (int i = 0; i < nMel + 2; i++) hz[i] = MelToHz(mlo + (mhi - mlo) * i / (nMel + 1));
        double step = (double)sampleRate / nFft;
        var fb = new float[nMel * bins];
        for (int m = 0; m < nMel; m++)
        {
            double l = hz[m], c = hz[m + 1], r = hz[m + 2];
            for (int k = 0; k < bins; k++)
            {
                double f = k * step, w = 0;
                if (f >= l && f <= c) w = (f - l) / Math.Max(1e-30, c - l);
                else if (f > c && f <= r) w = (r - f) / Math.Max(1e-30, r - c);
                fb[m * bins + k] = (float)w;
            }
        }
        return fb;
    }

    /// <summary>
    /// Gemma 4 audio features: 16 kHz, 20 ms periodic-Hann window (320) zero-padded to a 512-point FFT, 10 ms hop,
    /// semicausal left padding, magnitude spectrum, 128 HTK mel bins, natural log with floor 1e-3.
    /// Returns mel-major features [nMel][frames].
    /// </summary>
    public static (float[] Mel, int Frames) Gemma4LogMel(float[] samples, int nMel = 128)
    {
        const int nFft = 512, window = 320, hop = 160;
        int padLeft = window / 2;
        int nWithLeft = samples.Length + padLeft;
        int frames = (nWithLeft - (window + 1)) / hop + 1;
        if (frames <= 0) return (new float[nMel], 0);
        int needed = (frames - 1) * hop + nFft;
        int totalPad = Math.Max(needed - samples.Length, padLeft);
        var padded = new float[totalPad + samples.Length];
        Array.Copy(samples, 0, padded, padLeft, samples.Length);
        var hann = new double[nFft];
        for (int i = 0; i < window; i++) hann[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / window);
        var fb = MelFilterbankHtk(nMel, nFft, 16000);
        int bins = nFft / 2 + 1;
        var mel = new float[nMel * frames];
        Parallel.For(0, frames, () => (new double[nFft], new double[nFft], new double[bins]), (f, _, buf) =>
        {
            var (re, im, mag) = buf;
            int off = f * hop;
            for (int j = 0; j < nFft; j++)
            {
                int idx = off + j;
                re[j] = idx < padded.Length ? hann[j] * padded[idx] : 0;
                im[j] = 0;
            }
            Fft(re, im);
            for (int k = 0; k < bins; k++) mag[k] = Math.Sqrt(re[k] * re[k] + im[k] * im[k]);
            for (int m = 0; m < nMel; m++)
            {
                double s = 0;
                for (int k = 0; k < bins; k++) s += mag[k] * fb[m * bins + k];
                mel[m * frames + f] = (float)Math.Log(Math.Max(s, 0.001));
            }
            return buf;
        }, _ => { });
        return (mel, frames);
    }
}
