using System.Diagnostics;
using LammaNet.Compute;
using LammaNet.Inference;
using LammaNet.Memory;
using LammaNet.Models;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;

namespace LammaNet.Cli;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0) { Console.WriteLine("usage: lamma <command> ..."); return 1; }
        using var logging = CreateLogging();
        LammaNet.Diagnostics.LammaLogging.LoggerFactory = logging;
        return Run(args);
    }

    /// <summary>Library logs on stderr; LAMMA_LOG = trace|debug|information|warning|error|critical|none (default warning).</summary>
    private static ILoggerFactory CreateLogging()
    {
        var level = Environment.GetEnvironmentVariable("LAMMA_LOG") is { Length: > 0 } v && Enum.TryParse<LogLevel>(v, ignoreCase: true, out var l) ? l : LogLevel.Warning;
        return LoggerFactory.Create(b =>
        {
            b.SetMinimumLevel(level);
            b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss.fff "; });
            b.Services.Configure<ConsoleLoggerOptions>(o => o.LogToStandardErrorThreshold = LogLevel.Trace);
        });
    }

    private static int Run(string[] args)
    {
        int optionStart = args[0] switch { "ask" => 3, "bench" => 4, "gpucheck" => 2, _ => args.Length };
        int compact = Array.IndexOf(args, "--compact-kv", Math.Min(optionStart, args.Length));
        KvCacheType kv = compact >= 0 ? KvCacheType.Q8_0 : KvCacheType.F16;
        if (compact >= 0) args = [.. args[..compact], .. args[(compact + 1)..]];
        return args[0] switch
        {
            "logits" => Logits(args[1], args[2], args.Length > 3 ? int.Parse(args[3]) : 0, args.Contains("--all")),
            "tokenize" => Tokenize(args[1], args[2]),
            "probs" => Probs(args[1], args[2], args[3], Opt(args, "--mmproj"), Opt(args, "--image")),
            "info" => Info(ModelArg(args[1]), args.Contains("--tensors"), args.Contains("--keys"), Opt(args, "--vocab") is { } nv ? int.Parse(nv) : 0),
            "render" => Render(args[1], args[2]),
            "agent" => Demos.Agent(args[1]).GetAwaiter().GetResult(),
            "structured" => Demos.Structured(args[1]).GetAwaiter().GetResult(),
            "embed" => Demos.Embeddings(args[1]).GetAwaiter().GetResult(),
            "embed-json" => Demos.EmbedJson(args[1], args[2..]).GetAwaiter().GetResult(),
            "vision" => Demos.Vision(args[1], args[2], args[3]).GetAwaiter().GetResult(),
            "stt" => Demos.SpeechToText(args[1], args[2], args[3]).GetAwaiter().GetResult(),
            "realtime" => Demos.Realtime(args[1], args[2], args[3]).GetAwaiter().GetResult(),
            "ask" => Ask(args[1], args[2], args.Contains("--tools"), args.Contains("--think"),
                Opt(args, "--mmproj"), Opt(args, "--image"), Opt(args, "--audio"), args.Contains("--special"), kv,
                Opt(args, "--ctx") is { } ctx ? int.Parse(ctx) : 4096).GetAwaiter().GetResult(),
            "encode" => EncodeMedia(args[1], args[2], args[3], Opt(args, "--repeat") is { } rp ? int.Parse(rp) : 3, Opt(args, "--max-tokens") is { } mt ? int.Parse(mt) : null,
                Opt(args, "--save"), Opt(args, "--compare")),
            "allocs" => Allocs(args[1], args.Length > 2 ? int.Parse(args[2]) : 512, args.Length > 3 ? int.Parse(args[3]) : 256, args.Contains("--gpu")).GetAwaiter().GetResult(),
            "pull" => Pull(args[1], args.Contains("--dry-run"), args.Contains("--no-mmproj")).GetAwaiter().GetResult(),
            "bench" => Bench(args[1], int.Parse(args[2]), int.Parse(args[3]), kv),
            "enginecheck" => EngineCheck.Run(args[1], args.Length > 2 ? int.Parse(args[2]) : 64).GetAwaiter().GetResult(),
            "gpucheck" => GpuCheck(args[1], args.Length > 2 ? int.Parse(args[2]) : -1, args.Length > 3 ? int.Parse(args[3]) : 32, kv),
            "gpu-kstats" => PrintLines(LammaNet.Gpu.GpuBench.KernelStats()),
            "gpu-peak" => PrintLines(LammaNet.Gpu.GpuBenchPeaks.Run()),
            "spec" => Spec(args[1], args[2], args.Length > 3 && int.TryParse(args[3], out var ng) ? ng : 128, args.Contains("--gpu"),
                Opt(args, "--drafts") is { } nd ? int.Parse(nd) : null, Opt(args, "--pmin") is { } pm ? float.Parse(pm, System.Globalization.CultureInfo.InvariantCulture) : 0f).GetAwaiter().GetResult(),
            "bench-dattn" => PrintLines(from hd in new[] { int.Parse(args[1]) } from c in args[4..].Select(int.Parse)
                                        select LammaNet.Gpu.GpuBench.DecodeAttention(hd, int.Parse(args[2]), int.Parse(args[3]), c,
                                            Environment.GetEnvironmentVariable("LAMMA_ATTN_CHUNK") is { } ch ? int.Parse(ch) : 32)),
            "bench-gemm" => PrintLines(args[2].Split(',').Select(t => LammaNet.Gpu.GpuBench.Gemm(args[1], t, args.Length > 3 ? int.Parse(args[3]) : 512))),
            "bench-gpu" => BenchGpu(args[1], args[2], args.Length > 3 ? int.Parse(args[3]) : 1, args.Length > 4 ? int.Parse(args[4]) : 1),
            "bench-attn" => BenchAttn(args[1], int.Parse(args[2]), int.Parse(args[3])),
            "bench-matmul" => BenchMatMul(args[1], args[2], args.Length > 3 ? int.Parse(args[3]) : 1, args.Length > 4 ? int.Parse(args[4]) : 4),
            _ => Unknown(args[0]),
        };
    }

    /// <summary>Greedy generation without and with an MTP drafter: speed, acceptance and identical output.</summary>
    private static async Task<int> Spec(string modelPath, string drafterPath, int nGen, bool gpu, int? drafts, float pMin)
    {
        string[] prompts =
        [
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
            "Explain how a hash table works, with an example in C#.",
            "List the planets of the solar system and one fact about each.",
        ];
        async Task<(string[] Texts, double TokPerSec, long Drafted, long Accepted)> Run(string? drafter)
        {
            using var engine = InferenceEngine.Load(modelPath, new EngineOptions
            {
                ContextSize = 2048, AutoAccelerator = gpu, DraftModelPath = drafter, SpeculativeDraftTokens = drafter != null ? drafts : 0,
                DraftMinProbability = pMin,
            });
            var chat = new LammaNet.Chat.ChatEngine(engine);
            var texts = new string[prompts.Length];
            long tokens = 0; double seconds = 0;
            for (int i = 0; i < prompts.Length; i++)
            {
                var (prompt, _) = chat.BuildPrompt([LammaNet.Chat.ChatMessage.User(prompts[i])], new LammaNet.Chat.ChatRequestOptions { EnableThinking = false });
                var (text, stats) = await engine.GenerateTextAsync(new GenerationRequest { Prompt = prompt, MaxTokens = nGen, Sampling = LammaNet.Sampling.SamplingOptions.Greedy });
                texts[i] = text;
                if (i > 0 || prompts.Length == 1) { tokens += stats.CompletionTokens; seconds += stats.GenerationTime.TotalSeconds; }   // first one warms up
            }
            if (Environment.GetEnvironmentVariable("LAMMA_GPU_PROFILE") == "1") { Console.WriteLine($"--- GPU profile ({(drafter != null ? "with" : "without")} drafter)"); (engine.Offload as LammaNet.Gpu.GpuLayerOffload)?.ReportProfile(Console.Out); }
            return (texts, tokens / seconds, engine.TotalDraftTokens, engine.TotalAcceptedDraftTokens);
        }
        var plain = await Run(null);
        if (Profiler.Enabled) { Console.WriteLine("--- profile without drafter"); Profiler.Report(Console.Out); }
        var spec = await Run(drafterPath);
        if (Profiler.Enabled) { Console.WriteLine("--- profile with drafter"); Profiler.Report(Console.Out); }
        Console.WriteLine($"without drafter: {plain.TokPerSec,7:F1} tok/s");
        Console.WriteLine($"with drafter:    {spec.TokPerSec,7:F1} tok/s  ({spec.TokPerSec / plain.TokPerSec:F2}x), drafted {spec.Drafted}, accepted {spec.Accepted} ({(spec.Drafted > 0 ? 100.0 * spec.Accepted / spec.Drafted : 0):F0}%)");
        int same = plain.Texts.Zip(spec.Texts).Count(p => p.First == p.Second);
        Console.WriteLine($"identical outputs: {same}/{prompts.Length}");
        for (int i = 0; i < prompts.Length; i++)
            if (plain.Texts[i] != spec.Texts[i])
                Console.WriteLine($"--- {i} differs:" + Environment.NewLine + plain.Texts[i] + Environment.NewLine + "+++" + Environment.NewLine + spec.Texts[i]);
        return same == prompts.Length ? 0 : 1;
    }

    private static unsafe int BenchMatMul(string modelPath, string tensor, int nTok, int threads)
    {
        using var model = Model.Load(modelPath);
        var w = LammaNet.Tensors.WeightTensor.FromGguf(model.File, model.File.GetTensor(tensor));
        var pool = new ComputePool(threads);
        var mm = new MatMul(pool);
        using var x = new LammaNet.Tensors.NativeBuffer((long)nTok * w.Cols * 4);
        using var y = new LammaNet.Tensors.NativeBuffer((long)nTok * w.Rows * 4);
        var rnd = new Random(1);
        for (long i = 0; i < (long)nTok * w.Cols; i++) x.F[i] = (float)rnd.NextDouble() - 0.5f;
        for (int i = 0; i < 3; i++) mm.Multiply(w, x.F, nTok, y.F);
        var sw = Stopwatch.StartNew();
        int iters = 0;
        while (sw.ElapsedMilliseconds < 2000) { mm.Multiply(w, x.F, nTok, y.F); iters++; }
        double ms = sw.Elapsed.TotalMilliseconds / iters;
        double gb = w.ByteSize / 1e9;
        Console.WriteLine($"{w}: {ms:F3} ms/iter  {gb / (ms / 1000):F1} GB/s  {2.0 * w.Rows * w.Cols * nTok / (ms / 1000) / 1e9:F1} GFLOP/s");
        return 0;
    }

    private static unsafe int BenchAttn(string modelPath, int ctx, int threads)
    {
        using var model = Model.Load(modelPath, new ModelLoadOptions { Prefetch = false });
        var c = model.Config;
        int layer = Array.IndexOf(c.Kinds, LayerKind.Attention);
        var pool = new ComputePool(threads);
        using var kv = new KvCache(c, ctx + 64, Environment.GetEnvironmentVariable("LAMMA_KV") == "f32" ? KvCacheType.F32 : KvCacheType.F16);
        var seq = new SequenceState(kv, c);
        seq.EnsureCapacity(ctx);
        int kd = c.NHeadKv[layer] * c.HeadDimK[layer], vd = c.NHeadKv[layer] * c.HeadDimV[layer];
        var rnd = new Random(3);
        var kr = new float[kd]; var vr = new float[vd];
        fixed (float* kp = kr) fixed (float* vp = vr)
            for (int p = 0; p < ctx; p++)
            {
                for (int i = 0; i < kd; i++) kr[i] = (float)rnd.NextDouble() - 0.5f;
                for (int i = 0; i < vd; i++) vr[i] = (float)rnd.NextDouble() - 0.5f;
                kv.Write(layer, seq.Pages[p / kv.PageSize], p % kv.PageSize, kp, vp);
            }
        var attn = new AttentionKernel(pool);
        using var q = new LammaNet.Tensors.NativeBuffer(c.NHead[layer] * c.HeadDimK[layer] * 4);
        using var o = new LammaNet.Tensors.NativeBuffer(c.NHead[layer] * c.HeadDimV[layer] * 4);
        for (int i = 0; i < c.NHead[layer] * c.HeadDimK[layer]; i++) q.F[i] = (float)rnd.NextDouble() - 0.5f;
        var prm = new AttentionKernel.Params { Cache = kv, Layer = layer, NHead = c.NHead[layer], NKv = c.NHeadKv[layer], HeadDimK = c.HeadDimK[layer], HeadDimV = c.HeadDimV[layer], Scale = 0.1f, Q = q.F, QStride = 0, Out = o.F, OutStride = 0 };
        var seqs = new[] { seq }; var pos = new[] { ctx - 1 };
        for (int i = 0; i < 100; i++) attn.Run(prm, 1, seqs, pos);
        var sw = Stopwatch.StartNew();
        int iters = 0;
        while (sw.ElapsedMilliseconds < 2000) { attn.Run(prm, 1, seqs, pos); iters++; }
        double us = sw.Elapsed.TotalMicroseconds / iters;
        double bytes = (double)ctx * (kv.KRowBytes(layer) + kv.VRowBytes(layer));
        Console.WriteLine($"attn ctx={ctx} heads={prm.NHead}/{prm.NKv} hd={prm.HeadDimK}: {us:F1} us  ({bytes / us / 1e3:F2} GB/s)");
        return 0;
    }

    private static int Bench(string modelPath, int nPrompt, int nGen, KvCacheType kvType)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nPrompt);
        ArgumentOutOfRangeException.ThrowIfNegative(nGen);
        using var model = Model.Load(modelPath);
        int threads = int.TryParse(Environment.GetEnvironmentVariable("LAMMA_THREADS"), out var nt) ? nt : ComputePool.DefaultThreadCount();
        using var pool = new ComputePool(threads);
        int maxBatch = Math.Min(512, nPrompt), pageSize = 32, capacity = checked(nPrompt + nGen + 64);
        int pages = KvCache.PagesFor(capacity, pageSize);
        using var offload = GpuFromEnv(model, new OffloadGeometry(pageSize, pages, maxBatch) { KvCacheType = kvType });
        using var kv = new KvCache(model.Config, capacity, kvType, pageSize, offload == null ? null : l => !offload.OwnsKv(l), offload?.RecurrentStore);
        using var runner = new ModelRunner(model, kv, pool, maxBatch, 0, offload);
        var rnd = new Random(5);
        var batch = new Batch();
        // warmup
        using (var warm = new SequenceState(kv, model.Config))
        {
            for (int i = 0; i < Math.Min(32, nPrompt); i++) batch.Add(warm, rnd.Next(1000, 20000), i, i == Math.Min(32, nPrompt) - 1);
            runner.Forward(batch);
        }
        for (int rep = 0; rep < 3; rep++)
        {
            using var seq = new SequenceState(kv, model.Config);
            var sw = Stopwatch.StartNew();
            for (int start = 0; start < nPrompt; start += runner.MaxBatch)
            {
                batch.Clear();
                int end = Math.Min(nPrompt, start + runner.MaxBatch);
                for (int i = start; i < end; i++) batch.Add(seq, rnd.Next(1000, 20000), i, i == nPrompt - 1);
                runner.Forward(batch);
            }
            double pp = sw.Elapsed.TotalSeconds;
            sw.Restart();
            for (int g = 0; g < nGen; g++)
            {
                batch.Clear();
                batch.Add(seq, rnd.Next(1000, 20000), seq.Length, true);
                runner.Forward(batch);
            }
            double tg = sw.Elapsed.TotalSeconds;
            Console.WriteLine($"pp{nPrompt}: {nPrompt / pp,8:F1} tok/s   tg{nGen}: {nGen / Math.Max(tg, 1e-9),8:F1} tok/s");
            if (rep == 0) (offload as LammaNet.Gpu.GpuLayerOffload)?.ResetProfile(); // drop JIT / warm-up
        }
        Profiler.Report(Console.Out);
        (offload as LammaNet.Gpu.GpuLayerOffload)?.ReportProfile(Console.Out);
        return 0;
    }

    private static int Render(string templateFile, string contextJson)
    {
        var tpl = new LammaNet.Templates.Jinja.JinjaTemplate(File.ReadAllText(templateFile));
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(contextJson));
        var ctx = new Dictionary<string, object?>();
        foreach (var p in doc.RootElement.EnumerateObject()) ctx[p.Name] = LammaNet.Templates.Jinja.JinjaValue.FromJson(p.Value.Clone());
        try { Console.Out.Write(tpl.Render(ctx)); return 0; }
        catch (LammaNet.Templates.Jinja.JinjaException e) { Console.Out.Write("ERROR: " + e.Message); return 2; }
    }

    private static string? Opt(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    /// <summary>
    /// <c>pull hf:owner/repo[:QUANT] | owner/repo/file.gguf | https://... [--dry-run] [--no-mmproj]</c>: resolves a model through
    /// the model cache and downloads what is missing (<c>--dry-run</c>: only report what would be used).
    /// </summary>
    private static async Task<int> Pull(string reference, bool dryRun, bool noMmproj)
    {
        using var cache = new ModelCache(new ModelCacheOptions { Progress = new ConsoleProgress() });
        var sw = Stopwatch.StartNew();
        ModelResolution r;
        if (reference.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || reference.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            r = dryRun ? await cache.ResolveAsync(new Uri(reference)) : await cache.GetAsync(new Uri(reference));
        else
        {
            var m = HuggingFaceModel.Parse(reference) with { IncludeMmproj = !noMmproj };
            r = dryRun ? await cache.ResolveAsync(m) : await cache.GetAsync(m);
        }
        Console.WriteLine($"{r.Source}{(r.Commit != null ? " @ " + r.Commit : "")}{(r.FromCacheOnly ? " (from cache, hub not consulted)" : "")}  [{sw.Elapsed.TotalSeconds:F1} s]");
        foreach (var f in new[] { r.Model, r.Mmproj })
            if (f != null)
                Console.WriteLine($"  {(f.IsCached ? "cached " : "missing")} {f.RemotePath} ({(f.Size is long b ? $"{b / 1e6:F1} MB" : "?")}) -> {f.LocalPath}");
        return 0;
    }

    /// <summary>
    /// <c>encode model.gguf mmproj.gguf image|audio.wav [--repeat N] [--max-tokens N] [--save f32file] [--compare f32file]</c>:
    /// times the media encoder on the CPU and on the GPU and compares the embeddings (worst row cosine, largest difference)
    /// with the CPU result, or with embeddings saved earlier (raw float32).
    /// </summary>
    private static unsafe int EncodeMedia(string modelPath, string mmproj, string media, int repeat, int? maxTokens, string? save, string? compare)
    {
        float[]? golden = compare != null ? System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(compare)).ToArray() : null;
        static string Compare(float[] reference, LammaNet.Inference.EmbeddingSegment seg)
        {
            if (reference.Length != seg.Data.Length) return $"size {seg.Data.Length} != {reference.Length}";
            double worst = 1, maxAbs = 0;
            for (int t = 0; t < seg.Rows; t++)
            {
                var a = reference.AsSpan(t * seg.Dim, seg.Dim); var b = seg.Data.AsSpan(t * seg.Dim, seg.Dim);
                worst = Math.Min(worst, System.Numerics.Tensors.TensorPrimitives.CosineSimilarity(a, b));
                for (int i = 0; i < seg.Dim; i++) maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
            }
            return $"worst row cosine {worst:F5}, max |diff| {maxAbs:F4}";
        }
        using var model = Model.Load(modelPath, new ModelLoadOptions { Prefetch = false });
        using var off = LammaNet.Gpu.GpuLayerOffload.TryCreate(model, new OffloadGeometry(32, 64, 512), new LammaNet.Gpu.GpuOffloadOptions { Warmup = false, Layers = 1 });
        LammaNet.Chat.ChatPart part = media.EndsWith(".wav", StringComparison.OrdinalIgnoreCase)
            ? new LammaNet.Chat.AudioPart { EncodedBytes = File.ReadAllBytes(media) }
            : new LammaNet.Chat.ImagePart { EncodedBytes = File.ReadAllBytes(media) };
        float[]? reference = null;
        foreach (var acc in new[] { null, off as LammaNet.Multimodal.IVisionAccelerator })
        {
            var enc = LammaNet.Multimodal.MultimodalProjector.Load(mmproj, model, new LammaNet.Multimodal.ProjectorOptions { Accelerator = acc, MaxImageTokens = maxTokens });
            try
            {
                string device = enc switch
                {
                    LammaNet.Multimodal.Gemma4Projector g => part is LammaNet.Chat.AudioPart ? g.AudioDevice : g.VisionDevice,
                    LammaNet.Multimodal.Qwen3VLProjector q => q.EncoderDevice,
                    LammaNet.Multimodal.Idefics3Projector i => i.EncoderDevice,
                    _ => "?",
                };
                enc.Encode(part);   // warm-up
                Profiler.Report(TextWriter.Null);   // drop the warm-up timings
                var sw = Stopwatch.StartNew();
                LammaNet.Inference.EmbeddingSegment? seg = null;
                for (int r = 0; r < repeat; r++) seg = enc.Encode(part).OfType<LammaNet.Inference.EmbeddingSegment>().First();
                double ms = sw.Elapsed.TotalMilliseconds / repeat;
                string cmp = "";
                if (reference == null)
                {
                    reference = seg!.Data;
                    if (save != null) File.WriteAllBytes(save, System.Runtime.InteropServices.MemoryMarshal.AsBytes(reference.AsSpan()).ToArray());
                }
                else cmp = "  vs CPU: " + Compare(reference, seg!);
                if (golden != null) cmp += "  vs saved: " + Compare(golden, seg);
                Console.WriteLine($"{device,-36} {ms,8:F1} ms/encode, {seg!.Rows} tokens{cmp}");
                Profiler.Report(Console.Out);
            }
            finally { (enc as IDisposable)?.Dispose(); }
            if (acc == null && off == null) break;
        }
        return 0;
    }

    /// <summary>
    /// <c>allocs model.gguf [nPrompt] [nGen] [--gpu]</c>: managed allocations, GC collections and token-latency spread of a
    /// generation through the engine (prompt of random tokens, greedy, EOS ignored).
    /// </summary>
    private static async Task<int> Allocs(string modelPath, int nPrompt, int nGen, bool gpu)
    {
        using var engine = InferenceEngine.Load(modelPath, new EngineOptions { ContextSize = nPrompt + nGen + 64, AutoAccelerator = gpu });
        var rnd = new Random(7);
        GenerationRequest Request(int n, int gen) => new()
        {
            Prompt = new Prompt(Enumerable.Range(0, n).Select(_ => rnd.Next(1000, 20000))),
            Sampling = LammaNet.Sampling.SamplingOptions.Greedy, MaxTokens = gen, IgnoreEos = true, CachePrompt = false,
        };
        await engine.GenerateTextAsync(Request(64, 16));   // warm-up (JIT, kernels)
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        using var sampler = new AllocationSampler();
        int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
        long a0 = GC.GetTotalAllocatedBytes(true);
        var gaps = new List<double>(nGen);
        var sw = Stopwatch.StartNew();
        double last = 0, firstToken = 0;
        long aPrompt = 0;
        await foreach (var _ in engine.GenerateAsync(Request(nPrompt, nGen)))
        {
            double now = sw.Elapsed.TotalMilliseconds;
            if (gaps.Count == 0 && firstToken == 0) { firstToken = now; aPrompt = GC.GetTotalAllocatedBytes(true) - a0; }
            else gaps.Add(now - last);
            last = now;
        }
        long a1 = GC.GetTotalAllocatedBytes(true);
        gaps.Sort();
        double P(double q) => gaps.Count == 0 ? 0 : gaps[Math.Min(gaps.Count - 1, (int)(q * gaps.Count))];
        Console.WriteLine($"{(gpu ? engine.Offload?.DeviceName ?? "CPU" : "CPU")}: prompt {nPrompt} tok -> first token {firstToken:F1} ms, {aPrompt / 1024.0:F0} KB allocated");
        Console.WriteLine($"  generation {gaps.Count + 1} tok: {(a1 - a0 - aPrompt) / (double)(gaps.Count + 1):F0} B allocated per token");
        Console.WriteLine($"  GC collections gen0/1/2: {GC.CollectionCount(0) - g0}/{GC.CollectionCount(1) - g1}/{GC.CollectionCount(2) - g2}");
        Console.WriteLine($"  token gap p50 {P(0.5):F2} ms  p90 {P(0.9):F2} ms  p99 {P(0.99):F2} ms  max {(gaps.Count > 0 ? gaps[^1] : 0):F2} ms");
        foreach (var (type, kb) in sampler.Top(12)) Console.WriteLine($"  {kb,8:F0} KB (sampled)  {type}");
        return 0;
    }

    /// <summary>Samples allocations by type from the runtime's AllocationTick events (one event per ~100 KB allocated).</summary>
    private sealed class AllocationSampler : System.Diagnostics.Tracing.EventListener
    {
        private readonly Dictionary<string, double> _bytes = new();

        protected override void OnEventSourceCreated(System.Diagnostics.Tracing.EventSource source)
        {
            if (source.Name == "Microsoft-Windows-DotNETRuntime")
                EnableEvents(source, System.Diagnostics.Tracing.EventLevel.Verbose, (System.Diagnostics.Tracing.EventKeywords)0x1);
        }

        protected override void OnEventWritten(System.Diagnostics.Tracing.EventWrittenEventArgs e)
        {
            if (e.EventName?.StartsWith("GCAllocationTick") != true || e.Payload == null) return;
            int ti = e.PayloadNames!.IndexOf("TypeName"), ai = e.PayloadNames.IndexOf("AllocationAmount64");
            if (ti < 0) return;
            string type = e.Payload[ti]?.ToString() ?? "?";
            double amount = ai >= 0 ? Convert.ToDouble(e.Payload[ai]) : 100_000;
            lock (_bytes) _bytes[type] = _bytes.GetValueOrDefault(type) + amount;
        }

        public IEnumerable<(string Type, double KB)> Top(int n)
        {
            lock (_bytes) return _bytes.OrderByDescending(kv => kv.Value).Take(n).Select(kv => (kv.Key, kv.Value / 1024)).ToList();
        }
    }

    /// <summary>A model argument: a local path, or <c>hf:owner/repo[:QUANT]</c>, <c>hf://...</c>, <c>http(s)://...</c> fetched through the model cache.</summary>
    private static string ModelArg(string arg) => ModelArg(arg, false, out _);

    private static string ModelArg(string arg, bool withMmproj, out string? mmproj)
    {
        mmproj = null;
        bool url = arg.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || arg.StartsWith("https://", StringComparison.OrdinalIgnoreCase);
        if (!url && !arg.StartsWith("hf:", StringComparison.OrdinalIgnoreCase)) return arg;
        using var cache = new ModelCache(new ModelCacheOptions { Progress = new ConsoleProgress() });
        var r = url ? cache.GetAsync(new Uri(arg)).GetAwaiter().GetResult()
            : cache.GetAsync(HuggingFaceModel.Parse(arg) with { IncludeMmproj = withMmproj }).GetAwaiter().GetResult();
        Console.Error.WriteLine($"-- {r.Source}{(r.Commit != null ? " @ " + r.Commit[..12] : "")}: {r.ModelPath}");
        mmproj = r.MmprojPath;
        return r.ModelPath;
    }

    private sealed class ConsoleProgress : IProgress<ModelDownloadProgress>
    {
        public void Report(ModelDownloadProgress p) =>
            Console.Error.Write(p.TotalBytes is long t && t > 0
                ? $"\r-- {p.File}: {p.BytesDownloaded / 1e6:F0}/{t / 1e6:F0} MB ({100.0 * p.BytesDownloaded / t:F0}%)" + (p.BytesDownloaded >= t ? "\n" : "")
                : $"\r-- {p.File}: {p.BytesDownloaded / 1e6:F0} MB");
    }

    private static async Task<int> Ask(string modelPath, string question, bool tools, bool think, string? mmproj, string? image, string? audio, bool special, KvCacheType kvType, int contextSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextSize);
        modelPath = ModelArg(modelPath, mmproj == null && (image != null || audio != null), out var cachedMmproj);
        mmproj ??= cachedMmproj;
        using var engine = InferenceEngine.Load(modelPath, new EngineOptions
        {
            // The one-shot CLI serves one sequence. Its entire prompt and decode fit in one context.
            ContextSize = contextSize, KvCacheTokens = contextSize, KvCacheType = kvType,
            SpeculativeDraftTokens = int.TryParse(Environment.GetEnvironmentVariable("LAMMA_SPEC"), out var spec) ? spec : null,
            Offload = GpuFactoryFromEnv(),
        });
        Console.Error.WriteLine(engine.Offload is { } dev ? $"-- {dev.DeviceName}: {dev.LayerCount}/{engine.Config.NLayer} layers" : $"-- CPU ({LammaNet.Gpu.GpuLayerOffload.LastUnavailableReason ?? "no accelerator"})");
        if (engine.Offload is LammaNet.Gpu.GpuLayerOffload gpu)
            Console.Error.WriteLine($"-- {kvType} KV, context {contextSize}, allocated device buffers {gpu.DeviceBytes / (1024 * 1024)} MiB at startup");
        var chat = new LammaNet.Chat.ChatEngine(engine);
        if (mmproj != null) chat.MediaEncoders.Add(LammaNet.Multimodal.MultimodalProjector.Load(mmproj, engine));
        var toolDefs = tools ? new List<LammaNet.Chat.ToolDefinition>
        {
            new() { Name = "get_weather", Description = "Get the current weather for a city",
                Parameters = System.Text.Json.JsonDocument.Parse("""{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}""").RootElement },
        } : null;
        var userMsg = LammaNet.Chat.ChatMessage.User(question);
        if (image != null) userMsg.Parts.Insert(0, new LammaNet.Chat.ImagePart { EncodedBytes = File.ReadAllBytes(image) });
        if (audio != null) userMsg.Parts.Insert(0, new LammaNet.Chat.AudioPart { EncodedBytes = File.ReadAllBytes(audio) });
        var messages = new List<LammaNet.Chat.ChatMessage> { userMsg };
        var swEnc = Stopwatch.StartNew();
        var opts = new LammaNet.Chat.ChatRequestOptions { Tools = toolDefs, MaxTokens = 512, EnableThinking = think, Sampling = LammaNet.Sampling.SamplingOptions.Greedy, RenderSpecialTokens = special ? true : null };
        await foreach (var ev in chat.ChatAsync(messages, opts))
        {
            if (ev.Output is { } o)
            {
                switch (o.Kind)
                {
                    case LammaNet.Chat.ChatOutputKind.Text: Console.Write(o.Text); break;
                    case LammaNet.Chat.ChatOutputKind.Reasoning: Console.ForegroundColor = ConsoleColor.DarkGray; Console.Write(o.Text); Console.ResetColor(); break;
                    case LammaNet.Chat.ChatOutputKind.ToolCall: Console.WriteLine($"\n[tool call] {o.Call!.Name}({o.Call.ArgumentsJson})"); break;
                }
            }
            if (ev.Stats != null && engine.TotalAcceptedDraftTokens > 0)
                Console.Error.WriteLine($"\n-- speculative: {engine.TotalAcceptedDraftTokens} draft tokens accepted");
            if (ev.Stats is { } st)
                Console.Error.WriteLine($"\n-- {st.FinishReason}; prompt {st.PromptTokens} tok ({st.PromptTokensPerSecond:F0} tok/s), gen {st.CompletionTokens} tok ({st.TokensPerSecond:F1} tok/s)");
        }
        Profiler.Report(Console.Error);
        return 0;
    }

    /// <summary>Engine offload factory from LAMMA_GPU (see <see cref="GpuFromEnv"/>), or null.</summary>
    private static Func<Model, OffloadGeometry, ILayerOffload?>? GpuFactoryFromEnv()
    {
        var spec = Environment.GetEnvironmentVariable("LAMMA_GPU");
        if (string.IsNullOrEmpty(spec)) return null;
        return (model, geometry) => GpuFromEnv(model, geometry, spec);
    }

    /// <summary>LAMMA_GPU=&lt;layers&gt;[:cpu] → offload the first layers via ILGPU (-1 = all supported; ":cpu" = kernel emulation).</summary>
    private static ILayerOffload? GpuFromEnv(Model model, OffloadGeometry geometry, string? spec = null)
    {
        spec ??= Environment.GetEnvironmentVariable("LAMMA_GPU");
        if (string.IsNullOrEmpty(spec)) return null;
        if (LammaNet.Gpu.GpuLayerOffload.SupportedLayerPrefix(model.Config) == 0)
        {
            Console.Error.WriteLine($"Running {model.Config.Architecture} on CPU.");
            return null;
        }
        var parts = spec.Split(':');
        var offload = new LammaNet.Gpu.GpuLayerOffload(model, geometry, new LammaNet.Gpu.GpuOffloadOptions
        {
            Layers = int.Parse(parts[0]),
            ReserveBytes = (long.TryParse(Environment.GetEnvironmentVariable("LAMMA_GPU_RESERVE_MB"), out var rmb) ? rmb : 384) << 20,
            Device = parts.Length < 2 ? LammaNet.Gpu.GpuDevice.Auto
                : parts[1] == "cpu" ? LammaNet.Gpu.GpuDevice.CpuEmulation : LammaNet.Gpu.GpuDevice.Cuda,
        });
        Console.Error.WriteLine($"offloading {offload.LayerCount}/{model.Config.NLayer} layers to {offload.DeviceName} [{offload.Plan}]");
        if (offload.LayerCount > 0) return offload;
        offload.Dispose();
        return null;
    }

    /// <summary>GPU-vs-CPU logit check (see <see cref="LammaNet.Gpu.GpuValidation"/>); layers -1 = all supported.</summary>
    private static int GpuCheck(string modelPath, int layers, int nGen, KvCacheType kvType)
    {
        using var model = Model.Load(modelPath);
        var r = LammaNet.Gpu.GpuValidation.Compare(model, new LammaNet.Gpu.GpuOffloadOptions { Layers = layers }, nGen,
            Environment.GetEnvironmentVariable("LAMMA_PROMPT") ?? LammaNet.Gpu.GpuValidation.DefaultPrompt,
            int.TryParse(Environment.GetEnvironmentVariable("LAMMA_THREADS"), out var nt) ? nt : 0, kvType);
        Console.WriteLine($"CPU: {r.CpuText.ReplaceLineEndings(" ")}");
        Console.WriteLine($"cpu-dec  {r.CpuDecode}  (CPU decode, Q8 activations)");
        Console.WriteLine($"offloading {r.OffloadedLayers}/{model.Config.NLayer} layers to {r.Device}");
        Console.WriteLine($"decode   {r.GpuDecode}");
        Console.WriteLine($"prefill  {r.GpuPrefill}");
        Console.WriteLine(r.Passed ? "PASS" : "FAIL");
        return r.Passed ? 0 : 3;
    }

    /// <summary>GGUF metadata without loading the model: <c>info model.gguf [--tensors] [--keys] [--vocab N]</c>.</summary>
    private static int Info(string modelPath, bool tensors, bool keys, int vocab)
    {
        var info = ModelInfo.Read(modelPath, new ModelInfoOptions { LoadVocabulary = vocab > 0 || keys });
        Console.Write(info);
        if (keys)
            foreach (var (k, v) in info.Metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                string s = v switch
                {
                    string str => str.Length > 120 ? $"\"{str[..120].ReplaceLineEndings("\\n")}...\" ({str.Length} chars)" : $"\"{str.ReplaceLineEndings("\\n")}\"",
                    Array a => $"[{string.Join(", ", a.Cast<object>().Take(8))}{(a.Length > 8 ? $", ... ({a.Length})" : "")}]",
                    _ => Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture) ?? "",
                };
                Console.WriteLine($"  {k} = {s}");
            }
        if (tensors) foreach (var t in info.Tensors) Console.WriteLine("  " + t);
        if (vocab > 0 && info.Tokenizer?.Tokens is { } toks)
            for (int i = 0; i < Math.Min(vocab, toks.Count); i++)
                Console.WriteLine($"  {i,7} {info.Tokenizer.TokenTypes?[i].ToString() ?? "",-11} {toks[i]}");
        return 0;
    }
    private static int BenchGpu(string modelPath, string tensors, int nTok, int repeat)
    {
        foreach (var t in tensors.Split(','))
            Console.WriteLine(LammaNet.Gpu.GpuBench.MatVec(modelPath, t, nTok, repeat));
        return 0;
    }

    private static int PrintLines(IEnumerable<string> lines)
    {
        foreach (var l in lines) Console.WriteLine(l);
        return 0;
    }

    /// <summary>
    /// Teacher-forced next-token probabilities for comparing with a reference implementation: <paramref name="stepsJson"/>
    /// is [{"next": id, "cand": [ids]}, ...]; the prompt is evaluated, then each step prints the probabilities of the
    /// candidate ids and feeds "next". Output: JSON [[p(cand0), p(cand1), ...], ...].
    /// </summary>
    /// <summary>
    /// Teacher-forced probabilities of given candidate tokens (steps JSON: [{next, cand[]}]). With --mmproj/--image the
    /// prompt is a user message rendered with the chat template (generation prompt added) and the image encoded.
    /// </summary>
    private static int Probs(string modelPath, string prompt, string stepsJson, string? mmproj = null, string? image = null)
    {
        using var model = Model.Load(modelPath);
        var pool = new ComputePool(ComputePool.DefaultThreadCount());
        using var offload = GpuFromEnv(model, new OffloadGeometry(32, 4096 / 32, 512));
        using var kv = new KvCache(model.Config, 4096, KvCacheType.F16, 32, offload == null ? null : l => !offload.OwnsKv(l), offload?.RecurrentStore);
        using var runner = new ModelRunner(model, kv, pool, 512, 0, offload);
        using var seq = new SequenceState(kv, model.Config);
        using var doc = System.Text.Json.JsonDocument.Parse(File.Exists(stepsJson) ? File.ReadAllText(stepsJson) : stepsJson);
        var batch = new Batch();
        if (mmproj == null)
        {
            var toks = model.Vocab.Tokenize(prompt);
            for (int i = 0; i < toks.Count; i++) batch.Add(seq, toks[i], i, i == toks.Count - 1);
        }
        else
        {
            var chat = new LammaNet.Chat.ChatEngine(new InferenceEngine(model, new EngineOptions { ContextSize = 64, AutoAccelerator = false }));
            var enc = LammaNet.Multimodal.MultimodalProjector.Load(mmproj, model, new LammaNet.Multimodal.ProjectorOptions { Accelerator = offload as LammaNet.Multimodal.IVisionAccelerator });
            if (enc is LammaNet.Multimodal.Idefics3Projector ip) Console.Error.WriteLine($"vision encoder on {ip.EncoderDevice}");
            chat.MediaEncoders.Add(enc);
            var msg = new LammaNet.Chat.ChatMessage { Role = LammaNet.Chat.ChatRole.User, Parts = [new LammaNet.Chat.TextPart(prompt)] };
            if (image != null) msg.Parts.Insert(0, new LammaNet.Chat.ImagePart { EncodedBytes = File.ReadAllBytes(image) });
            var sw = Stopwatch.StartNew();
            var (p, _) = chat.BuildPrompt([msg]);
            Console.Error.WriteLine($"prompt {p.Length} positions, built in {sw.ElapsedMilliseconds} ms");
            Profiler.Report(Console.Error);
            int pos = 0;
            foreach (var seg in p.Segments)
            {
                if (seg is TokenSegment ts) foreach (var t in ts.Tokens) batch.Add(seq, t, pos++, false);
                else if (seg is EmbeddingSegment es) for (int r = 0; r < es.Rows; r++) batch.Add(seq, 0, pos++, false, es.Row(r));
            }
        }
        // feed in chunks of the runner's batch size; logits for the last prompt position only
        var all = batch;
        int total = all.Count;
        for (int c0 = 0; c0 < total; c0 += 512)
        {
            var part = new Batch();
            for (int i = c0; i < Math.Min(total, c0 + 512); i++) part.Add(seq, all.Tokens[i], all.Positions[i], i == total - 1, all.Embeddings[i]);
            runner.Forward(part);
        }
        batch = new Batch();
        var result = new List<float[]>();
        foreach (var step in doc.RootElement.EnumerateArray())
        {
            var lg = runner.Logits(0).ToArray();
            float mx = lg.Max();
            double sum = 0;
            foreach (var v in lg) sum += Math.Exp(v - mx);
            result.Add(step.GetProperty("cand").EnumerateArray().Select(c => (float)(Math.Exp(lg[c.GetInt32()] - mx) / sum)).ToArray());
            batch.Clear();
            batch.Add(seq, step.GetProperty("next").GetInt32(), seq.Length, true);
            runner.Forward(batch);
        }
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(result));
        return 0;
    }

    private static int Unknown(string cmd) { Console.Error.WriteLine($"unknown command {cmd}"); return 1; }

    private static int Tokenize(string modelPath, string text)
    {
        using var model = Model.Load(modelPath, new ModelLoadOptions { Prefetch = false });
        var toks = model.Vocab.Tokenize(text);
        Console.WriteLine("TOKENS " + string.Join(" ", toks));
        return 0;
    }

    private static unsafe int Logits(string modelPath, string prompt, int nGen, bool all)
    {
        var sw = Stopwatch.StartNew();
        using var model = Model.Load(modelPath);
        Console.Error.WriteLine($"loaded {model} in {sw.ElapsedMilliseconds} ms");
        var pool = new ComputePool(int.TryParse(Environment.GetEnvironmentVariable("LAMMA_THREADS"), out var nt) ? nt : ComputePool.DefaultThreadCount());
        int pageSize = 32, pages = 4096 / 32;
        var offload = GpuFromEnv(model, new OffloadGeometry(pageSize, pages, 512));
        var off = offload;
        using var kv = new KvCache(model.Config, 4096, KvCacheType.F16, pageSize, off == null ? null : l => !off.OwnsKv(l), off?.RecurrentStore);
        using var runner = new ModelRunner(model, kv, pool, 512, 0, offload);
        var seq = new SequenceState(kv, model.Config);
        var toks = model.Vocab.Tokenize(prompt);
        Console.WriteLine("TOKENS " + string.Join(" ", toks));
        var batch = new Batch();
        for (int i = 0; i < toks.Count; i++) batch.Add(seq, toks[i], i, all || i == toks.Count - 1);
        sw.Restart();
        runner.Forward(batch);
        Console.Error.WriteLine($"prefill {toks.Count} tokens: {sw.ElapsedMilliseconds} ms");
        int outIdx = 0;
        int last = 0;
        for (int i = 0; i < toks.Count; i++)
            if (all || i == toks.Count - 1) last = Dump(runner, outIdx++, i);
        sw.Restart();
        var dec = model.Vocab.CreateStreamingDecoder();
        for (int g = 0; g < nGen; g++)
        {
            batch.Clear();
            batch.Add(seq, last, seq.Length, true);
            runner.Forward(batch);
            last = all ? Dump(runner, 0, seq.Length - 1) : System.Numerics.Tensors.TensorPrimitives.IndexOfMax(runner.Logits(0));
            Console.Error.Write(dec.Push(last));
        }
        Console.Error.WriteLine();
        Profiler.Report(Console.Error);
        if (nGen > 0) Console.Error.WriteLine($"decode: {sw.ElapsedMilliseconds / (double)nGen:F1} ms/token ({nGen * 1000.0 / sw.ElapsedMilliseconds:F1} tok/s)");
        return 0;
    }

    private static int Dump(ModelRunner r, int row, int pos)
    {
        var lg = r.Logits(row);
        var idx = Enumerable.Range(0, lg.Length).ToArray();
        var arr = lg.ToArray();
        Array.Sort(arr.Select(v => -v).ToArray(), idx);
        Console.WriteLine($"POS {pos} " + string.Join(" ", idx.Take(8).Select(i => $"{i}:{arr[i]:F4}")));
        return idx[0];
    }
}
