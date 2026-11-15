using System.Diagnostics;
using LammaNet.Chat;
using LammaNet.Gpu;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Sampling;

namespace LammaNet.Cli;

/// <summary>
/// End-to-end check of the GPU offload through the real <see cref="InferenceEngine"/>: a multi-turn greedy chat
/// (prefix reuse → partial KV page copies on the device), speculative decoding on/off, and two concurrent requests
/// sharing a batch; outputs are compared with the CPU engine and between configurations.
/// </summary>
internal static class EngineCheck
{
    private static readonly string[] Turns =
    [
        "Write a haiku about graphics cards.",
        "Explain the haiku in two short sentences.",
        "Now list three facts about the number seven.",
    ];

    public static async Task<int> Run(string modelPath, int maxTokens)
    {
        using var model = Model.Load(modelPath);
        var cpu = await Chat(model, new EngineOptions { ContextSize = 2048, SpeculativeDraftTokens = 3, AutoAccelerator = false }, maxTokens);
        Report("cpu  spec=3", cpu);
        // pass/fail with float activations: int8 blocks amplify the last-bit differences between prompt and decode
        // kernels (attention chunking, fp16 GEMM) into probability shifts of a few percent, which flips near ties early
        // and would hide the bugs this check is for (KV page reuse, snapshots, batching)
        bool ok = await Compare(model, maxTokens, Int8Activations.Off, cpu, strict: true);
        // the default int8 activations: agreement reported, not enforced (accuracy is checked against llama.cpp's
        // probabilities by the tests; the kernels against an int8-rounded reference by bench-gpu)
        Console.WriteLine("-- int8 activations (default), report only");
        await Compare(model, maxTokens, new GpuOffloadOptions().Int8, cpu, strict: false);
        Console.WriteLine(ok ? "PASS" : "FAIL");
        return ok ? 0 : 3;
    }

    private static async Task<bool> Compare(Model model, int maxTokens, Int8Activations int8, (List<string> Replies, double Tps, int Cached) cpu, bool strict)
    {
        var gpu0 = await Chat(model, Gpu(0, int8), maxTokens);
        Report("gpu  spec=0", gpu0);
        var gpu3 = await Chat(model, Gpu(3, int8), maxTokens);
        Report("gpu  spec=3", gpu3);
        // the same conversation without prefix reuse: resuming from cached KV pages / recurrent snapshots must not
        // change the result
        var gpuNc = await Chat(model, Gpu(0, int8), maxTokens, cachePrompt: false);
        Report("gpu  nocache", gpuNc);

        bool ok = true;
        for (int i = 0; i < Turns.Length; i++)
        {
            int same = Common(gpu0.Replies[i], gpu3.Replies[i]), vsCpu = Common(gpu0.Replies[i], cpu.Replies[i]);
            int nc = Common(gpu0.Replies[i], gpuNc.Replies[i]);
            Console.WriteLine($"turn {i}: gpu spec0 vs spec3 common {same}/{gpu0.Replies[i].Length} chars, vs no-cache {nc}/{gpuNc.Replies[i].Length}, gpu vs cpu {vsCpu}/{cpu.Replies[i].Length}");
            // greedy GPU decoding must not depend on batching (n=1 vs n=4 verification) or prefix reuse beyond near-tie flips late in the text
            ok &= same >= Math.Min(gpu0.Replies[i].Length, 40) && nc >= Math.Min(gpu0.Replies[i].Length, 40);
            if (nc < Math.Min(gpu0.Replies[i].Length, 40))
                Console.WriteLine($"  cached:   {gpu0.Replies[i].ReplaceLineEndings(" ")}{Environment.NewLine}  no-cache: {gpuNc.Replies[i].ReplaceLineEndings(" ")}");
        }
        ok &= gpu0.Cached > 0 || cpu.Cached == 0; // some chat templates re-render earlier turns (no reusable prefix)

        // two concurrent requests share every decode batch
        using (var engine = new InferenceEngine(model, Gpu(0, int8)))
        {
            var chat = new ChatEngine(engine);
            var reqs = new[] { 0, 2 }.Select(i => new GenerationRequest
            {
                Prompt = chat.BuildPrompt([ChatMessage.User(Turns[i])], new ChatRequestOptions { EnableThinking = false }).Prompt,
                Sampling = SamplingOptions.Greedy, MaxTokens = maxTokens, CachePrompt = false,
            }).ToArray();
            var tasks = reqs.Select(r => engine.GenerateTextAsync(r)).ToArray();
            var results = await Task.WhenAll(tasks);
            for (int k = 0; k < 2; k++)
            {
                string solo = k == 0 ? gpu0.Replies[0] : (await engine.GenerateTextAsync(reqs[1])).Text;
                int c = Common(results[k].Text, solo);
                Console.WriteLine($"concurrent request {k}: common with solo run {c}/{solo.Length} chars");
                ok &= c >= Math.Min(solo.Length, 40);
            }
        }
        if (!strict && !ok) Console.WriteLine("(int8 run differs beyond the float threshold; not a failure)");
        return ok;
    }

    private static EngineOptions Gpu(int spec, Int8Activations int8) => new()
    {
        ContextSize = 2048, SpeculativeDraftTokens = spec, Offload = GpuLayerOffload.Factory(new GpuOffloadOptions { Int8 = int8 }),
    };

    private static void Report(string name, (List<string> Replies, double Tps, int Cached) r) =>
        Console.WriteLine($"{name}: {r.Tps,6:F1} tok/s, cached prompt tokens {r.Cached}, turn0: {r.Replies[0].ReplaceLineEndings(" ")[..Math.Min(80, r.Replies[0].Length)]}");

    private static async Task<(List<string> Replies, double Tps, int Cached)> Chat(Model model, EngineOptions opt, int maxTokens, bool cachePrompt = true)
    {
        using var engine = new InferenceEngine(model, opt);
        var chat = new ChatEngine(engine);
        var msgs = new List<ChatMessage>();
        var replies = new List<string>();
        int tokens = 0, cached = 0;
        double secs = 0;
        foreach (var turn in Turns)
        {
            msgs.Add(ChatMessage.User(turn));
            var (prompt, _) = chat.BuildPrompt(msgs, new ChatRequestOptions { EnableThinking = false });
            var (text, stats) = await engine.GenerateTextAsync(new GenerationRequest { Prompt = prompt, Sampling = SamplingOptions.Greedy, MaxTokens = maxTokens, CachePrompt = cachePrompt });
            replies.Add(text);
            msgs.Add(ChatMessage.Assistant(text));
            tokens += stats.CompletionTokens;
            secs += stats.GenerationTime.TotalSeconds;
            cached += stats.CachedPromptTokens;
        }
        return (replies, tokens / Math.Max(secs, 1e-9), cached);
    }

    private static int Common(string a, string b)
    {
        int n = 0;
        while (n < a.Length && n < b.Length && a[n] == b[n]) n++;
        return n;
    }
}
