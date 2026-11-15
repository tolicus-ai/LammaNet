using System.Collections.Concurrent;
using LammaNet.Chat;
using LammaNet.Extensions.AI;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Multimodal;
using Microsoft.Extensions.AI;

namespace LammaNet.Tests;

/// <summary>Several callers on shared objects at once (agents sharing a client, realtime sessions, several engines).</summary>
public class ConcurrencyTests
{
    private static byte[] TestImage(string name) => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", name));

    private static float[] Flatten(IReadOnlyList<PromptSegment> segments) =>
        segments.OfType<EmbeddingSegment>().SelectMany(s => Enumerable.Range(0, s.Rows).SelectMany(r => s.Row(r))).ToArray();

    /// <summary>
    /// Concurrent Encode calls on one projector (a realtime session transcribing while it answers, agents sharing a client)
    /// used to share the encoder's GEMM job and scratch buffers: wrong embeddings or an access violation.
    /// </summary>
    [Theory]
    [InlineData("Qwen3.5-0.8B-Q4_K_M.gguf", "mmproj-Qwen3.5-0.8B-F16.gguf")]
    [InlineData("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "mmproj-gemma-4-E2B-it-qat-F16.gguf")]
    public void ConcurrentEncodesOnOneProjectorMatchSequentialEncodes(string modelFile, string mmprojFile)
    {
        var path = TestModels.Find(modelFile);
        var mmproj = TestModels.Find(mmprojFile);
        if (path == null || mmproj == null) return;
        using var model = Model.Load(path);
        var encoder = MultimodalProjector.Load(mmproj, model, new ProjectorOptions { Threads = 4, MaxImageTokens = 64 });
        try
        {
            ImagePart[] images = [new() { EncodedBytes = TestImage("nyt-moon.jpeg") }, new() { EncodedBytes = TestImage("docpage.png") }];
            var expected = images.Select(i => Flatten(encoder.Encode(i))).ToArray();
            var results = new float[8][];
            Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = results.Length },
                i => results[i] = Flatten(encoder.Encode(images[i % 2])));
            for (int i = 0; i < results.Length; i++) Assert.Equal(expected[i % 2], results[i]);
        }
        finally { (encoder as IDisposable)?.Dispose(); }
    }

    /// <summary>Agents on different models at once: two engines, several concurrent requests each (CPU, and GPU when present).</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EnginesOnDifferentModelsServeConcurrentRequests(bool gpu)
    {
        var a = TestModels.Find(TestModels.Lfm2Q4);
        var b = TestModels.Find(TestModels.Qwen3Q4);
        if (a == null || b == null) return;
        EngineOptions Options() => new() { ContextSize = 512, Threads = 4, AutoAccelerator = gpu };
        using var e1 = InferenceEngine.Load(a, Options());
        using var e2 = InferenceEngine.Load(b, Options());
        if (gpu && (e1.Offload == null || e2.Offload == null)) return;   // no device
        string[] prompts = ["The capital of France is", "Write a haiku about rain:", "1, 2, 3, 4,", "Explain recursion briefly:"];
        var work = new List<Task<(string Text, GenerationStats Stats)>>();
        for (int round = 0; round < 2; round++)
            foreach (var engine in new[] { e1, e2 })
                foreach (var p in prompts)
                    work.Add(Task.Run(() => engine.GenerateTextAsync(new GenerationRequest
                    {
                        Prompt = new Prompt(engine.Tokenize(p)), MaxTokens = 24, Sampling = Sampling.SamplingOptions.Greedy,
                    })));
        var results = await Task.WhenAll(work);
        Assert.All(results, r => Assert.True(r.Stats.CompletionTokens > 0 && r.Stats.FinishReason is FinishReason.Stop or FinishReason.Length,
            $"{r.Stats.FinishReason}, {r.Stats.CompletionTokens} tokens"));
    }

    /// <summary>Dispose while requests run or wait: every request ends (cancelled), nothing touches freed memory.</summary>
    [Fact]
    public async Task DisposeDuringGenerationCancelsRunningAndQueuedRequests()
    {
        var path = TestModels.Find(TestModels.Qwen3Q4);
        if (path == null) return;
        var engine = InferenceEngine.Load(path, new EngineOptions { ContextSize = 2048, Threads = 4, AutoAccelerator = false, MaxConcurrentSequences = 1 });
        var prompt = new Prompt(engine.Tokenize(string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 60))));
        var handles = Enumerable.Range(0, 3).Select(_ => engine.Submit(new GenerationRequest { Prompt = prompt, MaxTokens = 500, IgnoreEos = true })).ToList();
        await Task.Delay(1500);   // the first request is in prefill or decoding, the others wait for a slot
        engine.Dispose();
        var finished = Task.WhenAll(handles.Select(h => h.Completion));
        Assert.Same(finished, await Task.WhenAny(finished, Task.Delay(TimeSpan.FromSeconds(60))));
        Assert.All(await finished, s => Assert.Equal(FinishReason.Cancelled, s.FinishReason));
        Assert.Throws<ObjectDisposedException>(() => engine.Submit(new GenerationRequest { Prompt = prompt }));
    }

    /// <summary>
    /// Conversations with images through one client at once (shared projector, template renderer and engine); on the GPU
    /// the vision encoder shares the engine's device context while the engine decodes other requests.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConcurrentImageConversationsOnOneClient(bool gpu)
    {
        var path = TestModels.Find(TestModels.Qwen35);
        var mmproj = TestModels.Find("mmproj-Qwen3.5-0.8B-F16.gguf");
        if (path == null || mmproj == null) return;
        using var client = LammaChatClient.Load(path, new EngineOptions { ContextSize = 1024, Threads = 4, AutoAccelerator = gpu },
            new LammaChatClientOptions { MmprojPath = mmproj, ProjectorOptions = new ProjectorOptions { MaxImageTokens = 64, Threads = 4 }, EnableThinking = false });
        if (gpu && client.Engine.Offload == null) return;   // no device
        var image = TestImage("nyt-moon.jpeg");
        var replies = await Task.WhenAll(Enumerable.Range(0, gpu ? 12 : 4).Select(i => Task.Run(() => client.GetResponseAsync(
            [new Microsoft.Extensions.AI.ChatMessage(Microsoft.Extensions.AI.ChatRole.User, new List<AIContent> { new DataContent(image, "image/jpeg"), new TextContent($"Question {i}: what is in this picture? One sentence.") })],
            new ChatOptions { MaxOutputTokens = 16, Temperature = 0 }))));
        Assert.All(replies, r => Assert.False(string.IsNullOrWhiteSpace(r.Text)));
    }

    /// <summary>
    /// The realtime scenario that crashed (access violation in the CPU GEMM): voice turns back to back with transcription,
    /// so the next turn's transcription encodes audio while the previous turn's response encodes the conversation.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RealtimeTurnsWithTranscriptionOverlapSafely(bool gpu)
    {
        var path = TestModels.Find("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
        var mmproj = TestModels.Find("mmproj-gemma-4-E2B-it-qat-F16.gguf");
        if (path == null || mmproj == null) return;
        using var client = LammaChatClient.Load(path, new EngineOptions { ContextSize = 2048, Threads = 4, AutoAccelerator = gpu },
            new LammaChatClientOptions { MmprojPath = mmproj, ProjectorOptions = new ProjectorOptions { Threads = 4 } });
        if (gpu && client.Engine.Offload == null) return;   // no device
        IRealtimeClient realtime = new LammaRealtimeClient(client);
        await using var session = await realtime.CreateSessionAsync(new RealtimeSessionOptions
        {
            InputAudioFormat = new RealtimeAudioFormat("audio/pcm", 16000),
            VoiceActivityDetection = new VoiceActivityDetectionOptions { Enabled = true, AllowInterruption = false },
            TranscriptionOptions = new TranscriptionOptions(),
            MaxOutputTokens = 8,
        });
        const int turns = 3;
        var messages = new ConcurrentQueue<RealtimeServerMessage>();
        var allDone = new TaskCompletionSource();
        int done = 0;
        var reader = Task.Run(async () =>
        {
            await foreach (var m in session.GetStreamingResponseAsync())
            {
                messages.Enqueue(m);
                if (m is ResponseCreatedRealtimeServerMessage { Type: var t } && t == RealtimeServerMessageType.ResponseDone && Interlocked.Increment(ref done) == turns)
                    allDone.TrySetResult();
            }
        });
        // 2 s of signal + 1 s of silence per turn, as 100 ms PCM16 chunks, sent without pacing
        var turn = VisionEncoderGpuTests.SyntheticAudio(2).Concat(new float[16000]).ToArray();
        for (int t = 0; t < turns; t++)
            for (int off = 0; off < turn.Length; off += 1600)
            {
                var bytes = new byte[1600 * 2];
                for (int i = 0; i < 1600; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), (short)Math.Clamp(turn[off + i] * 32767f, -32768f, 32767f));
                await session.SendAsync(new InputAudioBufferAppendRealtimeClientMessage(new DataContent(bytes, "audio/pcm")));
            }
        Assert.Same(allDone.Task, await Task.WhenAny(allDone.Task, Task.Delay(TimeSpan.FromMinutes(5))));
        Assert.DoesNotContain(messages, m => m is ErrorRealtimeServerMessage);
        Assert.Equal(turns, messages.Count(m => m.Type == RealtimeServerMessageType.InputAudioTranscriptionCompleted));
    }
}
