using System.ComponentModel;
using LammaNet.Extensions.AI;
using LammaNet.Inference;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace LammaNet.Cli;

/// <summary>End-to-end demos of the Microsoft.Extensions.AI / Agent Framework integration.</summary>
internal static class Demos
{
    [Description("Gets the current weather for a city.")]
    private static string GetWeather([Description("The city name")] string city) =>
        city.ToLowerInvariant() switch
        {
            "paris" => "18°C, light rain",
            "london" => "14°C, cloudy",
            "tokyo" => "24°C, sunny",
            _ => "22°C, clear sky",
        };

    [Description("Returns the current local date and time.")]
    private static string GetTime() => DateTime.Now.ToString("yyyy-MM-dd HH:mm");

    public static async Task<int> Agent(string modelPath)
    {
        using var client = LammaChatClient.Load(modelPath, new EngineOptions { ContextSize = 8192 });
        IChatClient chat = new ChatClientBuilder(client).UseFunctionInvocation().Build();
        AIAgent agent = chat.AsAIAgent(
            instructions: "You are a concise assistant. Use the tools when they help answer the question.",
            name: "LocalAgent",
            tools: [AIFunctionFactory.Create(GetWeather), AIFunctionFactory.Create(GetTime)]);
        var session = await agent.CreateSessionAsync();
        foreach (var q in new[] { "What's the weather in Paris and in Tokyo?", "Which of the two cities is warmer?" })
        {
            Console.WriteLine($"> {q}");
            await foreach (var update in agent.RunStreamingAsync(q, session))
                Console.Write(update.Text);
            Console.WriteLine();
        }
        Console.Error.WriteLine($"[engine] prompt tokens {client.Engine.TotalPromptTokens} (cached {client.Engine.TotalCachedPromptTokens}), generated {client.Engine.TotalGeneratedTokens}");
        return 0;
    }

    public sealed record CityInfo(string Name, string Country, int Population, string[] Landmarks);

    public static async Task<int> Structured(string modelPath)
    {
        using var client = LammaChatClient.Load(modelPath, new EngineOptions { ContextSize = 4096 });
        var resp = await client.GetResponseAsync<CityInfo>("Give me information about Paris: name, country, approximate population and three landmarks.",
            new ChatOptions { Temperature = 0 });
        Console.WriteLine(resp.Text);
        if (resp.TryGetResult(out var city))
            Console.WriteLine($"Parsed: {city.Name}, {city.Country}, pop {city.Population}, landmarks: {string.Join(", ", city.Landmarks)}");
        else Console.WriteLine("Could not parse result");
        return 0;
    }

    public static async Task<int> Vision(string modelPath, string mmproj, string imagePath)
    {
        using var client = LammaChatClient.Load(modelPath, new EngineOptions { ContextSize = 4096 }, new LammaChatClientOptions { MmprojPath = mmproj });
        var msg = new ChatMessage(ChatRole.User, [new DataContent(File.ReadAllBytes(imagePath), "image/jpeg"), new TextContent("What is shown in this image? Answer briefly.")]);
        await foreach (var u in client.GetStreamingResponseAsync([msg], new ChatOptions { Temperature = 0 })) Console.Write(u.Text);
        Console.WriteLine();
        return 0;
    }

    public static async Task<int> SpeechToText(string modelPath, string mmproj, string wavPath)
    {
        using var client = LammaChatClient.Load(modelPath, new EngineOptions { ContextSize = 4096 }, new LammaChatClientOptions { MmprojPath = mmproj });
        ISpeechToTextClient stt = new LammaSpeechToTextClient(client);
        await using var fs = File.OpenRead(wavPath);
        var resp = await stt.GetTextAsync(fs);
        Console.WriteLine(resp.Text);
        return 0;
    }

    public static async Task<int> Realtime(string modelPath, string mmproj, string wavPath)
    {
        using var client = LammaChatClient.Load(modelPath, new EngineOptions { ContextSize = 4096 }, new LammaChatClientOptions { MmprojPath = mmproj });
        IRealtimeClient realtime = new LammaRealtimeClient(client);
        await using var session = await realtime.CreateSessionAsync(new RealtimeSessionOptions
        {
            Instructions = "You are a helpful voice assistant. Answer in one or two short sentences.",
            InputAudioFormat = new RealtimeAudioFormat("audio/pcm", 16000),
            VoiceActivityDetection = new VoiceActivityDetectionOptions { Enabled = true, AllowInterruption = true },
            TranscriptionOptions = new TranscriptionOptions(),
        });
        var reader = Task.Run(async () =>
        {
            await foreach (var m in session.GetStreamingResponseAsync())
            {
                switch (m)
                {
                    case OutputTextAudioRealtimeServerMessage t when t.Type == RealtimeServerMessageType.OutputTextDelta: Console.Write(t.Text); break;
                    case InputAudioTranscriptionRealtimeServerMessage tr: Console.WriteLine($"[user said] {tr.Transcription}"); break;
                    case ResponseCreatedRealtimeServerMessage rc when rc.Type == RealtimeServerMessageType.ResponseDone:
                        Console.WriteLine($"\n[response {rc.Status}; {rc.Usage?.InputTokenCount} in / {rc.Usage?.OutputTokenCount} out]");
                        return;
                    case ErrorRealtimeServerMessage e: Console.WriteLine($"[error] {e.Error?.Message}"); break;
                    default: Console.WriteLine($"[{m.Type}]"); break;
                }
            }
        });
        // stream the clip as 100 ms PCM16 chunks (as a microphone would), followed by 1 s of silence
        var (pcm, rate) = LammaNet.Multimodal.AudioProcessing.DecodeWav(File.ReadAllBytes(wavPath));
        pcm = LammaNet.Multimodal.AudioProcessing.Resample(pcm, rate, 16000);
        var all = pcm.Concat(new float[16000]).ToArray();
        for (int off = 0; off < all.Length; off += 1600)
        {
            var chunk = all.AsSpan(off, Math.Min(1600, all.Length - off));
            var bytes = new byte[chunk.Length * 2];
            for (int i = 0; i < chunk.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), (short)Math.Clamp(chunk[i] * 32767f, -32768f, 32767f));
            await session.SendAsync(new InputAudioBufferAppendRealtimeClientMessage(new DataContent(bytes, "audio/pcm")));
        }
        await reader;
        return 0;
    }

    /// <summary>Prints the embeddings of the given texts as a JSON array (one vector per text).</summary>
    public static async Task<int> EmbedJson(string modelPath, string[] texts)
    {
        using var engine = InferenceEngine.Load(modelPath, new EngineOptions { ContextSize = 4096, BatchSize = 2048 });
        Console.Error.WriteLine(engine.Offload is { } dev ? $"-- {dev.DeviceName}" : "-- CPU");
        var vecs = new List<float[]>();
        foreach (var t in texts) vecs.Add(await engine.EmbedAsync(new LammaNet.Inference.Prompt(engine.Tokenize(t))));
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(vecs));
        return 0;
    }

    public static async Task<int> Embeddings(string modelPath)
    {
        using var engine = InferenceEngine.Load(modelPath, new EngineOptions { ContextSize = 2048 });
        using var gen = new LammaEmbeddingGenerator(engine);
        string[] docs = ["The cat sat on the mat.", "A feline rested on the rug.", "Stock markets fell sharply today.", "Equities dropped in heavy trading."];
        var emb = await gen.GenerateAsync(docs);
        for (int i = 0; i < docs.Length; i++)
        {
            var sims = Enumerable.Range(0, docs.Length).Select(j => System.Numerics.Tensors.TensorPrimitives.CosineSimilarity(emb[i].Vector.Span, emb[j].Vector.Span));
            Console.WriteLine($"{docs[i],-40} " + string.Join("  ", sims.Select(s => s.ToString("F3"))));
        }
        return 0;
    }
}
