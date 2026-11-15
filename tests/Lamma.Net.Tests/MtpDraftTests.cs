using LammaNet.Chat;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Sampling;

namespace LammaNet.Tests;

/// <summary>Speculative decoding with the Gemma 4 MTP drafter (gemma4-assistant) on the CPU and the GPU.</summary>
public class MtpDraftTests
{
    private const string Drafter = "mtp-gemma-4-E2B-it.gguf";
    private const string Target = "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";

    [Fact]
    public void DrafterLoadsAndChecksItsTarget()
    {
        var dp = TestModels.Find(Drafter);
        var tp = TestModels.Find(Target);
        if (dp == null || tp == null) return;
        using var drafter = MtpDrafterModel.Load(dp);
        Assert.Equal(4, drafter.Layers.Length);
        Assert.Equal(256, drafter.NEmbd);
        Assert.Equal(1536, drafter.NEmbdTarget);
        Assert.Equal([true, true, true, false], drafter.Layers.Select(l => l.IsSwa));
        using var target = Model.Load(tp, new ModelLoadOptions { Prefetch = false });
        drafter.Validate(target);
        if (TestModels.Find(TestModels.Qwen3Q4) is { } other)
        {
            using var wrong = Model.Load(other, new ModelLoadOptions { Prefetch = false });
            Assert.Throws<NotSupportedException>(() => drafter.Validate(wrong));
        }
    }

    /// <summary>Drafts are proposed and accepted, and the greedy text equals generation without drafting.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task DraftedGenerationMatchesPlainGeneration(bool gpu)
    {
        var dp = TestModels.Find(Drafter);
        var tp = TestModels.Find(Target);
        if (dp == null || tp == null) return;
        string[] prompts = ["List the planets of the solar system.", "Explain what a hash table is in two sentences."];
        async Task<(string[] Texts, long Drafted, long Accepted, bool OnDevice)> Run(string? drafter)
        {
            using var engine = InferenceEngine.Load(tp, new EngineOptions
            {
                ContextSize = 1024, Threads = 4, AutoAccelerator = gpu, DraftModelPath = drafter, SpeculativeDraftTokens = drafter != null ? 2 : 0,
            });
            var chat = new ChatEngine(engine);
            var texts = new List<string>();
            foreach (var p in prompts)
            {
                var (prompt, _) = chat.BuildPrompt([ChatMessage.User(p)], new ChatRequestOptions { EnableThinking = false });
                var (text, _) = await engine.GenerateTextAsync(new GenerationRequest { Prompt = prompt, MaxTokens = 32, Sampling = SamplingOptions.Greedy });
                texts.Add(text);
            }
            return (texts.ToArray(), engine.TotalDraftTokens, engine.TotalAcceptedDraftTokens, engine.Offload != null);
        }
        var plain = await Run(null);
        if (gpu && !plain.OnDevice) return;   // no device
        var drafted = await Run(dp);
        Assert.True(drafted.Drafted > 0, "no drafts proposed");
        Assert.True(drafted.Accepted > drafted.Drafted / 4, $"accepted {drafted.Accepted} of {drafted.Drafted}");
        Assert.Equal(plain.Texts, drafted.Texts);
    }
}
