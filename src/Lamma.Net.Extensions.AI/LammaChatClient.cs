using System.Runtime.CompilerServices;
using System.Text.Json;
using LammaNet.Chat;
using LammaNet.Inference;
using LammaNet.Models;
using LammaNet.Sampling;
using Microsoft.Extensions.AI;
using MeaiChatMessage = Microsoft.Extensions.AI.ChatMessage;
using MeaiChatRole = Microsoft.Extensions.AI.ChatRole;

namespace LammaNet.Extensions.AI;

/// <summary>Options for <see cref="LammaChatClient"/>.</summary>
public sealed class LammaChatClientOptions
{
    /// <summary>Model id reported in responses (defaults to the GGUF general.name).</summary>
    public string? ModelId { get; set; }
    /// <summary>Default sampling when a request does not specify values (defaults to the GGUF's recommended sampling, if any).</summary>
    public SamplingOptions? DefaultSampling { get; set; }
    /// <summary>Default for model "thinking" (null = chat template default).</summary>
    public bool? EnableThinking { get; set; }
    /// <summary>Default maximum output tokens.</summary>
    public int? DefaultMaxOutputTokens { get; set; }
    /// <summary>Override for the GGUF chat template (Jinja).</summary>
    public string? ChatTemplate { get; set; }
    /// <summary>Append the JSON schema to the prompt when a JSON response format with schema is requested.</summary>
    public bool IncludeSchemaInPrompt { get; set; } = true;
    /// <summary>Path to a multimodal projector GGUF (mmproj) enabling image / audio inputs.</summary>
    public string? MmprojPath { get; set; }
    /// <summary>Options for the multimodal encoders.</summary>
    public LammaNet.Multimodal.ProjectorOptions? ProjectorOptions { get; set; }

    internal LammaChatClientOptions Clone() => (LammaChatClientOptions)MemberwiseClone();
}

/// <summary>
/// <see cref="IChatClient"/> running a GGUF model fully in-process with Lamma.Net.
/// Supports streaming, tool calling (model-native formats), reasoning content, JSON output (constrained decoding),
/// multimodal inputs (images / audio via <see cref="DataContent"/> when a projector is loaded) and usage reporting.
/// Compose with <c>UseFunctionInvocation()</c> for automatic tool execution or use it as the client of a Microsoft Agent Framework agent.
/// </summary>
public sealed class LammaChatClient : IChatClient
{
    private readonly ChatEngine _chat;
    private readonly LammaChatClientOptions _options;
    private readonly ChatClientMetadata _metadata;
    private readonly bool _ownsEngine;
    private readonly Microsoft.Extensions.Logging.ILogger _log;

    public ChatEngine ChatEngine => _chat;
    public InferenceEngine Engine => _chat.Engine;

    public LammaChatClient(InferenceEngine engine, LammaChatClientOptions? options = null, bool ownsEngine = false)
    {
        _options = options ?? new LammaChatClientOptions();
        _log = LammaNet.Diagnostics.LammaLogging.Create<LammaChatClient>(engine.LoggerFactory);
        _chat = new ChatEngine(engine, _options.ChatTemplate);
        _ownsEngine = ownsEngine;
        if (_options.MmprojPath != null)
        {
            try { _chat.MediaEncoders.Add(LammaNet.Multimodal.MultimodalProjector.Load(_options.MmprojPath, engine, _options.ProjectorOptions)); }
            catch
            {
                if (ownsEngine) engine.Dispose();   // logged by the loader; do not leak the engine we were given to own
                throw;
            }
        }
        string modelId = _options.ModelId ?? engine.Config.Name;
        _metadata = new ChatClientMetadata("lamma.net", null, modelId);
    }

    public LammaChatClient(ChatEngine chatEngine, LammaChatClientOptions? options = null)
    {
        _options = options ?? new LammaChatClientOptions();
        _log = LammaNet.Diagnostics.LammaLogging.Create<LammaChatClient>(chatEngine.Engine.LoggerFactory);
        _chat = chatEngine;
        _metadata = new ChatClientMetadata("lamma.net", null, _options.ModelId ?? chatEngine.Engine.Config.Name);
    }

    /// <summary>Loads a GGUF file and creates a chat client that owns the engine.</summary>
    public static LammaChatClient Load(string ggufPath, EngineOptions? engineOptions = null, LammaChatClientOptions? options = null) =>
        new(InferenceEngine.Load(ggufPath, engineOptions), options, ownsEngine: true);

    /// <summary>
    /// Loads a model from Hugging Face through the model cache: the cached copy of the resolved version is used when present,
    /// otherwise the file (and, unless <see cref="HuggingFaceModel.IncludeMmproj"/> is false or <see cref="LammaChatClientOptions.MmprojPath"/>
    /// is set, the repository's matching mmproj) is downloaded into the cache first. Blocks while downloading; see <see cref="LoadAsync(HuggingFaceModel, EngineOptions?, LammaChatClientOptions?, ModelCache?, CancellationToken)"/>.
    /// </summary>
    public static LammaChatClient Load(HuggingFaceModel model, EngineOptions? engineOptions = null, LammaChatClientOptions? options = null, ModelCache? cache = null) =>
        LoadAsync(model, engineOptions, options, cache).GetAwaiter().GetResult();

    /// <summary>Loads a GGUF file from a URL through the model cache (downloaded once per server version / ETag). Blocks while downloading.</summary>
    public static LammaChatClient Load(Uri modelUrl, EngineOptions? engineOptions = null, LammaChatClientOptions? options = null, ModelCache? cache = null) =>
        LoadAsync(modelUrl, engineOptions, options, cache).GetAwaiter().GetResult();

    /// <inheritdoc cref="Load(HuggingFaceModel, EngineOptions?, LammaChatClientOptions?, ModelCache?)"/>
    public static async Task<LammaChatClient> LoadAsync(HuggingFaceModel model, EngineOptions? engineOptions = null, LammaChatClientOptions? options = null,
        ModelCache? cache = null, CancellationToken cancellationToken = default)
    {
        if (options?.MmprojPath != null) model = model with { IncludeMmproj = false };
        using var own = OwnCache(cache, engineOptions);
        var files = await (cache ?? own ?? ModelCache.Shared).GetAsync(model, cancellationToken).ConfigureAwait(false);
        return FromCache(files, engineOptions, options);
    }

    /// <inheritdoc cref="Load(Uri, EngineOptions?, LammaChatClientOptions?, ModelCache?)"/>
    public static async Task<LammaChatClient> LoadAsync(Uri modelUrl, EngineOptions? engineOptions = null, LammaChatClientOptions? options = null,
        ModelCache? cache = null, CancellationToken cancellationToken = default)
    {
        using var own = OwnCache(cache, engineOptions);
        var files = await (cache ?? own ?? ModelCache.Shared).GetAsync(modelUrl, cancellationToken).ConfigureAwait(false);
        return FromCache(files, engineOptions, options);
    }

    /// <summary>A cache logging to the engine's factory when the caller set one and passed no cache (same directories as the shared cache).</summary>
    private static ModelCache? OwnCache(ModelCache? cache, EngineOptions? engineOptions) =>
        cache == null && engineOptions?.LoggerFactory is { } lf ? new ModelCache(new ModelCacheOptions { LoggerFactory = lf }) : null;

    private static LammaChatClient FromCache(ModelResolution files, EngineOptions? engineOptions, LammaChatClientOptions? options)
    {
        if (files.MmprojPath != null && options?.MmprojPath == null)
        {
            // never modify the caller's options object
            options = options?.Clone() ?? new LammaChatClientOptions();
            options.MmprojPath = files.MmprojPath;
        }
        return Load(files.ModelPath, engineOptions, options);
    }

    public async Task<ChatResponse> GetResponseAsync(IEnumerable<MeaiChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in GetStreamingResponseAsync(messages, options, cancellationToken).ConfigureAwait(false)) updates.Add(u);
        var response = updates.ToChatResponse();
        return response;
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<MeaiChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var coreMessages = MessageMapper.ToCore(messages, options);
        var tools = MessageMapper.ToTools(options);
        var request = BuildRequest(options, tools, coreMessages);
        bool showReasoning = options?.Reasoning?.Output != ReasoningOutput.None;
        string responseId = Guid.NewGuid().ToString("N");
        string messageId = Guid.NewGuid().ToString("N");
        var created = DateTimeOffset.UtcNow;
        bool anyToolCall = false;

        ChatResponseUpdate Update(params AIContent[] contents) => new(MeaiChatRole.Assistant, contents)
        {
            ResponseId = responseId,
            MessageId = messageId,
            ModelId = _metadata.DefaultModelId,
            CreatedAt = created,
        };

        await foreach (var ev in _chat.ChatAsync(coreMessages, request, cancellationToken).ConfigureAwait(false))
        {
            if (ev.Output is { } o)
            {
                switch (o.Kind)
                {
                    case ChatOutputKind.Text:
                        yield return Update(new TextContent(o.Text));
                        break;
                    case ChatOutputKind.Reasoning when showReasoning:
                        yield return Update(new TextReasoningContent(o.Text));
                        break;
                    case ChatOutputKind.ToolCall:
                        anyToolCall = true;
                        yield return Update(new FunctionCallContent(o.Call!.Id, o.Call.Name, MessageMapper.ParseArguments(o.Call.ArgumentsJson, _log)));
                        break;
                }
            }
            else if (ev.Stats is { } s)
            {
                var usage = new UsageDetails
                {
                    InputTokenCount = s.PromptTokens,
                    OutputTokenCount = s.CompletionTokens,
                    TotalTokenCount = s.PromptTokens + s.CompletionTokens,
                    CachedInputTokenCount = s.CachedPromptTokens,
                };
                var final = Update(new UsageContent(usage));
                final.FinishReason = anyToolCall ? ChatFinishReason.ToolCalls : s.FinishReason switch
                {
                    FinishReason.Length or FinishReason.ContextFull => ChatFinishReason.Length,
                    _ => ChatFinishReason.Stop,
                };
                final.AdditionalProperties = new AdditionalPropertiesDictionary
                {
                    ["prompt_tokens_per_second"] = s.PromptTokensPerSecond,
                    ["tokens_per_second"] = s.TokensPerSecond,
                };
                yield return final;
            }
        }
    }

    private ChatRequestOptions BuildRequest(ChatOptions? o, List<ToolDefinition>? tools, List<Chat.ChatMessage> messages)
    {
        var baseS = _options.DefaultSampling ?? RecommendedSampling(Engine) ?? new SamplingOptions();
        var props = o?.AdditionalProperties;
        T? Prop<T>(string key) where T : struct
        {
            if (props == null || !props.TryGetValue(key, out var v) || v == null) return null;
            try { return v is JsonElement je ? je.Deserialize<T>() : (T)Convert.ChangeType(v, typeof(T), System.Globalization.CultureInfo.InvariantCulture); }
            catch (Exception e) when (e is JsonException or InvalidCastException or FormatException or OverflowException or NotSupportedException)
            {
                LammaNet.Diagnostics.Log.ChatOptionIgnored(_log, e, key, typeof(T).Name);
                return null;
            }
        }
        var sampling = baseS with
        {
            Temperature = o?.Temperature ?? baseS.Temperature,
            TopP = o?.TopP ?? baseS.TopP,
            TopK = o?.TopK ?? baseS.TopK,
            FrequencyPenalty = o?.FrequencyPenalty ?? baseS.FrequencyPenalty,
            PresencePenalty = o?.PresencePenalty ?? baseS.PresencePenalty,
            Seed = o?.Seed is long seed ? (int)seed : baseS.Seed,
            MinP = Prop<float>("min_p") ?? baseS.MinP,
            TypicalP = Prop<float>("typical_p") ?? baseS.TypicalP,
            RepetitionPenalty = Prop<float>("repetition_penalty") ?? baseS.RepetitionPenalty,
            DryMultiplier = Prop<float>("dry_multiplier") ?? baseS.DryMultiplier,
        };

        bool? thinking = _options.EnableThinking;
        if (Prop<bool>("enable_thinking") is bool et) thinking = et;
        if (o?.Reasoning?.Effort is ReasoningEffort effort) thinking = effort != ReasoningEffort.None;

        ITokenConstraint? constraint = null;
        if (o?.ResponseFormat is ChatResponseFormatJson json)
        {
            bool requireObject = json.Schema is not JsonElement schema || !schema.TryGetProperty("type", out var t) || t.ValueKind != JsonValueKind.String || t.GetString() == "object";
            constraint = new JsonConstraint(Engine.Vocab, requireObject);
            if (json.Schema is JsonElement sch && _options.IncludeSchemaInPrompt)
            {
                string instruction = "Respond only with JSON conforming to this JSON schema:\n" + sch.GetRawText();
                if (json.SchemaName != null) instruction = $"{instruction}\nSchema name: {json.SchemaName}";
                messages.Insert(messages.Count > 0 && messages[0].Role == Chat.ChatRole.System ? 1 : 0, Chat.ChatMessage.System(instruction));
                MergeSystemMessages(messages);
            }
            tools = null; // structured output and tool calls are mutually exclusive for local models
        }

        return new ChatRequestOptions
        {
            Sampling = sampling,
            MaxTokens = o?.MaxOutputTokens ?? _options.DefaultMaxOutputTokens,
            StopSequences = o?.StopSequences?.ToList() ?? [],
            Tools = tools,
            EnableThinking = thinking,
            Constraint = constraint,
            RenderSpecialTokens = Prop<bool>("render_special_tokens"),
        };
    }

    /// <summary>Many templates accept a single leading system message: merge consecutive system messages.</summary>
    private static void MergeSystemMessages(List<Chat.ChatMessage> messages)
    {
        if (messages.Count < 2 || messages[0].Role != Chat.ChatRole.System || messages[1].Role != Chat.ChatRole.System) return;
        var merged = Chat.ChatMessage.System(messages[0].Text + "\n\n" + messages[1].Text);
        messages.RemoveRange(0, 2);
        messages.Insert(0, merged);
    }

    /// <summary>Sampling parameters recommended in the GGUF metadata (general.sampling.*), if present.</summary>
    public static SamplingOptions? RecommendedSampling(InferenceEngine engine)
    {
        var f = engine.Model.File;
        if (!f.Metadata.Keys.Any(k => k.StartsWith("general.sampling.", StringComparison.Ordinal))) return null;
        var d = new SamplingOptions();
        return d with
        {
            Temperature = f.Get("general.sampling.temp", d.Temperature),
            TopK = f.Get("general.sampling.top_k", d.TopK),
            TopP = f.Get("general.sampling.top_p", d.TopP),
            MinP = f.Get("general.sampling.min_p", d.MinP),
        };
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey != null) return null;
        if (serviceType == typeof(ChatClientMetadata)) return _metadata;
        if (serviceType.IsInstanceOfType(this)) return this;
        if (serviceType.IsInstanceOfType(_chat)) return _chat;
        if (serviceType.IsInstanceOfType(_chat.Engine)) return _chat.Engine;
        if (serviceType.IsInstanceOfType(_chat.Engine.Model)) return _chat.Engine.Model;
        return null;
    }

    public void Dispose()
    {
        if (_ownsEngine)
        {
            foreach (var e in _chat.MediaEncoders.OfType<IDisposable>()) e.Dispose();
            _chat.Engine.Dispose();
        }
    }
}
