using System.Runtime.CompilerServices;
using LammaNet.Inference;
using LammaNet.Sampling;

namespace LammaNet.Chat;

/// <summary>Per-request chat options.</summary>
public sealed class ChatRequestOptions
{
    public SamplingOptions Sampling { get; init; } = new();
    public int? MaxTokens { get; init; }
    public IReadOnlyList<ToolDefinition>? Tools { get; init; }
    public IReadOnlyList<string> StopSequences { get; init; } = [];
    public bool? EnableThinking { get; init; }
    public ITokenConstraint? Constraint { get; init; }
    public IReadOnlyDictionary<string, object?>? TemplateVariables { get; init; }
    /// <summary>
    /// Keep control tokens in the output text (end-of-turn tokens are still removed). Needed for models whose output
    /// markup is made of special tokens, e.g. granite-docling's DocTags (&lt;doctag&gt;, &lt;text&gt;, &lt;otsl&gt;, ...).
    /// Default: only when the template's tool-call / reasoning markers need them.
    /// </summary>
    public bool? RenderSpecialTokens { get; init; }
}

/// <summary>Encodes media parts into embedding segments (implemented by the multimodal projector).</summary>
public interface IMediaEncoder
{
    /// <summary>True if this encoder can handle the part.</summary>
    bool CanEncode(ChatPart part);
    /// <summary>Produces the prompt pieces (tokens / embeddings) that replace the media marker.</summary>
    IReadOnlyList<PromptSegment> Encode(ChatPart part);
}

/// <summary>A streamed chat event (text, reasoning or tool call) plus the terminal statistics.</summary>
public readonly record struct ChatStreamEvent(ChatOutputEvent? Output, GenerationStats? Stats);

/// <summary>
/// High-level chat interface over an <see cref="InferenceEngine"/>: renders conversations with the model's template,
/// encodes media, generates, and parses reasoning / tool calls from the output stream.
/// </summary>
public sealed class ChatEngine
{
    public InferenceEngine Engine { get; }
    public ChatTemplateRenderer Renderer { get; }
    public IList<IMediaEncoder> MediaEncoders { get; } = new List<IMediaEncoder>();

    private readonly Microsoft.Extensions.Logging.ILogger _log;

    public ChatEngine(InferenceEngine engine, string? templateOverride = null)
    {
        Engine = engine;
        _log = Diagnostics.LammaLogging.Create<ChatEngine>(engine.LoggerFactory);
        Renderer = new ChatTemplateRenderer(engine.Model, templateOverride) { Logger = Diagnostics.LammaLogging.Create<ChatTemplateRenderer>(engine.LoggerFactory) };
    }

    /// <summary>Builds the model prompt for a conversation (renders the template, tokenizes, inserts media embeddings).</summary>
    public (Prompt Prompt, string Text) BuildPrompt(IReadOnlyList<ChatMessage> messages, ChatRequestOptions? options = null)
    {
        try { return BuildPromptCore(messages, options); }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Diagnostics.Log.PromptBuildFailed(_log, e, messages.Count, messages.Sum(m => m.Parts.Count(p => p is ImagePart or AudioPart)));
            throw;
        }
    }

    private (Prompt Prompt, string Text) BuildPromptCore(IReadOnlyList<ChatMessage> messages, ChatRequestOptions? options)
    {
        options ??= new ChatRequestOptions();
        string text = Renderer.Render(messages, options.Tools, new ChatRenderOptions
        {
            EnableThinking = options.EnableThinking,
            ExtraVariables = options.TemplateVariables,
        });
        var media = messages.SelectMany(m => m.Parts).Where(p => p is ImagePart or AudioPart).ToList();
        var vocab = Engine.Vocab;
        bool addSpecial = Renderer.BosText.Length == 0 || !text.StartsWith(Renderer.BosText, StringComparison.Ordinal);
        var prompt = new Prompt();
        if (media.Count == 0)
        {
            prompt.Add(vocab.Tokenize(text, addSpecial, parseSpecial: true));
            return (prompt, text);
        }

        // split at media markers in order of appearance
        int pos = 0, mi = 0;
        bool first = true;
        while (pos <= text.Length)
        {
            int next = -1; string? marker = null;
            if (mi < media.Count)
            {
                marker = Renderer.MediaMarker(media[mi] is AudioPart ? "audio" : "image")
                    ?? throw new NotSupportedException($"The chat template of this model does not support {(media[mi] is AudioPart ? "audio" : "image")} input");
                next = text.IndexOf(marker, pos, StringComparison.Ordinal);
                if (next < 0) throw new InvalidOperationException("Media marker not found in rendered prompt");
            }
            string piece = next < 0 ? text[pos..] : text[pos..next];
            if (piece.Length > 0 || first) prompt.Add(vocab.Tokenize(piece, first && addSpecial, parseSpecial: true));
            first = false;
            if (next < 0) break;
            var part = media[mi++];
            var enc = MediaEncoders.FirstOrDefault(e => e.CanEncode(part))
                ?? throw new NotSupportedException($"No media encoder (multimodal projector) loaded for {part.GetType().Name}");
            foreach (var seg in enc.Encode(part))
            {
                if (seg is TokenSegment ts) prompt.Add(ts.Tokens);
                else if (seg is EmbeddingSegment es) prompt.Add(es);
            }
            pos = next + marker!.Length;
        }
        return (prompt, text);
    }

    /// <summary>Streams parsed output for a conversation turn.</summary>
    public async IAsyncEnumerable<ChatStreamEvent> ChatAsync(IReadOnlyList<ChatMessage> messages, ChatRequestOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new ChatRequestOptions();
        var (prompt, text) = BuildPrompt(messages, options);
        var syntax = Renderer.Syntax;
        bool inReasoning = syntax.ReasoningStart != null && text.TrimEnd().EndsWith(syntax.ReasoningStart, StringComparison.Ordinal);
        var parser = new ChatOutputParser(syntax, options.Tools, inReasoning) { Logger = _log };
        var req = new GenerationRequest
        {
            Prompt = prompt,
            Sampling = options.Sampling,
            MaxTokens = options.MaxTokens,
            StopSequences = options.StopSequences,
            Constraint = options.Constraint,
            // tool-call / reasoning markers are often special tokens: render them so the parser can see them
            RenderSpecialTokens = options.RenderSpecialTokens ?? (syntax.ToolCalls != ToolCallSyntax.None || syntax.ReasoningStart != null),
        };
        var handle = Engine.Submit(req, cancellationToken);
        await foreach (var t in handle.Tokens.ReadAllAsync(CancellationToken.None).ConfigureAwait(false))
        {
            if (t.Text.Length == 0) continue;
            string visible = req.RenderSpecialTokens ? StripTurnTokens(t.Text) : t.Text;
            foreach (var ev in parser.Push(visible)) yield return new ChatStreamEvent(ev, null);
        }
        foreach (var ev in parser.Complete()) yield return new ChatStreamEvent(ev, null);
        var stats = await handle.Completion.ConfigureAwait(false);
        yield return new ChatStreamEvent(null, stats);
    }

    /// <summary>Removes rendered end-of-turn control tokens that are not part of the content.</summary>
    private string StripTurnTokens(string text)
    {
        var v = Engine.Vocab;
        foreach (var id in v.EndOfGeneration)
        {
            string s = v.TokenText(id);
            if (s.Length > 0 && text.Contains(s, StringComparison.Ordinal)) text = text.Replace(s, "", StringComparison.Ordinal);
        }
        return text;
    }
}
