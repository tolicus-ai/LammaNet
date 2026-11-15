using LammaNet.Models;
using LammaNet.Templates.Jinja;

namespace LammaNet.Chat;

/// <summary>How tool calls are written by a model family.</summary>
public enum ToolCallSyntax
{
    None,
    /// <summary><c>&lt;tool_call&gt;{"name":..,"arguments":{..}}&lt;/tool_call&gt;</c> (Qwen2.5/3, Granite, Hermes).</summary>
    HermesJson,
    /// <summary><c>&lt;tool_call&gt;&lt;function=name&gt;&lt;parameter=p&gt;v&lt;/parameter&gt;&lt;/function&gt;&lt;/tool_call&gt;</c> (Qwen3.5/3.6, Qwen3-Coder).</summary>
    QwenXml,
    /// <summary>Bare JSON object <c>{"name":..,"parameters":{..}}</c> (Llama 3.x).</summary>
    LlamaJson,
    /// <summary><c>&lt;|tool_call&gt;call:name{k:&lt;|"|&gt;v&lt;|"|&gt;}&lt;tool_call|&gt;</c> (Gemma 4).</summary>
    Gemma4,
}

/// <summary>Describes the output conventions of a chat model.</summary>
public sealed record ChatSyntax
{
    public ToolCallSyntax ToolCalls { get; init; }
    public string? ReasoningStart { get; init; }
    public string? ReasoningEnd { get; init; }
    public string? ToolCallStart { get; init; }
    public string? ToolCallEnd { get; init; }

    public static ChatSyntax Detect(string template)
    {
        ToolCallSyntax tc = ToolCallSyntax.None;
        string? ts = null, te = null;
        if (template.Contains("<|tool_call>call:") || template.Contains("'<|tool_call>call:'")) { tc = ToolCallSyntax.Gemma4; ts = "<|tool_call>"; te = "<tool_call|>"; }
        else if (template.Contains("<function=")) { tc = ToolCallSyntax.QwenXml; ts = "<tool_call>"; te = "</tool_call>"; }
        else if (template.Contains("<tool_call>")) { tc = ToolCallSyntax.HermesJson; ts = "<tool_call>"; te = "</tool_call>"; }
        else if (template.Contains("\"parameters\": dictionary of argument name") || template.Contains("ipython")) tc = ToolCallSyntax.LlamaJson;

        string? rs = null, re = null;
        if (template.Contains("<|channel>")) { rs = "<|channel>thought"; re = "<channel|>"; }
        else if (template.Contains("<think>")) { rs = "<think>"; re = "</think>"; }
        return new ChatSyntax { ToolCalls = tc, ToolCallStart = ts, ToolCallEnd = te, ReasoningStart = rs, ReasoningEnd = re };
    }
}

/// <summary>Options for rendering a conversation into a prompt.</summary>
public sealed class ChatRenderOptions
{
    public bool AddGenerationPrompt { get; init; } = true;
    /// <summary>Enable/disable model "thinking" when the template supports it (null = template default).</summary>
    public bool? EnableThinking { get; init; }
    /// <summary>Extra template variables (e.g. <c>reasoning_effort</c>, <c>custom_instructions</c>).</summary>
    public IReadOnlyDictionary<string, object?>? ExtraVariables { get; init; }
}

/// <summary>
/// Renders conversations with the model's own Jinja chat template (from GGUF) — or a supplied override —
/// and locates where media parts are placed in the rendered prompt.
/// </summary>
public sealed class ChatTemplateRenderer
{
    private readonly JinjaTemplate _template;
    private readonly string _bos, _eos;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, string?> _markers = new();   // prompts are built concurrently

    public string Source { get; }
    public ChatSyntax Syntax { get; }
    public string BosText => _bos;

    /// <summary>Logger for template fallbacks (set by <see cref="ChatEngine"/>; default: the default factory's).</summary>
    internal Microsoft.Extensions.Logging.ILogger Logger { get; set; } = Diagnostics.LammaLogging.Create<ChatTemplateRenderer>(null);

    public ChatTemplateRenderer(Model model, string? templateOverride = null)
        : this(templateOverride ?? model.ChatTemplate ?? DefaultChatML,
               model.Vocab.Bos >= 0 ? model.Vocab.TokenText(model.Vocab.Bos) : "",
               model.Vocab.Eos >= 0 ? model.Vocab.TokenText(model.Vocab.Eos) : "")
    {
    }

    public ChatTemplateRenderer(string template, string bosToken, string eosToken)
    {
        Source = template;
        _template = new JinjaTemplate(template);
        _bos = bosToken;
        _eos = eosToken;
        Syntax = ChatSyntax.Detect(template);
    }

    /// <summary>ChatML fallback used when a GGUF carries no template.</summary>
    public const string DefaultChatML =
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}" +
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    /// <summary>Renders the conversation; media parts appear as the template's own placeholder markers.</summary>
    public string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition>? tools = null, ChatRenderOptions? options = null)
    {
        options ??= new ChatRenderOptions();
        var vars = new Dictionary<string, object?>
        {
            ["messages"] = messages.Select(ToTemplateMessage).ToList(),
            ["add_generation_prompt"] = options.AddGenerationPrompt,
            ["bos_token"] = _bos,
            ["eos_token"] = _eos,
        };
        if (tools is { Count: > 0 }) vars["tools"] = tools.Select(ToTemplateTool).ToList();
        if (options.EnableThinking is bool et) { vars["enable_thinking"] = et; vars["thinking"] = et; }
        if (options.ExtraVariables != null) foreach (var (k, v) in options.ExtraVariables) vars[k] = v;
        return _template.Render(vars);
    }

    /// <summary>
    /// Discovers the placeholder text the template emits for a media part of the given modality
    /// ("image" or "audio") by rendering a probe conversation with and without the part. Null if unsupported.
    /// </summary>
    public string? MediaMarker(string modality)
    {
        if (_markers.TryGetValue(modality, out var m)) return m;
        try
        {
            ChatPart media = modality == "audio" ? new AudioPart() : new ImagePart();
            var withMedia = Render([new ChatMessage { Role = ChatRole.User, Parts = [media, new TextPart("\u0001probe")] }], null, new() { AddGenerationPrompt = false });
            var without = Render([new ChatMessage { Role = ChatRole.User, Parts = [new TextPart("\u0001probe")] }], null, new() { AddGenerationPrompt = false });
            int i = 0;
            while (i < without.Length && i < withMedia.Length && without[i] == withMedia[i]) i++;
            int extra = withMedia.Length - without.Length;
            m = extra > 0 ? withMedia.Substring(i, extra) : null;
            if (m != null && m.Trim().Length == 0) m = null;
        }
        catch (JinjaException e)
        {
            Diagnostics.Log.MediaMarkerUnavailable(Logger, e, modality);
            m = null;
        }
        _markers[modality] = m;
        return m;
    }

    private object ToTemplateMessage(ChatMessage msg)
    {
        var d = new JinjaDict
        {
            ["role"] = msg.Role switch { ChatRole.System => "system", ChatRole.User => "user", ChatRole.Assistant => "assistant", _ => "tool" },
        };
        bool onlyText = msg.Parts.All(p => p is TextPart);
        if (onlyText) d["content"] = msg.Text;
        else
        {
            var parts = new List<object?>();
            foreach (var p in msg.Parts)
            {
                var pd = new JinjaDict();
                switch (p)
                {
                    case TextPart t: pd["type"] = "text"; pd["text"] = t.Text; break;
                    case ImagePart: pd["type"] = "image"; break;
                    case AudioPart: pd["type"] = "audio"; break;
                }
                parts.Add(pd);
            }
            d["content"] = parts;
        }
        if (msg.ToolCalls.Count > 0)
        {
            d["tool_calls"] = msg.ToolCalls.Select(tc =>
            {
                object? args;
                try { using var doc = System.Text.Json.JsonDocument.Parse(string.IsNullOrWhiteSpace(tc.ArgumentsJson) ? "{}" : tc.ArgumentsJson); args = JinjaValue.FromJson(doc.RootElement.Clone()); }
                catch (Exception e) when (e is System.Text.Json.JsonException or ArgumentException)
                {
                    Diagnostics.Log.ToolArgumentsNotJson(Logger, e, tc.Name);
                    args = tc.ArgumentsJson;
                }
                var fn = new JinjaDict { ["name"] = tc.Name, ["arguments"] = args };
                return (object?)new JinjaDict { ["id"] = tc.Id, ["type"] = "function", ["function"] = fn };
            }).ToList();
        }
        if (msg.ToolCallId != null) d["tool_call_id"] = msg.ToolCallId;
        if (msg.Name != null) d["name"] = msg.Name;
        if (msg.Reasoning != null) d["reasoning_content"] = msg.Reasoning;
        return d;
    }

    private static object ToTemplateTool(ToolDefinition t)
    {
        var fn = new JinjaDict { ["name"] = t.Name };
        if (t.Description != null) fn["description"] = t.Description;
        fn["parameters"] = t.Parameters is System.Text.Json.JsonElement je
            ? JinjaValue.FromJson(je)
            : new JinjaDict { ["type"] = "object", ["properties"] = new JinjaDict() };
        return new JinjaDict { ["type"] = "function", ["function"] = fn };
    }
}
