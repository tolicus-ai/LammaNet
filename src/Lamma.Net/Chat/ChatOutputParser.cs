using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LammaNet.Chat;

/// <summary>Kinds of parsed output events.</summary>
public enum ChatOutputKind { Text, Reasoning, ToolCall }

/// <summary>A parsed piece of model output.</summary>
public readonly record struct ChatOutputEvent(ChatOutputKind Kind, string Text, ToolCall? Call = null);

/// <summary>
/// Incrementally splits generated text into visible text, reasoning and tool calls according to the model's
/// <see cref="ChatSyntax"/>. Partial markers are held back so they never leak into streamed text.
/// </summary>
public sealed class ChatOutputParser
{
    private readonly ChatSyntax _syntax;
    private readonly IReadOnlyList<ToolDefinition>? _tools;
    private readonly StringBuilder _buf = new();
    private Mode _mode;
    private int _callCounter;
    private bool _sawNonSpace;
    private readonly StringBuilder _llamaJson = new();

    private enum Mode { Text, Reasoning, Tool, LlamaJsonCandidate }

    /// <param name="startInReasoning">True when the prompt already opened a reasoning block (e.g. ends with "&lt;think&gt;").</param>
    /// <summary>Logger for malformed tool calls (set by <see cref="ChatEngine"/>).</summary>
    internal Microsoft.Extensions.Logging.ILogger Logger { get; init; } = Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;

    public ChatOutputParser(ChatSyntax syntax, IReadOnlyList<ToolDefinition>? tools = null, bool startInReasoning = false)
    {
        _syntax = syntax;
        _tools = tools;
        _mode = startInReasoning ? Mode.Reasoning : Mode.Text;
    }

    public IEnumerable<ChatOutputEvent> Push(string text)
    {
        _buf.Append(text);
        return Drain(final: false);
    }

    public IEnumerable<ChatOutputEvent> Complete() => Drain(final: true);

    private List<ChatOutputEvent> Drain(bool final)
    {
        var events = new List<ChatOutputEvent>();
        while (true)
        {
            string s = _buf.ToString();
            if (s.Length == 0) break;
            switch (_mode)
            {
                case Mode.Text:
                {
                    // Llama-style bare JSON tool call: output starts with '{' or <|python_tag|>
                    if (_syntax.ToolCalls == ToolCallSyntax.LlamaJson && !_sawNonSpace && _tools is { Count: > 0 })
                    {
                        string t = s.TrimStart();
                        if (t.Length == 0) { if (final) { _buf.Clear(); } goto done; }
                        if (t.StartsWith('{') || t.StartsWith("<|python_tag|>"))
                        {
                            _mode = Mode.LlamaJsonCandidate;
                            continue;
                        }
                        if ("<|python_tag|>".StartsWith(t, StringComparison.Ordinal) && !final) goto done;
                    }
                    var (idx, marker, next) = FirstMarker(s, (_syntax.ReasoningStart, Mode.Reasoning), (_syntax.ToolCallStart, Mode.Tool));
                    if (idx >= 0)
                    {
                        Add(events, ChatOutputKind.Text, s[..idx]);
                        _buf.Remove(0, idx + marker!.Length);
                        _mode = next;
                        continue;
                    }
                    int hold = final ? 0 : HoldBack(s, _syntax.ReasoningStart, _syntax.ToolCallStart);
                    Add(events, ChatOutputKind.Text, s[..^hold]);
                    _buf.Remove(0, s.Length - hold);
                    goto done;
                }
                case Mode.Reasoning:
                {
                    int idx = _syntax.ReasoningEnd != null ? s.IndexOf(_syntax.ReasoningEnd, StringComparison.Ordinal) : -1;
                    if (idx >= 0)
                    {
                        Add(events, ChatOutputKind.Reasoning, s[..idx]);
                        _buf.Remove(0, idx + _syntax.ReasoningEnd!.Length);
                        _mode = Mode.Text;
                        continue;
                    }
                    int hold = final ? 0 : HoldBack(s, _syntax.ReasoningEnd);
                    Add(events, ChatOutputKind.Reasoning, s[..^hold]);
                    _buf.Remove(0, s.Length - hold);
                    goto done;
                }
                case Mode.Tool:
                {
                    int idx = _syntax.ToolCallEnd != null ? s.IndexOf(_syntax.ToolCallEnd, StringComparison.Ordinal) : -1;
                    if (idx < 0 && !final) goto done;
                    string body = idx >= 0 ? s[..idx] : s;
                    _buf.Remove(0, idx >= 0 ? idx + _syntax.ToolCallEnd!.Length : s.Length);
                    _mode = Mode.Text;
                    foreach (var call in ParseToolBody(body)) events.Add(new ChatOutputEvent(ChatOutputKind.ToolCall, "", call));
                    continue;
                }
                case Mode.LlamaJsonCandidate:
                {
                    if (!final) goto done; // need the full output to decide
                    string t = s.Trim();
                    _buf.Clear();
                    if (t.StartsWith("<|python_tag|>")) t = t["<|python_tag|>".Length..].Trim();
                    var calls = ParseLlamaJson(t);
                    if (calls.Count > 0) foreach (var c in calls) events.Add(new ChatOutputEvent(ChatOutputKind.ToolCall, "", c));
                    else Add(events, ChatOutputKind.Text, s);
                    _mode = Mode.Text;
                    goto done;
                }
            }
        }
    done:
        return events;
    }

    private void Add(List<ChatOutputEvent> ev, ChatOutputKind kind, string text)
    {
        if (text.Length == 0) return;
        if (kind == ChatOutputKind.Text && !_sawNonSpace)
        {
            if (text.Trim().Length == 0) return; // drop leading whitespace before content
            text = text.TrimStart();
            _sawNonSpace = true;
        }
        if (ev.Count > 0 && ev[^1].Kind == kind && kind != ChatOutputKind.ToolCall)
            ev[^1] = ev[^1] with { Text = ev[^1].Text + text };
        else ev.Add(new ChatOutputEvent(kind, text));
    }

    private static (int Index, string? Marker, Mode Next) FirstMarker(string s, params (string? Marker, Mode Next)[] markers)
    {
        int best = -1; string? bm = null; Mode bn = Mode.Text;
        foreach (var (m, n) in markers)
        {
            if (string.IsNullOrEmpty(m)) continue;
            int i = s.IndexOf(m, StringComparison.Ordinal);
            if (i >= 0 && (best < 0 || i < best)) { best = i; bm = m; bn = n; }
        }
        return (best, bm, bn);
    }

    /// <summary>Length of the longest suffix of <paramref name="s"/> that is a proper prefix of any marker.</summary>
    private static int HoldBack(string s, params string?[] markers)
    {
        int hold = 0;
        foreach (var m in markers)
        {
            if (string.IsNullOrEmpty(m)) continue;
            for (int k = Math.Min(m.Length - 1, s.Length); k > hold; k--)
                if (s.AsSpan(s.Length - k).SequenceEqual(m.AsSpan(0, k))) { hold = k; break; }
        }
        return hold;
    }

    private string NextId() => $"call_{++_callCounter}_{Guid.NewGuid().ToString("N")[..8]}";

    // ------------------------------------------------------------------ tool-call body parsers

    private List<ToolCall> ParseToolBody(string body) => _syntax.ToolCalls switch
    {
        ToolCallSyntax.HermesJson => ParseHermes(body),
        ToolCallSyntax.QwenXml => ParseQwenXml(body),
        ToolCallSyntax.Gemma4 => ParseGemma4(body),
        _ => ParseHermes(body),
    };

    private List<ToolCall> ParseHermes(string body)
    {
        var res = new List<ToolCall>();
        try
        {
            using var doc = JsonDocument.Parse(body.Trim());
            AddJsonCall(doc.RootElement, res);
        }
        catch (JsonException e)
        {
            Diagnostics.Log.MalformedToolCall(Logger, e, "JSON", body.Length, "the call is dropped");
        }
        return res;
    }

    private void AddJsonCall(JsonElement e, List<ToolCall> res)
    {
        if (e.ValueKind == JsonValueKind.Array) { foreach (var x in e.EnumerateArray()) AddJsonCall(x, res); return; }
        if (e.ValueKind != JsonValueKind.Object || !e.TryGetProperty("name", out var n)) return;
        string args = "{}";
        if (e.TryGetProperty("arguments", out var a) || e.TryGetProperty("parameters", out a))
            args = a.ValueKind == JsonValueKind.String ? a.GetString() ?? "{}" : a.GetRawText();
        res.Add(new ToolCall(NextId(), n.GetString() ?? "", args));
    }

    private List<ToolCall> ParseLlamaJson(string text)
    {
        var res = new List<ToolCall>();
        foreach (var piece in text.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            try
            {
                using var doc = JsonDocument.Parse(piece);
                if (doc.RootElement.ValueKind == JsonValueKind.Object && doc.RootElement.TryGetProperty("name", out var n)
                    && _tools!.Any(t => t.Name == n.GetString()))
                    AddJsonCall(doc.RootElement, res);
                else return [];
            }
            catch (JsonException) { return []; }   // not a tool call: the text is plain content
        }
        return res;
    }

    private List<ToolCall> ParseQwenXml(string body)
    {
        var res = new List<ToolCall>();
        int pos = 0;
        while (true)
        {
            int f = body.IndexOf("<function=", pos, StringComparison.Ordinal);
            if (f < 0) break;
            int nameEnd = body.IndexOf('>', f);
            if (nameEnd < 0) break;
            string name = body[(f + 10)..nameEnd].Trim();
            int fEnd = body.IndexOf("</function>", nameEnd, StringComparison.Ordinal);
            string inner = fEnd >= 0 ? body[(nameEnd + 1)..fEnd] : body[(nameEnd + 1)..];
            var args = new JsonObject();
            var schema = _tools?.FirstOrDefault(t => t.Name == name)?.Parameters;
            int p = 0;
            while (true)
            {
                int ps = inner.IndexOf("<parameter=", p, StringComparison.Ordinal);
                if (ps < 0) break;
                int pnEnd = inner.IndexOf('>', ps);
                string pname = inner[(ps + 11)..pnEnd].Trim();
                int pe = inner.IndexOf("</parameter>", pnEnd, StringComparison.Ordinal);
                string val = pe >= 0 ? inner[(pnEnd + 1)..pe] : inner[(pnEnd + 1)..];
                if (val.StartsWith('\n')) val = val[1..];
                if (val.EndsWith('\n')) val = val[..^1];
                args[pname] = ConvertParam(val, ParamType(schema, pname));
                p = pe >= 0 ? pe + 12 : inner.Length;
            }
            res.Add(new ToolCall(NextId(), name, args.ToJsonString()));
            pos = fEnd >= 0 ? fEnd + 11 : body.Length;
        }
        return res;
    }

    private static string? ParamType(JsonElement? schema, string name)
    {
        if (schema is not JsonElement s || s.ValueKind != JsonValueKind.Object) return null;
        if (!s.TryGetProperty("properties", out var props) || !props.TryGetProperty(name, out var p)) return null;
        return p.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString() : null;
    }

    private static JsonNode? ConvertParam(string raw, string? type)
    {
        if (type == "string") return JsonValue.Create(raw);
        string t = raw.Trim();
        try
        {
            var node = JsonNode.Parse(t);
            if (node != null) return node;
        }
        catch (JsonException) { }   // not JSON: the parameter is a plain string
        return JsonValue.Create(raw);
    }

    // Gemma 4: call:name{key:<|"|>str<|"|>,n:5,list:[...],obj:{...}}
    private List<ToolCall> ParseGemma4(string body)
    {
        var res = new List<ToolCall>();
        string s = body.Trim();
        int pos = 0;
        while (pos < s.Length)
        {
            int c = s.IndexOf("call:", pos, StringComparison.Ordinal);
            if (c < 0) break;
            int brace = s.IndexOf('{', c);
            if (brace < 0) break;
            string name = s[(c + 5)..brace].Trim();
            int p = brace;
            JsonNode? args;
            try { args = ParseGemmaValue(s, ref p); }
            catch (Exception e)
            {
                Diagnostics.Log.MalformedToolCall(Logger, e, "Gemma 4", body.Length, $"call {name} gets empty arguments");
                args = new JsonObject();
                p = s.Length;
            }
            res.Add(new ToolCall(NextId(), name, (args ?? new JsonObject()).ToJsonString()));
            pos = p;
        }
        return res;
    }

    private const string GQ = "<|\"|>";

    private static JsonNode? ParseGemmaValue(string s, ref int p)
    {
        SkipWs(s, ref p);
        if (s.AsSpan(p).StartsWith(GQ))
        {
            int end = s.IndexOf(GQ, p + GQ.Length, StringComparison.Ordinal);
            string str = s[(p + GQ.Length)..end];
            p = end + GQ.Length;
            return JsonValue.Create(str);
        }
        char ch = s[p];
        if (ch == '{')
        {
            p++;
            var obj = new JsonObject();
            while (true)
            {
                SkipWs(s, ref p);
                if (s[p] == '}') { p++; break; }
                string key;
                if (s.AsSpan(p).StartsWith(GQ))
                {
                    int end = s.IndexOf(GQ, p + GQ.Length, StringComparison.Ordinal);
                    key = s[(p + GQ.Length)..end];
                    p = end + GQ.Length;
                }
                else
                {
                    int colon = s.IndexOf(':', p);
                    key = s[p..colon].Trim();
                    p = colon;
                }
                SkipWs(s, ref p);
                if (s[p] == ':') p++;
                obj[key] = ParseGemmaValue(s, ref p);
                SkipWs(s, ref p);
                if (p < s.Length && s[p] == ',') p++;
            }
            return obj;
        }
        if (ch == '[')
        {
            p++;
            var arr = new JsonArray();
            while (true)
            {
                SkipWs(s, ref p);
                if (s[p] == ']') { p++; break; }
                arr.Add(ParseGemmaValue(s, ref p));
                SkipWs(s, ref p);
                if (p < s.Length && s[p] == ',') p++;
            }
            return arr;
        }
        int start = p;
        while (p < s.Length && s[p] != ',' && s[p] != '}' && s[p] != ']') p++;
        string tok = s[start..p].Trim();
        return tok switch
        {
            "true" => JsonValue.Create(true),
            "false" => JsonValue.Create(false),
            "null" => null,
            _ => double.TryParse(tok, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var d)
                ? (tok.Contains('.') || tok.Contains('e') || tok.Contains('E') ? JsonValue.Create(d) : JsonValue.Create((long)d))
                : JsonValue.Create(tok),
        };
    }

    private static void SkipWs(string s, ref int p)
    {
        while (p < s.Length && char.IsWhiteSpace(s[p])) p++;
    }
}
