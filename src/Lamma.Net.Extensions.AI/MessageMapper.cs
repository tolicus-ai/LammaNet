using System.Text.Json;
using Microsoft.Extensions.AI;
using Core = LammaNet.Chat;
using MeaiChatMessage = Microsoft.Extensions.AI.ChatMessage;
using MeaiChatRole = Microsoft.Extensions.AI.ChatRole;

namespace LammaNet.Extensions.AI;

/// <summary>Converts between Microsoft.Extensions.AI types and Lamma.Net chat types.</summary>
internal static class MessageMapper
{
    public static List<Core.ChatMessage> ToCore(IEnumerable<MeaiChatMessage> messages, ChatOptions? options)
    {
        var list = new List<Core.ChatMessage>();
        var callNames = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!string.IsNullOrWhiteSpace(options?.Instructions))
            list.Add(Core.ChatMessage.System(options!.Instructions!));

        foreach (var m in messages)
        {
            var role = m.Role == MeaiChatRole.System ? Core.ChatRole.System
                : m.Role == MeaiChatRole.Assistant ? Core.ChatRole.Assistant
                : m.Role == MeaiChatRole.Tool ? Core.ChatRole.Tool
                : Core.ChatRole.User;

            var parts = new List<Core.ChatPart>();
            var calls = new List<Core.ToolCall>();
            string? reasoning = null;
            var results = new List<FunctionResultContent>();
            foreach (var c in m.Contents)
            {
                switch (c)
                {
                    case TextContent t: parts.Add(new Core.TextPart(t.Text)); break;
                    case TextReasoningContent r: reasoning = (reasoning ?? "") + r.Text; break;
                    case DataContent d when d.HasTopLevelMediaType("image"):
                        parts.Add(new Core.ImagePart { EncodedBytes = d.Data.ToArray() }); break;
                    case DataContent d when d.HasTopLevelMediaType("audio"):
                        parts.Add(new Core.AudioPart { EncodedBytes = d.Data.ToArray() }); break;
                    case DataContent d when d.HasTopLevelMediaType("text"):
                        parts.Add(new Core.TextPart(System.Text.Encoding.UTF8.GetString(d.Data.Span))); break;
                    case UriContent u when u.Uri.IsFile && File.Exists(u.Uri.LocalPath):
                    {
                        var bytes = File.ReadAllBytes(u.Uri.LocalPath);
                        if (u.HasTopLevelMediaType("audio")) parts.Add(new Core.AudioPart { EncodedBytes = bytes });
                        else parts.Add(new Core.ImagePart { EncodedBytes = bytes });
                        break;
                    }
                    case FunctionCallContent fc:
                        callNames[fc.CallId] = fc.Name;
                        calls.Add(new Core.ToolCall(fc.CallId, fc.Name, JsonSerializer.Serialize(fc.Arguments ?? new Dictionary<string, object?>(), AIJsonUtilities.DefaultOptions)));
                        break;
                    case FunctionResultContent fr: results.Add(fr); break;
                }
            }

            if (results.Count > 0)
            {
                // each function result becomes its own tool message
                foreach (var r in results)
                {
                    string text = r.Result switch
                    {
                        null => r.Exception != null ? $"Error: {r.Exception.Message}" : "",
                        string s => s,
                        JsonElement je => je.ValueKind == JsonValueKind.String ? je.GetString() ?? "" : je.GetRawText(),
                        _ => JsonSerializer.Serialize(r.Result, AIJsonUtilities.DefaultOptions),
                    };
                    list.Add(new Core.ChatMessage
                    {
                        Role = Core.ChatRole.Tool,
                        ToolCallId = r.CallId,
                        Name = callNames.GetValueOrDefault(r.CallId),
                        Parts = [new Core.TextPart(text)],
                    });
                }
                if (parts.Count == 0) continue;
            }
            list.Add(new Core.ChatMessage { Role = role, Parts = parts, ToolCalls = calls, Reasoning = reasoning });
        }
        return list;
    }

    public static List<Core.ToolDefinition>? ToTools(ChatOptions? options)
    {
        if (options?.Tools is not { Count: > 0 } tools || options.ToolMode is NoneChatToolMode) return null;
        var res = new List<Core.ToolDefinition>();
        foreach (var t in tools)
        {
            if (t is AIFunctionDeclaration f)
            {
                if (options.ToolMode is RequiredChatToolMode { RequiredFunctionName: string req } && req != f.Name) continue;
                res.Add(new Core.ToolDefinition { Name = f.Name, Description = f.Description, Parameters = f.JsonSchema });
            }
        }
        return res.Count > 0 ? res : null;
    }

    public static Dictionary<string, object?> ParseArguments(string json, Microsoft.Extensions.Logging.ILogger? log = null)
    {
        try
        {
            using var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(json) ? "{}" : json);
            var d = new Dictionary<string, object?>();
            if (doc.RootElement.ValueKind == JsonValueKind.Object)
                foreach (var p in doc.RootElement.EnumerateObject()) d[p.Name] = p.Value.Clone();
            return d;
        }
        catch (JsonException e)
        {
            if (log != null) LammaNet.Diagnostics.Log.ToolArgumentsRaw(log, e);
            return new Dictionary<string, object?> { ["raw"] = json };
        }
    }
}
