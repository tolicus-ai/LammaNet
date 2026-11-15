using System.Text.Json;

namespace LammaNet.Chat;

public enum ChatRole { System, User, Assistant, Tool }

/// <summary>A part of a chat message.</summary>
public abstract record ChatPart;

public sealed record TextPart(string Text) : ChatPart;

/// <summary>An image given as encoded bytes (PNG/JPEG/BMP/GIF...) or raw RGB pixels.</summary>
public sealed record ImagePart : ChatPart
{
    public byte[]? EncodedBytes { get; init; }
    /// <summary>Raw interleaved RGB (3 bytes per pixel), row-major.</summary>
    public byte[]? Rgb { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }
}

/// <summary>Audio as mono float PCM samples at <see cref="SampleRate"/> Hz, or encoded bytes (WAV).</summary>
public sealed record AudioPart : ChatPart
{
    public float[]? Samples { get; init; }
    public int SampleRate { get; init; } = 16000;
    public byte[]? EncodedBytes { get; init; }
}

/// <summary>A function call produced by (or replayed to) the model.</summary>
public sealed record ToolCall(string Id, string Name, string ArgumentsJson);

/// <summary>A chat message in engine-neutral form.</summary>
public sealed class ChatMessage
{
    public ChatRole Role { get; init; }
    public List<ChatPart> Parts { get; init; } = new();
    public List<ToolCall> ToolCalls { get; init; } = new();
    /// <summary>For <see cref="ChatRole.Tool"/> messages: the call being answered.</summary>
    public string? ToolCallId { get; init; }
    /// <summary>For tool messages: the function name.</summary>
    public string? Name { get; init; }
    /// <summary>Reasoning text of an assistant message (rendered only if the template supports it).</summary>
    public string? Reasoning { get; init; }

    public string Text => string.Concat(Parts.OfType<TextPart>().Select(p => p.Text));

    public static ChatMessage System(string text) => new() { Role = ChatRole.System, Parts = [new TextPart(text)] };
    public static ChatMessage User(string text) => new() { Role = ChatRole.User, Parts = [new TextPart(text)] };
    public static ChatMessage Assistant(string text) => new() { Role = ChatRole.Assistant, Parts = [new TextPart(text)] };
    public static ChatMessage ToolResult(string callId, string name, string result) =>
        new() { Role = ChatRole.Tool, ToolCallId = callId, Name = name, Parts = [new TextPart(result)] };
}

/// <summary>A function the model may call.</summary>
public sealed class ToolDefinition
{
    public required string Name { get; init; }
    public string? Description { get; init; }
    /// <summary>JSON schema of the parameters object.</summary>
    public JsonElement? Parameters { get; init; }
}
