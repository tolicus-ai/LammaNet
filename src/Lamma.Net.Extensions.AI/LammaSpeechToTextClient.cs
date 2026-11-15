using System.Runtime.CompilerServices;
using LammaNet.Chat;
using LammaNet.Sampling;
using Microsoft.Extensions.AI;

namespace LammaNet.Extensions.AI;

/// <summary>
/// <see cref="ISpeechToTextClient"/> using an audio-capable multimodal model (e.g. Gemma 4 E2B/E4B with its mmproj)
/// to transcribe — or translate — speech fully in-process.
/// </summary>
public sealed class LammaSpeechToTextClient : ISpeechToTextClient
{
    private readonly ChatEngine _chat;
    private readonly SpeechToTextClientMetadata _metadata;

    public LammaSpeechToTextClient(LammaChatClient chatClient)
        : this(chatClient.ChatEngine) { }

    public LammaSpeechToTextClient(ChatEngine chat)
    {
        _chat = chat;
        if (!chat.MediaEncoders.Any(e => e.CanEncode(new AudioPart())))
            throw new InvalidOperationException("The chat engine has no audio encoder; load an audio-capable mmproj (e.g. Gemma 4).");
        _metadata = new SpeechToTextClientMetadata("lamma.net", null, chat.Engine.Config.Name);
    }

    public async Task<SpeechToTextResponse> GetTextAsync(Stream audioSpeechStream, SpeechToTextOptions? options = null, CancellationToken cancellationToken = default)
    {
        var sb = new System.Text.StringBuilder();
        string id = Guid.NewGuid().ToString("N");
        await foreach (var u in GetStreamingTextAsync(audioSpeechStream, options, cancellationToken).ConfigureAwait(false))
            if (u.Kind == SpeechToTextResponseUpdateKind.TextUpdating) sb.Append(u.Text);
        return new SpeechToTextResponse(sb.ToString().Trim()) { ResponseId = id, ModelId = _metadata.DefaultModelId };
    }

    public async IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(Stream audioSpeechStream, SpeechToTextOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var ms = new MemoryStream();
        await audioSpeechStream.CopyToAsync(ms, cancellationToken).ConfigureAwait(false);
        var bytes = ms.ToArray();
        AudioPart audio = IsWav(bytes)
            ? new AudioPart { EncodedBytes = bytes }
            : new AudioPart { Samples = Pcm16ToFloat(bytes), SampleRate = options?.SpeechSampleRate ?? 16000 };

        string instruction = options?.TextLanguage is string tl && tl.Length > 0 && tl != options.SpeechLanguage
            ? $"Transcribe the speech and translate it into {tl}. Output only the translation."
            : "Transcribe this audio verbatim. Output only the transcription.";
        var msg = new Chat.ChatMessage { Role = Chat.ChatRole.User, Parts = [audio, new TextPart(instruction)] };
        string id = Guid.NewGuid().ToString("N");
        yield return new SpeechToTextResponseUpdate { Kind = SpeechToTextResponseUpdateKind.SessionOpen, ResponseId = id };
        await foreach (var ev in _chat.ChatAsync([msg], new ChatRequestOptions { Sampling = SamplingOptions.Greedy, EnableThinking = false }, cancellationToken).ConfigureAwait(false))
        {
            if (ev.Output is { Kind: ChatOutputKind.Text } o)
                yield return new SpeechToTextResponseUpdate(o.Text) { Kind = SpeechToTextResponseUpdateKind.TextUpdating, ResponseId = id, ModelId = _metadata.DefaultModelId };
        }
        yield return new SpeechToTextResponseUpdate { Kind = SpeechToTextResponseUpdateKind.SessionClose, ResponseId = id };
    }

    internal static bool IsWav(byte[] b) => b.Length > 12 && b[0] == 'R' && b[1] == 'I' && b[2] == 'F' && b[3] == 'F';

    internal static float[] Pcm16ToFloat(ReadOnlySpan<byte> pcm)
    {
        var s = new float[pcm.Length / 2];
        for (int i = 0; i < s.Length; i++) s[i] = BitConverter.ToInt16(pcm.Slice(i * 2, 2)) / 32768f;
        return s;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey == null && serviceType == typeof(SpeechToTextClientMetadata) ? _metadata
        : serviceKey == null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}
