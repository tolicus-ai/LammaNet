using System.Runtime.CompilerServices;
using System.Threading.Channels;
using LammaNet.Chat;
using LammaNet.Sampling;
using Microsoft.Extensions.AI;
using Core = LammaNet.Chat;
using ChatRole = Microsoft.Extensions.AI.ChatRole;

namespace LammaNet.Extensions.AI;

/// <summary>
/// <see cref="IRealtimeClient"/> providing full-duplex, multimodal conversation sessions on a local model:
/// stream microphone audio (with voice-activity detection and barge-in), text and images in; receive streamed text
/// (and audio when a text-to-speech client is supplied) out, with tool calling.
/// </summary>
public sealed class LammaRealtimeClient : IRealtimeClient
{
    private readonly ChatEngine _chat;
    private readonly ITextToSpeechClient? _tts;

    /// <param name="chat">Chat engine (load an mmproj with audio support for speech input).</param>
    /// <param name="textToSpeech">Optional TTS used when audio output is requested.</param>
    public LammaRealtimeClient(ChatEngine chat, ITextToSpeechClient? textToSpeech = null)
    {
        _chat = chat;
        _tts = textToSpeech;
    }

    public LammaRealtimeClient(LammaChatClient client, ITextToSpeechClient? textToSpeech = null) : this(client.ChatEngine, textToSpeech) { }

    public Task<IRealtimeClientSession> CreateSessionAsync(RealtimeSessionOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.FromResult<IRealtimeClientSession>(new LammaRealtimeSession(_chat, options ?? new RealtimeSessionOptions(), _tts));

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey == null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}

/// <summary>A realtime conversation session (see <see cref="LammaRealtimeClient"/>).</summary>
public sealed class LammaRealtimeSession : IRealtimeClientSession
{
    private readonly ChatEngine _chat;
    private readonly ITextToSpeechClient? _tts;
    private readonly Channel<RealtimeServerMessage> _out = Channel.CreateUnbounded<RealtimeServerMessage>();
    private readonly List<Core.ChatMessage> _conversation = new();
    private readonly Dictionary<string, string> _callNames = new();
    private readonly List<float> _audio = new();       // 16 kHz input buffer
    private readonly object _lock = new();
    private readonly EnergyVad _vad = new();
    private readonly Microsoft.Extensions.Logging.ILogger _log;
    private static int s_sessionCount;
    private readonly int _id = Interlocked.Increment(ref s_sessionCount);
    private CancellationTokenSource? _responseCts;
    private Task? _responseTask;
    private bool _disposed;

    public RealtimeSessionOptions Options { get; private set; }

    internal LammaRealtimeSession(ChatEngine chat, RealtimeSessionOptions options, ITextToSpeechClient? tts)
    {
        _chat = chat;
        _log = LammaNet.Diagnostics.LammaLogging.Create<LammaRealtimeSession>(chat.Engine.LoggerFactory);
        _tts = tts;
        Options = options;
    }

    public IAsyncEnumerable<RealtimeServerMessage> GetStreamingResponseAsync(CancellationToken cancellationToken = default) =>
        _out.Reader.ReadAllAsync(cancellationToken);

    public async Task SendAsync(RealtimeClientMessage message, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        try { await HandleAsync(message, cancellationToken).ConfigureAwait(false); }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            LammaNet.Diagnostics.Log.RealtimeMessageFailed(_log, e, _id, message.GetType().Name);
            throw;
        }
    }

    private async Task HandleAsync(RealtimeClientMessage message, CancellationToken cancellationToken)
    {
        switch (message)
        {
            case SessionUpdateRealtimeClientMessage su:
                Options = su.Options;
                break;
            case CreateConversationItemRealtimeClientMessage ci:
                AddItem(ci.Item);
                break;
            case InputAudioBufferAppendRealtimeClientMessage ap:
                await AppendAudioAsync(ap.Content, cancellationToken).ConfigureAwait(false);
                break;
            case InputAudioBufferCommitRealtimeClientMessage:
                await CommitAudioAsync(cancellationToken).ConfigureAwait(false);
                break;
            case CreateResponseRealtimeClientMessage cr:
                StartResponse(cr);
                break;
            default:
                Emit(new ErrorRealtimeServerMessage { Error = new ErrorContent($"Unsupported client message {message.GetType().Name}"), OriginatingMessageId = message.MessageId });
                break;
        }
    }

    // ------------------------------------------------------------------ conversation items

    private void AddItem(RealtimeConversationItem item)
    {
        var role = item.Role == ChatRole.Assistant ? Core.ChatRole.Assistant : item.Role == ChatRole.System ? Core.ChatRole.System : Core.ChatRole.User;
        var parts = new List<ChatPart>();
        foreach (var c in item.Contents)
        {
            switch (c)
            {
                case TextContent t: parts.Add(new Core.TextPart(t.Text)); break;
                case DataContent d when d.HasTopLevelMediaType("audio"): parts.Add(ToAudioPart(d.Data.ToArray(), d.MediaType)); break;
                case DataContent d when d.HasTopLevelMediaType("image"): parts.Add(new ImagePart { EncodedBytes = d.Data.ToArray() }); break;
                case FunctionResultContent fr:
                    lock (_lock)
                        _conversation.Add(Core.ChatMessage.ToolResult(fr.CallId, _callNames.GetValueOrDefault(fr.CallId, ""), fr.Result?.ToString() ?? ""));
                    break;
                case FunctionCallContent fc:
                    lock (_lock)
                    {
                        _callNames[fc.CallId] = fc.Name;
                        _conversation.Add(new Core.ChatMessage
                        {
                            Role = Core.ChatRole.Assistant,
                            ToolCalls = [new ToolCall(fc.CallId, fc.Name, System.Text.Json.JsonSerializer.Serialize(fc.Arguments ?? new Dictionary<string, object?>(), AIJsonUtilities.DefaultOptions))],
                        });
                    }
                    break;
            }
        }
        if (parts.Count > 0) lock (_lock) _conversation.Add(new Core.ChatMessage { Role = role, Parts = parts });
        Emit(new ResponseOutputItemRealtimeServerMessage(RealtimeServerMessageType.ConversationItemAdded) { Item = item });
    }

    private AudioPart ToAudioPart(byte[] data, string mediaType)
    {
        if (LammaSpeechToTextClient.IsWav(data)) return new AudioPart { EncodedBytes = data };
        int rate = Options.InputAudioFormat?.SampleRate ?? 24000;
        return new AudioPart { Samples = LammaSpeechToTextClient.Pcm16ToFloat(data), SampleRate = rate };
    }

    // ------------------------------------------------------------------ audio input + VAD

    private async Task AppendAudioAsync(DataContent content, CancellationToken ct)
    {
        var part = ToAudioPart(content.Data.ToArray(), content.MediaType);
        float[] pcm = part.Samples ?? LammaNet.Multimodal.AudioProcessing.DecodeWav(part.EncodedBytes!).Samples;
        int rate = part.Samples != null ? part.SampleRate : LammaNet.Multimodal.AudioProcessing.DecodeWav(part.EncodedBytes!).SampleRate;
        if (rate != 16000) pcm = LammaNet.Multimodal.AudioProcessing.Resample(pcm, rate, 16000);
        lock (_lock) _audio.AddRange(pcm);

        if (Options.VoiceActivityDetection is not { Enabled: true } vad) return;
        foreach (var ev in _vad.Process(pcm))
        {
            if (ev == VadEvent.SpeechStarted && vad.AllowInterruption) CancelResponse();
            if (ev == VadEvent.SpeechEnded)
            {
                await CommitAudioAsync(ct).ConfigureAwait(false);
                StartResponse(new CreateResponseRealtimeClientMessage());
            }
        }
    }

    private async Task CommitAudioAsync(CancellationToken ct)
    {
        float[] pcm;
        lock (_lock)
        {
            if (_audio.Count == 0) return;
            pcm = _audio.ToArray();
            _audio.Clear();
        }
        string itemId = Guid.NewGuid().ToString("N");
        lock (_lock) _conversation.Add(new Core.ChatMessage { Role = Core.ChatRole.User, Parts = [new AudioPart { Samples = pcm, SampleRate = 16000 }] });
        Emit(new ResponseOutputItemRealtimeServerMessage(RealtimeServerMessageType.ConversationItemAdded)
        {
            Item = new RealtimeConversationItem([new DataContent(ToWav(pcm, 16000), "audio/wav")], itemId, ChatRole.User),
        });
        if (Options.TranscriptionOptions != null)
        {
            try
            {
                var stt = new LammaSpeechToTextClient(_chat);
                using var ms = new MemoryStream(ToWav(pcm, 16000));
                var text = await stt.GetTextAsync(ms, new SpeechToTextOptions { SpeechLanguage = Options.TranscriptionOptions.SpeechLanguage }, ct).ConfigureAwait(false);
                Emit(new InputAudioTranscriptionRealtimeServerMessage(RealtimeServerMessageType.InputAudioTranscriptionCompleted) { ItemId = itemId, Transcription = text.Text });
            }
            catch (Exception ex)
            {
                LammaNet.Diagnostics.Log.RealtimeTranscriptionFailed(_log, ex, _id, pcm.Length / 16000.0);
                Emit(new InputAudioTranscriptionRealtimeServerMessage(RealtimeServerMessageType.InputAudioTranscriptionFailed) { ItemId = itemId, Error = new ErrorContent(ex.Message) });
            }
        }
    }

    // ------------------------------------------------------------------ responses

    private void CancelResponse()
    {
        CancellationTokenSource? cts;
        lock (_lock) cts = _responseCts;
        cts?.Cancel();
    }

    private void StartResponse(CreateResponseRealtimeClientMessage request)
    {
        CancelResponse();
        var cts = new CancellationTokenSource();
        lock (_lock)   // responses run one after another, in request order
        {
            var previous = _responseTask;
            _responseCts = cts;
            _responseTask = Task.Run(() => RunAfterAsync(previous, request, cts.Token));
        }
    }

    private async Task RunAfterAsync(Task? previous, CreateResponseRealtimeClientMessage request, CancellationToken ct)
    {
        try
        {
            if (previous != null) await previous.ConfigureAwait(false);   // never faults: failures are logged below
            await RunResponseAsync(request, ct).ConfigureAwait(false);
        }
        catch (Exception e)
        {
            // nothing awaits this task: a failure here would otherwise go unobserved
            LammaNet.Diagnostics.Log.RealtimeTaskFailed(_log, e, _id);
            Emit(new ErrorRealtimeServerMessage { Error = new ErrorContent(e.Message) });
        }
    }

    private async Task RunResponseAsync(CreateResponseRealtimeClientMessage req, CancellationToken ct)
    {
        string responseId = Guid.NewGuid().ToString("N");
        string itemId = Guid.NewGuid().ToString("N");
        Emit(new ResponseCreatedRealtimeServerMessage(RealtimeServerMessageType.ResponseCreated) { ResponseId = responseId });

        List<Core.ChatMessage> convo;
        lock (_lock)
        {
            convo = new List<Core.ChatMessage>();
            string? instr = req.Instructions ?? Options.Instructions;
            if (!string.IsNullOrWhiteSpace(instr)) convo.Add(Core.ChatMessage.System(instr));
            convo.AddRange(_conversation);
            if (req.Items != null)
                foreach (var it in req.Items)
                    convo.Add(new Core.ChatMessage { Role = Core.ChatRole.User, Parts = it.Contents.OfType<TextContent>().Select(t => (ChatPart)new Core.TextPart(t.Text)).ToList() });
        }
        var tools = (req.Tools ?? Options.Tools?.ToList())?.OfType<AIFunctionDeclaration>()
            .Select(f => new ToolDefinition { Name = f.Name, Description = f.Description, Parameters = f.JsonSchema }).ToList();
        if (Options.ToolMode is NoneChatToolMode || req.ToolMode is NoneChatToolMode) tools = null;

        var text = new System.Text.StringBuilder();
        var calls = new List<ToolCall>();
        UsageDetails? usage = null;
        string status = RealtimeResponseStatus.Completed;
        var sentence = new System.Text.StringBuilder();
        bool wantAudio = _tts != null && ((IEnumerable<string>?)req.OutputModalities ?? Options.OutputModalities)?.Contains("audio") == true;
        try
        {
            await foreach (var ev in _chat.ChatAsync(convo, new ChatRequestOptions
            {
                Tools = tools is { Count: > 0 } ? tools : null,
                MaxTokens = req.MaxOutputTokens ?? Options.MaxOutputTokens,
                EnableThinking = false,
                Sampling = new SamplingOptions { Temperature = 0.7f },
            }, ct).ConfigureAwait(false))
            {
                if (ev.Output is { Kind: ChatOutputKind.Text } o)
                {
                    text.Append(o.Text);
                    Emit(new OutputTextAudioRealtimeServerMessage(RealtimeServerMessageType.OutputTextDelta) { ResponseId = responseId, ItemId = itemId, Text = o.Text });
                    if (wantAudio)
                    {
                        sentence.Append(o.Text);
                        if (EndsSentence(sentence)) { await SpeakAsync(sentence.ToString(), responseId, itemId, ct).ConfigureAwait(false); sentence.Clear(); }
                    }
                }
                else if (ev.Output is { Kind: ChatOutputKind.ToolCall, Call: { } call }) calls.Add(call);
                else if (ev.Stats is { } s)
                {
                    usage = new UsageDetails { InputTokenCount = s.PromptTokens, OutputTokenCount = s.CompletionTokens, TotalTokenCount = s.PromptTokens + s.CompletionTokens, CachedInputTokenCount = s.CachedPromptTokens };
                    if (s.FinishReason == LammaNet.Inference.FinishReason.Cancelled) status = RealtimeResponseStatus.Cancelled;
                }
            }
            if (wantAudio && sentence.Length > 0) await SpeakAsync(sentence.ToString(), responseId, itemId, ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            status = RealtimeResponseStatus.Cancelled;
            LammaNet.Diagnostics.Log.RealtimeResponseCancelled(_log, _id, responseId);
        }
        catch (Exception ex)
        {
            status = RealtimeResponseStatus.Failed;
            LammaNet.Diagnostics.Log.RealtimeResponseFailed(_log, ex, _id, responseId);
            Emit(new ErrorRealtimeServerMessage { Error = new ErrorContent(ex.Message) });
        }
        if (ct.IsCancellationRequested) status = RealtimeResponseStatus.Cancelled;

        var contents = new List<AIContent>();
        if (text.Length > 0)
        {
            Emit(new OutputTextAudioRealtimeServerMessage(RealtimeServerMessageType.OutputTextDone) { ResponseId = responseId, ItemId = itemId, Text = text.ToString() });
            contents.Add(new TextContent(text.ToString()));
        }
        foreach (var c in calls)
        {
            lock (_lock) _callNames[c.Id] = c.Name;
            contents.Add(new FunctionCallContent(c.Id, c.Name, MessageMapper.ParseArguments(c.ArgumentsJson, _log)));
        }
        // the (possibly interrupted) assistant turn stays in the conversation
        lock (_lock)
            if (text.Length > 0 || calls.Count > 0)
                _conversation.Add(new Core.ChatMessage { Role = Core.ChatRole.Assistant, Parts = text.Length > 0 ? [new Core.TextPart(text.ToString())] : [], ToolCalls = calls });
        var item = new RealtimeConversationItem(contents, itemId, ChatRole.Assistant);
        Emit(new ResponseOutputItemRealtimeServerMessage(RealtimeServerMessageType.ResponseOutputItemDone) { ResponseId = responseId, Item = item });
        Emit(new ResponseCreatedRealtimeServerMessage(RealtimeServerMessageType.ResponseDone) { ResponseId = responseId, Status = status, Usage = usage, Items = [item] });
    }

    private static bool EndsSentence(System.Text.StringBuilder sb)
    {
        if (sb.Length < 8) return false;
        char c = sb[^1];
        return c is '.' or '!' or '?' or '\n' or '。';
    }

    private async Task SpeakAsync(string text, string responseId, string itemId, CancellationToken ct)
    {
        if (_tts == null || string.IsNullOrWhiteSpace(text)) return;
        await foreach (var u in _tts.GetStreamingAudioAsync(text, null, ct).ConfigureAwait(false))
            foreach (var d in u.Contents.OfType<DataContent>())
                Emit(new OutputTextAudioRealtimeServerMessage(RealtimeServerMessageType.OutputAudioDelta)
                {
                    ResponseId = responseId, ItemId = itemId, Audio = Convert.ToBase64String(d.Data.Span),
                });
    }

    private void Emit(RealtimeServerMessage m) => _out.Writer.TryWrite(m);

    internal static byte[] ToWav(float[] pcm, int rate)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("RIFF"u8.ToArray()); w.Write(36 + pcm.Length * 2); w.Write("WAVE"u8.ToArray());
        w.Write("fmt "u8.ToArray()); w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(rate); w.Write(rate * 2); w.Write((short)2); w.Write((short)16);
        w.Write("data"u8.ToArray()); w.Write(pcm.Length * 2);
        foreach (var s in pcm) w.Write((short)Math.Clamp(s * 32767f, -32768f, 32767f));
        w.Flush();
        return ms.ToArray();
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey == null && serviceType.IsInstanceOfType(this) ? this : null;

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        CancelResponse();
        Task? running;
        lock (_lock) running = _responseTask;
        if (running != null) await running.ConfigureAwait(false);   // never faults (failures are logged inside)
        _out.Writer.TryComplete();
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();
}

internal enum VadEvent { SpeechStarted, SpeechEnded }

/// <summary>Adaptive energy-based voice activity detector (20 ms frames at 16 kHz).</summary>
internal sealed class EnergyVad
{
    private const int Frame = 320;
    private readonly List<float> _pending = new();
    private double _noise = 1e-4;
    private bool _inSpeech;
    private int _speechFrames, _silenceFrames;
    public int MinSpeechFrames { get; set; } = 3;      // 60 ms
    public int HangoverFrames { get; set; } = 35;      // 700 ms of silence ends a turn
    public double ThresholdRatio { get; set; } = 4.0;  // energy over noise floor

    public IEnumerable<VadEvent> Process(float[] samples)
    {
        _pending.AddRange(samples);
        var events = new List<VadEvent>();
        while (_pending.Count >= Frame)
        {
            double e = 0;
            for (int i = 0; i < Frame; i++) e += _pending[i] * _pending[i];
            _pending.RemoveRange(0, Frame);
            e /= Frame;
            bool speech = e > _noise * ThresholdRatio && e > 1e-6;
            if (!speech) _noise = 0.95 * _noise + 0.05 * e; // track the noise floor in silence
            if (speech)
            {
                _silenceFrames = 0;
                if (!_inSpeech && ++_speechFrames >= MinSpeechFrames) { _inSpeech = true; events.Add(VadEvent.SpeechStarted); }
            }
            else
            {
                _speechFrames = 0;
                if (_inSpeech && ++_silenceFrames >= HangoverFrames) { _inSpeech = false; _silenceFrames = 0; events.Add(VadEvent.SpeechEnded); }
            }
        }
        return events;
    }
}
