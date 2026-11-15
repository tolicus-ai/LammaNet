using LammaNet.Inference;
using LammaNet.Memory;
using Microsoft.Extensions.Logging;

namespace LammaNet.Diagnostics;

/// <summary>Log messages of Lamma.Net (source generated; ids grouped by area, see <see cref="LammaLogging"/>).</summary>
internal static partial class Log
{
    // ------------------------------------------------------------------ 1xxx engine

    [LoggerMessage(1000, LogLevel.Information, "Engine {EngineId} started: {Model} ({Architecture}, {Layers} layers), context {ContextSize}, KV cache {KvTokens} tokens ({KvType}), {MaxSequences} sequences, batch {BatchSize}, {Threads} CPU threads, {Device}")]
    public static partial void EngineStarted(ILogger logger, int engineId, string model, string architecture, int layers, int contextSize, int kvTokens,
        KvCacheType kvType, int maxSequences, int batchSize, int threads, string device);

    [LoggerMessage(1001, LogLevel.Information, "Engine {EngineId} stopped: {PromptTokens} prompt tokens ({CachedTokens} from the prefix cache), {GeneratedTokens} generated")]
    public static partial void EngineStopped(ILogger logger, int engineId, long promptTokens, long cachedTokens, long generatedTokens);

    [LoggerMessage(1002, LogLevel.Warning, "Engine {EngineId}: the scheduler did not stop within {Seconds} s of Dispose; its memory is released when the running step ends")]
    public static partial void SchedulerStopTimeout(ILogger logger, int engineId, int seconds);

    [LoggerMessage(1003, LogLevel.Critical, "Engine {EngineId}: the scheduler stopped unexpectedly; the engine is faulted and its requests fail")]
    public static partial void SchedulerCrashed(ILogger logger, Exception exception, int engineId);

    [LoggerMessage(1004, LogLevel.Error, "Engine {EngineId} could not be created for {Model}")]
    public static partial void EngineStartFailed(ILogger logger, Exception exception, int engineId, string model);

    [LoggerMessage(1010, LogLevel.Debug, "Engine {EngineId} request {RequestId} started: {Positions} prompt positions, {Reused} reused from the prefix cache")]
    public static partial void RequestStarted(ILogger logger, int engineId, long requestId, int positions, int reused);

    [LoggerMessage(1011, LogLevel.Warning, "Engine {EngineId} request {RequestId} rejected: {Reason}")]
    public static partial void RequestRejected(ILogger logger, Exception exception, int engineId, long requestId, string reason);

    [LoggerMessage(1012, LogLevel.Error, "Engine {EngineId} request {RequestId} could not start")]
    public static partial void RequestStartFailed(ILogger logger, Exception exception, int engineId, long requestId);

    [LoggerMessage(1013, LogLevel.Debug, "Engine {EngineId} request {RequestId} finished ({Reason}): {PromptTokens} prompt tokens ({CachedTokens} cached) in {PromptMs:F0} ms, {GeneratedTokens} tokens in {GenerationMs:F0} ms")]
    public static partial void RequestFinished(ILogger logger, int engineId, long requestId, FinishReason reason, int promptTokens, int cachedTokens,
        double promptMs, int generatedTokens, double generationMs);

    [LoggerMessage(1014, LogLevel.Warning, "Engine {EngineId} request {RequestId} stopped at {Positions} positions: the KV cache is full ({Pages} pages, {Active} active sequences)")]
    public static partial void KvCacheFull(ILogger logger, int engineId, long requestId, int positions, int pages, int active);

    [LoggerMessage(1015, LogLevel.Debug, "Engine {EngineId}: KV cache full, evicted a cached prompt prefix")]
    public static partial void PrefixEvicted(ILogger logger, int engineId);

    [LoggerMessage(1016, LogLevel.Debug, "Engine {EngineId} request {RequestId}: context shift discarded {Removed} positions")]
    public static partial void ContextShifted(ILogger logger, int engineId, long requestId, int removed);

    [LoggerMessage(1017, LogLevel.Warning, "Engine {EngineId} request {RequestId}: embedding stopped ({Reason})")]
    public static partial void EmbeddingStopped(ILogger logger, int engineId, long requestId, FinishReason reason);

    [LoggerMessage(1018, LogLevel.Error, "Engine {EngineId} request {RequestId} failed while sampling or decoding its output; other requests continue")]
    public static partial void RequestFailed(ILogger logger, Exception exception, int engineId, long requestId);

    [LoggerMessage(1020, LogLevel.Error, "Engine {EngineId}: inference step failed ({Rows} batch rows, requests {RequestIds}); these requests fail")]
    public static partial void StepFailed(ILogger logger, Exception exception, int engineId, int rows, string requestIds);

    [LoggerMessage(1021, LogLevel.Warning, "Engine {EngineId}: rejected a request: the engine is faulted")]
    public static partial void SubmitFaulted(ILogger logger, int engineId);

    [LoggerMessage(1030, LogLevel.Information, "Engine {EngineId}: MTP drafter {Name} on {Device}, up to {MaxDrafts} draft tokens per step")]
    public static partial void DrafterReady(ILogger logger, int engineId, string name, string device, int maxDrafts);

    [LoggerMessage(1031, LogLevel.Warning, "Engine {EngineId}: MTP drafter {Name} cannot run with this device split ({Reason}); generating without it")]
    public static partial void DrafterUnavailable(ILogger logger, int engineId, string name, string reason);

    // ------------------------------------------------------------------ 2xxx accelerator

    [LoggerMessage(2000, LogLevel.Information, "Accelerator {Device}: {Plan}")]
    public static partial void AcceleratorSelected(ILogger logger, string device, string plan);

    [LoggerMessage(2001, LogLevel.Information, "No accelerator, running on the CPU: {Reason}")]
    public static partial void AcceleratorUnavailable(ILogger logger, string reason);

    [LoggerMessage(2002, LogLevel.Warning, "Accelerator setup failed; running on the CPU")]
    public static partial void AcceleratorFailed(ILogger logger, Exception exception);

    [LoggerMessage(2003, LogLevel.Warning, "Loading the accelerator package failed; running on the CPU")]
    public static partial void AcceleratorDiscoveryFailed(ILogger logger, Exception exception);

    [LoggerMessage(2010, LogLevel.Information, "{Encoder} runs on the CPU: it needs {NeedMb} MB of device memory, {FreeMb} MB are free")]
    public static partial void EncoderNoDeviceMemory(ILogger logger, string encoder, long needMb, long freeMb);

    [LoggerMessage(2011, LogLevel.Warning, "{Encoder} could not be created on {Device}; it runs on the CPU")]
    public static partial void EncoderCreateFailed(ILogger logger, Exception exception, string encoder, string device);

    [LoggerMessage(2012, LogLevel.Debug, "Page-locking {Bytes} bytes of host memory failed; transfers use pageable memory")]
    public static partial void PinFailed(ILogger logger, Exception exception, long bytes);

    // ------------------------------------------------------------------ 3xxx model loading

    [LoggerMessage(3000, LogLevel.Information, "Loaded {Path}: {Name} ({Architecture}, {Layers} layers, {SizeMb} MB) in {Ms:F0} ms")]
    public static partial void ModelLoaded(ILogger logger, string path, string name, string architecture, int layers, long sizeMb, double ms);

    [LoggerMessage(3001, LogLevel.Error, "Failed to load model {Path}")]
    public static partial void ModelLoadFailed(ILogger logger, Exception exception, string path);

    [LoggerMessage(3010, LogLevel.Debug, "{Source}: the metadata does not describe a model configuration this version can read")]
    public static partial void ModelConfigUnreadable(ILogger logger, Exception exception, string source);

    // ------------------------------------------------------------------ 4xxx model cache

    [LoggerMessage(4000, LogLevel.Information, "Downloading {File} ({Bytes} bytes) from {Url}")]
    public static partial void DownloadStarting(ILogger logger, string file, long? bytes, string url);

    [LoggerMessage(4001, LogLevel.Information, "Downloaded {File}: {Bytes} bytes in {Seconds:F1} s")]
    public static partial void DownloadCompleted(ILogger logger, string file, long bytes, double seconds);

    [LoggerMessage(4002, LogLevel.Warning, "Download of {File} failed (attempt {Attempt} of {Attempts}); retrying")]
    public static partial void DownloadAttemptFailed(ILogger logger, Exception exception, string file, int attempt, int attempts);

    [LoggerMessage(4003, LogLevel.Error, "Download of {File} failed")]
    public static partial void DownloadFailed(ILogger logger, Exception exception, string file);

    [LoggerMessage(4004, LogLevel.Error, "Download of {File} is corrupt (SHA-256 mismatch); the partial file was deleted")]
    public static partial void DownloadCorrupt(ILogger logger, Exception exception, string file);

    [LoggerMessage(4005, LogLevel.Warning, "{Source}: the server is unreachable; using the cached copy {Path}")]
    public static partial void ServedFromCache(ILogger logger, Exception exception, string source, string path);

    [LoggerMessage(4006, LogLevel.Error, "{Source}: the server is unreachable and there is no cached copy")]
    public static partial void UnreachableNoCache(ILogger logger, Exception exception, string source);

    [LoggerMessage(4007, LogLevel.Debug, "Symbolic links are not available; cached files are hard-linked or moved instead")]
    public static partial void SymlinksUnsupported(ILogger logger, Exception? exception);

    [LoggerMessage(4008, LogLevel.Debug, "Cache file operation failed ({Operation} {Path})")]
    public static partial void CacheFileOperationFailed(ILogger logger, Exception exception, string operation, string path);

    [LoggerMessage(4009, LogLevel.Information, "Waiting for another process that is downloading {File}")]
    public static partial void WaitingForOtherDownload(ILogger logger, string file);

    [LoggerMessage(4010, LogLevel.Debug, "{Source} resolved to {Path} (cached: {Cached})")]
    public static partial void Resolved(ILogger logger, string source, string path, bool cached);

    // ------------------------------------------------------------------ 5xxx multimodal

    [LoggerMessage(5000, LogLevel.Information, "Loaded projector {Path} ({Type}); encoders run on {Device}")]
    public static partial void ProjectorLoaded(ILogger logger, string path, string type, string device);

    [LoggerMessage(5002, LogLevel.Error, "Failed to load projector {Path}")]
    public static partial void ProjectorLoadFailed(ILogger logger, Exception exception, string path);

    [LoggerMessage(5001, LogLevel.Warning, "{Encoder} failed on {Device}; this and later inputs are encoded on the CPU")]
    public static partial void EncoderDeviceFailed(ILogger logger, Exception exception, string encoder, string device);

    // ------------------------------------------------------------------ 6xxx chat and templates

    [LoggerMessage(6000, LogLevel.Error, "Building the prompt failed ({Messages} messages, {MediaParts} media parts)")]
    public static partial void PromptBuildFailed(ILogger logger, Exception exception, int messages, int mediaParts);

    [LoggerMessage(6001, LogLevel.Warning, "Malformed tool call in the model output ({Format}, {Length} chars): {Outcome}")]
    public static partial void MalformedToolCall(ILogger logger, Exception exception, string format, int length, string outcome);

    [LoggerMessage(6002, LogLevel.Debug, "The chat template cannot render {Modality} input")]
    public static partial void MediaMarkerUnavailable(ILogger logger, Exception exception, string modality);

    [LoggerMessage(6003, LogLevel.Debug, "Arguments of tool call {Name} are not JSON; the template receives them as a string")]
    public static partial void ToolArgumentsNotJson(ILogger logger, Exception exception, string name);

    [LoggerMessage(6004, LogLevel.Trace, "Template filter {Filter} could not convert its input; using the default")]
    public static partial void TemplateConversionDefault(ILogger logger, Exception exception, string filter);

    // ------------------------------------------------------------------ 7xxx Microsoft.Extensions.AI adapters

    [LoggerMessage(7000, LogLevel.Warning, "Chat option {Key} ignored: its value cannot be converted to {Type}")]
    public static partial void ChatOptionIgnored(ILogger logger, Exception exception, string key, string type);

    [LoggerMessage(7001, LogLevel.Debug, "Tool call arguments are not a JSON object; passed on as raw text")]
    public static partial void ToolArgumentsRaw(ILogger logger, Exception exception);

    [LoggerMessage(7010, LogLevel.Error, "Realtime session {SessionId}: transcription of {Seconds:F1} s of audio failed")]
    public static partial void RealtimeTranscriptionFailed(ILogger logger, Exception exception, int sessionId, double seconds);

    [LoggerMessage(7011, LogLevel.Error, "Realtime session {SessionId}: response {ResponseId} failed")]
    public static partial void RealtimeResponseFailed(ILogger logger, Exception exception, int sessionId, string responseId);

    [LoggerMessage(7012, LogLevel.Debug, "Realtime session {SessionId}: response {ResponseId} cancelled")]
    public static partial void RealtimeResponseCancelled(ILogger logger, int sessionId, string responseId);

    [LoggerMessage(7013, LogLevel.Error, "Realtime session {SessionId}: a response task failed outside the response loop")]
    public static partial void RealtimeTaskFailed(ILogger logger, Exception exception, int sessionId);

    [LoggerMessage(7014, LogLevel.Error, "Realtime session {SessionId}: processing a {Message} message failed")]
    public static partial void RealtimeMessageFailed(ILogger logger, Exception exception, int sessionId, string message);

    // ------------------------------------------------------------------ 8xxx compute

    [LoggerMessage(8000, LogLevel.Debug, "Physical core detection failed; using {Cores} compute threads")]
    public static partial void CpuTopologyFallback(ILogger logger, Exception exception, int cores);

    [LoggerMessage(8010, LogLevel.Debug, "Adaptive resource limit: {Threads} compute threads (other processes use {OtherCores:F1} cores)")]
    public static partial void AdaptiveThreads(ILogger logger, int threads, double otherCores);

    [LoggerMessage(8011, LogLevel.Information, "Adaptive resource limiting is not available on this platform (no system CPU counters)")]
    public static partial void AdaptiveUnavailable(ILogger logger);

    [LoggerMessage(8012, LogLevel.Debug, "Sampling the machine's CPU load failed")]
    public static partial void AdaptiveSampleFailed(ILogger logger, Exception exception);
}
