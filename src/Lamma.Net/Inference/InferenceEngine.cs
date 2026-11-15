using System.Diagnostics;
using System.Threading.Channels;
using LammaNet.Compute;
using LammaNet.Diagnostics;
using LammaNet.Memory;
using LammaNet.Models;
using LammaNet.Sampling;
using LammaNet.Tokenization;
using Microsoft.Extensions.Logging;

namespace LammaNet.Inference;

/// <summary>Configuration of an <see cref="InferenceEngine"/>.</summary>
public sealed class EngineOptions
{
    /// <summary>Maximum positions per sequence (prompt + generated). Defaults to min(training context, 8192).</summary>
    public int? ContextSize { get; init; }
    /// <summary>Total KV cache capacity in tokens across all sequences (default: ContextSize × 2).</summary>
    public int? KvCacheTokens { get; init; }
    /// <summary>Maximum number of sequences decoded concurrently (continuous batching).</summary>
    public int MaxConcurrentSequences { get; init; } = 4;
    /// <summary>Maximum positions evaluated per forward pass (prefill chunk size).</summary>
    public int BatchSize { get; init; } = 512;
    /// <summary>Compute threads (default: physical cores).</summary>
    public int? Threads { get; init; }
    public KvCacheType KvCacheType { get; init; } = KvCacheType.F16;
    public int KvPageSize { get; init; } = 32;
    /// <summary>Number of finished sequences kept for prompt-prefix reuse.</summary>
    public int PrefixCacheEntries { get; init; } = 8;
    /// <summary>
    /// Explicit accelerator offload (CPU/GPU layer split), e.g. <c>GpuLayerOffload.Factory(new() { Layers = 20 })</c>
    /// from the Lamma.Net.Gpu package; a factory may return null to run on the CPU. When null, see <see cref="AutoAccelerator"/>.
    /// </summary>
    public Func<Model, OffloadGeometry, ILayerOffload?>? Offload { get; init; }
    /// <summary>
    /// Without an explicit <see cref="Offload"/>, use an installed accelerator package automatically when a supported
    /// device is present (layers fitted to its free memory); false forces the CPU. See <see cref="Accelerators"/>.
    /// </summary>
    public bool AutoAccelerator { get; init; } = true;
    /// <summary>
    /// When a generating sequence reaches <see cref="ContextSize"/>, discard the oldest half of the context after the
    /// first <see cref="ContextShiftKeep"/> tokens (whole KV pages) and continue, instead of stopping. Keys of the kept
    /// tail are re-rotated to their new positions. Not available for recurrent/hybrid models or sequences with media.
    /// </summary>
    public bool ContextShift { get; init; }
    /// <summary>Tokens at the start kept by a context shift (system prompt); rounded up to whole KV pages.</summary>
    public int ContextShiftKeep { get; init; } = 1;
    /// <summary>
    /// Speculative decoding by prompt lookup: up to this many draft tokens are proposed from n-gram matches in the
    /// context and verified in one forward pass (exact for any sampler). 0 disables. Not used for recurrent/hybrid models.
    /// Default (null): 3 on the CPU; off with an accelerator offload, where verifying several tokens currently costs
    /// more than the accepted drafts save.
    /// </summary>
    public int? SpeculativeDraftTokens { get; init; }
    /// <summary>
    /// Logging for the engine and what it creates (model loading through <see cref="InferenceEngine.Load"/>, the accelerator,
    /// projectors, chat clients). Default: <see cref="LammaLogging.LoggerFactory"/>.
    /// </summary>
    public ILoggerFactory? LoggerFactory { get; init; }
    /// <summary>
    /// Limits on the machine's resources this engine shares with other engines and the application (threads, compute
    /// share, concurrent steps, device memory). Default: <see cref="ResourceLimiter.Default"/>.
    /// </summary>
    public ResourceLimiter? Resources { get; init; }
    /// <summary>
    /// A multi-token-prediction drafter GGUF for speculative decoding (Gemma 4: the <c>gemma4-assistant</c> published next
    /// to the model, e.g. <c>mtp-gemma-4-E2B-it.gguf</c>). Up to <see cref="SpeculativeDraftTokens"/> tokens are drafted per
    /// step and verified in one pass; the tokens are the model's own choices (verification batches several rows, so a
    /// near-tie can occasionally resolve differently than in one-token steps, as in llama.cpp). Default: 2 drafts on the CPU
    /// (1.5x generation for Gemma-4-E2B); with an accelerator drafting is off unless <see cref="SpeculativeDraftTokens"/> is set.
    /// </summary>
    public string? DraftModelPath { get; init; }
    /// <summary>MTP drafting stops before a draft token whose drafter probability is below this (0 = always draft the maximum).</summary>
    public float DraftMinProbability { get; init; }
}

/// <summary>Handle for a submitted request: a token stream plus final statistics.</summary>
public sealed class GenerationHandle
{
    internal GenerationHandle(ChannelReader<GeneratedToken> reader, Task<GenerationStats> completion, long requestId)
    {
        Tokens = reader;
        Completion = completion;
        RequestId = requestId;
    }

    /// <summary>Id of the request in the engine's log messages.</summary>
    public long RequestId { get; }

    public ChannelReader<GeneratedToken> Tokens { get; }
    public Task<GenerationStats> Completion { get; }
}

/// <summary>
/// Thread-safe inference engine over one <see cref="Model"/>. A background scheduler runs continuous batching:
/// every step it decodes one token for each active sequence and fills the remaining batch budget with chunks
/// of pending prompts (chunked prefill). Prompt prefixes are reused automatically from in-flight and recently
/// finished sequences (paged KV sharing; recurrent-state snapshots for hybrid models).
/// </summary>
public sealed class InferenceEngine : IDisposable
{
    private readonly Model _model;
    private readonly EngineOptions _opt;
    private readonly ComputePool _pool;
    private readonly KvCache _kv;
    private readonly ModelRunner _runner;
    private readonly Thread _thread;
    private readonly Channel<Work> _incoming = Channel.CreateUnbounded<Work>(new UnboundedChannelOptions { SingleReader = true });
    private readonly List<Active> _active = new();
    private readonly LinkedList<SequenceState> _retired = new();
    private readonly CancellationTokenSource _shutdown = new();
    private readonly object _statsLock = new();
    private volatile bool _disposed;
    private int _released;
    private readonly ILogger _log;
    private static int s_engineCount;
    private long _nextRequestId;
    private volatile Exception? _faulted;

    /// <summary>Id of this engine in log messages (engines of a process are numbered from 1).</summary>
    public int Id { get; } = Interlocked.Increment(ref s_engineCount);

    /// <summary>The logger factory of this engine (see <see cref="EngineOptions.LoggerFactory"/>).</summary>
    public ILoggerFactory LoggerFactory { get; }

    /// <summary>The resource limits this engine follows (see <see cref="EngineOptions.Resources"/>).</summary>
    public ResourceLimiter Resources { get; }

    public Model Model => _model;
    public ModelConfig Config => _model.Config;
    public Vocabulary Vocab => _model.Vocab;
    public int ContextSize { get; }
    public EngineOptions Options => _opt;

    /// <summary>Cumulative counters (useful for monitoring).</summary>
    public long TotalPromptTokens { get; private set; }
    public long TotalCachedPromptTokens { get; private set; }
    public long TotalGeneratedTokens { get; private set; }

    public InferenceEngine(Model model, EngineOptions? options = null)
    {
        _model = model;
        _opt = options ?? new EngineOptions();
        LoggerFactory = _opt.LoggerFactory ?? LammaLogging.LoggerFactory;
        _log = LammaLogging.Create<InferenceEngine>(LoggerFactory);
        Resources = _opt.Resources ?? ResourceLimiter.Default;
        ContextSize = _opt.ContextSize ?? Math.Min(model.Config.NCtxTrain > 0 ? model.Config.NCtxTrain : 4096, 8192);
        int kvTokens = _opt.KvCacheTokens ?? ContextSize * 2;
        ILayerOffload? off = null;
        try
        {
            _pool = new ComputePool(_opt.Threads ?? ComputePool.DefaultThreadCount(), Resources);
            int pageCount = Math.Max(1, (kvTokens + _opt.KvPageSize - 1) / _opt.KvPageSize);
            var geometry = new OffloadGeometry(_opt.KvPageSize, pageCount, _opt.BatchSize) { LoggerFactory = LoggerFactory, Resources = Resources };
            _offload = off = _opt.Offload != null ? _opt.Offload(model, geometry) : _opt.AutoAccelerator ? Accelerators.TryCreate(model, geometry) : null;
            _kv = new KvCache(model.Config, kvTokens, _opt.KvCacheType, _opt.KvPageSize, off == null ? null : l => !off.OwnsKv(l), off?.RecurrentStore);
            if (off != null) _kv.PageCopied += off.CopyPage;
            _runner = new ModelRunner(model, _kv, _pool, _opt.BatchSize, Math.Max(_opt.BatchSize, _opt.MaxConcurrentSequences), off);
        }
        catch (Exception e)
        {
            // release what exists (compute threads, KV memory, the device context) instead of leaking it with the half-built engine
            Log.EngineStartFailed(_log, e, Id, model.Config.Name);
            _runner?.Dispose();
            _kv?.Dispose();
            off?.Dispose();
            _pool?.Dispose();
            throw;
        }
        if (_opt.DraftModelPath is { } draftPath)
        {
            try { _drafter = CreateDrafter(draftPath, model, off); }
            catch
            {
                _runner.Dispose(); _kv.Dispose(); off?.Dispose(); _pool.Dispose();
                throw;
            }
        }
        // MTP: 2 drafts on the CPU (verifying more rows costs more than they save there); on an accelerator only when
        // asked for: verifying k+1 rows still costs about k+1 single-token steps there
        _draftTokens = _opt.SpeculativeDraftTokens ?? (off == null ? (_drafter != null ? 2 : 3) : 0);
        if (_drafter != null) Log.DrafterReady(_log, Id, _drafterModel!.Name, _drafter.Device, _draftTokens);
        Log.EngineStarted(_log, Id, model.Config.Name, model.Config.Architecture, model.Config.NLayer, ContextSize, kvTokens, _opt.KvCacheType,
            _opt.MaxConcurrentSequences, _opt.BatchSize, _pool.ThreadCount,
            off == null ? "CPU only" : $"{off.DeviceName}: {off.LayerCount}/{model.Config.NLayer} layers on the device");
        _thread = new Thread(SchedulerLoop) { IsBackground = true, Name = "lamma-scheduler" };
        _thread.Start();
    }

    /// <summary>Loads a GGUF model and creates an engine for it.</summary>
    public static InferenceEngine Load(string path, EngineOptions? options = null, ModelLoadOptions? loadOptions = null)
    {
        var load = loadOptions ?? new ModelLoadOptions();
        if (load.LoggerFactory == null && options?.LoggerFactory != null)
            load = new ModelLoadOptions { Prefetch = load.Prefetch, LoggerFactory = options.LoggerFactory };
        var model = Model.Load(path, load);
        try { return new(model, options) { _ownsModel = true }; }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    private bool _ownsModel;
    private readonly int _draftTokens;
    private readonly IDraftRunner? _drafter;
    private MtpDrafterModel? _drafterModel;
    private long _forwardCount;   // forward passes so far (an output row index is only valid for the latest one)

    /// <summary>Loads the MTP drafter and places it with the target's KV cache (device or host).</summary>
    private IDraftRunner? CreateDrafter(string path, Model model, ILayerOffload? off)
    {
        var dm = MtpDrafterModel.Load(path);
        try
        {
            dm.Validate(model);
            IDraftRunner? runner = null;
            string? why = null;
            if (off != null && off.ComputesHead) runner = off.CreateMtpDrafter(dm, model) ?? throw new NotSupportedException("the accelerator cannot run this drafter");
            else if (off == null || Enumerable.Range(0, dm.Layers.Length).All(il => !off.OwnsKv(dm.TargetKvLayer(il, model.Config))))
                runner = new CpuMtpDrafter(dm, model, _kv, _runner, _pool);
            else why = "the KV cache it reads is on the device but the output head is not";
            if (runner == null)
            {
                Log.DrafterUnavailable(_log, Id, dm.Name, why ?? "unsupported");
                dm.Dispose();
                return null;
            }
            _drafterModel = dm;
            return runner;
        }
        catch
        {
            dm.Dispose();
            throw;
        }
    }
    private readonly ILayerOffload? _offload;

    /// <summary>The accelerator running the first layers, if any.</summary>
    public ILayerOffload? Offload => _offload;

    // ===================================================================== public API

    /// <summary>Submits a request; tokens are streamed through the returned handle.</summary>
    public GenerationHandle Submit(GenerationRequest request, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ThrowIfFaulted();
        var ch = Channel.CreateUnbounded<GeneratedToken>(new UnboundedChannelOptions { SingleWriter = true });
        var tcs = new TaskCompletionSource<GenerationStats>(TaskCreationOptions.RunContinuationsAsynchronously);
        var work = new Work { Id = Interlocked.Increment(ref _nextRequestId), Request = request, Output = ch.Writer, Completion = tcs, Cancel = cancellationToken };
        if (!_incoming.Writer.TryWrite(work)) { ThrowIfFaulted(); throw new InvalidOperationException("Engine is shutting down"); }
        return new GenerationHandle(ch.Reader, tcs.Task, work.Id);
    }

    private void ThrowIfFaulted()
    {
        if (_faulted is not { } f) return;
        Log.SubmitFaulted(_log, Id);
        throw new InvalidOperationException($"Engine {Id} is faulted: {f.Message}", f);
    }

    /// <summary>Streams generated tokens.</summary>
    public async IAsyncEnumerable<GeneratedToken> GenerateAsync(GenerationRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var h = Submit(request, cancellationToken);
        await foreach (var t in h.Tokens.ReadAllAsync(CancellationToken.None))
            yield return t;
        await h.Completion.ConfigureAwait(false);
    }

    /// <summary>Generates a complete text (convenience).</summary>
    public async Task<(string Text, GenerationStats Stats)> GenerateTextAsync(GenerationRequest request, CancellationToken cancellationToken = default)
    {
        var h = Submit(request, cancellationToken);
        var sb = new System.Text.StringBuilder();
        await foreach (var t in h.Tokens.ReadAllAsync(CancellationToken.None)) sb.Append(t.Text);
        return (sb.ToString(), await h.Completion.ConfigureAwait(false));
    }

    /// <summary>Computes a pooled embedding of the prompt (normalized hidden state of the last layer).</summary>
    public Task<float[]> EmbedAsync(Prompt prompt, EmbeddingPooling pooling = EmbeddingPooling.Mean, bool normalize = true, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ThrowIfFaulted();
        var tcs = new TaskCompletionSource<GenerationStats>(TaskCreationOptions.RunContinuationsAsynchronously);
        var result = new TaskCompletionSource<float[]>(TaskCreationOptions.RunContinuationsAsynchronously);
        var work = new Work
        {
            Id = Interlocked.Increment(ref _nextRequestId),
            Request = new GenerationRequest { Prompt = prompt, CachePrompt = false },
            Completion = tcs, Cancel = cancellationToken, Embedding = result, Pooling = pooling, Normalize = normalize,
        };
        if (!_incoming.Writer.TryWrite(work)) { ThrowIfFaulted(); throw new InvalidOperationException("Engine is shutting down"); }
        return result.Task;
    }

    /// <summary>Tokenizes text with the model's vocabulary.</summary>
    public List<int> Tokenize(string text, bool addSpecial = true, bool parseSpecial = true) => _model.Vocab.Tokenize(text, addSpecial, parseSpecial);

    // ===================================================================== internals

    private sealed class Work
    {
        public long Id;
        public required GenerationRequest Request;
        public ChannelWriter<GeneratedToken>? Output;
        public required TaskCompletionSource<GenerationStats> Completion;
        public CancellationToken Cancel;
        public TaskCompletionSource<float[]>? Embedding;
        public EmbeddingPooling Pooling;
        public bool Normalize;
    }

    private enum Phase { Prefill, Decode }

    private sealed class Active
    {
        public required Work Work;
        public required SequenceState Seq;
        public required List<int> Keys;            // position keys of the full prompt
        public required List<(int Token, float[]? Emb, int[]? MPos)> Inputs; // per prompt position
        public int Cursor;                         // next prompt position to evaluate
        public int Cached;
        public Phase Phase = Phase.Prefill;
        public Sampler? Sampler;
        public StreamingDecoder? Decoder;
        public StopDetector? Stops;
        public int LastToken;
        public int Generated;
        public Stopwatch Clock = Stopwatch.StartNew();
        public TimeSpan PromptTime;
        public float[]? EmbAccum;
        public int EmbCount;
        public FinishReason Finish;
        public int? RopeShift;        // M-RoPE: rope position offset for generated tokens
        public int[]? Draft;          // draft tokens proposed this step
        public int DraftBase;         // sequence length before this step
        public bool DraftDone;        // verification ended (mismatch or finish)
        public int DraftAccepted;     // statistics
        public int OutRow = -1;       // output row of the last sampled token in forward pass OutRowForward (MTP seed)
        public long OutRowForward = -1;
    }

    /// <summary>Draft tokens accepted by speculative decoding (cumulative).</summary>
    public long TotalAcceptedDraftTokens { get; private set; }

    /// <summary>Draft tokens proposed for verification (cumulative; acceptance = accepted / drafted).</summary>
    public long TotalDraftTokens { get; private set; }

    /// <summary>MTP drafting for <paramref name="a"/>: the drafter continues from its last sampled token and that row's hidden state.</summary>
    private int[]? MtpDraft(Active a, int maxTokens)
    {
        if (maxTokens <= 0 || a.OutRowForward != _forwardCount || a.OutRow < 0) return null;
        Span<int> buf = stackalloc int[maxTokens];
        int n;
        using (Profiler.Section("engine.mtp_draft")) n = _drafter!.Draft(a.Seq, a.Seq.Length, a.LastToken, a.OutRow, _opt.DraftMinProbability, buf);
        return n > 0 ? buf[..n].ToArray() : null;
    }

    /// <summary>Prompt-lookup drafting: continue the most recent earlier occurrence of the trailing n-gram.</summary>
    private int[]? FindDraft(Active a, int maxTokens)
    {
        if (maxTokens <= 0) return null;
        var h = a.Seq.TokenList;
        int len = h.Count;
        if (len < 4) return null;
        for (int ngram = 4; ngram >= 2; ngram--)
        {
            // suffix = h[len-ngram+1 .. len-1] + LastToken
            for (int start = len - ngram; start >= 0; start--)
            {
                bool ok = h[start + ngram - 1] == a.LastToken;
                for (int j = 0; ok && j < ngram - 1; j++) ok = h[start + j] == h[len - ngram + 1 + j];
                if (!ok) continue;
                int from = start + ngram;
                int cnt = Math.Min(maxTokens, len - from);
                if (cnt <= 0) continue;
                var d = new int[cnt];
                for (int j = 0; j < cnt; j++)
                {
                    d[j] = h[from + j];
                    if (d[j] < 0) return null; // media positions are not tokens
                }
                return d;
            }
        }
        return null;
    }

    private void SchedulerLoop()
    {
        try { RunScheduler(); }
        catch (Exception ex)
        {
            // only a failure inside the failure handling itself gets here: no request can be served any more
            _faulted = ex;
            Log.SchedulerCrashed(_log, ex, Id);
            _incoming.Writer.TryComplete();
            foreach (var a in _active.ToList())
                try { Fail(a, ex); } catch (Exception) { /* keep failing the others */ }
            while (_incoming.Reader.TryRead(out var w)) FailWork(w, ex);
        }
        finally
        {
            // requests submitted but never admitted would otherwise wait forever
            while (_incoming.Reader.TryRead(out var w))
            {
                w.Output?.TryWrite(new GeneratedToken(-1, "", FinishReason.Cancelled));
                w.Output?.TryComplete();
                w.Completion.TrySetResult(new GenerationStats { FinishReason = FinishReason.Cancelled });
                w.Embedding?.TrySetCanceled();
            }
            // Dispose stopped waiting for this thread: the native memory goes now that no step uses it any more
            if (_disposed) ReleaseResources();
        }
    }

    private void RunScheduler()
    {
        var batch = new Batch();
        var rowOwners = new List<(Active A, bool PromptDone, int Pos)>();
        while (!_shutdown.IsCancellationRequested)
        {
            try
            {
                Admit();
                if (_active.Count == 0)
                {
                    try
                    {
                        if (!_incoming.Reader.WaitToReadAsync(_shutdown.Token).AsTask().GetAwaiter().GetResult()) break;
                    }
                    catch (OperationCanceledException) { break; }
                    continue;
                }

                // drop cancelled
                foreach (var a in _active.ToList())
                    if (a.Work.Cancel.IsCancellationRequested) Finish(a, FinishReason.Cancelled);

                if (_active.Count == 0) continue;
                // one step slot (ResourceLimiter.MaxConcurrentSteps: engines sharing the limiter take turns) and one
                // utilization interval cover the whole step: building the batch (MTP drafting), the forward pass, the outputs
                long stepStart = Stopwatch.GetTimestamp();
                ResourceLimiter.StepSlot slot;
                try { slot = Resources.EnterStep(_shutdown.Token); }
                catch (OperationCanceledException) when (_shutdown.IsCancellationRequested) { break; }
                using (slot)
                {
                    using (Profiler.Section("engine.build")) BuildBatch(batch, rowOwners);
                    if (batch.Count == 0) continue;
                    bool embedOnly = rowOwners.All(r => r.A.Work.Embedding != null);
                    try
                    {
                        _runner.Forward(batch, embedOnly);
                        _forwardCount++;
                    }
                    catch (KvCacheFullException)
                    {
                        RollbackBatch(batch);
                        if (EvictRetired()) { Log.PrefixEvicted(_log, Id); continue; }
                        // out of memory: stop the sequence holding the most pages
                        var victim = _active.OrderByDescending(x => x.Seq.Length).First();
                        Log.KvCacheFull(_log, Id, victim.Work.Id, victim.Seq.Length, _kv.PageCount, _active.Count);
                        Finish(victim, FinishReason.ContextFull);
                        continue;
                    }
                    using (Profiler.Section("engine.outputs")) ProcessOutputs(rowOwners);
                }
                Resources.PauseAfterStep(Stopwatch.GetElapsedTime(stepStart), _shutdown.Token);   // ResourceLimiter.MaxUtilization
            }
            catch (Exception ex)
            {
                Log.StepFailed(_log, ex, Id, batch.Count, string.Join(", ", _active.Select(a => a.Work.Id)));
                foreach (var a in _active.ToList()) Fail(a, ex);
            }
        }
        foreach (var a in _active.ToList()) Finish(a, FinishReason.Cancelled);
    }

    /// <summary>Context shift of a full sequence (see <see cref="EngineOptions.ContextShift"/>).</summary>
    private bool TryShiftContext(Active a)
    {
        if (!_opt.ContextShift || Config.HasRecurrentLayers || a.RopeShift != null || a.Seq.TokenList.Any(t => t < 0)) return false;
        if (Config.Rope.Any(r => r != null && r.IsMulti)) return false;
        int ps = _kv.PageSize;
        int keepPages = (Math.Max(1, _opt.ContextShiftKeep) + ps - 1) / ps;
        int discardPages = (a.Seq.Length / ps - keepPages) / 2;
        if (discardPages <= 0) return false;
        int removed = a.Seq.DiscardPages(keepPages, discardPages);
        int from = keepPages * ps;
        _kv.RotateKeys(a.Seq.PageTable, from, a.Seq.Length, -removed);
        _offload?.RotateKeys(a.Seq.PageTable, from, a.Seq.Length, -removed);
        a.Seq.Snapshots.Clear();
        TotalContextShifts++;
        Log.ContextShifted(_log, Id, a.Work.Id, removed);
        return true;
    }

    /// <summary>Number of context shifts performed (cumulative).</summary>
    public long TotalContextShifts { get; private set; }

    private readonly List<(Active A, int Cursor)> _cursorUndo = new();

    /// <summary>Undo prompt-cursor advances when a forward pass could not run (KV cache full; nothing was committed).</summary>
    private void RollbackBatch(Batch batch)
    {
        foreach (var (a, c) in _cursorUndo) a.Cursor = c;
        _cursorUndo.Clear();
    }

    private void Admit()
    {
        while (_active.Count < _opt.MaxConcurrentSequences && _incoming.Reader.TryRead(out var w))
        {
            if (w.Cancel.IsCancellationRequested)
            {
                w.Output?.TryComplete();
                w.Completion.TrySetCanceled(w.Cancel);
                w.Embedding?.TrySetCanceled(w.Cancel);
                continue;
            }
            try { _active.Add(Start(w)); }
            catch (Exception ex)
            {
                if (ex is ArgumentException or NotSupportedException) Log.RequestRejected(_log, ex, Id, w.Id, ex.Message);
                else Log.RequestStartFailed(_log, ex, Id, w.Id);
                FailWork(w, ex);
            }
        }
    }

    private static void FailWork(Work w, Exception ex)
    {
        w.Output?.TryWrite(new GeneratedToken(-1, "", FinishReason.Error));
        w.Output?.TryComplete(ex);
        w.Completion.TrySetException(ex);
        w.Embedding?.TrySetException(ex);
    }

    private Active Start(Work w)
    {
        var prompt = w.Request.Prompt;
        var keys = prompt.PositionKeys();
        if (keys.Count == 0) throw new ArgumentException("Prompt is empty");
        if (keys.Count >= ContextSize) throw new ArgumentException($"Prompt ({keys.Count} positions) exceeds the context size ({ContextSize})");
        if (!Config.Causal && keys.Count > _opt.BatchSize)
            throw new ArgumentException($"Input ({keys.Count} positions) exceeds the batch size ({_opt.BatchSize}); bidirectional models process the whole input in one batch");

        // flatten inputs; models with multi-section rope (Qwen-VL / Qwen3.5) get explicit 4D positions
        bool mrope = Config.Rope.Any(r => r?.IsMulti == true);
        var inputs = new List<(int, float[]?, int[]?)>(keys.Count);
        int pos = 0, ropePos = 0;
        foreach (var seg in prompt.Segments)
        {
            if (seg is TokenSegment ts)
                foreach (var t in ts.Tokens)
                {
                    inputs.Add((t, null, mrope ? [ropePos, ropePos, ropePos, ropePos] : null));
                    pos++; ropePos++;
                }
            else if (seg is EmbeddingSegment es)
            {
                if (es.Dim != Config.NEmbd) throw new ArgumentException($"Embedding dim {es.Dim} does not match model ({Config.NEmbd})");
                for (int i = 0; i < es.Rows; i++)
                {
                    int[]? mp = null;
                    if (mrope)
                        mp = es.RopePositions?[i] is int[] rp
                            ? [ropePos + rp[0], ropePos + rp[1], ropePos + rp[2], rp.Length > 3 ? rp[3] : 0]
                            : [ropePos + i, ropePos + i, ropePos + i, ropePos + i];
                    inputs.Add((keys[pos], es.Row(i), mp));
                    pos++;
                }
                ropePos += mrope ? (es.RopeSpan ?? es.Rows) : es.Rows;
            }
        }
        int ropeShift = ropePos - pos; // rope position = KV position + shift for generated tokens

        var seq = new SequenceState(_kv, Config);
        int reuse = 0;
        if (w.Embedding == null) reuse = ReusePrefix(seq, keys);
        var a = new Active { Work = w, Seq = seq, Keys = keys, Inputs = inputs, Cursor = reuse, Cached = reuse, RopeShift = mrope ? ropeShift : (int?)null };
        if (w.Embedding == null)
        {
            a.Sampler = new Sampler(w.Request.Sampling) { Constraint = w.Request.Constraint, EndTokens = _model.Vocab.EndOfGeneration };
            a.Sampler.AcceptPrompt(keys.Where(k => k >= 0));
            a.Decoder = _model.Vocab.CreateStreamingDecoder(w.Request.RenderSpecialTokens);
            a.Stops = new StopDetector(w.Request.StopSequences);
        }
        Log.RequestStarted(_log, Id, w.Id, keys.Count, reuse);
        return a;
    }

    /// <summary>Finds the longest reusable prefix among active and retired sequences and copies it into <paramref name="seq"/>.</summary>
    private int ReusePrefix(SequenceState seq, List<int> keys)
    {
        int limit = keys.Count - 1; // always evaluate at least one position to get logits
        SequenceState? best = null;
        int bestLen = 0;
        IEnumerable<SequenceState> candidates = _retired.Concat(_active.Select(x => x.Seq));
        foreach (var c in candidates)
        {
            int lcp = 0, max = Math.Min(Math.Min(c.Length, c.TokenList.Count), limit);
            while (lcp < max && c.TokenList[lcp] == keys[lcp]) lcp++;
            int usable = lcp;
            if (Config.HasRecurrentLayers)
            {
                // only positions where a recurrent snapshot exists can be resumed
                usable = 0;
                if (c.Length <= lcp && c.Recurrent != null) usable = c.Length;
                foreach (var k in c.Snapshots.Keys) if (k <= lcp && k > usable) usable = k;
            }
            if (usable > bestLen) { bestLen = usable; best = c; }
        }
        if (best == null || bestLen == 0) return 0;
        // make room if necessary
        while (!seq.CopyPrefixFrom(best, bestLen, Config))
            if (!EvictRetired(except: best)) return 0;
        var node = _retired.Find(best);
        if (node != null) { _retired.Remove(node); _retired.AddFirst(node); }
        return bestLen;
    }

    private bool EvictRetired(SequenceState? except = null)
    {
        for (var n = _retired.Last; n != null; n = n.Previous)
        {
            if (n.Value == except) continue;
            n.Value.Dispose();
            _retired.Remove(n);
            return true;
        }
        return false;
    }

    private static int[]? MPos(Active a, int kvPos)
    {
        if (a.RopeShift is not int sh) return null;
        int r = kvPos + sh;
        return [r, r, r, r];
    }

    private void BuildBatch(Batch batch, List<(Active, bool, int)> owners)
    {
        batch.Clear();
        owners.Clear();
        int budget = _opt.BatchSize;
        // decode steps first (latency), then prefill chunks
        _cursorUndo.Clear();
        foreach (var a in _active.ToList())
        {
            if (a.Phase != Phase.Decode || budget == 0) continue;
            if (a.Seq.Length + 1 > ContextSize && !TryShiftContext(a)) { Finish(a, FinishReason.Length); continue; }
            a.Draft = null; a.DraftDone = false; a.DraftBase = a.Seq.Length;
            if (_draftTokens > 0 && !Config.HasRecurrentLayers && a.Work.Request.Constraint == null)
            {
                int room = Math.Min(Math.Min(_draftTokens, budget - 1), ContextSize - a.Seq.Length - 1);
                if (a.Work.Request.MaxTokens is int mt) room = Math.Min(room, mt - a.Generated - 1);
                a.Draft = _drafter != null ? MtpDraft(a, room) : FindDraft(a, room);
                if (a.Draft != null) TotalDraftTokens += a.Draft.Length;
            }
            batch.Add(a.Seq, a.LastToken, a.Seq.Length, true, null, MPos(a, a.Seq.Length));
            owners.Add((a, true, a.Seq.Length));
            budget--;
            if (a.Draft != null)
                for (int j = 0; j < a.Draft.Length; j++)
                {
                    batch.Add(a.Seq, a.Draft[j], a.DraftBase + 1 + j, true, null, MPos(a, a.DraftBase + 1 + j));
                    owners.Add((a, true, a.DraftBase + 1 + j));
                    budget--;
                }
        }
        foreach (var a in _active.ToList())
        {
            if (a.Phase != Phase.Prefill || budget == 0) continue;
            int n = Math.Min(budget, a.Inputs.Count - a.Cursor);
            bool embed = a.Work.Embedding != null;
            // bidirectional models see the whole input at once: never split a prompt across batches
            if (!Config.Causal && n < a.Inputs.Count - a.Cursor) continue;
            for (int i = 0; i < n; i++)
            {
                int p = a.Cursor + i;
                bool lastOfPrompt = p == a.Inputs.Count - 1;
                bool want = embed ? (a.Work.Pooling == EmbeddingPooling.Mean || lastOfPrompt) : lastOfPrompt;
                var (tok, emb, mp) = a.Inputs[p];
                batch.Add(a.Seq, tok, p, want, emb, mp);
                if (want) owners.Add((a, lastOfPrompt, p));
            }
            _cursorUndo.Add((a, a.Cursor));
            a.Cursor += n;
            budget -= n;
        }
    }

    private void ProcessOutputs(List<(Active A, bool PromptDone, int Pos)> owners)
    {
        for (int row = 0; row < owners.Count; row++)
        {
            var a = owners[row].A;
            if (a.Finish != FinishReason.None) continue;
            // a failure in one request's sampling / constraint / detokenization fails that request only
            try { ProcessRow(owners[row], row); }
            catch (Exception ex)
            {
                Log.RequestFailed(_log, ex, Id, a.Work.Id);
                Fail(a, ex);
            }
        }
    }

    private unsafe void ProcessRow((Active A, bool PromptDone, int Pos) owner, int row)
    {
        var (a, promptDone, rowPos) = owner;
        if (a.Phase == Phase.Decode && a.Draft != null && a.DraftDone) return; // rejected draft rows
        if (a.Work.Embedding != null)
        {
            int d = _runner.EmbeddingSize;
            a.EmbAccum ??= new float[d];
            float* h = _runner.HiddenPtr(row);
            if (a.Work.Pooling == EmbeddingPooling.Last) { for (int i = 0; i < d; i++) a.EmbAccum[i] = h[i]; a.EmbCount = 1; }
            else { for (int i = 0; i < d; i++) a.EmbAccum[i] += h[i]; a.EmbCount++; }
            if (promptDone) CompleteEmbedding(a);
            return;
        }
        if (a.Phase == Phase.Prefill)
        {
            if (!promptDone) return;
            a.Phase = Phase.Decode;
            a.PromptTime = a.Clock.Elapsed;
            a.Clock.Restart();
            if (Config.HasRecurrentLayers) a.Seq.Snapshot(Config);
        }
        int tok = a.Sampler!.Sample(_runner.Logits(row));
        a.Sampler.Accept(tok);
        a.OutRow = row;
        a.OutRowForward = _forwardCount;
        a.Generated++;
        a.LastToken = tok;
        if (a.Draft != null)
        {
            int i = rowPos - a.DraftBase;
            bool match = i < a.Draft.Length && tok == a.Draft[i];
            if (match) { a.DraftAccepted++; TotalAcceptedDraftTokens++; }
            else
            {
                // KV for rejected draft positions is discarded (paged cache: just drop the tail)
                a.Seq.Truncate(a.DraftBase + i + 1);
                a.DraftDone = true;
            }
        }
        var req = a.Work.Request;
        bool eog = !req.IgnoreEos && (_model.Vocab.IsEog(tok) || (req.StopTokens?.Contains(tok) ?? false));
        if (eog)
        {
            Emit(a, a.Decoder!.Flush());
            Finish(a, FinishReason.Stop);
            return;
        }
        string text = a.Stops!.Push(a.Decoder!.Push(tok));
        if (text.Length > 0 || req.RenderSpecialTokens) Emit(a, text, tok);
        if (a.Stops.Stopped) { Finish(a, FinishReason.StopSequence); return; }
        if (req.Constraint?.IsComplete == true) { Finish(a, FinishReason.ConstraintComplete); return; }
        if (req.MaxTokens is int mt && a.Generated >= mt) { Finish(a, FinishReason.Length); return; }
    }

    private static void Emit(Active a, string text, int token = -1)
    {
        if (text.Length == 0 && token < 0) return;
        a.Work.Output?.TryWrite(new GeneratedToken(token, text));
    }

    private void CompleteEmbedding(Active a)
    {
        var v = a.EmbAccum!;
        if (a.EmbCount > 1) for (int i = 0; i < v.Length; i++) v[i] /= a.EmbCount;
        if (a.Work.Normalize)
        {
            double n = 0;
            foreach (var x in v) n += x * x;
            float inv = n > 0 ? (float)(1 / Math.Sqrt(n)) : 0;
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }
        a.Work.Embedding!.TrySetResult(v);
        a.Finish = FinishReason.Stop;
        a.Work.Completion.TrySetResult(new GenerationStats { PromptTokens = a.Keys.Count, PromptTime = a.Clock.Elapsed, FinishReason = FinishReason.Stop });
        _active.Remove(a);
        a.Seq.Dispose();
    }

    private void Finish(Active a, FinishReason reason)
    {
        if (a.Finish != FinishReason.None) return;
        a.Finish = reason;
        _active.Remove(a);
        if (a.Work.Embedding != null)
        {
            if (reason == FinishReason.Cancelled) a.Work.Embedding.TrySetCanceled();
            else
            {
                Log.EmbeddingStopped(_log, Id, a.Work.Id, reason);
                a.Work.Embedding.TrySetException(new InvalidOperationException($"Embedding failed: {reason}"));
            }
            a.Work.Completion.TrySetResult(new GenerationStats { FinishReason = reason });
            a.Seq.Dispose();
            return;
        }
        if (reason != FinishReason.StopSequence)
        {
            var rest = a.Stops?.Flush() ?? "";
            if (rest.Length > 0) Emit(a, rest);
        }
        var stats = new GenerationStats
        {
            PromptTokens = a.Keys.Count,
            CachedPromptTokens = a.Cached,
            CompletionTokens = a.Generated,
            PromptTime = a.Phase == Phase.Prefill ? a.Clock.Elapsed : a.PromptTime,
            GenerationTime = a.Phase == Phase.Decode ? a.Clock.Elapsed : TimeSpan.Zero,
            FinishReason = reason,
        };
        lock (_statsLock)
        {
            TotalPromptTokens += a.Keys.Count;
            TotalCachedPromptTokens += a.Cached;
            TotalGeneratedTokens += a.Generated;
        }
        Log.RequestFinished(_log, Id, a.Work.Id, reason, stats.PromptTokens, stats.CachedPromptTokens, stats.PromptTime.TotalMilliseconds,
            stats.CompletionTokens, stats.GenerationTime.TotalMilliseconds);
        a.Work.Output?.TryWrite(new GeneratedToken(-1, "", reason));
        a.Work.Output?.TryComplete();
        if (reason == FinishReason.Cancelled) a.Work.Completion.TrySetResult(stats);
        else a.Work.Completion.TrySetResult(stats);
        Retire(a);
    }

    private void Fail(Active a, Exception ex)
    {
        if (a.Finish != FinishReason.None) return;
        a.Finish = FinishReason.Error;
        _active.Remove(a);
        a.Work.Output?.TryComplete(ex);
        a.Work.Completion.TrySetException(ex);
        a.Work.Embedding?.TrySetException(ex);
        a.Seq.Dispose();
    }

    private void Retire(Active a)
    {
        if (!a.Work.Request.CachePrompt || a.Seq.Length == 0 || _opt.PrefixCacheEntries <= 0)
        {
            a.Seq.Dispose();
            return;
        }
        // the last sampled token was never evaluated; the cached state covers Seq.Length positions
        if (Config.HasRecurrentLayers) a.Seq.Snapshot(Config);
        _retired.AddFirst(a.Seq);
        while (_retired.Count > _opt.PrefixCacheEntries) EvictRetired();
    }

    /// <summary>
    /// Stops the engine: running requests finish as cancelled. Waits up to 10 s for the step in progress; if it takes
    /// longer, the scheduler thread releases the native memory when that step ends (never while it is in use).
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _incoming.Writer.TryComplete();
        _shutdown.Cancel();
        if (_thread.Join(TimeSpan.FromSeconds(10))) ReleaseResources();
        else Log.SchedulerStopTimeout(_log, Id, 10);
    }

    /// <summary>Frees the native resources once, after the scheduler has stopped (called by Dispose or by the scheduler).</summary>
    private void ReleaseResources()
    {
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        foreach (var s in _retired) s.Dispose();
        _retired.Clear();
        _drafter?.Dispose();
        _drafterModel?.Dispose();
        _runner.Dispose();
        _kv.Dispose();
        _offload?.Dispose();
        _pool.Dispose();
        if (_ownsModel) _model.Dispose();
        Log.EngineStopped(_log, Id, TotalPromptTokens, TotalCachedPromptTokens, TotalGeneratedTokens);
    }
}

/// <summary>How token hidden states are pooled into a single embedding.</summary>
public enum EmbeddingPooling { Mean, Last }
