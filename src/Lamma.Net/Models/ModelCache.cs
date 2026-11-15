using System.Collections.Concurrent;
using System.Net;
using System.Net.Http.Headers;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using LammaNet.Diagnostics;
using Microsoft.Extensions.Logging;

namespace LammaNet.Models;

/// <summary>Options for <see cref="ModelCache"/>. Unset values come from the environment (as huggingface_hub / llama.cpp read it).</summary>
public sealed class ModelCacheOptions
{
    /// <summary>
    /// Root of the Hugging Face hub cache (<c>models--owner--repo/{blobs,refs,snapshots}</c>), shared with huggingface_hub
    /// and llama.cpp. Default: LAMMA_CACHE, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME/hub, XDG_CACHE_HOME/huggingface/hub,
    /// ~/.cache/huggingface/hub.
    /// </summary>
    public string? CacheDirectory { get; init; }
    /// <summary>Where plain URL downloads are kept. Default: LAMMA_CACHE/downloads, else ~/.cache/lamma.net/downloads.</summary>
    public string? DownloadDirectory { get; init; }
    /// <summary>Hugging Face access token (gated / private repositories). Default: HF_TOKEN, or the token saved by <c>hf auth login</c>.</summary>
    public string? Token { get; init; }
    /// <summary>Hub endpoint. Default: HF_ENDPOINT or https://huggingface.co.</summary>
    public string? Endpoint { get; init; }
    /// <summary>Never touch the network; serve from the cache only. Default: HF_HUB_OFFLINE.</summary>
    public bool? Offline { get; init; }
    /// <summary>HTTP client to use (default: an internal client without an overall timeout; stalled transfers are retried).</summary>
    public HttpClient? HttpClient { get; init; }
    /// <summary>Logging of downloads, retries and cache fallbacks (default: <see cref="LammaLogging.LoggerFactory"/>).</summary>
    public ILoggerFactory? LoggerFactory { get; init; }
    /// <summary>Download progress.</summary>
    public IProgress<ModelDownloadProgress>? Progress { get; init; }
    /// <summary>A transfer receiving no data for this long is aborted and resumed (up to 3 attempts).</summary>
    public TimeSpan StallTimeout { get; init; } = TimeSpan.FromSeconds(60);
    /// <summary>Fill unset values from environment variables (tests turn this off).</summary>
    public bool UseEnvironment { get; init; } = true;
}

/// <summary>Progress of one file download.</summary>
public sealed record ModelDownloadProgress(string File, long BytesDownloaded, long? TotalBytes);

/// <summary>One file of a resolved model.</summary>
public sealed record CachedModelFile
{
    /// <summary>Path in the repository, or the URL.</summary>
    public required string RemotePath { get; init; }
    /// <summary>Where the file is (or will be, once downloaded) in the cache.</summary>
    public required string LocalPath { get; init; }
    public long? Size { get; init; }
    /// <summary>The content is available locally (no download needed).</summary>
    public bool IsCached { get; init; }
    /// <summary>LFS SHA-256 / git blob id of the content (Hugging Face) or the HTTP ETag (URLs).</summary>
    public string? ContentId { get; init; }
    internal string? Url { get; init; }
    internal string? ExistingSource { get; init; }
}

/// <summary>The files of a model and whether they are already cached.</summary>
public sealed record ModelResolution
{
    /// <summary>The repository (<c>owner/repo</c>) or URL.</summary>
    public required string Source { get; init; }
    /// <summary>Hugging Face commit the files belong to.</summary>
    public string? Commit { get; init; }
    public required CachedModelFile Model { get; init; }
    public CachedModelFile? Mmproj { get; init; }
    /// <summary>The resolution came from the cache alone (offline or the hub was unreachable).</summary>
    public bool FromCacheOnly { get; init; }
    public bool IsCached => Model.IsCached && (Mmproj?.IsCached ?? true);
    public string ModelPath => Model.LocalPath;
    public string? MmprojPath => Mmproj?.LocalPath;
}

/// <summary>
/// Local cache of GGUF models from Hugging Face and plain URLs. A model version that is already cached is used without
/// downloading; otherwise the files are downloaded (resumable, SHA-256 verified) into the cache and then used from there.
/// Hugging Face files use the hub cache layout (content in <c>blobs/{sha256}</c>, linked into <c>snapshots/{commit}/</c>),
/// so downloads are shared with huggingface_hub and llama.cpp, and a new commit that leaves a file unchanged reuses it.
/// URL downloads are versioned by ETag (an updated file is stored next to the old one, which may still be in use).
/// </summary>
public sealed partial class ModelCache : IDisposable
{
    private static readonly Lazy<ModelCache> s_shared = new(() => new ModelCache());
    /// <summary>A process-wide cache with default (environment) settings.</summary>
    public static ModelCache Shared => s_shared.Value;

    private static readonly ConcurrentDictionary<string, SemaphoreSlim> s_fileLocks = new(StringComparer.OrdinalIgnoreCase);
    private static volatile bool s_symlinksUnsupported;

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string? _token;
    private readonly bool _offline;
    private readonly IProgress<ModelDownloadProgress>? _progress;
    private readonly TimeSpan _stall;
    private readonly ILogger _log;

    /// <summary>Root of the Hugging Face hub cache.</summary>
    public string CacheDirectory { get; }
    /// <summary>Root of the URL download cache.</summary>
    public string DownloadDirectory { get; }
    /// <summary>Hub endpoint (with a trailing slash).</summary>
    public string Endpoint { get; }

    public ModelCache(ModelCacheOptions? options = null)
    {
        options ??= new ModelCacheOptions();
        _log = LammaLogging.Create<ModelCache>(options.LoggerFactory);
        bool env = options.UseEnvironment;
        string? E(string name) => env && Environment.GetEnvironmentVariable(name) is { Length: > 0 } v ? v : null;
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string? hfHome = E("HF_HOME");
        string? xdg = E("XDG_CACHE_HOME");
        string? lammaCache = E("LAMMA_CACHE");

        CacheDirectory = Path.GetFullPath(options.CacheDirectory ?? lammaCache ?? E("HF_HUB_CACHE") ?? E("HUGGINGFACE_HUB_CACHE")
            ?? (hfHome != null ? Path.Combine(hfHome, "hub") : null)
            ?? Path.Combine(xdg ?? Path.Combine(home, ".cache"), "huggingface", "hub"));
        DownloadDirectory = Path.GetFullPath(options.DownloadDirectory
            ?? (lammaCache != null ? Path.Combine(lammaCache, "downloads") : Path.Combine(xdg ?? Path.Combine(home, ".cache"), "lamma.net", "downloads")));
        string endpoint = options.Endpoint ?? E("HF_ENDPOINT") ?? "https://huggingface.co";
        Endpoint = endpoint.EndsWith('/') ? endpoint : endpoint + "/";
        _token = options.Token ?? E("HF_TOKEN") ?? E("HUGGING_FACE_HUB_TOKEN") ?? (env ? ReadSavedToken(hfHome, xdg, home) : null);
        _offline = options.Offline ?? (E("HF_HUB_OFFLINE") is { } off && off is "1" or "true" or "TRUE" or "yes" or "on");
        _progress = options.Progress;
        _stall = options.StallTimeout;
        if (options.HttpClient != null) _http = options.HttpClient;
        else
        {
            _http = new HttpClient(new SocketsHttpHandler { ConnectTimeout = TimeSpan.FromSeconds(30), AutomaticDecompression = DecompressionMethods.None })
            {
                Timeout = Timeout.InfiniteTimeSpan,
            };
            _http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("lamma.net", typeof(ModelCache).Assembly.GetName().Version?.ToString(3) ?? "0"));
            _ownsHttp = true;
        }
    }

    private string? ReadSavedToken(string? hfHome, string? xdg, string home)
    {
        string path = hfHome != null ? Path.Combine(hfHome, "token") : Path.Combine(xdg ?? Path.Combine(home, ".cache"), "huggingface", "token");
        try { return System.IO.File.Exists(path) ? System.IO.File.ReadAllText(path).Trim() is { Length: > 0 } t ? t : null : null; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Log.CacheFileOperationFailed(_log, e, "read token", path);
            return null;
        }
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    // ===================================================================== Hugging Face

    /// <summary>Which files <paramref name="model"/> resolves to and whether they are cached; downloads nothing.</summary>
    public Task<ModelResolution> ResolveAsync(HuggingFaceModel model, CancellationToken cancellationToken = default) =>
        ResolveHfAsync(model, cancellationToken);

    /// <summary>Returns the local files of <paramref name="model"/>, downloading whatever the cache does not hold yet.</summary>
    public async Task<ModelResolution> GetAsync(HuggingFaceModel model, CancellationToken cancellationToken = default)
    {
        var r = await ResolveHfAsync(model, cancellationToken).ConfigureAwait(false);
        var m = await MaterializeHfAsync(model.Repository, r.Model, cancellationToken).ConfigureAwait(false);
        var p = r.Mmproj == null ? null : await MaterializeHfAsync(model.Repository, r.Mmproj, cancellationToken).ConfigureAwait(false);
        Log.Resolved(_log, model.ToString(), m.LocalPath, r.IsCached);
        return r with { Model = m, Mmproj = p };
    }

    private string RepoDir(string repo) => Path.Combine(CacheDirectory, "models--" + repo.Replace("/", "--"));

    private async Task<ModelResolution> ResolveHfAsync(HuggingFaceModel model, CancellationToken ct)
    {
        string repo = model.Repository;
        if (!HuggingFaceModel.IsValidRepositoryId(repo)) throw new ArgumentException($"Invalid Hugging Face repository id '{repo}'");
        if (model.File != null) CheckRelativePath(model.File);
        if (model.MmprojFile != null) CheckRelativePath(model.MmprojFile);
        string repoDir = RepoDir(repo);
        bool pinned = IsCommitHash(model.Revision);

        // a pinned commit that is fully in the cache needs no network at all
        if (pinned || _offline)
        {
            var cached = TryResolveFromCache(model, repoDir);
            if (cached != null) return cached;
            if (_offline)
                throw new FileNotFoundException($"{model} is not in the model cache ({CacheDirectory}) and the cache is offline");
        }

        RepoListing listing;
        try { listing = await ListRepositoryAsync(repo, model.Revision, ct).ConfigureAwait(false); }
        catch (Exception e) when (e is HttpRequestException or IOException or JsonException or InvalidDataException
            || (e is OperationCanceledException && !ct.IsCancellationRequested))
        {
            // hub unreachable / refused: serve the cached copy if there is one
            if (TryResolveFromCache(model, repoDir) is { } c)
            {
                Log.ServedFromCache(_log, e, model.ToString(), c.ModelPath);
                return c with { FromCacheOnly = true };
            }
            Log.UnreachableNoCache(_log, e, model.ToString());
            throw;
        }

        if (!pinned) WriteRef(repoDir, model.Revision, listing.Commit);
        var names = listing.Files.Keys.ToList();
        string modelPath = model.File ?? SelectModelFile(names, model.Quantization)
            ?? throw new FileNotFoundException($"No GGUF model{(model.Quantization != null ? $" with quantization '{model.Quantization}'" : "")} in {repo}. Available: {string.Join(", ", names.Where(n => n.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)))}");
        if (!listing.Files.ContainsKey(modelPath)) throw new FileNotFoundException($"'{modelPath}' is not in {repo}@{model.Revision}");
        CheckNotSplit(modelPath);
        string? mmprojPath = model.MmprojFile ?? (model.IncludeMmproj
            ? SelectMmproj(names, modelPath, p => IsInCache(repoDir, listing.Commit, listing.Files[p])) : null);
        if (mmprojPath != null && !listing.Files.ContainsKey(mmprojPath)) throw new FileNotFoundException($"'{mmprojPath}' is not in {repo}@{model.Revision}");

        return new ModelResolution
        {
            Source = repo,
            Commit = listing.Commit,
            Model = DescribeHfFile(repoDir, listing.Commit, repo, listing.Files[modelPath]),
            Mmproj = mmprojPath == null ? null : DescribeHfFile(repoDir, listing.Commit, repo, listing.Files[mmprojPath]),
        };
    }

    private sealed record RemoteFile(string Path, long Size, string? Oid);
    private sealed record RepoListing(string Commit, Dictionary<string, RemoteFile> Files);

    private async Task<RepoListing> ListRepositoryAsync(string repo, string revision, CancellationToken ct)
    {
        using var req = new HttpRequestMessage(HttpMethod.Get, $"{Endpoint}api/models/{repo}/revision/{Uri.EscapeDataString(revision)}?blobs=true");
        if (_token != null) req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _token);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(60));
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, timeout.Token).ConfigureAwait(false);
        if (resp.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden or HttpStatusCode.NotFound)
            throw new HttpRequestException($"Hugging Face repository {repo}@{revision} not found or not accessible ({(int)resp.StatusCode})" +
                (_token == null ? "; gated and private repositories need a token (HF_TOKEN)" : ""), null, resp.StatusCode);
        resp.EnsureSuccessStatusCode();
        await using var body = await resp.Content.ReadAsStreamAsync(timeout.Token).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(body, default, timeout.Token).ConfigureAwait(false);
        var root = doc.RootElement;
        string commit = root.TryGetProperty("sha", out var sha) && sha.GetString() is { } s && IsCommitHash(s)
            ? s : throw new InvalidDataException($"Hub response for {repo} has no valid commit");
        var files = new Dictionary<string, RemoteFile>(StringComparer.Ordinal);
        if (root.TryGetProperty("siblings", out var sib) && sib.ValueKind == JsonValueKind.Array)
            foreach (var f in sib.EnumerateArray())
            {
                if (!f.TryGetProperty("rfilename", out var fn) || fn.GetString() is not { Length: > 0 } path) continue;
                if (!IsSafeRelativePath(path)) continue;   // never let a listing escape the snapshot directory
                long size = f.TryGetProperty("size", out var sz) && sz.TryGetInt64(out var l) ? l : -1;
                string? oid = f.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object && lfs.TryGetProperty("sha256", out var h) ? h.GetString()
                    : f.TryGetProperty("blobId", out var b) ? b.GetString() : null;
                if (oid != null && !IsHex(oid, 40) && !IsHex(oid, 64)) oid = null;
                files[path] = new RemoteFile(path, size, oid?.ToLowerInvariant());
            }
        return new RepoListing(commit, files);
    }

    private CachedModelFile DescribeHfFile(string repoDir, string commit, string repo, RemoteFile f)
    {
        string final = SnapshotPath(repoDir, commit, f.Path);
        string url = $"{Endpoint}{repo}/resolve/{commit}/{string.Join('/', f.Path.Split('/').Select(Uri.EscapeDataString))}";
        long? size = f.Size >= 0 ? f.Size : null;
        string? source = null;
        if (ExistingLength(final) is long len && (size == null || len == size)) source = final;
        else if (f.Oid != null && ExistingLength(Path.Combine(repoDir, "blobs", f.Oid)) is long bl && (size == null || bl == size))
            source = Path.Combine(repoDir, "blobs", f.Oid);
        else if (f.Oid != null && size != null && FindInOtherSnapshots(repoDir, commit, f.Path, size.Value, f.Oid) is { } other)
            source = other;
        return new CachedModelFile
        {
            RemotePath = f.Path, LocalPath = final, Size = size, ContentId = f.Oid, Url = url,
            IsCached = source != null, ExistingSource = source,
        };
    }

    /// <summary>
    /// An identical file in an older snapshot (e.g. stored without blobs by huggingface_hub on Windows): same path and size,
    /// and — when the content id is a SHA-256 — the same hash.
    /// </summary>
    private static string? FindInOtherSnapshots(string repoDir, string commit, string path, long size, string oid)
    {
        var snaps = Path.Combine(repoDir, "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (var dir in Directory.EnumerateDirectories(snaps).OrderByDescending(Directory.GetLastWriteTimeUtc))
        {
            if (string.Equals(Path.GetFileName(dir), commit, StringComparison.OrdinalIgnoreCase)) continue;
            var candidate = Path.Combine(dir, path.Replace('/', Path.DirectorySeparatorChar));
            if (ExistingLength(candidate) != size) continue;
            if (!IsHex(oid, 64) || Sha256Of(candidate) == oid) return candidate;
        }
        return null;
    }

    private ModelResolution? TryResolveFromCache(HuggingFaceModel model, string repoDir)
    {
        string? commit = IsCommitHash(model.Revision) ? model.Revision.ToLowerInvariant() : ReadRef(repoDir, model.Revision);
        if (commit == null) return null;
        string snapDir = Path.Combine(repoDir, "snapshots", commit);
        if (!Directory.Exists(snapDir)) return null;
        var names = Directory.EnumerateFiles(snapDir, "*", SearchOption.AllDirectories)
            .Where(p => !p.EndsWith(".incomplete", StringComparison.Ordinal) && ExistingLength(p) != null)
            .Select(p => Path.GetRelativePath(snapDir, p).Replace(Path.DirectorySeparatorChar, '/')).Order(StringComparer.Ordinal).ToList();
        string? modelPath = model.File ?? SelectModelFile(names, model.Quantization);
        if (modelPath == null || !names.Contains(modelPath)) return null;
        CheckNotSplit(modelPath);
        string? mmprojPath = model.MmprojFile ?? (model.IncludeMmproj ? SelectMmproj(names, modelPath) : null);
        if (mmprojPath != null && !names.Contains(mmprojPath)) return null;
        CachedModelFile Local(string p)
        {
            string full = SnapshotPath(repoDir, commit, p);
            return new CachedModelFile { RemotePath = p, LocalPath = full, Size = ExistingLength(full), IsCached = true, ExistingSource = full };
        }
        return new ModelResolution { Source = model.Repository, Commit = commit, Model = Local(modelPath), Mmproj = mmprojPath == null ? null : Local(mmprojPath), FromCacheOnly = true };
    }

    private async Task<CachedModelFile> MaterializeHfAsync(string repo, CachedModelFile f, CancellationToken ct)
    {
        string final = f.LocalPath;
        if (f.ExistingSource == final) return EnsureValid(f);
        string repoDir = RepoDir(repo);
        string blobs = Path.Combine(repoDir, "blobs");
        Directory.CreateDirectory(blobs);
        string blob = Path.Combine(blobs, f.ContentId ?? ("nohash-" + Sha256Hex(Encoding.UTF8.GetBytes(f.Url ?? final))[..40]));

        var gate = s_fileLocks.GetOrAdd(blob, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (ExistingLength(final) is long have && (f.Size == null || have == f.Size)) return EnsureValid(f);
            if (f.ExistingSource != null && f.ExistingSource != blob && ExistingLength(blob) == null)
            {
                // content found in an older snapshot: make it the blob (hard link, else move) so later commits find it too
                if (!FileLinks.TryHardLink(f.ExistingSource, blob, _log)) System.IO.File.Move(f.ExistingSource, blob);
            }
            if (ExistingLength(blob) is not long bl || (f.Size != null && bl != f.Size))
            {
                if (f.Url == null) throw new FileNotFoundException($"{f.RemotePath} is not cached");
                await DownloadAsync(f.Url, blob, f.Size, f.ContentId is { } oid && IsHex(oid, 64) ? oid : null, f.RemotePath, resume: f.ContentId != null, ct).ConfigureAwait(false);
            }
            LinkIntoSnapshot(blob, final);
            return EnsureValid(f with { IsCached = true, ExistingSource = final, Size = ExistingLength(final) });
        }
        finally { gate.Release(); }
    }

    /// <summary>symlink (the hub layout) → hard link → move: the snapshot entry points at the blob.</summary>
    private void LinkIntoSnapshot(string blob, string final)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(final)!);
        if (System.IO.File.Exists(final)) System.IO.File.Delete(final);
        if (!s_symlinksUnsupported)
        {
            Exception? error = null;
            try
            {
                System.IO.File.CreateSymbolicLink(final, Path.GetRelativePath(Path.GetDirectoryName(final)!, blob));
                if (ExistingLength(final) != null) return;
                System.IO.File.Delete(final);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or PlatformNotSupportedException) { error = e; }
            s_symlinksUnsupported = true;
            Log.SymlinksUnsupported(_log, error);
        }
        if (FileLinks.TryHardLink(blob, final, _log)) return;
        System.IO.File.Move(blob, final);
    }

    private static CachedModelFile EnsureValid(CachedModelFile f)
    {
        if (f.LocalPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && ModelInfo.Read(f.LocalPath, new ModelInfoOptions { LoadVocabulary = false }).IsComplete == false)
            throw new InvalidDataException($"Cached file {f.LocalPath} is incomplete (shorter than its tensor table requires)");
        return f;
    }

    private static string SnapshotPath(string repoDir, string commit, string relative)
    {
        CheckRelativePath(relative);
        var snap = Path.GetFullPath(Path.Combine(repoDir, "snapshots", commit));
        var full = Path.GetFullPath(Path.Combine(snap, relative.Replace('/', Path.DirectorySeparatorChar)));
        if (!full.StartsWith(snap + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"Path '{relative}' escapes the snapshot directory");
        return full;
    }

    private void WriteRef(string repoDir, string revision, string commit)
    {
        if (!IsSafeRelativePath(revision)) return;
        string refs = Path.Combine(repoDir, "refs");
        string path = Path.GetFullPath(Path.Combine(refs, revision.Replace('/', Path.DirectorySeparatorChar)));
        if (!path.StartsWith(Path.GetFullPath(refs) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) return;
        try
        {
            if (System.IO.File.Exists(path) && System.IO.File.ReadAllText(path).Trim() == commit) return;
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            string tmp = path + "." + Guid.NewGuid().ToString("N")[..8] + ".tmp";
            System.IO.File.WriteAllText(tmp, commit);
            System.IO.File.Move(tmp, path, overwrite: true);
        }
        catch (IOException e) { Log.CacheFileOperationFailed(_log, e, "write ref", path); }   // e.g. a concurrent writer wrote the same ref
    }

    private string? ReadRef(string repoDir, string revision)
    {
        if (!IsSafeRelativePath(revision)) return null;
        string path = Path.Combine(repoDir, "refs", revision.Replace('/', Path.DirectorySeparatorChar));
        try { return System.IO.File.Exists(path) && System.IO.File.ReadAllText(path).Trim() is var c && IsCommitHash(c) ? c.ToLowerInvariant() : null; }
        catch (IOException e)
        {
            Log.CacheFileOperationFailed(_log, e, "read ref", path);
            return null;
        }
    }

    // ---------------------------------------------------------------- file selection (llama.cpp -hf semantics)

    private static readonly string[] s_notModels = ["mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-"];

    private static bool IsModelFile(string path)
    {
        if (!path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) return false;
        string name = path[(path.LastIndexOf('/') + 1)..];
        return !s_notModels.Any(k => name.Contains(k, StringComparison.OrdinalIgnoreCase));
    }

    [GeneratedRegex(@"-(\d{5})-of-(\d{5})\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex SplitRegex();

    private static void CheckNotSplit(string path)
    {
        if (SplitRegex().IsMatch(path))
            throw new NotSupportedException($"'{path}' is part of a split GGUF; split models are not supported yet (merge with llama-gguf-split --merge)");
    }

    /// <summary>The model file for a quantization tag (default Q4_K_M, then Q8_0, then the first model); split GGUFs are skipped.</summary>
    internal static string? SelectModelFile(IReadOnlyList<string> files, string? quantization)
    {
        var models = files.Where(IsModelFile).Where(p => !SplitRegex().IsMatch(p)).Order(StringComparer.Ordinal).ToList();
        string[] tags = quantization != null ? [quantization] : ["Q4_K_M", "Q8_0"];
        foreach (var tag in tags)
        {
            var re = new Regex(Regex.Escape(tag) + "[.-]", RegexOptions.IgnoreCase);
            // prefer a file named exactly "...-TAG.gguf", else any file containing "TAG." / "TAG-"
            var hit = models.FirstOrDefault(p => p.EndsWith("-" + tag + ".gguf", StringComparison.OrdinalIgnoreCase))
                ?? models.FirstOrDefault(p => re.IsMatch(p));
            if (hit != null) return hit;
        }
        if (quantization != null)
        {
            if (files.Any(p => SplitRegex().IsMatch(p) && new Regex(Regex.Escape(quantization) + "[.-]", RegexOptions.IgnoreCase).IsMatch(p)))
                throw new NotSupportedException($"Quantization '{quantization}' is only available as a split GGUF, which is not supported yet");
            return null;
        }
        return models.FirstOrDefault();
    }

    /// <summary>Cheap check (no hashing) whether some copy of the file is in the repository's cache.</summary>
    private static bool IsInCache(string repoDir, string commit, RemoteFile f)
    {
        if (ExistingLength(SnapshotPath(repoDir, commit, f.Path)) != null) return true;
        if (f.Oid != null && ExistingLength(Path.Combine(repoDir, "blobs", f.Oid)) != null) return true;
        var snaps = Path.Combine(repoDir, "snapshots");
        return Directory.Exists(snaps) && Directory.EnumerateDirectories(snaps)
            .Any(d => ExistingLength(Path.Combine(d, f.Path.Replace('/', Path.DirectorySeparatorChar))) == f.Size);
    }

    /// <summary>
    /// The mmproj next to the model whose precision is closest to the model's (llama.cpp's choice); among equally close
    /// ones a file already in the cache wins, then the first by name.
    /// </summary>
    internal static string? SelectMmproj(IReadOnlyList<string> files, string modelPath, Func<string, bool>? isCached = null)
    {
        string dir = modelPath.Contains('/') ? modelPath[..modelPath.LastIndexOf('/')] : "";
        int modelBits = QuantBits(modelPath);
        string? best = null;
        int bestDiff = int.MaxValue;
        bool bestCached = false;
        foreach (var f in files.Order(StringComparer.Ordinal))
        {
            if (!f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) || !f[(f.LastIndexOf('/') + 1)..].Contains("mmproj", StringComparison.OrdinalIgnoreCase)) continue;
            string fdir = f.Contains('/') ? f[..f.LastIndexOf('/')] : "";
            if (fdir != dir) continue;
            int diff = Math.Abs(QuantBits(f) - modelBits);
            bool cached = isCached?.Invoke(f) ?? false;
            if (diff < bestDiff || (diff == bestDiff && cached && !bestCached)) { best = f; bestDiff = diff; bestCached = cached; }
        }
        return best;
    }

    [GeneratedRegex(@"[-.]([A-Z0-9_]+)$", RegexOptions.IgnoreCase)]
    private static partial Regex TagRegex();

    /// <summary>Q4_K_M → 4, F16 → 16, BF16 → 16 (the first number of the file's quantization tag).</summary>
    private static int QuantBits(string path)
    {
        string p = path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? path[..^5] : path;
        var m = TagRegex().Match(p);
        if (!m.Success) return 0;
        var d = Regex.Match(m.Groups[1].Value, @"\d+");
        return d.Success && int.TryParse(d.Value, out int bits) ? bits : 0;
    }

    // ===================================================================== plain URLs

    /// <summary>Where the file at <paramref name="url"/> is (or would be) cached; contacts the server only for its version.</summary>
    public Task<ModelResolution> ResolveAsync(Uri url, CancellationToken cancellationToken = default)
    {
        if (HuggingFaceModel.TryFromUrl(url, out var hf, new Uri(Endpoint).Host)) return ResolveAsync(hf!, cancellationToken);
        return ResolveUrlAsync(url, cancellationToken);
    }

    /// <summary>Returns the local copy of the file at <paramref name="url"/>, downloading it if this version is not cached.</summary>
    public async Task<ModelResolution> GetAsync(Uri url, CancellationToken cancellationToken = default)
    {
        if (HuggingFaceModel.TryFromUrl(url, out var hf, new Uri(Endpoint).Host)) return await GetAsync(hf!, cancellationToken).ConfigureAwait(false);
        var r = await ResolveUrlAsync(url, cancellationToken).ConfigureAwait(false);
        if (r.Model.IsCached) return r with { Model = EnsureValid(r.Model) };
        var f = r.Model;
        var gate = s_fileLocks.GetOrAdd(f.LocalPath, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (ExistingLength(f.LocalPath) == null)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(f.LocalPath)!);
                await DownloadAsync(f.Url!, f.LocalPath, f.Size, null, Redact(f.RemotePath), resume: f.ContentId != null, cancellationToken).ConfigureAwait(false);
                System.IO.File.WriteAllText(Path.Combine(Path.GetDirectoryName(f.LocalPath)!, "source.txt"),
                    $"url={url}\nversion={f.ContentId}\nsize={ExistingLength(f.LocalPath)}\ndownloaded={DateTimeOffset.UtcNow:O}\n");
            }
            Log.Resolved(_log, Redact(url.ToString()), f.LocalPath, r.IsCached);
            return r with { Model = EnsureValid(f with { IsCached = true, Size = ExistingLength(f.LocalPath) }) };
        }
        finally { gate.Release(); }
    }

    private async Task<ModelResolution> ResolveUrlAsync(Uri url, CancellationToken ct)
    {
        if (url.Scheme != Uri.UriSchemeHttps && url.Scheme != Uri.UriSchemeHttp) throw new ArgumentException($"Only http(s) URLs can be downloaded: {url}");
        string clean = url.GetComponents(UriComponents.HttpRequestUrl, UriFormat.UriEscaped);
        string fileName = Uri.UnescapeDataString(url.Segments.Length > 0 ? url.Segments[^1].TrimEnd('/') : "");
        if (fileName.Length == 0 || fileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || fileName is "." or "..") fileName = "model.gguf";
        string urlDir = Path.Combine(DownloadDirectory, Sanitize(url.Host), Sha256Hex(Encoding.UTF8.GetBytes(clean))[..16]);

        CachedModelFile Version(string? token, long? size, bool fromCache = false)
        {
            string dir = Path.Combine(urlDir, token == null ? "unversioned" : Sha256Hex(Encoding.UTF8.GetBytes(token))[..16]);
            string path = Path.Combine(dir, fileName);
            return new CachedModelFile
            {
                RemotePath = clean, LocalPath = path, Size = size, ContentId = token, Url = clean,
                IsCached = ExistingLength(path) is long l && (size == null || l == size), ExistingSource = fromCache ? path : null,
            };
        }
        CachedModelFile? Newest()
        {
            if (!Directory.Exists(urlDir)) return null;
            var best = Directory.EnumerateDirectories(urlDir).Select(d => Path.Combine(d, fileName)).Where(p => ExistingLength(p) != null)
                .OrderByDescending(System.IO.File.GetLastWriteTimeUtc).FirstOrDefault();
            return best == null ? null : new CachedModelFile { RemotePath = clean, LocalPath = best, Size = ExistingLength(best), IsCached = true, Url = clean, ExistingSource = best };
        }

        if (_offline)
            return Newest() is { } c ? new ModelResolution { Source = clean, Model = c, FromCacheOnly = true }
                : throw new FileNotFoundException($"{clean} is not in the download cache ({DownloadDirectory}) and the cache is offline");

        HttpResponseMessage? head = null;
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeout.CancelAfter(TimeSpan.FromSeconds(60));
            head = await _http.SendAsync(new HttpRequestMessage(HttpMethod.Head, clean), HttpCompletionOption.ResponseHeadersRead, timeout.Token).ConfigureAwait(false);
            if (head.StatusCode == HttpStatusCode.MethodNotAllowed)
            {
                head.Dispose();
                head = await _http.SendAsync(new HttpRequestMessage(HttpMethod.Get, clean), HttpCompletionOption.ResponseHeadersRead, timeout.Token).ConfigureAwait(false);
            }
            head.EnsureSuccessStatusCode();
        }
        catch (Exception e) when (e is HttpRequestException or IOException || (e is OperationCanceledException && !ct.IsCancellationRequested))
        {
            head?.Dispose();
            if (Newest() is { } c)
            {
                Log.ServedFromCache(_log, e, Redact(clean), c.LocalPath);
                return new ModelResolution { Source = clean, Model = c, FromCacheOnly = true };
            }
            Log.UnreachableNoCache(_log, e, Redact(clean));
            throw;
        }
        using (head)
        {
            long? size = head.Content.Headers.ContentLength;
            string? token = head.Headers.ETag?.ToString()
                ?? (head.Content.Headers.LastModified is { } lm ? $"{lm:R}|{size}" : null);
            var v = Version(token, size);
            // no version information: any cached copy is as good as another
            if (token == null && !v.IsCached && Newest() is { } any) v = any;
            return new ModelResolution { Source = clean, Model = v };
        }
    }

    private static string Sanitize(string s) => new(s.Select(c => char.IsLetterOrDigit(c) || c is '.' or '-' ? c : '_').ToArray());

    /// <summary>A URL for log messages: without the query (signed download URLs carry credentials there).</summary>
    private static string Redact(string url) =>
        Uri.TryCreate(url, UriKind.Absolute, out var u) ? u.GetLeftPart(UriPartial.Path) + (u.Query.Length > 1 ? "?…" : "") : "(invalid URL)";

    // ===================================================================== download

    /// <summary>
    /// Downloads <paramref name="url"/> to <paramref name="destination"/> via an exclusively opened <c>.incomplete</c> file
    /// (resumed with a Range request when allowed), verifying size and SHA-256, then renames it into place.
    /// </summary>
    private async Task DownloadAsync(string url, string destination, long? expectedSize, string? sha256, string displayName, bool resume, CancellationToken ct)
    {
        string temp = destination + ".incomplete";
        Exception? last = null;
        const int attempts = 3;
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        Log.DownloadStarting(_log, displayName, expectedSize, Redact(url));
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            if (attempt > 0) await Task.Delay(TimeSpan.FromSeconds(attempt), ct).ConfigureAwait(false);
            var fs = await OpenExclusiveAsync(temp, destination, displayName, ct).ConfigureAwait(false);
            if (fs == null) return;   // another process finished the same download
            IncrementalHash? hash = null;
            try
            {
                await using (fs)
                {
                    long existing = resume ? fs.Length : 0;
                    if (expectedSize is long es && existing > es) existing = 0;
                    fs.SetLength(existing);
                    if (sha256 != null)
                    {
                        hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                        if (existing > 0) await HashPrefixAsync(fs, existing, hash, ct).ConfigureAwait(false);
                    }
                    fs.Position = existing;

                    using var req = new HttpRequestMessage(HttpMethod.Get, url);
                    if (_token != null && url.StartsWith(Endpoint, StringComparison.OrdinalIgnoreCase)) req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _token);
                    if (existing > 0) req.Headers.Range = new RangeHeaderValue(existing, null);
                    using var stall = CancellationTokenSource.CreateLinkedTokenSource(ct);
                    stall.CancelAfter(_stall);
                    using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, stall.Token).ConfigureAwait(false);
                    resp.EnsureSuccessStatusCode();
                    if (existing > 0 && resp.StatusCode != HttpStatusCode.PartialContent)
                    {
                        // the server ignored the range: start over
                        existing = 0;
                        fs.SetLength(0);
                        fs.Position = 0;
                        if (hash != null) { hash.Dispose(); hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256); }
                    }
                    long? total = expectedSize ?? (resp.Content.Headers.ContentLength is long cl ? existing + cl : null);
                    await using var body = await resp.Content.ReadAsStreamAsync(stall.Token).ConfigureAwait(false);
                    var buf = new byte[1 << 20];
                    long done = existing, reported = existing;
                    _progress?.Report(new ModelDownloadProgress(displayName, done, total));
                    while (true)
                    {
                        stall.CancelAfter(_stall);   // the timeout covers a stalled read, not the whole transfer
                        int n = await body.ReadAsync(buf, stall.Token).ConfigureAwait(false);
                        if (n == 0) break;
                        await fs.WriteAsync(buf.AsMemory(0, n), ct).ConfigureAwait(false);
                        hash?.AppendData(buf, 0, n);
                        done += n;
                        if (done - reported >= 8 << 20) { reported = done; _progress?.Report(new ModelDownloadProgress(displayName, done, total)); }
                    }
                    if (reported != done) _progress?.Report(new ModelDownloadProgress(displayName, done, total));
                    if (expectedSize is long size && done != size)
                        throw new IOException($"Download of {displayName} ended at {done} of {size} bytes");
                    if (hash != null && Convert.ToHexStringLower(hash.GetHashAndReset()) is var got && got != sha256)
                        throw new InvalidDataException($"SHA-256 mismatch for {displayName}: expected {sha256}, got {got}");
                }
                System.IO.File.Move(temp, destination, overwrite: true);
                Log.DownloadCompleted(_log, displayName, ExistingLength(destination) ?? 0, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalSeconds);
                return;
            }
            catch (InvalidDataException e)
            {
                Log.DownloadCorrupt(_log, e, displayName);
                TryDelete(temp);
                throw;
            }
            catch (Exception e) when (e is HttpRequestException or IOException || (e is OperationCanceledException && !ct.IsCancellationRequested))
            {
                last = e;
                if (e is HttpRequestException { StatusCode: { } sc } && (int)sc is >= 400 and < 500 and not 408 and not 429) break;
                if (attempt + 1 < attempts) Log.DownloadAttemptFailed(_log, e, displayName, attempt + 1, attempts);
            }
            finally { hash?.Dispose(); }
        }
        var failed = new HttpRequestException($"Download of {displayName} failed: {last?.Message}", last);
        Log.DownloadFailed(_log, failed, displayName);
        throw failed;
    }

    /// <summary>
    /// Opens the partial download exclusively. While another process holds it, waits for that download to finish
    /// (returns null once <paramref name="destination"/> exists) and gives up when the partial file stops growing.
    /// </summary>
    private async Task<FileStream?> OpenExclusiveAsync(string temp, string destination, string displayName, CancellationToken ct)
    {
        long lastLength = -1;
        var lastChange = DateTime.UtcNow;
        bool waiting = false;
        while (true)
        {
            try { return new FileStream(temp, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1 << 16, FileOptions.Asynchronous); }
            catch (IOException) when (System.IO.File.Exists(temp))
            {
                if (!waiting) { waiting = true; Log.WaitingForOtherDownload(_log, displayName); }
                if (ExistingLength(destination) != null) return null;
                long len = ExistingLength(temp) ?? -1;
                if (len != lastLength) { lastLength = len; lastChange = DateTime.UtcNow; }
                else if (DateTime.UtcNow - lastChange > TimeSpan.FromMinutes(5)) throw;
                await Task.Delay(1000, ct).ConfigureAwait(false);
            }
        }
    }

    private static async Task HashPrefixAsync(FileStream fs, long length, IncrementalHash hash, CancellationToken ct)
    {
        fs.Position = 0;
        var buf = new byte[1 << 20];
        long left = length;
        while (left > 0)
        {
            int n = await fs.ReadAsync(buf.AsMemory(0, (int)Math.Min(buf.Length, left)), ct).ConfigureAwait(false);
            if (n == 0) throw new IOException("Partial download shrank while hashing");
            hash.AppendData(buf, 0, n);
            left -= n;
        }
    }

    // ===================================================================== helpers

    /// <summary>Length of a file (following symlinks), or null when it does not exist.</summary>
    internal static long? ExistingLength(string path)
    {
        try
        {
            var fi = new FileInfo(path);
            if (!fi.Exists) return null;
            if (fi.LinkTarget != null) return fi.ResolveLinkTarget(true) is FileInfo { Exists: true } t ? t.Length : null;
            return fi.Length;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Log.CacheFileOperationFailed(LammaLogging.Static("LammaNet.Models.ModelCache"), e, "stat", path);
            return null;
        }
    }

    private static string Sha256Of(string path)
    {
        using var s = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, 1 << 20, FileOptions.SequentialScan);
        return Convert.ToHexStringLower(SHA256.HashData(s));
    }

    private static string Sha256Hex(byte[] data) => Convert.ToHexStringLower(SHA256.HashData(data));

    private void TryDelete(string path)
    {
        try { System.IO.File.Delete(path); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { Log.CacheFileOperationFailed(_log, e, "delete", path); }
    }

    private static bool IsHex(string s, int length) => s.Length == length && s.All(char.IsAsciiHexDigit);
    private static bool IsCommitHash(string s) => IsHex(s, 40);

    /// <summary>A relative '/'-separated path without '..', empty or rooted parts (no escaping the cache).</summary>
    internal static bool IsSafeRelativePath(string p) =>
        p.Length > 0 && !p.StartsWith('/') && !p.Contains('\\') && !p.Contains(':') && !p.Contains('\0')
        && p.Split('/').All(s => s.Length > 0 && s != "." && s != "..");

    private static void CheckRelativePath(string p)
    {
        if (!IsSafeRelativePath(p)) throw new ArgumentException($"Unsafe repository path '{p}'");
    }

    private static partial class FileLinks
    {
        [LibraryImport("kernel32.dll", EntryPoint = "CreateHardLinkW", StringMarshalling = StringMarshalling.Utf16, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static partial bool CreateHardLink(string newFile, string existingFile, nint securityAttributes);

        [LibraryImport("libc", EntryPoint = "link", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
        private static partial int Link(string existing, string newPath);

        /// <summary>Creates <paramref name="link"/> as a hard link to <paramref name="existing"/> (false when unsupported).</summary>
        public static bool TryHardLink(string existing, string link, ILogger log)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(link)!);
                return OperatingSystem.IsWindows() ? CreateHardLink(link, existing, 0) : Link(existing, link) == 0;
            }
            catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or IOException)
            {
                Log.CacheFileOperationFailed(log, e, "hard link", link);
                return false;
            }
        }
    }
}
