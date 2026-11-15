using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using LammaNet.Extensions.AI;
using LammaNet.Gguf;
using LammaNet.Models;
using LammaNet.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;

namespace LammaNet.Tests;

/// <summary>
/// <see cref="ModelCache"/> against a local HTTP server playing the Hugging Face hub (and a plain file server): no network,
/// no environment (explicit cache directories, token and offline flag).
/// </summary>
public sealed class ModelCacheTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "lamma-cache-test-" + Guid.NewGuid().ToString("N")[..8]);
    private readonly FakeHub _hub = new();

    public void Dispose()
    {
        _hub.Dispose();
        try { Directory.Delete(_root, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private ModelCache Cache(bool offline = false, HttpClient? http = null, TimeSpan? stall = null, ILoggerFactory? loggerFactory = null) => new(new ModelCacheOptions
    {
        LoggerFactory = loggerFactory,
        CacheDirectory = Path.Combine(_root, "hub"),
        DownloadDirectory = Path.Combine(_root, "downloads"),
        Endpoint = _hub.Url,
        Offline = offline,
        UseEnvironment = false,
        HttpClient = http,
        StallTimeout = stall ?? TimeSpan.FromSeconds(30),
    });

    private static byte[] Gguf(string arch, int extraBytes = 4096)
    {
        var w = new GgufWriter();
        w.Add("general.architecture", arch);
        w.Add("general.name", "tiny " + arch);
        w.AddTensor("weights", GgmlType.F32, [extraBytes / 4]);
        return w.ToArray();
    }

    private const string Repo = "test-org/Tiny-GGUF";

    private FakeHub.RepoState TinyRepo() => _hub.AddRepo(Repo, new()
    {
        ["Tiny-Q4_K_M.gguf"] = Gguf("llama", 64 << 10),
        ["Tiny-Q8_0.gguf"] = Gguf("llama", 96 << 10),
        ["Tiny-UD-Q4_K_XL.gguf"] = Gguf("llama", 72 << 10),
        ["mmproj-F16.gguf"] = Gguf("clip", 16 << 10),
        ["Big-Q2_K-00001-of-00002.gguf"] = Gguf("llama"),
        ["Big-Q2_K-00002-of-00002.gguf"] = Gguf("llama"),
        ["README.md"] = Encoding.UTF8.GetBytes("# tiny"),
    });

    // ------------------------------------------------------------------ Hugging Face

    [Fact]
    public async Task DownloadsOnceThenServesFromCache()
    {
        var repo = TinyRepo();
        using var cache = Cache();
        var r = await cache.ResolveAsync(new HuggingFaceModel(Repo));
        Assert.False(r.IsCached);
        Assert.Equal("Tiny-Q4_K_M.gguf", r.Model.RemotePath);
        Assert.Equal("mmproj-F16.gguf", r.Mmproj!.RemotePath);
        Assert.Equal(0, _hub.FileGets);

        var got = await cache.GetAsync(new HuggingFaceModel(Repo));
        Assert.True(got.IsCached);
        Assert.Equal(repo.Files["Tiny-Q4_K_M.gguf"], File.ReadAllBytes(got.ModelPath));
        Assert.Equal(repo.Files["mmproj-F16.gguf"], File.ReadAllBytes(got.MmprojPath!));
        Assert.Equal(Path.Combine(_root, "hub", "models--test-org--Tiny-GGUF", "snapshots", repo.Commit, "Tiny-Q4_K_M.gguf"), got.ModelPath);
        Assert.Equal(repo.Commit, File.ReadAllText(Path.Combine(_root, "hub", "models--test-org--Tiny-GGUF", "refs", "main")));
        Assert.Equal(2, _hub.FileGets);
        Assert.True(ModelInfo.Read(got.ModelPath).IsComplete);

        var again = await cache.GetAsync(new HuggingFaceModel(Repo));
        Assert.Equal(got.ModelPath, again.ModelPath);
        Assert.Equal(2, _hub.FileGets);                       // only the listing was fetched
        Assert.False(again.FromCacheOnly);
    }

    [Fact]
    public async Task NewCommitWithUnchangedFileIsNotDownloadedAgain()
    {
        var repo = TinyRepo();
        using var cache = Cache();
        var first = await cache.GetAsync(new HuggingFaceModel(Repo) { IncludeMmproj = false });
        int gets = _hub.FileGets;
        string oldCommit = repo.Commit;
        repo.Files["README.md"] = Encoding.UTF8.GetBytes("# tiny, updated");   // a README-only commit
        repo.NewCommit();

        var second = await cache.GetAsync(new HuggingFaceModel(Repo) { IncludeMmproj = false });
        Assert.NotEqual(oldCommit, repo.Commit);
        Assert.Contains(repo.Commit, second.ModelPath);
        Assert.Equal(gets, _hub.FileGets);
        Assert.Equal(File.ReadAllBytes(first.ModelPath), File.ReadAllBytes(second.ModelPath));
    }

    [Fact]
    public async Task VerifiedCopyInAnOlderSnapshotIsReused()
    {
        // a cache written without blobs (huggingface_hub on Windows without symlinks): only snapshots/{old}/file
        var repo = TinyRepo();
        string snap = Path.Combine(_root, "hub", "models--test-org--Tiny-GGUF", "snapshots", new string('a', 40));
        Directory.CreateDirectory(snap);
        File.WriteAllBytes(Path.Combine(snap, "Tiny-Q4_K_M.gguf"), repo.Files["Tiny-Q4_K_M.gguf"]);
        var corrupt = (byte[])repo.Files["Tiny-Q8_0.gguf"].Clone();
        corrupt[^1] ^= 0xFF;   // same size, different content
        File.WriteAllBytes(Path.Combine(snap, "Tiny-Q8_0.gguf"), corrupt);

        using var cache = Cache();
        var r = await cache.ResolveAsync(new HuggingFaceModel(Repo) { IncludeMmproj = false });
        Assert.True(r.IsCached);
        await cache.GetAsync(new HuggingFaceModel(Repo) { IncludeMmproj = false });
        Assert.Equal(0, _hub.FileGets);

        // the corrupted copy fails verification and is downloaded
        var q8 = await cache.GetAsync(new HuggingFaceModel(Repo) { Quantization = "Q8_0", IncludeMmproj = false });
        Assert.Equal(1, _hub.FileGets);
        Assert.Equal(repo.Files["Tiny-Q8_0.gguf"], File.ReadAllBytes(q8.ModelPath));
    }

    [Fact]
    public async Task PinnedCommitInCacheMakesNoRequests()
    {
        var repo = TinyRepo();
        using var cache = Cache();
        await cache.GetAsync(new HuggingFaceModel(Repo));
        int requests = _hub.Requests.Count;
        var pinned = await cache.GetAsync(new HuggingFaceModel(Repo) { Revision = repo.Commit });
        Assert.Equal(requests, _hub.Requests.Count);
        Assert.True(pinned.IsCached);
    }

    [Fact]
    public async Task OfflineAndUnreachableHubUseTheCache()
    {
        TinyRepo();
        using (var online = Cache()) await online.GetAsync(new HuggingFaceModel(Repo));
        int requests = _hub.Requests.Count;

        using (var offline = Cache(offline: true))
        {
            var r = await offline.GetAsync(new HuggingFaceModel(Repo));
            Assert.True(r.FromCacheOnly);
            Assert.Equal("mmproj-F16.gguf", r.Mmproj!.RemotePath);
            await Assert.ThrowsAsync<FileNotFoundException>(() => offline.GetAsync(new HuggingFaceModel(Repo) { Quantization = "Q8_0" }));
        }
        Assert.Equal(requests, _hub.Requests.Count);

        _hub.Down = true;
        using var cache = Cache();
        var fallback = await cache.GetAsync(new HuggingFaceModel(Repo));
        Assert.True(fallback.FromCacheOnly);
        await Assert.ThrowsAnyAsync<HttpRequestException>(() => cache.GetAsync(new HuggingFaceModel(Repo) { Quantization = "Q8_0" }));
    }

    [Fact]
    public async Task MissingRepositoryReportsClearly()
    {
        using var cache = Cache();
        var e = await Assert.ThrowsAsync<HttpRequestException>(() => cache.ResolveAsync(new HuggingFaceModel("nobody/nothing")));
        Assert.Contains("not found or not accessible", e.Message);
    }

    [Fact]
    public async Task ResumesPartialDownloadAndRejectsBadContent()
    {
        var repo = TinyRepo();
        byte[] content = repo.Files["Tiny-Q8_0.gguf"];
        string sha = Convert.ToHexStringLower(SHA256.HashData(content));
        string blobs = Path.Combine(_root, "hub", "models--test-org--Tiny-GGUF", "blobs");
        Directory.CreateDirectory(blobs);
        File.WriteAllBytes(Path.Combine(blobs, sha + ".incomplete"), content[..(content.Length / 2)]);

        using var cache = Cache();
        var r = await cache.GetAsync(new HuggingFaceModel(Repo) { Quantization = "Q8_0", IncludeMmproj = false });
        Assert.Equal(content, File.ReadAllBytes(r.ModelPath));
        Assert.Contains(_hub.Requests, q => q.Range == $"bytes={content.Length / 2}-");

        // the server sends bytes that do not match the advertised SHA-256
        repo.Corrupt = "Tiny-UD-Q4_K_XL.gguf";
        var bad = new HuggingFaceModel(Repo) { Quantization = "UD-Q4_K_XL", IncludeMmproj = false };
        await Assert.ThrowsAsync<InvalidDataException>(() => cache.GetAsync(bad));
        var res = await cache.ResolveAsync(bad);
        Assert.False(File.Exists(res.ModelPath));
        Assert.Empty(Directory.GetFiles(blobs, "*.incomplete"));
    }

    [Fact]
    public async Task SelectsFilesLikeLlamaCpp()
    {
        TinyRepo();
        using var cache = Cache();
        async Task<string> Pick(string? quant) => (await cache.ResolveAsync(new HuggingFaceModel(Repo) { Quantization = quant })).Model.RemotePath;
        Assert.Equal("Tiny-Q4_K_M.gguf", await Pick(null));
        Assert.Equal("Tiny-Q8_0.gguf", await Pick("Q8_0"));
        Assert.Equal("Tiny-Q8_0.gguf", await Pick("q8_0"));
        Assert.Equal("Tiny-UD-Q4_K_XL.gguf", await Pick("UD-Q4_K_XL"));
        await Assert.ThrowsAsync<NotSupportedException>(() => Pick("Q2_K"));            // only as a split GGUF
        await Assert.ThrowsAsync<FileNotFoundException>(() => Pick("IQ1_S"));
        await Assert.ThrowsAsync<NotSupportedException>(() => cache.ResolveAsync(new HuggingFaceModel(Repo) { File = "Big-Q2_K-00001-of-00002.gguf" }));

        Assert.Equal("mmproj-F16.gguf", ModelCache.SelectMmproj(["mmproj-F32.gguf", "mmproj-F16.gguf", "m-Q4_K_M.gguf"], "m-Q4_K_M.gguf"));
        Assert.Equal("mmproj-BF16.gguf", ModelCache.SelectMmproj(["mmproj-F16.gguf", "mmproj-BF16.gguf"], "m-Q4_K_M.gguf"));
        Assert.Equal("mmproj-F16.gguf", ModelCache.SelectMmproj(["mmproj-F16.gguf", "mmproj-BF16.gguf"], "m-Q4_K_M.gguf", p => p == "mmproj-F16.gguf"));
        Assert.Equal("mmproj-Q8_0.gguf", ModelCache.SelectMmproj(["mmproj-F16.gguf", "mmproj-Q8_0.gguf"], "m-Q8_0.gguf"));
        Assert.Null(ModelCache.SelectMmproj(["sub/mmproj-F16.gguf"], "m-Q4_K_M.gguf"));
    }

    [Fact]
    public async Task RejectsUnsafeNames()
    {
        _hub.AddRepo("evil/repo", new() { ["../../escape.gguf"] = Gguf("llama"), ["ok-Q4_K_M.gguf"] = Gguf("llama") });
        using var cache = Cache();
        await Assert.ThrowsAsync<ArgumentException>(() => cache.ResolveAsync(new HuggingFaceModel("evil/repo") { File = "../../escape.gguf" }));
        await Assert.ThrowsAsync<ArgumentException>(() => cache.ResolveAsync(new HuggingFaceModel("evil/../repo")));
        // the listing entry is ignored, the safe file still resolves
        Assert.Equal("ok-Q4_K_M.gguf", (await cache.ResolveAsync(new HuggingFaceModel("evil/repo"))).Model.RemotePath);
        Assert.False(ModelCache.IsSafeRelativePath("C:/x.gguf"));
        Assert.False(ModelCache.IsSafeRelativePath("/abs.gguf"));
        Assert.False(ModelCache.IsSafeRelativePath("a\\..\\b.gguf"));
        Assert.True(ModelCache.IsSafeRelativePath("sub/dir/m.gguf"));
    }

    [Fact]
    public void ParsesReferences()
    {
        var a = HuggingFaceModel.Parse("unsloth/Qwen3-0.6B-GGUF:Q8_0");
        Assert.Equal(("unsloth/Qwen3-0.6B-GGUF", "Q8_0", (string?)null, "main"), (a.Repository, a.Quantization, a.File, a.Revision));
        var b = HuggingFaceModel.Parse("hf://unsloth/Qwen3-0.6B-GGUF@v2/sub/Qwen3-0.6B-Q4_K_M.gguf");
        Assert.Equal(("unsloth/Qwen3-0.6B-GGUF", "sub/Qwen3-0.6B-Q4_K_M.gguf", "v2"), (b.Repository, b.File, b.Revision));
        var c = HuggingFaceModel.Parse("https://huggingface.co/unsloth/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_K_M.gguf");
        Assert.Equal(("unsloth/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_K_M.gguf", "main", false), (c.Repository, c.File, c.Revision, c.IncludeMmproj));
        Assert.Equal("unsloth/x", HuggingFaceModel.Parse("hf:unsloth/x").Repository);
        Assert.False(HuggingFaceModel.TryParse("model.gguf", out _));
        Assert.False(HuggingFaceModel.TryParse("a/b:", out _));
        Assert.False(HuggingFaceModel.TryParse("a/-b", out _));
        Assert.Equal("unsloth/x:Q8_0", new HuggingFaceModel("unsloth/x") { Quantization = "Q8_0" }.ToString());
    }

    // ------------------------------------------------------------------ plain URLs

    [Fact]
    public async Task UrlDownloadsAreVersionedByEtag()
    {
        var v1 = Gguf("llama", 32 << 10);
        _hub.UrlFiles["/files/model.gguf"] = (v1, "\"v1\"");
        var url = new Uri(_hub.Url + "files/model.gguf");
        using var cache = Cache();
        var first = await cache.GetAsync(url);
        Assert.Equal(v1, File.ReadAllBytes(first.ModelPath));
        Assert.Equal("model.gguf", Path.GetFileName(first.ModelPath));

        int gets = _hub.FileGets;
        var same = await cache.GetAsync(url);
        Assert.Equal(first.ModelPath, same.ModelPath);
        Assert.Equal(gets, _hub.FileGets);   // HEAD only

        // the server publishes a new version while the old file is memory-mapped (in use)
        using (var inUse = new GgufFile(first.ModelPath))
        {
            var v2 = Gguf("llama", 40 << 10);
            _hub.UrlFiles["/files/model.gguf"] = (v2, "\"v2\"");
            var second = await cache.GetAsync(url);
            Assert.NotEqual(first.ModelPath, second.ModelPath);
            Assert.Equal(v2, File.ReadAllBytes(second.ModelPath));
            Assert.Equal(v1.Length, inUse.FileSize);
        }

        _hub.Down = true;
        var fallback = await cache.GetAsync(url);
        Assert.True(fallback.FromCacheOnly);
        using var newest = new GgufFile(fallback.ModelPath);
        Assert.Equal(40 << 10, newest.Tensors[0].ByteSize);
    }

    [Fact]
    public async Task HuggingFaceUrlsUseTheHubLayout()
    {
        var repo = TinyRepo();
        using var cache = Cache();
        var r = await cache.GetAsync(new Uri($"{_hub.Url}{Repo}/resolve/main/Tiny-Q8_0.gguf"));
        Assert.Contains(Path.Combine("snapshots", repo.Commit), r.ModelPath);
        Assert.Null(r.Mmproj);
    }

    [Fact]
    public async Task SlowTransfersOutliveTheClientTimeoutAndStallsResume()
    {
        var data = Gguf("llama", 256 << 10);
        _hub.UrlFiles["/files/slow.gguf"] = (data, "\"s1\"");
        _hub.ChunkDelay = TimeSpan.FromMilliseconds(150);   // ~4 chunks of 64 KB: the body takes ~0.6 s
        using (var http = new HttpClient { Timeout = TimeSpan.FromMilliseconds(400) })
        using (var cache = Cache(http: http))
        {
            var r = await cache.GetAsync(new Uri(_hub.Url + "files/slow.gguf"));
            Assert.Equal(data, File.ReadAllBytes(r.ModelPath));
        }

        // the first transfer stalls after 100 KB; the stall timeout aborts it and the retry resumes
        _hub.ChunkDelay = TimeSpan.Zero;
        _hub.UrlFiles["/files/stall.gguf"] = (data, "\"s2\"");
        _hub.StallOnceAt = 100 << 10;
        using (var cache = Cache(stall: TimeSpan.FromMilliseconds(500)))
        {
            var r = await cache.GetAsync(new Uri(_hub.Url + "files/stall.gguf"));
            Assert.Equal(data, File.ReadAllBytes(r.ModelPath));
        }
        Assert.Contains(_hub.Requests, q => q.Path == "/files/stall.gguf" && q.Range != null);
    }

    [Fact]
    public async Task LogsRetriesCorruptDownloadsAndCacheFallbacks()
    {
        using var capture = new LogCapture();
        using var lf = capture.CreateFactory();

        // a stalled transfer is retried (warning) and completes (information)
        _hub.UrlFiles["/files/stall.gguf"] = (Gguf("llama", 256 << 10), "\"s3\"");
        _hub.StallOnceAt = 100 << 10;
        using (var cache = Cache(stall: TimeSpan.FromMilliseconds(500), loggerFactory: lf))
            await cache.GetAsync(new Uri(_hub.Url + "files/stall.gguf?sig=secret"));
        Assert.Contains(capture.Entries, e => e.Id.Id == 4002 && e.Level == LogLevel.Warning && e.Exception != null);
        Assert.Contains(capture.Entries, e => e.Id.Id == 4001 && e.Level == LogLevel.Information);
        Assert.DoesNotContain(capture.Entries, e => e.Message.Contains("secret"));   // URL queries are not logged

        // bytes that do not match the advertised SHA-256
        var repo = TinyRepo();
        repo.Corrupt = "Tiny-Q8_0.gguf";
        var q8 = new HuggingFaceModel(Repo) { Quantization = "Q8_0", IncludeMmproj = false };
        using (var cache = Cache(loggerFactory: lf))
            await Assert.ThrowsAsync<InvalidDataException>(() => cache.GetAsync(q8));
        Assert.Contains(capture.Entries, e => e.Id.Id == 4004 && e.Level == LogLevel.Error && e.Exception is InvalidDataException);

        // hub down: the cached copy is served with a warning
        repo.Corrupt = null;
        using (var cache = Cache(loggerFactory: lf))
        {
            await cache.GetAsync(q8);
            _hub.Down = true;
            Assert.True((await cache.GetAsync(q8)).FromCacheOnly);
            await Assert.ThrowsAnyAsync<Exception>(() => cache.GetAsync(new HuggingFaceModel("test-org/Never-Cached") { IncludeMmproj = false }));
        }
        Assert.Contains(capture.Entries, e => e.Id.Id == 4005 && e.Level == LogLevel.Warning);
        Assert.Contains(capture.Entries, e => e.Id.Id == 4006 && e.Level == LogLevel.Error);
    }

    // ------------------------------------------------------------------ LammaChatClient

    [Fact]
    public async Task ChatClientLoadsFromTheHub()
    {
        var path = TestModels.Find(TestModels.Lfm2Q4);
        if (path == null) return;
        _hub.AddRepo("liquid/LFM-GGUF", new() { [TestModels.Lfm2Q4] = File.ReadAllBytes(path) });
        using var cache = Cache();
        var options = new LammaChatClientOptions();
        using (var client = await LammaChatClient.LoadAsync(HuggingFaceModel.Parse("liquid/LFM-GGUF:Q4_K_M"), new Inference.EngineOptions { ContextSize = 512 }, options, cache))
        {
            var reply = await client.GetResponseAsync("What is the capital of France? Answer in one sentence.", new ChatOptions { Temperature = 0f, MaxOutputTokens = 24 });
            Assert.Contains("Paris", reply.Text);
        }
        Assert.Null(options.MmprojPath);
        Assert.Equal(1, _hub.FileGets);
        using (LammaChatClient.Load(new HuggingFaceModel("liquid/LFM-GGUF"), null, null, cache)) { }
        Assert.Equal(1, _hub.FileGets);
    }
}

/// <summary>
/// A minimal HTTP/1.1 server on a loopback socket: the Hugging Face API (<c>/api/models/{repo}/revision/{rev}?blobs=true</c>),
/// <c>/{repo}/resolve/{commit}/{path}</c> (redirected to <c>/cdn/{sha256}</c>, like the hub) and plain files with ETags,
/// with Range support, slow and stalling bodies.
/// </summary>
internal sealed class FakeHub : IDisposable
{
    public sealed class RepoState(Dictionary<string, byte[]> files)
    {
        public Dictionary<string, byte[]> Files { get; } = files;
        public string Commit { get; private set; } = Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(20));
        public string? Corrupt { get; set; }
        public void NewCommit() => Commit = Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(20));
    }

    public sealed record Request(string Method, string Path, string? Range);

    private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
    private readonly CancellationTokenSource _cts = new();
    private readonly ConcurrentDictionary<string, RepoState> _repos = new();
    private int _fileGets;

    public string Url { get; }
    public ConcurrentQueue<Request> Requests { get; } = new();
    public ConcurrentDictionary<string, (byte[] Data, string ETag)> UrlFiles { get; } = new();
    public int FileGets => _fileGets;
    public bool Down { get; set; }
    public TimeSpan ChunkDelay { get; set; }
    public long StallOnceAt { get; set; } = -1;

    public FakeHub()
    {
        _listener.Start();
        Url = $"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}/";
        _ = AcceptLoop();
    }

    public RepoState AddRepo(string id, Dictionary<string, byte[]> files) => _repos[id] = new RepoState(files);

    public void Dispose()
    {
        _cts.Cancel();
        _listener.Stop();
    }

    private async Task AcceptLoop()
    {
        while (!_cts.IsCancellationRequested)
        {
            TcpClient c;
            try { c = await _listener.AcceptTcpClientAsync(_cts.Token); }
            catch (Exception) { return; }
            _ = Task.Run(() => Serve(c));
        }
    }

    private async Task Serve(TcpClient client)
    {
        using var _ = client;
        var s = client.GetStream();
        try
        {
            var head = new StringBuilder();
            var one = new byte[1];
            while (!head.ToString().EndsWith("\r\n\r\n"))
            {
                if (await s.ReadAsync(one) == 0) return;
                head.Append((char)one[0]);
            }
            var lines = head.ToString().Split("\r\n");
            var first = lines[0].Split(' ');
            string method = first[0], target = first[1];
            var headers = lines.Skip(1).Where(l => l.Contains(':')).ToDictionary(l => l[..l.IndexOf(':')].Trim(), l => l[(l.IndexOf(':') + 1)..].Trim(), StringComparer.OrdinalIgnoreCase);
            string path = Uri.UnescapeDataString(target.Split('?')[0]);
            headers.TryGetValue("Range", out var range);
            Requests.Enqueue(new Request(method, path, range));
            if (Down) { client.Client.Close(); return; }
            await Route(s, method, path, range);
        }
        catch (IOException) { }
        catch (SocketException) { }
        catch (ObjectDisposedException) { }
    }

    private async Task Route(NetworkStream s, string method, string path, string? range)
    {
        var seg = path.Trim('/').Split('/');
        if (seg is ["api", "models", var owner, var name, "revision", var rev])
        {
            if (!_repos.TryGetValue($"{owner}/{name}", out var repo) || (rev != "main" && rev != repo.Commit))
            {
                await Respond(s, method, 401, Encoding.UTF8.GetBytes("""{"error":"Repository not found"}"""), null, null);
                return;
            }
            var siblings = repo.Files.Select(kv => new Dictionary<string, object>
            {
                ["rfilename"] = kv.Key, ["size"] = kv.Value.Length,
                ["lfs"] = new Dictionary<string, object> { ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(kv.Value)), ["size"] = kv.Value.Length },
            }).ToArray();
            var json = JsonSerializer.SerializeToUtf8Bytes(new Dictionary<string, object> { ["id"] = $"{owner}/{name}", ["sha"] = repo.Commit, ["siblings"] = siblings });
            await Respond(s, method, 200, json, null, null);
            return;
        }
        if (seg.Length >= 5 && seg[2] == "resolve" && _repos.TryGetValue($"{seg[0]}/{seg[1]}", out var r) && r.Files.TryGetValue(string.Join('/', seg[4..]), out var content))
        {
            string file = string.Join('/', seg[4..]);
            if (r.Corrupt == file) { content = (byte[])content.Clone(); content[^1] ^= 0x55; }
            // the hub redirects to its CDN
            string cdn = "/cdn/" + Guid.NewGuid().ToString("N");
            _cdn[cdn] = content;
            await Respond(s, method, 302, [], new() { ["Location"] = Url.TrimEnd('/') + cdn }, null);
            return;
        }
        if (seg is ["cdn", _] && _cdn.TryGetValue(path, out var blob))
        {
            if (method == "GET") Interlocked.Increment(ref _fileGets);
            await Respond(s, method, 200, blob, null, range);
            return;
        }
        if (UrlFiles.TryGetValue(path, out var f))
        {
            if (method == "GET") Interlocked.Increment(ref _fileGets);
            await Respond(s, method, 200, f.Data, new() { ["ETag"] = f.ETag }, range);
            return;
        }
        await Respond(s, method, 404, [], null, null);
    }

    private readonly ConcurrentDictionary<string, byte[]> _cdn = new();

    private async Task Respond(NetworkStream s, string method, int status, byte[] body, Dictionary<string, string>? extra, string? range)
    {
        long start = 0;
        if (range != null && status == 200 && range.StartsWith("bytes=") && long.TryParse(range[6..].TrimEnd('-'), out var from) && from < body.Length)
        {
            start = from;
            status = 206;
        }
        var sb = new StringBuilder($"HTTP/1.1 {status} {(status is 200 or 206 ? "OK" : "X")}\r\n");
        sb.Append($"Content-Length: {body.Length - start}\r\nConnection: close\r\nAccept-Ranges: bytes\r\n");
        if (status == 206) sb.Append($"Content-Range: bytes {start}-{body.Length - 1}/{body.Length}\r\n");
        foreach (var (k, v) in extra ?? []) sb.Append($"{k}: {v}\r\n");
        sb.Append("\r\n");
        await s.WriteAsync(Encoding.ASCII.GetBytes(sb.ToString()));
        if (method == "HEAD") return;
        long sent = start;
        while (sent < body.Length)
        {
            if (StallOnceAt >= 0 && sent >= StallOnceAt)
            {
                StallOnceAt = -1;
                await s.FlushAsync();
                await Task.Delay(TimeSpan.FromSeconds(5));   // longer than the client's stall timeout
                return;
            }
            int n = (int)Math.Min(64 << 10, body.Length - sent);
            if (StallOnceAt >= 0) n = (int)Math.Min(n, Math.Max(1, StallOnceAt - sent));
            await s.WriteAsync(body.AsMemory((int)sent, n));
            sent += n;
            if (ChunkDelay > TimeSpan.Zero) { await s.FlushAsync(); await Task.Delay(ChunkDelay); }
        }
    }
}
