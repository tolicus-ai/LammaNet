using System.Text;
using LammaNet.Gguf;
using LammaNet.Models;
using LammaNet.Tensors;
using LammaNet.Tokenization;

namespace LammaNet.Tests;

/// <summary><see cref="ModelInfo"/>: GGUF metadata read from the header only (path, stream and async paths).</summary>
public class ModelInfoTests
{
    // ---------------------------------------------------------------- synthetic files

    private static GgufWriter SyntheticModel(int dataBytes = 0, GgmlType extraType = GgmlType.F32)
    {
        var w = new GgufWriter();
        w.Add("general.architecture", "llama");
        w.Add("general.name", "Tiny Test");
        w.Add("general.size_label", "1K");
        w.Add("general.file_type", 7u);
        w.Add("general.tags", new[] { "test", "tiny" });
        w.Add("general.base_model.count", 1u);
        w.Add("general.base_model.0.name", "Tiny Base");
        w.Add("general.sampling.temp", 0.7f);
        w.Add("llama.block_count", 2u);
        w.Add("llama.embedding_length", 64u);
        w.Add("llama.context_length", 2048u);
        w.Add("llama.feed_forward_length", 128u);
        w.Add("llama.attention.head_count", 4u);
        w.Add("llama.attention.head_count_kv", 2u);
        w.Add("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        w.Add("llama.rope.freq_base", 500000f);
        w.Add("llama.rope.scaling.type", "yarn");
        w.Add("llama.rope.scaling.factor", 4f);
        w.Add("llama.rope.scaling.original_context_length", 512u);
        w.Add("tokenizer.ggml.model", "gpt2");
        w.Add("tokenizer.ggml.pre", "llama-bpe");
        var tokens = Enumerable.Range(0, 5000).Select(i => i switch { 0 => "<s>", 1 => "</s>", _ => $"t{i}" }).ToArray();
        w.Add("tokenizer.ggml.tokens", tokens);
        w.Add("tokenizer.ggml.token_type", tokens.Select((_, i) => i < 2 ? 3 : 1).ToArray());
        w.Add("tokenizer.ggml.merges", Enumerable.Range(0, 4800).Select(i => $"t{i} t{i + 1}").ToArray());
        w.Add("tokenizer.ggml.bos_token_id", 0u);
        w.Add("tokenizer.ggml.eos_token_id", 1u);
        w.Add("tokenizer.ggml.add_bos_token", true);
        w.Add("tokenizer.chat_template", "{{ messages }}");
        w.Add("tokenizer.chat_template.tool_use", "{{ tools }}");
        w.AddTensor("token_embd.weight", GgmlType.Q8_0, [64, 5000]);
        w.AddTensor("output_norm.weight", GgmlType.F32, [64]);
        for (int l = 0; l < 2; l++)
        {
            w.AddTensor($"blk.{l}.attn_q.weight", GgmlType.Q8_0, [64, 64]);
            w.AddTensor($"blk.{l}.ffn_up.weight", GgmlType.Q4_K, [256, 128]);
        }
        if (dataBytes > 0) w.AddTensor("extra", extraType, [dataBytes / 4], dataBytes);
        return w;
    }

    private static string WriteTemp(byte[] bytes)
    {
        var path = Path.Combine(Path.GetTempPath(), $"lamma-info-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private static void AssertSynthetic(ModelInfo info, bool vocabulary)
    {
        Assert.Equal("llama", info.Architecture);
        Assert.Equal("Tiny Test", info.General.Name);
        Assert.Equal(["test", "tiny"], info.General.Tags);
        Assert.Equal("Tiny Base", Assert.Single(info.General.BaseModels).Name);
        Assert.Equal(0.7f, Convert.ToSingle(info.General.RecommendedSampling["temp"]));
        Assert.Equal(2048, info.ContextLength);
        Assert.Equal(64, info.EmbeddingLength);
        Assert.Equal(2, info.LayerCount);
        Assert.Equal(4, info.HeadCount);
        Assert.Equal(2, info.HeadCountKv);
        Assert.Equal(16, info.HeadDim);
        Assert.Equal(128, info.FeedForwardLength);
        Assert.Equal("AA", info.LayerPattern);
        Assert.Equal(5000, info.VocabularySize);

        var rope = info.Rope!;
        Assert.Equal(500000f, rope.FreqBase);
        Assert.Equal(16, rope.Dimensions);
        Assert.Equal("yarn", rope.ScalingType);
        Assert.Equal(4f, rope.ScalingFactor);
        Assert.Equal(512, rope.OriginalContextLength);

        var tk = info.Tokenizer!;
        Assert.Equal("gpt2", tk.Model);
        Assert.Equal("llama-bpe", tk.Pre);
        Assert.Equal(5000, tk.VocabularySize);
        Assert.Equal(4800, tk.MergeCount);
        Assert.Equal(0, tk.SpecialTokens["bos"]);
        Assert.Equal(1, tk.SpecialTokens["eos"]);
        Assert.True(tk.AddBos);
        Assert.Null(tk.AddEos);
        Assert.Equal("{{ messages }}", tk.ChatTemplate);
        Assert.Equal("{{ tools }}", tk.ChatTemplates["tool_use"]);
        if (vocabulary)
        {
            Assert.Equal("</s>", tk.TokenText(1));
            Assert.Equal(TokenType.Control, tk.TokenTypes![0]);
            Assert.Equal(2, tk.ControlTokenCount);
        }
        else
        {
            Assert.Null(tk.Tokens);
            Assert.Null(tk.TokenTypes);
            Assert.Equal(new GgufArraySummary(GgufValueType.String, 5000), info.Metadata["tokenizer.ggml.tokens"]);
        }

        var w = info.Weights;
        Assert.Equal("Q8_0", w.FileTypeName);
        Assert.True(w.IsQuantized);
        Assert.Equal(GgmlType.Q8_0, w.PredominantType);
        Assert.Equal(64L * 5000 + 64 + 2 * (64 * 64 + 256 * 128), info.Tensors.Take(6).Sum(t => t.ElementCount));
        var embd = info.Tensors[0];
        Assert.Equal(("token_embd.weight", GgmlType.Q8_0), (embd.Name, embd.Type));
        Assert.Equal([64L, 5000], embd.Dims);
    }

    [Fact]
    public void ReadsHeaderFromPathStreamAndAsync()
    {
        var bytes = SyntheticModel(dataBytes: 4096).ToArray();
        var path = WriteTemp(bytes);
        try
        {
            var fromPath = ModelInfo.Read(path);
            AssertSynthetic(fromPath, vocabulary: true);
            Assert.Equal(bytes.Length, fromPath.FileSize);
            Assert.True(fromPath.IsComplete);

            using (var s = new NonSeekableStream(new MemoryStream(bytes)))
            {
                var fromStream = ModelInfo.Read(s, source: "memory");
                AssertSynthetic(fromStream, vocabulary: true);
                Assert.Null(fromStream.FileSize);
                Assert.Null(fromStream.IsComplete);
                Assert.Equal(fromPath.DataOffset, fromStream.DataOffset);
            }

            using (var s = new NonSeekableStream(new MemoryStream(bytes)))
                AssertSynthetic(ModelInfo.ReadAsync(s).GetAwaiter().GetResult(), vocabulary: true);

            var noVocab = new ModelInfoOptions { LoadVocabulary = false };
            AssertSynthetic(ModelInfo.Read(path, noVocab), vocabulary: false);
            using (var s = new MemoryStream(bytes))
                AssertSynthetic(ModelInfo.Read(s, noVocab), vocabulary: false);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void StreamReadStopsAtTheHeader()
    {
        // 32 MB of tensor data behind a ~150 KB header: only the first chunk (1 MB) may be read
        var bytes = SyntheticModel(dataBytes: 32 << 20).ToArray();
        using var counting = new NonSeekableStream(new MemoryStream(bytes));
        var info = ModelInfo.Read(counting);
        Assert.InRange(info.DataOffset, 100_000, 400_000);
        Assert.InRange(counting.BytesRead, info.DataOffset, 1 << 20);
        Assert.Equal(counting.BytesRead, info.BytesRead);
    }

    [Fact]
    public void DetectsTruncatedFilesAndBadHeaders()
    {
        var bytes = SyntheticModel(dataBytes: 1 << 16).ToArray();
        var full = ModelInfo.Read(new MemoryStream(bytes));
        Assert.Equal(bytes.Length, full.ExpectedFileSize);
        Assert.True(full.IsComplete);

        // a partial download: header intact, data cut short
        var partial = ModelInfo.Read(new MemoryStream(bytes, 0, bytes.Length - 100));
        Assert.False(partial.IsComplete);
        var path = WriteTemp(bytes[..^100]);
        try { Assert.False(ModelInfo.Read(path).IsComplete); }
        finally { File.Delete(path); }

        // header itself cut, or not a GGUF at all
        Assert.Throws<InvalidDataException>(() => ModelInfo.Read(new MemoryStream(bytes, 0, (int)full.DataOffset / 2)));
        Assert.Throws<InvalidDataException>(() => ModelInfo.Read(new NonSeekableStream(new MemoryStream(bytes, 0, 1000))));
        Assert.Throws<InvalidDataException>(() => ModelInfo.Read(new MemoryStream(Encoding.ASCII.GetBytes("PK\x03\x04 not a gguf file at all"))));
    }

    [Fact]
    public void UnknownTensorTypesDoNotBreakInspection()
    {
        var bytes = SyntheticModel(dataBytes: 1024, extraType: (GgmlType)250).ToArray();
        var info = ModelInfo.Read(new MemoryStream(bytes));
        AssertSynthetic(info, vocabulary: true);
        var unknown = info.Weights.ByType.Single(t => t.Type == (GgmlType)250);
        Assert.Null(unknown.ByteSize);
        Assert.Equal("type 250", unknown.TypeName);
        Assert.Null(info.ExpectedFileSize);
        Assert.Null(info.IsComplete);
        Assert.Contains("type 250", info.ToString());
    }

    [Fact]
    public void FileTypeNamesFollowLlamaCpp()
    {
        Assert.Equal("Q4_K_M", WeightsInfo.FileTypeNameOf(15));
        Assert.Equal("Q8_0", WeightsInfo.FileTypeNameOf(7));
        Assert.Equal("BF16", WeightsInfo.FileTypeNameOf(32));
        Assert.Equal("IQ4_XS", WeightsInfo.FileTypeNameOf(30));
        Assert.Equal("MXFP4_MOE", WeightsInfo.FileTypeNameOf(38));
    }

    // ---------------------------------------------------------------- real files

    [Fact]
    public void RealModelsMatchEngineDerivation()
    {
        if (!Directory.Exists(TestModels.Dir)) return;
        var files = Directory.GetFiles(TestModels.Dir, "*.gguf");
        foreach (var path in files)
        {
            string name = Path.GetFileName(path);
            var info = ModelInfo.Read(path);
            Assert.True(info.IsComplete, name);

            using var f = new GgufFile(path);
            Assert.Equal(f.DataOffset, info.DataOffset);
            Assert.Equal(f.Tensors.Count, info.Tensors.Count);

            // stream path == memory-mapped path, reading little more than the header
            using (var s = new NonSeekableStream(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16)))
            {
                var si = ModelInfo.Read(s, source: name);
                Assert.Equal(info.DataOffset, si.DataOffset);
                Assert.True(s.BytesRead <= Math.Max(4 * info.DataOffset, 1 << 20), $"{name}: read {s.BytesRead} for a {info.DataOffset}-byte header");
                Assert.Equal(info.Metadata.Count, si.Metadata.Count);
                foreach (var (k, v) in info.Metadata)
                {
                    var o = si.Metadata[k];
                    if (v is Array a) Assert.True(a.Cast<object>().SequenceEqual(((Array)o).Cast<object>()), $"{name}: {k}");
                    else Assert.Equal(v, o);
                }
                for (int i = 0; i < info.Tensors.Count; i++)
                {
                    GgufTensorInfo x = info.Tensors[i], y = si.Tensors[i];
                    Assert.Equal((x.Name, x.Type, x.Offset), (y.Name, y.Type, y.Offset));
                    Assert.Equal(x.Dims, y.Dims);
                }
            }

            if (info.Architecture == "clip")
            {
                Assert.Null(info.Config);
                Assert.NotNull(info.Projector);
                Assert.True(info.Projector!.Vision != null || info.Projector.Audio != null, name);
                Assert.NotNull((info.Projector.Vision ?? info.Projector.Audio)!.ProjectorType);
                continue;
            }
            // the hyper-parameters are the engine's own derivation
            var cfg = ModelConfig.FromGguf(f);
            var ic = info.Config!;
            Assert.Equal(cfg.ToString(), ic.ToString());
            Assert.Equal(cfg.NHead, ic.NHead);
            Assert.Equal(cfg.NHeadKv, ic.NHeadKv);
            Assert.Equal(cfg.HeadDimK, ic.HeadDimK);
            Assert.Equal(cfg.IsSwa, ic.IsSwa);
            Assert.Equal(cfg.RmsEps, ic.RmsEps);
            Assert.Equal(cfg.SlidingWindow, ic.SlidingWindow);
            Assert.Equal(cfg.Kinds, ic.Kinds);
            Assert.Equal(cfg.SsmConvKernel, ic.SsmConvKernel);
            for (int l = 0; l < cfg.NLayer; l++)
                Assert.Equal((cfg.Rope[l]?.Dims, cfg.Rope[l]?.FreqBase, cfg.Rope[l]?.FreqFactors?.Length),
                    (ic.Rope[l]?.Dims, ic.Rope[l]?.FreqBase, ic.Rope[l]?.FreqFactors?.Length));
            Assert.NotNull(info.Tokenizer);
            Assert.Equal(Tokenization.Vocabulary.FromGguf(f).Count, info.Tokenizer!.VocabularySize);
            Assert.False(string.IsNullOrEmpty(info.ToString()));
        }

        string? Ftype(string file) => TestModels.Find(file) is { } p ? ModelInfo.Read(p, new ModelInfoOptions { LoadVocabulary = false }).Weights.FileTypeName : null;
        if (TestModels.Find(TestModels.Qwen3Q4) != null) Assert.Equal("Q4_K_M", Ftype(TestModels.Qwen3Q4));
        if (TestModels.Find(TestModels.Qwen3Q8) != null) Assert.Equal("Q8_0", Ftype(TestModels.Qwen3Q8));
        if (TestModels.Find(TestModels.Lfm2Bf16) != null) Assert.Equal("BF16", Ftype(TestModels.Lfm2Bf16));
    }

    [Fact]
    public void ModelExposesInfo()
    {
        var path = TestModels.Find(TestModels.Qwen3Q4);
        if (path == null) return;
        using var model = Model.Load(path, new ModelLoadOptions { Prefetch = false });
        var info = model.Info;
        Assert.Same(info, model.Info);
        Assert.Equal(model.Config.NCtxTrain, info.ContextLength);
        Assert.Equal(model.Vocab.Count, info.VocabularySize);
        Assert.Equal(model.ChatTemplate, info.Tokenizer!.ChatTemplate);
        Assert.Equal(model.WeightBytes, info.Weights.ByteSize);
    }
}

/// <summary>Minimal GGUF v3 writer for tests (tensor data is zero-filled).</summary>
internal sealed class GgufWriter
{
    private readonly List<(string Key, object Value)> _kv = [];
    private readonly List<(string Name, GgmlType Type, long[] Dims, long Bytes)> _tensors = [];

    public void Add(string key, object value) => _kv.Add((key, value));

    public void AddTensor(string name, GgmlType type, long[] dims, long? bytes = null)
    {
        long n = dims.Aggregate(1L, (a, b) => a * b);
        _tensors.Add((name, type, dims, bytes ?? GgmlTypeInfo.RowSize(type, dims[0]) * (n / dims[0])));
    }

    public byte[] ToArray(int alignment = 32)
    {
        var ms = new MemoryStream();
        var w = new BinaryWriter(ms, Encoding.UTF8);
        void Str(string s) { var b = Encoding.UTF8.GetBytes(s); w.Write((ulong)b.Length); w.Write(b); }
        w.Write(0x46554747u);
        w.Write(3u);
        w.Write((ulong)_tensors.Count);
        w.Write((ulong)_kv.Count);
        foreach (var (k, v) in _kv)
        {
            Str(k);
            switch (v)
            {
                case string s: w.Write((uint)GgufValueType.String); Str(s); break;
                case uint u: w.Write((uint)GgufValueType.UInt32); w.Write(u); break;
                case int i: w.Write((uint)GgufValueType.Int32); w.Write(i); break;
                case float f: w.Write((uint)GgufValueType.Float32); w.Write(f); break;
                case bool b: w.Write((uint)GgufValueType.Bool); w.Write(b); break;
                case string[] sa:
                    w.Write((uint)GgufValueType.Array); w.Write((uint)GgufValueType.String); w.Write((ulong)sa.Length);
                    foreach (var s in sa) Str(s);
                    break;
                case int[] ia:
                    w.Write((uint)GgufValueType.Array); w.Write((uint)GgufValueType.Int32); w.Write((ulong)ia.Length);
                    foreach (var i in ia) w.Write(i);
                    break;
                default: throw new NotSupportedException(v.GetType().Name);
            }
        }
        long offset = 0, end = 0;
        foreach (var t in _tensors)
        {
            Str(t.Name);
            w.Write((uint)t.Dims.Length);
            foreach (var d in t.Dims) w.Write((ulong)d);
            w.Write((uint)t.Type);
            w.Write((ulong)offset);
            end = offset + t.Bytes;
            offset = (end + alignment - 1) / alignment * alignment;
        }
        w.Flush();
        long header = (ms.Length + alignment - 1) / alignment * alignment;
        // the file ends right after the last tensor (no trailing padding)
        ms.SetLength(header + end);
        return ms.ToArray();
    }
}

/// <summary>Wraps a stream hiding its length / seeking, and counts the bytes read.</summary>
internal sealed class NonSeekableStream(Stream inner) : Stream
{
    public long BytesRead { get; private set; }
    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
    public override int Read(byte[] buffer, int offset, int count) { int n = inner.Read(buffer, offset, count); BytesRead += n; return n; }
    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        int n = await inner.ReadAsync(buffer, cancellationToken);
        BytesRead += n;
        return n;
    }
    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    protected override void Dispose(bool disposing) { if (disposing) inner.Dispose(); base.Dispose(disposing); }
}
