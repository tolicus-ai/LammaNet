using System.Globalization;
using System.Text;
using LammaNet.Compute;
using LammaNet.Gguf;
using LammaNet.Tensors;
using LammaNet.Tokenization;

namespace LammaNet.Models;

/// <summary>Options for <see cref="ModelInfo"/>.</summary>
public sealed class ModelInfoOptions
{
    /// <summary>
    /// Load the vocabulary arrays (token strings, scores, token types, merges). When false only their lengths are kept,
    /// which saves the allocations (the bytes still have to be read: they precede the tensor table).
    /// </summary>
    public bool LoadVocabulary { get; init; } = true;

    internal long MaxArrayLength => LoadVocabulary ? long.MaxValue : 4096;
}

/// <summary>
/// Everything a GGUF file says about its model, read from the header only (no weights are loaded or touched):
/// general / provenance metadata, the hyper-parameters as the engine derives them, tokenizer configuration and vocabulary,
/// the tensor table (names, shapes, types) with a quantization summary, and multimodal projector (mmproj) details.
/// <code>
/// var info = ModelInfo.Read("Qwen3-0.6B-Q4_K_M.gguf");
/// Console.WriteLine($"{info.Architecture} {info.Weights.FileTypeName} ctx={info.ContextLength} heads={info.HeadCount}");
/// </code>
/// </summary>
public sealed class ModelInfo
{
    /// <summary>File path or URL the information was read from (null for an anonymous stream).</summary>
    public string? Source { get; private init; }
    public uint GgufVersion { get; private init; }
    /// <summary>Size of the file, when known (local files).</summary>
    public long? FileSize { get; private init; }
    /// <summary>Offset of the tensor data: the header size including alignment padding.</summary>
    public long DataOffset { get; private init; }
    public int Alignment { get; private init; }
    /// <summary>All key/value metadata (arrays not loaded appear as <see cref="GgufArraySummary"/>).</summary>
    public GgufMetadata Metadata { get; private init; } = null!;
    /// <summary>Tensor names, shapes (ggml order: innermost first), types and data offsets.</summary>
    public IReadOnlyList<GgufTensorInfo> Tensors { get; private init; } = [];

    public GeneralInfo General { get; private init; } = null!;
    /// <summary>The model architecture (<c>general.architecture</c>; <c>clip</c> for multimodal projectors).</summary>
    public string Architecture => General.Architecture;
    /// <summary>
    /// Hyper-parameters exactly as <see cref="Model.Load"/> derives them (per-layer heads, kinds, windows, rope, ...);
    /// null for files that are not decoder / embedding models (projectors) or that lack the required keys.
    /// From a stream the rope frequency factors (<c>rope_freqs.weight</c>, tensor data) are not available.
    /// </summary>
    public ModelConfig? Config { get; private init; }
    public TokenizerInfo? Tokenizer { get; private init; }
    public WeightsInfo Weights { get; private init; } = null!;
    /// <summary>Multimodal projector details (mmproj files), otherwise null.</summary>
    public ProjectorInfo? Projector { get; private init; }
    /// <summary>RoPE configuration of the full-attention layers (null for NoPE / non-attention models).</summary>
    public RopeInfo? Rope { get; private init; }

    // ---------------- convenience accessors (the largest per-layer value where layers differ)

    /// <summary>Training context length.</summary>
    public int? ContextLength => Config?.NCtxTrain;
    public int? EmbeddingLength => Config?.NEmbd;
    public int? LayerCount => Config?.NLayer;
    public int? HeadCount => Config is { NHead.Length: > 0 } c ? c.NHead.Max() : null;
    public int? HeadCountKv => Config is { NHeadKv.Length: > 0 } c ? c.NHeadKv.Max() : null;
    public int? HeadDim => Config is { HeadDimK.Length: > 0 } c ? c.HeadDimK.Max() : null;
    public int? FeedForwardLength => Config is { NFf.Length: > 0 } c ? c.NFf.Max() : null;
    public int? VocabularySize => Config?.NVocab ?? Tokenizer?.VocabularySize;
    public int? SlidingWindow => Config is { SlidingWindow: > 0 } c ? c.SlidingWindow : null;
    /// <summary>Mixture-of-experts: experts per layer and experts used per token (0 for dense models).</summary>
    public int ExpertCount { get; private init; }
    public int ExpertUsedCount { get; private init; }
    /// <summary>Layer mixers in order: A attention, D Gated DeltaNet, M Mamba2, C short convolution.</summary>
    public string? LayerPattern => Config == null ? null : new string(Config.Kinds.Select(k => k switch
    {
        LayerKind.Attention => 'A', LayerKind.GatedDeltaNet => 'D', LayerKind.ShortConv => 'C', _ => 'M',
    }).ToArray());

    /// <summary>Bytes a complete file must have (header + tensor data); null when a tensor type is unknown.</summary>
    public long? ExpectedFileSize { get; private init; }
    /// <summary>False when the file is shorter than its tensor table requires (truncated / partial download); null if unknown.</summary>
    public bool? IsComplete => FileSize is long fs && ExpectedFileSize is long es ? fs >= es : null;
    /// <summary>Bytes consumed from the source to read the header.</summary>
    internal long BytesRead { get; private init; }

    // ===================================================================== readers

    /// <summary>Reads the header of a local GGUF file (memory-mapped; only the header pages are touched).</summary>
    public static ModelInfo Read(string path, ModelInfoOptions? options = null)
    {
        options ??= new ModelInfoOptions();
        using var f = new GgufFile(path, options.MaxArrayLength);
        return FromGguf(f);
    }

    /// <summary>
    /// Reads the header from a stream (file, network, ...), consuming roughly the header bytes and not the tensor data.
    /// The stream needs no seeking; it is left positioned somewhere after the header.
    /// </summary>
    public static ModelInfo Read(Stream stream, ModelInfoOptions? options = null, string? source = null)
    {
        options ??= new ModelInfoOptions();
        var h = GgufHeader.Read(stream, options.MaxArrayLength, source, out long read);
        return Create(h, source, TryLength(stream), read, null);
    }

    /// <inheritdoc cref="Read(Stream, ModelInfoOptions?, string?)"/>
    public static async Task<ModelInfo> ReadAsync(Stream stream, ModelInfoOptions? options = null, string? source = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ModelInfoOptions();
        var (h, read) = await GgufHeader.ReadAsync(stream, options.MaxArrayLength, source, cancellationToken).ConfigureAwait(false);
        return Create(h, source, TryLength(stream), read, null);
    }

    /// <summary>Information about an already opened file (e.g. <see cref="Model.File"/>).</summary>
    public static ModelInfo FromGguf(GgufFile file)
    {
        var h = new GgufHeader
        {
            Version = file.Version, Alignment = file.Alignment, Metadata = file.Metadata,
            Tensors = file.Tensors.ToList(), DataOffset = file.DataOffset,
        };
        float[]? freqFactors = null;
        if (file.TryGetTensor("rope_freqs.weight", out var rf) && rf.HasKnownType)
            freqFactors = WeightTensor.FromGguf(file, rf).ToFloatArray();
        return Create(h, file.Path, file.FileSize, file.DataOffset, freqFactors);
    }

    private static long? TryLength(Stream s)
    {
        try { return s.CanSeek ? s.Length : null; }
        catch (NotSupportedException) { return null; }   // a non-seekable stream: the size is just unknown
    }

    private static ModelInfo Create(GgufHeader h, string? source, long? fileSize, long bytesRead, float[]? freqFactors)
    {
        var m = h.Metadata;
        var general = GeneralInfo.From(m);
        string arch = general.Architecture;
        ModelConfig? config = null;
        if (arch != "clip" && m.Has($"{arch}.block_count") && m.Has($"{arch}.embedding_length"))
        {
            try { config = ModelConfig.FromMetadata(m, freqFactors); }
            catch (Exception e) when (e is KeyNotFoundException or InvalidCastException or FormatException or OverflowException
                or InvalidOperationException or IndexOutOfRangeException or ArgumentException)
            {
                Diagnostics.Log.ModelConfigUnreadable(Diagnostics.LammaLogging.Static("LammaNet.Models.ModelInfo"), e, source ?? "GGUF");
            }
        }

        long? expected = h.DataOffset;
        foreach (var t in h.Tensors)
        {
            if (!t.HasKnownType) { expected = null; break; }
            expected = Math.Max(expected.Value, h.DataOffset + (long)t.Offset + t.ByteSize);
        }

        return new ModelInfo
        {
            Source = source,
            GgufVersion = h.Version,
            FileSize = fileSize,
            DataOffset = h.DataOffset,
            Alignment = h.Alignment,
            Metadata = m,
            Tensors = h.Tensors,
            General = general,
            Config = config,
            Tokenizer = TokenizerInfo.From(m),
            Weights = WeightsInfo.From(h.Tensors, m),
            Projector = arch == "clip" ? ProjectorInfo.From(m) : null,
            Rope = config == null ? null : RopeInfo.From(config, m),
            ExpertCount = m.Get($"{arch}.expert_count", 0),
            ExpertUsedCount = m.Get($"{arch}.expert_used_count", 0),
            ExpectedFileSize = expected,
            BytesRead = bytesRead,
        };
    }

    // ===================================================================== description

    /// <summary>A multi-line human-readable summary.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        var g = General;
        void Line(string label, object? value)
        {
            if (value is null || value is string { Length: 0 }) return;
            sb.Append("  ").Append(label.PadRight(18)).Append(' ').AppendLine(Convert.ToString(value, CultureInfo.InvariantCulture));
        }
        static string Mb(long bytes) => (bytes / 1e6).ToString("F1", CultureInfo.InvariantCulture) + " MB";

        sb.AppendLine($"{g.Name ?? Source ?? "GGUF"} ({Architecture}{(g.Type is { } ty && ty != "model" ? ", " + ty : "")})");
        Line("file", FileSize is long fs ? $"{Mb(fs)}, GGUF v{GgufVersion}, header {Mb(DataOffset)}" + (IsComplete == false ? " - INCOMPLETE" : "") : $"GGUF v{GgufVersion}, header {Mb(DataOffset)}");
        Line("basename", g.Basename); Line("finetune", g.Finetune); Line("size label", g.SizeLabel);
        Line("organization", g.Organization); Line("author", g.Author); Line("license", g.License);
        Line("quantized by", g.QuantizedBy); Line("repo", g.RepoUrl);
        if (g.BaseModels.Count > 0) Line("base models", string.Join(", ", g.BaseModels.Select(b => b.Name ?? b.RepoUrl)));
        if (g.Tags.Count > 0) Line("tags", string.Join(", ", g.Tags));

        var w = Weights;
        Line("weights", $"{w.FileTypeName ?? "?"} - {w.ParameterCount / 1e6:F1} M parameters, {Mb(w.ByteSize)}, {w.BitsPerWeight:F2} bits/weight" +
            (w.IsQuantized ? ", quantized" : ", unquantized"));
        foreach (var t in w.ByType)
            Line("  " + t.TypeName, $"{t.TensorCount} tensors, {t.ParameterCount / 1e6:F1} M parameters" + (t.ByteSize is long b ? $", {Mb(b)}" : ""));

        if (Config is { } c)
        {
            Line("layers", $"{c.NLayer} [{LayerPattern}]");
            Line("embedding", c.NEmbd + (c.NEmbdOut > 0 ? $" (output {c.NEmbdOut})" : ""));
            Line("context (train)", c.NCtxTrain);
            static string Distinct(int[] v) => string.Join("/", v.Where(x => x > 0).Distinct().Order());
            Line("attention heads", $"{Distinct(c.NHead)} (kv {Distinct(c.NHeadKv)}), head dim {Distinct(c.HeadDimK)}" +
                (!c.HeadDimV.SequenceEqual(c.HeadDimK) ? $" (v {Distinct(c.HeadDimV)})" : ""));
            Line("feed forward", FeedForwardLength);
            if (SlidingWindow is int sw) Line("sliding window", $"{sw} ({c.IsSwa.Count(x => x)} of {c.NLayer} layers{(c.SymmetricSwa ? ", symmetric" : "")})");
            if (ExpertCount > 0) Line("experts", $"{ExpertCount} ({ExpertUsedCount} used)");
            if (!c.Causal) Line("attention", "bidirectional");
            if (c.PoolingType > 0) Line("pooling", c.PoolingType switch { 1 => "mean", 2 => "cls", 3 => "last", 4 => "rank", var p => p.ToString(CultureInfo.InvariantCulture) });
            Line("norm epsilon", c.RmsEps);
        }
        if (Rope is { } r)
            Line("rope", $"{r.Mode}, {r.Dimensions} dims, base {r.FreqBase:G}" + (r.FreqBaseSwa is float bs ? $" (swa {bs:G})" : "") +
                (r.ScalingType != "none" ? $", {r.ScalingType} x{r.ScalingFactor:G} from {r.OriginalContextLength}" : "") +
                (r.HasFreqFactors ? ", freq factors" : "") + (r.Sections.Any(s => s > 0) ? $", sections [{string.Join(",", r.Sections)}]" : ""));

        if (Tokenizer is { } tk)
        {
            Line("tokenizer", $"{tk.Model}{(tk.Pre != null ? $" / {tk.Pre}" : "")}, {tk.VocabularySize} tokens" + (tk.MergeCount > 0 ? $", {tk.MergeCount} merges" : "") +
                (tk.ControlTokenCount is int cc ? $", {cc} control" : ""));
            Line("special tokens", string.Join(", ", tk.SpecialTokens.Select(kv => $"{kv.Key}={kv.Value}" + (tk.TokenText(kv.Value) is { } s ? $" '{s}'" : ""))));
            Line("add bos/eos", $"{tk.AddBos?.ToString() ?? "-"}/{tk.AddEos?.ToString() ?? "-"}");
            if (tk.ChatTemplate != null) Line("chat template", $"{tk.ChatTemplate.Length} chars" + (tk.ChatTemplates.Count > 0 ? $" (+ {string.Join(", ", tk.ChatTemplates.Keys)})" : ""));
        }

        if (Projector is { } pj)
        {
            foreach (var (label, e) in new[] { ("vision", pj.Vision), ("audio", pj.Audio) })
                if (e != null)
                    Line(label, $"{e.ProjectorType}: {e.BlockCount} blocks, {e.EmbeddingLength} wide, {e.HeadCount} heads -> {e.ProjectionDim}" +
                        (e.ImageSize is int im ? $", image {im}px / patch {e.PatchSize}" : "") + (e.MelBins is int mb ? $", {mb} mel bins" : ""));
        }
        return sb.ToString();
    }
}

/// <summary>Provenance metadata (<c>general.*</c>).</summary>
public sealed record GeneralInfo
{
    public required string Architecture { get; init; }
    /// <summary><c>model</c>, <c>mmproj</c>, <c>adapter</c>, ... (null when not stated).</summary>
    public string? Type { get; init; }
    public string? Name { get; init; }
    public string? Basename { get; init; }
    public string? Finetune { get; init; }
    public string? Description { get; init; }
    public string? Author { get; init; }
    public string? Organization { get; init; }
    public string? Version { get; init; }
    public string? SizeLabel { get; init; }
    public string? License { get; init; }
    public string? LicenseLink { get; init; }
    public string? Url { get; init; }
    public string? RepoUrl { get; init; }
    public string? QuantizedBy { get; init; }
    public IReadOnlyList<string> Tags { get; init; } = [];
    public IReadOnlyList<string> Languages { get; init; } = [];
    public IReadOnlyList<BaseModelInfo> BaseModels { get; init; } = [];
    /// <summary>Recommended sampling (<c>general.sampling.*</c>, key without the prefix → value).</summary>
    public IReadOnlyDictionary<string, object> RecommendedSampling { get; init; } = new Dictionary<string, object>();

    internal static GeneralInfo From(GgufMetadata m)
    {
        string? S(string k) => m.TryGetValue(k, out var v) && v is string s ? s : null;
        string[] A(string k) => m.TryGetValue(k, out var v) && v is string[] a ? a : [];
        var bases = new List<BaseModelInfo>();
        int nb = m.Get("general.base_model.count", 0);
        for (int i = 0; i < nb; i++)
        {
            string p = $"general.base_model.{i}.";
            bases.Add(new BaseModelInfo(S(p + "name"), S(p + "organization"), S(p + "repo_url")));
        }
        const string sp = "general.sampling.";
        return new GeneralInfo
        {
            Architecture = m.Get("general.architecture", "unknown"),
            Type = S("general.type"),
            Name = S("general.name"),
            Basename = S("general.basename"),
            Finetune = S("general.finetune"),
            Description = S("general.description"),
            Author = S("general.author"),
            Organization = S("general.organization"),
            Version = S("general.version"),
            SizeLabel = S("general.size_label"),
            License = S("general.license") ?? S("general.license.name"),
            LicenseLink = S("general.license.link"),
            Url = S("general.url"),
            RepoUrl = S("general.repo_url") ?? S("general.source.repo_url"),
            QuantizedBy = S("general.quantized_by"),
            Tags = A("general.tags"),
            Languages = A("general.languages"),
            BaseModels = bases,
            RecommendedSampling = m.Where(kv => kv.Key.StartsWith(sp, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key[sp.Length..], kv => kv.Value),
        };
    }
}

/// <summary>A model this one was derived from (<c>general.base_model.N.*</c>).</summary>
public sealed record BaseModelInfo(string? Name, string? Organization, string? RepoUrl);

/// <summary>Tokenizer configuration and vocabulary (<c>tokenizer.*</c>).</summary>
public sealed record TokenizerInfo
{
    /// <summary>Tokenizer model: <c>gpt2</c> (byte-level BPE), <c>llama</c> (SentencePiece), <c>gemma4</c>, <c>bert</c>, ...</summary>
    public required string Model { get; init; }
    /// <summary>Pre-tokenizer (regex family) for BPE tokenizers.</summary>
    public string? Pre { get; init; }
    public int VocabularySize { get; init; }
    /// <summary>Token strings by id; null when not loaded (<see cref="ModelInfoOptions.LoadVocabulary"/>).</summary>
    public IReadOnlyList<string>? Tokens { get; init; }
    /// <summary>Token types by id; null when absent or not loaded.</summary>
    public IReadOnlyList<TokenType>? TokenTypes { get; init; }
    public int MergeCount { get; init; }
    public bool HasScores { get; init; }
    public bool? AddBos { get; init; }
    public bool? AddEos { get; init; }
    public bool? AddSpacePrefix { get; init; }
    /// <summary>Special token ids by role (bos, eos, eot, eom, unk, sep, pad, mask, fim_pre, ...).</summary>
    public IReadOnlyDictionary<string, int> SpecialTokens { get; init; } = new Dictionary<string, int>();
    /// <summary>The default chat template (Jinja).</summary>
    public string? ChatTemplate { get; init; }
    /// <summary>Additional named chat templates (<c>tokenizer.chat_template.NAME</c>, e.g. tool_use, rag).</summary>
    public IReadOnlyDictionary<string, string> ChatTemplates { get; init; } = new Dictionary<string, string>();

    public int? ControlTokenCount => TokenTypes?.Count(t => t == TokenType.Control);
    public int? UserDefinedTokenCount => TokenTypes?.Count(t => t == TokenType.UserDefined);

    /// <summary>The text of a token id, when the vocabulary was loaded.</summary>
    public string? TokenText(int id) => Tokens != null && id >= 0 && id < Tokens.Count ? Tokens[id] : null;

    internal static TokenizerInfo? From(GgufMetadata m)
    {
        if (!m.Has("tokenizer.ggml.model") && !m.Has("tokenizer.ggml.tokens")) return null;
        bool? B(string k) => m.TryGetValue(k, out var v) && v is bool b ? b : null;
        var special = new Dictionary<string, int>();
        foreach (var role in new[] { "bos", "eos", "eot", "eom", "unknown", "seperator", "padding", "mask", "cls",
                     "fim_pre", "fim_suf", "fim_mid", "fim_pad", "fim_rep", "fim_sep" })
        {
            if (!m.TryGetValue($"tokenizer.ggml.{role}_token_id", out var v)) continue;
            int id;
            try { id = Convert.ToInt32(v, CultureInfo.InvariantCulture); } catch (OverflowException) { continue; }   // not a token id: no such special token
            if (id < 0) continue;
            special[role switch { "unknown" => "unk", "seperator" => "sep", "padding" => "pad", _ => role }] = id;
        }
        const string ct = "tokenizer.chat_template.";
        var types = m.TryGetValue("tokenizer.ggml.token_type", out var tt) && tt is Array ta
            ? ta.Cast<object>().Select(o => (TokenType)Convert.ToInt32(o, CultureInfo.InvariantCulture)).ToArray() : null;
        return new TokenizerInfo
        {
            Model = m.Get("tokenizer.ggml.model", "unknown"),
            Pre = m.TryGetValue("tokenizer.ggml.pre", out var pre) ? pre as string : null,
            VocabularySize = (int)Math.Max(0, m.ArrayLength("tokenizer.ggml.tokens")),
            Tokens = m.TryGetValue("tokenizer.ggml.tokens", out var toks) ? toks as string[] : null,
            TokenTypes = types,
            MergeCount = (int)Math.Max(0, m.ArrayLength("tokenizer.ggml.merges")),
            HasScores = m.Has("tokenizer.ggml.scores"),
            AddBos = B("tokenizer.ggml.add_bos_token"),
            AddEos = B("tokenizer.ggml.add_eos_token"),
            AddSpacePrefix = B("tokenizer.ggml.add_space_prefix"),
            SpecialTokens = special,
            ChatTemplate = m.TryGetValue("tokenizer.chat_template", out var tpl) ? tpl as string : null,
            ChatTemplates = m.Where(kv => kv.Key.StartsWith(ct, StringComparison.Ordinal) && kv.Value is string)
                .ToDictionary(kv => kv.Key[ct.Length..], kv => (string)kv.Value),
        };
    }
}

/// <summary>Tensors of one storage type.</summary>
public sealed record TensorTypeSummary(GgmlType Type, int TensorCount, long ParameterCount, long? ByteSize)
{
    /// <summary>The type name (a number for types unknown to this library).</summary>
    public string TypeName => GgmlTypeInfo.IsKnown(Type) ? Type.ToString() : $"type {(uint)Type}";
}

/// <summary>Weight storage: quantization type, parameter count and per-type breakdown.</summary>
public sealed record WeightsInfo
{
    /// <summary><c>general.file_type</c> (llama.cpp <c>llama_ftype</c>), if stated.</summary>
    public int? FileType { get; init; }
    /// <summary>The quantization scheme, e.g. <c>Q4_K_M</c>, <c>Q8_0</c>, <c>BF16</c> (from file_type, else the predominant tensor type).</summary>
    public string? FileTypeName { get; init; }
    public int? QuantizationVersion { get; init; }
    public long ParameterCount { get; init; }
    /// <summary>Bytes of tensor data (tensors of unknown types excluded).</summary>
    public long ByteSize { get; init; }
    public double BitsPerWeight { get; init; }
    /// <summary>True when the matrices are block-quantized (Q*/IQ*/TQ*/MXFP4) rather than stored as floats.</summary>
    public bool IsQuantized { get; init; }
    /// <summary>The type holding the most parameters.</summary>
    public GgmlType? PredominantType { get; init; }
    /// <summary>Per-type breakdown, largest first.</summary>
    public IReadOnlyList<TensorTypeSummary> ByType { get; init; } = [];

    internal static WeightsInfo From(IReadOnlyList<GgufTensorInfo> tensors, GgufMetadata m)
    {
        var byType = tensors.GroupBy(t => t.Type).Select(g => new TensorTypeSummary(g.Key, g.Count(), g.Sum(t => t.ElementCount),
                GgmlTypeInfo.IsKnown(g.Key) ? g.Sum(t => t.ByteSize) : null))
            .OrderByDescending(s => s.ByteSize ?? 0).ThenByDescending(s => s.ParameterCount).ToList();
        long pars = byType.Sum(s => s.ParameterCount);
        long bytes = byType.Sum(s => s.ByteSize ?? 0);
        long knownPars = byType.Where(s => s.ByteSize != null).Sum(s => s.ParameterCount);
        var main = byType.Count > 0 ? byType.MaxBy(s => s.ParameterCount)!.Type : (GgmlType?)null;
        int? ftype = m.TryGetValue("general.file_type", out var ft) ? Convert.ToInt32(ft, CultureInfo.InvariantCulture) : null;
        // quantized = matrices (2-D+) stored in a block format
        bool quantized = tensors.Any(t => t.Dims.Length >= 2 && (!GgmlTypeInfo.IsKnown(t.Type) || GgmlTypeInfo.IsQuantized(t.Type)));
        // general.file_type can disagree with the tensors (e.g. a BF16-labelled file holding Q4_0 matrices): the tensors win
        string? declared = ftype is int f ? FileTypeNameOf(f) : null;
        bool declaredFloat = declared is "F32" or "F16" or "BF16";
        string? name = declared == null || (declaredFloat == quantized && main != null)
            ? (declared == null ? main?.ToString() : $"{main} (file_type says {declared})")
            : declared;
        return new WeightsInfo
        {
            FileType = ftype,
            FileTypeName = name,
            QuantizationVersion = m.TryGetValue("general.quantization_version", out var qv) ? Convert.ToInt32(qv, CultureInfo.InvariantCulture) : null,
            ParameterCount = pars,
            ByteSize = bytes,
            BitsPerWeight = knownPars > 0 ? bytes * 8.0 / knownPars : 0,
            IsQuantized = quantized,
            PredominantType = main,
            ByType = byType,
        };
    }

    /// <summary>Name of a llama.cpp <c>llama_ftype</c> value (include/llama.h).</summary>
    public static string FileTypeNameOf(int fileType) => fileType switch
    {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 7 => "Q8_0", 8 => "Q5_0", 9 => "Q5_1",
        10 => "Q2_K", 11 => "Q3_K_S", 12 => "Q3_K_M", 13 => "Q3_K_L", 14 => "Q4_K_S", 15 => "Q4_K_M",
        16 => "Q5_K_S", 17 => "Q5_K_M", 18 => "Q6_K", 19 => "IQ2_XXS", 20 => "IQ2_XS", 21 => "Q2_K_S",
        22 => "IQ3_XS", 23 => "IQ3_XXS", 24 => "IQ1_S", 25 => "IQ4_NL", 26 => "IQ3_S", 27 => "IQ3_M",
        28 => "IQ2_S", 29 => "IQ2_M", 30 => "IQ4_XS", 31 => "IQ1_M", 32 => "BF16",
        36 => "TQ1_0", 37 => "TQ2_0", 38 => "MXFP4_MOE", 39 => "NVFP4", 40 => "Q1_0", 41 => "Q2_0",
        _ => $"file type {fileType}",
    };
}

/// <summary>Rotary position embedding of the full-attention layers (and the sliding-window layers where they differ).</summary>
public sealed record RopeInfo
{
    public RopeMode Mode { get; init; }
    /// <summary>Rotated dimensions per head.</summary>
    public int Dimensions { get; init; }
    public float FreqBase { get; init; }
    public int? DimensionsSwa { get; init; }
    public float? FreqBaseSwa { get; init; }
    /// <summary><c>none</c>, <c>linear</c>, <c>yarn</c>, ... (<c>rope.scaling.type</c>).</summary>
    public string ScalingType { get; init; } = "none";
    public float ScalingFactor { get; init; } = 1f;
    public int? OriginalContextLength { get; init; }
    /// <summary>M-RoPE / interleaved M-RoPE sections (t, h, w, e).</summary>
    public IReadOnlyList<int> Sections { get; init; } = [];
    /// <summary>Per-dimension frequency factors are stored in a <c>rope_freqs.weight</c> tensor.</summary>
    public bool HasFreqFactors { get; init; }

    internal static RopeInfo? From(ModelConfig c, GgufMetadata m)
    {
        RopeConfig? full = null, swa = null;
        for (int i = 0; i < c.NLayer; i++)
        {
            if (c.Rope[i] is not { } r) continue;
            if (c.IsSwa[i]) swa ??= r; else full ??= r;
        }
        var main = full ?? swa;
        if (main == null) return null;
        string P(string k) => $"{c.Architecture}.{k}";
        float factor = m.Get(P("rope.scaling.factor"), m.Get(P("rope.scale_linear"), 0f));
        string type = m.Get(P("rope.scaling.type"), factor != 0f ? "linear" : "none");
        return new RopeInfo
        {
            Mode = main.Mode,
            Dimensions = main.Dims,
            FreqBase = main.FreqBase,
            DimensionsSwa = full != null && swa != null && swa.Dims != full.Dims ? swa.Dims : null,
            FreqBaseSwa = full != null && swa != null && swa.FreqBase != full.FreqBase ? swa.FreqBase : null,
            ScalingType = type,
            ScalingFactor = factor == 0f ? 1f : factor,
            OriginalContextLength = m.TryGet<int>(P("rope.scaling.original_context_length"), out var oc) ? oc : null,
            Sections = main.Sections,
            HasFreqFactors = main.FreqFactors != null || m.Has(P("rope.freq_factors")),
        };
    }
}

/// <summary>Multimodal projector (mmproj, <c>general.architecture = clip</c>) encoders.</summary>
public sealed record ProjectorInfo
{
    public EncoderInfo? Vision { get; init; }
    public EncoderInfo? Audio { get; init; }

    internal static ProjectorInfo From(GgufMetadata m)
    {
        EncoderInfo? Enc(string kind, bool present)
        {
            if (!present) return null;
            string P(string k) => $"clip.{kind}.{k}";
            int? I(string k) => m.TryGet<int>(P(k), out var v) ? v : null;
            return new EncoderInfo
            {
                ProjectorType = m.Get<string?>(P("projector_type"), null) ?? m.Get<string?>("clip.projector_type", null),
                EmbeddingLength = I("embedding_length"),
                FeedForwardLength = I("feed_forward_length"),
                BlockCount = I("block_count"),
                HeadCount = I("attention.head_count"),
                ProjectionDim = I("projection_dim"),
                Epsilon = m.TryGet<float>(P("attention.layer_norm_epsilon"), out var e) ? e : null,
                ImageSize = I("image_size"),
                PatchSize = I("patch_size"),
                MelBins = I("num_mel_bins"),
            };
        }
        bool vision = m.Get("clip.has_vision_encoder", m.Has("clip.vision.block_count"));
        bool audio = m.Get("clip.has_audio_encoder", m.Has("clip.audio.block_count"));
        return new ProjectorInfo { Vision = Enc("vision", vision), Audio = Enc("audio", audio) };
    }
}

/// <summary>One encoder of a multimodal projector.</summary>
public sealed record EncoderInfo
{
    /// <summary>The projector / preprocessing family (gemma4v, qwen3vl_merger, idefics3, gemma4a, ...).</summary>
    public string? ProjectorType { get; init; }
    public int? EmbeddingLength { get; init; }
    public int? FeedForwardLength { get; init; }
    public int? BlockCount { get; init; }
    public int? HeadCount { get; init; }
    /// <summary>Width of the embeddings handed to the language model.</summary>
    public int? ProjectionDim { get; init; }
    public float? Epsilon { get; init; }
    public int? ImageSize { get; init; }
    public int? PatchSize { get; init; }
    public int? MelBins { get; init; }
}
