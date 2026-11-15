using System.Collections.Concurrent;
using System.Text;
using System.Text.RegularExpressions;
using LammaNet.Gguf;

namespace LammaNet.Tokenization;

/// <summary>Token attribute as stored in GGUF (<c>tokenizer.ggml.token_type</c>).</summary>
public enum TokenType
{
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

/// <summary>Tokenizer algorithm.</summary>
public enum TokenizerKind
{
    /// <summary>SentencePiece unigram/BPE with scores (Llama 1/2, Gemma 1-3, Mistral v1...).</summary>
    Spm,
    /// <summary>Byte-level BPE (GPT-2 style) with a regex pre-tokenizer (Llama 3, Qwen, Granite, ...).</summary>
    Bpe,
    /// <summary>SentencePiece-style BPE over raw UTF-8 with ▁ for spaces (Gemma 4).</summary>
    BpeRawUtf8,
}

/// <summary>
/// Model vocabulary and tokenizer, loaded from GGUF metadata. Thread-safe.
/// Behaviour mirrors llama.cpp's <c>llama_vocab</c> (special-token partitioning, pre-tokenizer regexes, byte fallback).
/// </summary>
public sealed partial class Vocabulary
{
    private readonly string[] _tokens;
    private readonly float[] _scores;
    private readonly TokenType[] _types;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<string, int> _mergeRanks = new(StringComparer.Ordinal);
    private readonly Regex[] _preTokenizers;
    private readonly bool _ignoreMerges;
    private readonly bool _escapeWhitespace;
    private readonly bool _addSpacePrefix;
    private readonly string[] _specialSorted; // control/user-defined tokens, longest first
    private readonly HashSet<int> _eog = new();
    private readonly ConcurrentDictionary<string, int[]> _wordCache = new(StringComparer.Ordinal);
    private readonly int[] _byteTokens = new int[256];

    public TokenizerKind Kind { get; }
    public string Model { get; }
    public string PreTokenizer { get; }
    public int Count => _tokens.Length;
    public int Bos { get; } = -1;
    public int Eos { get; } = -1;
    public int Eot { get; } = -1;
    public int Unk { get; } = -1;
    public int Pad { get; } = -1;
    public bool AddBos { get; }
    public bool AddEos { get; }
    public IReadOnlySet<int> EndOfGeneration => _eog;

    private Vocabulary(GgufFile f)
    {
        Model = f.Get("tokenizer.ggml.model", "llama");
        _tokens = f.Get<string[]>("tokenizer.ggml.tokens");
        _scores = f.Has("tokenizer.ggml.scores") ? f.Get<float[]>("tokenizer.ggml.scores") : new float[_tokens.Length];
        var tt = f.Has("tokenizer.ggml.token_type") ? f.Get<int[]>("tokenizer.ggml.token_type") : null;
        _types = new TokenType[_tokens.Length];
        for (int i = 0; i < _tokens.Length; i++) _types[i] = tt != null ? (TokenType)tt[i] : TokenType.Normal;

        _tokenToId = new Dictionary<string, int>(_tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < _tokens.Length; i++) _tokenToId.TryAdd(_tokens[i], i);

        string pre = f.Get("tokenizer.ggml.pre", "default");
        bool addBosDefault;
        switch (Model)
        {
            case "gpt2":
                Kind = TokenizerKind.Bpe;
                addBosDefault = false;
                break;
            case "gemma4":
                Kind = TokenizerKind.BpeRawUtf8;
                pre = "gemma4";
                addBosDefault = true;
                _escapeWhitespace = true;
                break;
            case "llama":
            default:
                Kind = TokenizerKind.Spm;
                addBosDefault = true;
                _escapeWhitespace = true;
                _addSpacePrefix = f.Get("tokenizer.ggml.add_space_prefix", true);
                break;
        }
        PreTokenizer = pre;

        if (Kind != TokenizerKind.Spm && f.Has("tokenizer.ggml.merges"))
        {
            var merges = f.Get<string[]>("tokenizer.ggml.merges");
            for (int i = 0; i < merges.Length; i++)
            {
                string m = merges[i];
                int sp = m.IndexOf(' ', 1);
                if (sp < 0) continue;
                _mergeRanks.TryAdd(m, i);
            }
        }

        _preTokenizers = Kind == TokenizerKind.Spm ? [] : BuildPreTokenizers(pre, out _ignoreMerges, ref addBosDefault);

        Bos = f.Get("tokenizer.ggml.bos_token_id", -1);
        Eos = f.Get("tokenizer.ggml.eos_token_id", -1);
        Eot = f.Get("tokenizer.ggml.eot_token_id", -1);
        Unk = f.Get("tokenizer.ggml.unknown_token_id", -1);
        Pad = f.Get("tokenizer.ggml.padding_token_id", -1);
        AddBos = f.Get("tokenizer.ggml.add_bos_token", addBosDefault);
        if (Kind == TokenizerKind.BpeRawUtf8) AddBos = true; // gemma4 workaround (llama.cpp #21500)
        AddEos = f.Get("tokenizer.ggml.add_eos_token", false);

        // byte fallback tokens
        Array.Fill(_byteTokens, -1);
        for (int b = 0; b < 256; b++)
        {
            if (Kind == TokenizerKind.Bpe)
            {
                if (_tokenToId.TryGetValue(ByteEncoder.ByteToChar[b].ToString(), out int id)) _byteTokens[b] = id;
            }
            else if (_tokenToId.TryGetValue($"<0x{b:X2}>", out int id)) _byteTokens[b] = id;
        }

        // end-of-generation tokens (same heuristics as llama.cpp)
        if (Eos >= 0) _eog.Add(Eos);
        if (Eot >= 0) _eog.Add(Eot);
        if (f.TryGet<int>("tokenizer.ggml.eom_token_id", out var eom) && eom >= 0) _eog.Add(eom);
        string[] eogNames =
        [
            // same list as llama.cpp (llama-vocab.cpp); matched by name regardless of declared type
            "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
            "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
            "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~[", "<|ifm|im_end|>",
        ];
        foreach (var name in eogNames)
        {
            if (!_tokenToId.TryGetValue(name, out int id)) continue;
            _eog.Add(id);
            if (_types[id] is not (TokenType.Control or TokenType.UserDefined)) _types[id] = TokenType.Control; // llama.cpp overrides the type too
        }
        if (Eot < 0)
        {
            foreach (var name in new[] { "<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<turn|>", "<|end|>" })
                if (_tokenToId.TryGetValue(name, out int id)) { Eot = id; break; }
        }

        _specialSorted = Enumerable.Range(0, _tokens.Length)
            .Where(i => _types[i] is TokenType.Control or TokenType.UserDefined or TokenType.Unknown && _tokens[i].Length > 0)
            .Select(i => _tokens[i])
            .Distinct()
            .OrderByDescending(s => s.Length)
            .ToArray();
    }

    public static Vocabulary FromGguf(GgufFile f) => new(f);

    public string TokenText(int id) => _tokens[id];
    public TokenType TokenTypeOf(int id) => _types[id];
    public float Score(int id) => _scores[id];
    public bool IsEog(int id) => _eog.Contains(id);
    public bool IsControl(int id) => (uint)id < (uint)_types.Length && _types[id] == TokenType.Control;
    public int? Find(string text) => _tokenToId.TryGetValue(text, out int id) ? id : null;

    // ------------------------------------------------------------------ pre-tokenizers

    private const string Contractions = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";

    private static Regex[] BuildPreTokenizers(string pre, out bool ignoreMerges, ref bool addBos)
    {
        ignoreMerges = false;
        string[] patterns;
        switch (pre)
        {
            case "llama3":
            case "llama-v3":
            case "llama-bpe":
            case "falcon3":
            case "falcon-h1":
            case "pixtral":
            case "lfm2":
                ignoreMerges = true;
                addBos = true;
                patterns = [Contractions + @"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"];
                break;
            case "dbrx":
            case "smaug-bpe":
                patterns = [Contractions + @"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"];
                break;
            case "qwen2":
            case "deepseek-r1-qwen":
            case "stablelm2":
            case "hunyuan":
                patterns = [Contractions + @"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"];
                break;
            case "qwen35":
                patterns = [Contractions + @"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"];
                break;
            case "gemma4":
                patterns = [@"[^\n]+|[\n]+"];
                break;
            case "gpt-2":
            case "granite-docling":
            case "phi-2":
            case "olmo":
            case "mpt":
            case "jais":
                patterns = [@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"];
                break;
            case "starcoder":
            case "refact":
            case "command-r":
            case "smollm":
            case "codeshell":
            case "exaone":
            case "minerva-7b":
                patterns = [@"\p{N}", @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"];
                break;
            case "tekken":
                ignoreMerges = true;
                patterns = [@"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"];
                break;
            case "gpt-4o":
            case "minimax-m2":
                patterns = [@"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"];
                break;
            case "chatglm-bpe":
            case "glm4":
                patterns = [Contractions + @"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"];
                break;
            default:
                patterns = [@"[\p{P}\$\+<=>\^~\|]+", @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)", @"\p{N}+", "[0-9][0-9][0-9]"];
                break;
        }
        return patterns.Select(p => new Regex(p, RegexOptions.Compiled | RegexOptions.CultureInvariant)).ToArray();
    }

    /// <summary>Applies the regex chain; text not matched by a regex is kept as its own fragment (llama.cpp semantics).</summary>
    private List<string> PreTokenize(string text)
    {
        var words = new List<string> { text };
        foreach (var rx in _preTokenizers)
        {
            var next = new List<string>(words.Count * 2);
            foreach (var w in words)
            {
                int last = 0;
                for (var m = rx.Match(w); m.Success; m = m.NextMatch())
                {
                    if (m.Length == 0) continue;
                    if (m.Index > last) next.Add(w.Substring(last, m.Index - last));
                    next.Add(m.Value);
                    last = m.Index + m.Length;
                }
                if (last < w.Length) next.Add(w.Substring(last));
            }
            words = next;
        }
        return words;
    }

    // ------------------------------------------------------------------ tokenize

    /// <summary>
    /// Converts text to token ids.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <param name="addSpecial">Add BOS/EOS according to the model's configuration.</param>
    /// <param name="parseSpecial">Recognize control tokens written as text (e.g. "&lt;|im_start|&gt;").</param>
    public List<int> Tokenize(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        var output = new List<int>(text.Length / 3 + 4);
        if (addSpecial && AddBos && Bos >= 0) output.Add(Bos);
        bool isPrevSpecial = true;
        foreach (var (frag, tok) in Partition(text, parseSpecial))
        {
            if (tok >= 0) { output.Add(tok); isPrevSpecial = true; continue; }
            switch (Kind)
            {
                case TokenizerKind.Spm:
                {
                    string t = frag;
                    if (_addSpacePrefix && isPrevSpecial) t = " " + t;
                    t = t.Replace(' ', '▁');
                    TokenizeSpm(t, output);
                    break;
                }
                default:
                {
                    string t = _escapeWhitespace ? frag.Replace(' ', '▁') : frag;
                    TokenizeBpe(t, output);
                    break;
                }
            }
            isPrevSpecial = false;
        }
        if (addSpecial && AddEos && Eos >= 0) output.Add(Eos);
        return output;
    }

    /// <summary>Splits text into raw fragments and special-token matches (longest token first).</summary>
    private List<(string Text, int Token)> Partition(string text, bool parseSpecial)
    {
        var frags = new List<(string, int)> { (text, -1) };
        if (text.Length == 0) return frags;
        foreach (var special in _specialSorted)
        {
            int id = _tokenToId[special];
            if (!parseSpecial && _types[id] != TokenType.UserDefined) continue;
            for (int fi = 0; fi < frags.Count; fi++)
            {
                var (ft, ftok) = frags[fi];
                if (ftok >= 0) continue;
                int idx = ft.IndexOf(special, StringComparison.Ordinal);
                if (idx < 0) continue;
                var parts = new List<(string, int)>();
                int start = 0;
                while (idx >= 0)
                {
                    if (idx > start) parts.Add((ft.Substring(start, idx - start), -1));
                    parts.Add((special, id));
                    start = idx + special.Length;
                    idx = ft.IndexOf(special, start, StringComparison.Ordinal);
                }
                if (start < ft.Length) parts.Add((ft.Substring(start), -1));
                frags.RemoveAt(fi);
                frags.InsertRange(fi, parts);
                fi += parts.Count - 1;
            }
        }
        return frags;
    }

    private void TokenizeBpe(string text, List<int> output)
    {
        foreach (var word in PreTokenize(text))
        {
            string w = Kind == TokenizerKind.Bpe ? ByteEncoder.Encode(word) : word;
            if (_wordCache.TryGetValue(w, out var cached)) { output.AddRange(cached); continue; }
            var ids = new List<int>();
            BpeWord(w, ids);
            var arr = ids.ToArray();
            if (_wordCache.Count < 200_000) _wordCache.TryAdd(w, arr);
            output.AddRange(arr);
        }
    }

    private void BpeWord(string word, List<int> output)
    {
        if (_ignoreMerges && _tokenToId.TryGetValue(word, out int whole)) { output.Add(whole); return; }
        if (Kind == TokenizerKind.BpeRawUtf8 && word.Length > 0 && word.Trim('\n').Length == 0 && _tokenToId.TryGetValue(word, out int nl))
        {
            output.Add(nl);
            return;
        }

        // symbols are unicode scalar values (surrogate pairs kept together)
        var symStart = new List<int>();
        var symLen = new List<int>();
        for (int i = 0; i < word.Length;)
        {
            int len = char.IsHighSurrogate(word[i]) && i + 1 < word.Length ? 2 : 1;
            symStart.Add(i); symLen.Add(len);
            i += len;
        }
        int n = symStart.Count;
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }

        var queue = new PriorityQueue<(int Left, int Right, int LeftLen, int RightLen), (int Rank, int Left)>();
        void AddBigram(int l, int r)
        {
            if (l < 0 || r < 0) return;
            string key = string.Concat(word.AsSpan(symStart[l], symLen[l]), " ", word.AsSpan(symStart[r], symLen[r]));
            if (_mergeRanks.TryGetValue(key, out int rank)) queue.Enqueue((l, r, symLen[l], symLen[r]), (rank, l));
        }
        for (int i = 1; i < n; i++) AddBigram(i - 1, i);
        while (queue.TryDequeue(out var bg, out _))
        {
            if (symLen[bg.Left] == 0 || symLen[bg.Right] == 0) continue;
            if (symLen[bg.Left] != bg.LeftLen || symLen[bg.Right] != bg.RightLen || next[bg.Left] != bg.Right) continue; // stale
            symLen[bg.Left] += symLen[bg.Right];
            symLen[bg.Right] = 0;
            next[bg.Left] = next[bg.Right];
            if (next[bg.Right] >= 0) prev[next[bg.Right]] = bg.Left;
            AddBigram(prev[bg.Left], bg.Left);
            AddBigram(bg.Left, next[bg.Left]);
        }
        for (int i = 0; i != -1 && n > 0; i = next[i])
        {
            if (symLen[i] == 0) continue;
            string s = word.Substring(symStart[i], symLen[i]);
            if (_tokenToId.TryGetValue(s, out int id)) { output.Add(id); continue; }
            // byte fallback
            if (Kind == TokenizerKind.Bpe)
            {
                foreach (char c in s) if (_tokenToId.TryGetValue(c.ToString(), out int bid)) output.Add(bid);
            }
            else
            {
                foreach (byte b in Encoding.UTF8.GetBytes(s))
                {
                    int bid = _byteTokens[b];
                    if (bid >= 0) output.Add(bid);
                    else if (Unk >= 0) output.Add(Unk);
                }
            }
        }
    }

    private void TokenizeSpm(string text, List<int> output)
    {
        if (text.Length == 0) return;
        var symStart = new List<int>();
        var symLen = new List<int>();
        for (int i = 0; i < text.Length;)
        {
            int len = char.IsHighSurrogate(text[i]) && i + 1 < text.Length ? 2 : 1;
            symStart.Add(i); symLen.Add(len);
            i += len;
        }
        int n = symStart.Count;
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }
        var revMerge = new Dictionary<string, (int, int)>(StringComparer.Ordinal);
        var queue = new PriorityQueue<(int Left, int Right, int Size), (float NegScore, int Left)>();
        void TryAdd(int l, int r)
        {
            if (l < 0 || r < 0) return;
            string s = text.Substring(symStart[l], symLen[l] + symLen[r]);
            if (!_tokenToId.TryGetValue(s, out int id)) return;
            queue.Enqueue((l, r, s.Length), (-_scores[id], l));
            revMerge[s] = (l, r);
        }
        for (int i = 1; i < n; i++) TryAdd(i - 1, i);
        while (queue.TryDequeue(out var bg, out _))
        {
            if (symLen[bg.Left] == 0 || symLen[bg.Right] == 0 || symLen[bg.Left] + symLen[bg.Right] != bg.Size) continue;
            symLen[bg.Left] += symLen[bg.Right];
            symLen[bg.Right] = 0;
            next[bg.Left] = next[bg.Right];
            if (next[bg.Right] >= 0) prev[next[bg.Right]] = bg.Left;
            TryAdd(prev[bg.Left], bg.Left);
            TryAdd(bg.Left, next[bg.Left]);
        }
        void Resegment(int start, int len)
        {
            string s = text.Substring(start, len);
            if (_tokenToId.TryGetValue(s, out int id)) { output.Add(id); return; }
            if (!revMerge.TryGetValue(s, out var p))
            {
                foreach (byte b in Encoding.UTF8.GetBytes(s))
                {
                    int bid = _byteTokens[b];
                    output.Add(bid >= 0 ? bid : Math.Max(Unk, 0));
                }
                return;
            }
            Resegment(symStart[p.Item1], symStartLen(p.Item1, p.Item2));
            Resegment(symStart[p.Item2], start + len - symStart[p.Item2]);
        }
        int symStartLen(int l, int r) => symStart[r] - symStart[l];
        for (int i = 0; i != -1; i = next[i])
            if (symLen[i] > 0) Resegment(symStart[i], symLen[i]);
    }

    // ------------------------------------------------------------------ detokenize

    /// <summary>Returns the raw bytes of a token (UTF-8, possibly an incomplete sequence).</summary>
    public byte[] TokenBytes(int id, bool renderSpecial = false)
    {
        if ((uint)id >= (uint)_tokens.Length) return [];
        var type = _types[id];
        string t = _tokens[id];
        switch (type)
        {
            case TokenType.Control:
                return renderSpecial ? Encoding.UTF8.GetBytes(t) : [];
            case TokenType.Unknown:
                return renderSpecial ? Encoding.UTF8.GetBytes(t) : Encoding.UTF8.GetBytes("�");
            case TokenType.UserDefined:
                return Encoding.UTF8.GetBytes(t);
            case TokenType.Unused:
                return [];
            case TokenType.Byte:
                if (Kind != TokenizerKind.Bpe && t.Length == 6 && t.StartsWith("<0x"))
                    return [Convert.ToByte(t.Substring(3, 2), 16)];
                break;
        }
        return Kind switch
        {
            TokenizerKind.Bpe => ByteEncoder.Decode(t),
            _ => Encoding.UTF8.GetBytes(t.Replace('▁', ' ')),
        };
    }

    /// <summary>Decodes a full token sequence to text.</summary>
    public string Decode(IEnumerable<int> ids, bool renderSpecial = false, bool stripLeadingSpace = false)
    {
        var bytes = new List<byte>();
        bool first = true;
        foreach (var id in ids)
        {
            var b = TokenBytes(id, renderSpecial);
            if (first && stripLeadingSpace && Kind == TokenizerKind.Spm && _addSpacePrefix && b.Length > 0 && b[0] == (byte)' ')
                b = b[1..];
            bytes.AddRange(b);
            if (b.Length > 0) first = false;
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>Creates an incremental decoder that only emits complete UTF-8 characters.</summary>
    public StreamingDecoder CreateStreamingDecoder(bool renderSpecial = false) => new(this, renderSpecial);
}

/// <summary>Incremental token → text decoder that buffers partial UTF-8 sequences.</summary>
public sealed class StreamingDecoder
{
    private readonly Vocabulary _vocab;
    private readonly bool _renderSpecial;
    private readonly Decoder _utf8 = new UTF8Encoding(false, false).GetDecoder();
    private readonly char[] _chars = new char[64];

    internal StreamingDecoder(Vocabulary vocab, bool renderSpecial) { _vocab = vocab; _renderSpecial = renderSpecial; }

    public string Push(int token)
    {
        var bytes = _vocab.TokenBytes(token, _renderSpecial);
        if (bytes.Length == 0) return "";
        var sb = new StringBuilder();
        int off = 0;
        while (off < bytes.Length)
        {
            _utf8.Convert(bytes, off, bytes.Length - off, _chars, 0, _chars.Length, false, out int used, out int nChars, out _);
            sb.Append(_chars, 0, nChars);
            off += used;
        }
        return sb.ToString();
    }

    public string Flush()
    {
        int n = _utf8.GetChars([], 0, 0, _chars, 0, true);
        return new string(_chars, 0, n);
    }
}

/// <summary>GPT-2 byte ↔ unicode mapping used by byte-level BPE vocabularies.</summary>
internal static class ByteEncoder
{
    public static readonly char[] ByteToChar = Build();
    private static readonly Dictionary<char, byte> CharToByte = ByteToChar.Select((c, i) => (c, (byte)i)).ToDictionary(x => x.c, x => x.Item2);

    private static char[] Build()
    {
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n); n++; }
        }
        var map = new char[256];
        for (int i = 0; i < bs.Count; i++) map[bs[i]] = (char)cs[i];
        return map;
    }

    public static string Encode(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) chars[i] = ByteToChar[bytes[i]];
        return new string(chars);
    }

    public static byte[] Decode(string token)
    {
        var res = new byte[token.Length];
        int n = 0;
        foreach (char c in token)
        {
            if (CharToByte.TryGetValue(c, out byte b)) res[n++] = b;
            else
            {
                var extra = Encoding.UTF8.GetBytes(c.ToString());
                Array.Resize(ref res, res.Length + extra.Length);
                foreach (var e in extra) res[n++] = e;
            }
        }
        Array.Resize(ref res, n);
        return res;
    }
}
