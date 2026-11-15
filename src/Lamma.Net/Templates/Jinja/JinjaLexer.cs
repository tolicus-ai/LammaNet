using System.Text;

namespace LammaNet.Templates.Jinja;

internal enum TokKind
{
    Text, VarStart, VarEnd, BlockStart, BlockEnd,
    Name, String, Int, Float, Op, Eof,
}

internal readonly record struct Tok(TokKind Kind, string Value, int Pos);

/// <summary>
/// Jinja2 lexer with whitespace control ("-" modifiers) plus HF-style trim_blocks / lstrip_blocks.
/// </summary>
internal sealed class JinjaLexer
{
    private readonly string _src;
    private int _pos;
    private readonly List<Tok> _out = new();
    private readonly bool _trimBlocks, _lstripBlocks;

    public JinjaLexer(string src, bool trimBlocks = true, bool lstripBlocks = true)
    {
        _src = src.Replace("\r\n", "\n");
        _trimBlocks = trimBlocks;
        _lstripBlocks = lstripBlocks;
    }

    public List<Tok> Lex()
    {
        var text = new StringBuilder();
        bool stripNextLeading = false; // from "-%}" / "-}}"
        bool trimNextNewline = false;  // trim_blocks after a block tag
        while (_pos < _src.Length)
        {
            int next = IndexOfTagStart(_pos);
            string chunk = next < 0 ? _src[_pos..] : _src[_pos..next];
            if (trimNextNewline && chunk.StartsWith('\n')) chunk = chunk[1..];
            trimNextNewline = false;
            if (stripNextLeading) { chunk = chunk.TrimStart(); stripNextLeading = false; }
            if (next < 0) { EmitText(chunk); break; }

            char kind = _src[next + 1];
            bool leftStrip = next + 2 < _src.Length && _src[next + 2] == '-';
            bool leftKeep = next + 2 < _src.Length && _src[next + 2] == '+';
            if (leftStrip) chunk = chunk.TrimEnd();
            else if (kind != '{' && _lstripBlocks && !leftKeep)
            {
                // lstrip_blocks: strip spaces/tabs between line start and a block/comment tag
                int b = next;
                while (b > 0 && (_src[b - 1] == ' ' || _src[b - 1] == '\t')) b--;
                if (b == 0 || _src[b - 1] == '\n')
                {
                    int ws = next - b;
                    int trail = 0;
                    while (trail < chunk.Length && trail < ws && (chunk[^(trail + 1)] == ' ' || chunk[^(trail + 1)] == '\t')) trail++;
                    chunk = chunk[..^trail];
                }
            }
            EmitText(chunk);
            _pos = next + 2 + (leftStrip || leftKeep ? 1 : 0);

            if (kind == '#')
            {
                int end = _src.IndexOf("#}", _pos, StringComparison.Ordinal);
                if (end < 0) throw new JinjaException("Unclosed comment");
                bool rs = end > 0 && _src[end - 1] == '-';
                _pos = end + 2;
                if (rs) stripNextLeading = true;
                else if (_trimBlocks) trimNextNewline = true;
                continue;
            }

            if (kind == '%' && TryRaw(out var raw)) { EmitText(raw); continue; }

            _out.Add(new Tok(kind == '{' ? TokKind.VarStart : TokKind.BlockStart, "", next));
            string close = kind == '{' ? "}}" : "%}";
            bool rightStrip = LexExpression(close);
            _out.Add(new Tok(kind == '{' ? TokKind.VarEnd : TokKind.BlockEnd, "", _pos));
            if (rightStrip) stripNextLeading = true;
            else if (kind == '%' && _trimBlocks) trimNextNewline = true;
        }
        _out.Add(new Tok(TokKind.Eof, "", _pos));
        return _out;
    }

    private void EmitText(string s)
    {
        if (s.Length > 0) _out.Add(new Tok(TokKind.Text, s, _pos));
    }

    private int IndexOfTagStart(int from)
    {
        for (int i = from; i < _src.Length - 1; i++)
            if (_src[i] == '{' && (_src[i + 1] == '{' || _src[i + 1] == '%' || _src[i + 1] == '#')) return i;
        return -1;
    }

    private bool TryRaw(out string raw)
    {
        raw = "";
        int p = _pos;
        while (p < _src.Length && char.IsWhiteSpace(_src[p])) p++;
        if (!_src.AsSpan(p).StartsWith("raw")) return false;
        int close = _src.IndexOf("%}", p, StringComparison.Ordinal);
        if (close < 0) return false;
        string head = _src[p..close].Trim().TrimEnd('-').Trim();
        if (head != "raw") return false;
        int bodyStart = close + 2;
        int endTag = _src.IndexOf("endraw", bodyStart, StringComparison.Ordinal);
        if (endTag < 0) throw new JinjaException("Unclosed raw block");
        int tagOpen = _src.LastIndexOf("{%", endTag, StringComparison.Ordinal);
        raw = _src[bodyStart..tagOpen];
        int endClose = _src.IndexOf("%}", endTag, StringComparison.Ordinal);
        _pos = endClose + 2;
        return true;
    }

    private static readonly string[] Ops3 = ["**=", "..."];
    private static readonly string[] Ops2 = ["==", "!=", "<=", ">=", "//", "**", "+=", "-="];

    /// <summary>Lexes tokens until the closing delimiter; returns true when the closer had a '-' modifier.</summary>
    private bool LexExpression(string close)
    {
        while (true)
        {
            while (_pos < _src.Length && char.IsWhiteSpace(_src[_pos])) _pos++;
            if (_pos >= _src.Length) throw new JinjaException($"Unclosed tag, expected '{close}'");
            if (_src[_pos] == '-' && _src.AsSpan(_pos + 1).StartsWith(close)) { _pos += 1 + close.Length; return true; }
            if (_src[_pos] == '+' && _src.AsSpan(_pos + 1).StartsWith(close)) { _pos += 1 + close.Length; return false; }
            if (_src.AsSpan(_pos).StartsWith(close)) { _pos += close.Length; return false; }
            char c = _src[_pos];
            int start = _pos;
            if (char.IsLetter(c) || c == '_')
            {
                while (_pos < _src.Length && (char.IsLetterOrDigit(_src[_pos]) || _src[_pos] == '_')) _pos++;
                _out.Add(new Tok(TokKind.Name, _src[start.._pos], start));
            }
            else if (char.IsDigit(c))
            {
                while (_pos < _src.Length && (char.IsDigit(_src[_pos]) || _src[_pos] == '_')) _pos++;
                bool isFloat = false;
                if (_pos + 1 < _src.Length && _src[_pos] == '.' && char.IsDigit(_src[_pos + 1]))
                {
                    isFloat = true; _pos++;
                    while (_pos < _src.Length && char.IsDigit(_src[_pos])) _pos++;
                }
                if (_pos < _src.Length && (_src[_pos] == 'e' || _src[_pos] == 'E'))
                {
                    isFloat = true; _pos++;
                    if (_src[_pos] == '+' || _src[_pos] == '-') _pos++;
                    while (_pos < _src.Length && char.IsDigit(_src[_pos])) _pos++;
                }
                _out.Add(new Tok(isFloat ? TokKind.Float : TokKind.Int, _src[start.._pos].Replace("_", ""), start));
            }
            else if (c == '"' || c == '\'')
            {
                _out.Add(new Tok(TokKind.String, ReadString(c), start));
            }
            else
            {
                string op = Ops2.FirstOrDefault(o => _src.AsSpan(_pos).StartsWith(o)) ?? c.ToString();
                _pos += op.Length;
                _out.Add(new Tok(TokKind.Op, op, start));
            }
        }
    }

    private string ReadString(char quote)
    {
        var sb = new StringBuilder();
        _pos++;
        while (_pos < _src.Length && _src[_pos] != quote)
        {
            char c = _src[_pos++];
            if (c == '\\' && _pos < _src.Length)
            {
                char e = _src[_pos++];
                switch (e)
                {
                    case 'n': sb.Append('\n'); break;
                    case 't': sb.Append('\t'); break;
                    case 'r': sb.Append('\r'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case '0': sb.Append('\0'); break;
                    case '\\': sb.Append('\\'); break;
                    case '\'': sb.Append('\''); break;
                    case '"': sb.Append('"'); break;
                    case 'u' when _pos + 4 <= _src.Length:
                        sb.Append((char)Convert.ToInt32(_src.Substring(_pos, 4), 16)); _pos += 4; break;
                    case 'x' when _pos + 2 <= _src.Length:
                        sb.Append((char)Convert.ToInt32(_src.Substring(_pos, 2), 16)); _pos += 2; break;
                    default: sb.Append('\\').Append(e); break;
                }
            }
            else sb.Append(c);
        }
        _pos++; // closing quote
        return sb.ToString();
    }
}

/// <summary>Template syntax or runtime error.</summary>
public sealed class JinjaException(string message) : Exception(message);
