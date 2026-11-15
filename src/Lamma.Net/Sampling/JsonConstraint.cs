using System.Text;
using LammaNet.Tokenization;

namespace LammaNet.Sampling;

/// <summary>
/// Constrains generation to a single syntactically valid JSON value (optionally requiring a top-level object).
/// Works on token text: each candidate token is simulated against a copy of the incremental JSON parser state.
/// </summary>
public sealed class JsonConstraint : ITokenConstraint
{
    private readonly Vocabulary _vocab;
    private readonly string?[] _texts;
    private JsonState _state;

    public JsonConstraint(Vocabulary vocab, bool requireObject = true)
    {
        _vocab = vocab;
        _texts = new string?[vocab.Count];
        _state = JsonState.Initial(requireObject);
    }

    private string? Text(int id)
    {
        if ((uint)id >= (uint)_texts.Length) return null;
        var t = _texts[id];
        if (t != null) return t;
        if (_vocab.IsControl(id)) return _texts[id] = "\u0000"; // never allowed inside JSON
        var bytes = _vocab.TokenBytes(id);
        return _texts[id] = Encoding.UTF8.GetString(bytes);
    }

    public bool IsComplete => _state.Done;
    public bool AllowsEndOfGeneration => _state.Done || _state.CanFinish;

    public bool IsAllowed(int token)
    {
        var t = Text(token);
        if (string.IsNullOrEmpty(t) || t[0] == '\u0000') return false;
        var s = _state.Clone();
        foreach (char c in t) if (!s.Feed(c)) return false;
        return true;
    }

    public void Accept(int token)
    {
        var t = Text(token);
        if (t == null) return;
        foreach (char c in t) _state.Feed(c);
    }

    /// <summary>Character-level incremental JSON recognizer (pushdown automaton).</summary>
    private sealed class JsonState
    {
        // container stack: 'o' object, 'a' array
        private char[] _stack = new char[16];
        private int _depth;
        private Expect _expect;
        private Lex _lex;
        private string _literal = "";
        private int _litPos;
        private int _unicode;
        private bool _numHasDigits, _numHasDot, _numHasExp, _numExpDigits, _numLeadingZero;
        private bool _requireObject;
        private bool _started;
        private int _wsRun;
        public bool Done { get; private set; }

        private enum Expect { Value, ObjKeyOrEnd, ObjKey, Colon, CommaOrEnd, End, ArrValueOrEnd }
        private enum Lex { None, String, StringEscape, StringUnicode, Number, Literal }
        private bool _stringIsKey;

        public bool CanFinish => Done || (_depth == 0 && _lex == Lex.Number && _numHasDigits && (!_numHasExp || _numExpDigits) && !_requireObject);

        public static JsonState Initial(bool requireObject) => new() { _expect = Expect.Value, _requireObject = requireObject };

        public JsonState Clone()
        {
            var c = (JsonState)MemberwiseClone();
            c._stack = (char[])_stack.Clone();
            return c;
        }

        private void Push(char c)
        {
            if (_depth == _stack.Length) Array.Resize(ref _stack, _depth * 2);
            _stack[_depth++] = c;
        }

        private void ValueEnded()
        {
            if (_depth == 0) { Done = true; _expect = Expect.End; }
            else _expect = Expect.CommaOrEnd;
        }

        public bool Feed(char c)
        {
            switch (_lex)
            {
                case Lex.String:
                    if (c == '"') { _lex = Lex.None; if (_stringIsKey) _expect = Expect.Colon; else ValueEnded(); return true; }
                    if (c == '\\') { _lex = Lex.StringEscape; return true; }
                    return c >= 0x20;
                case Lex.StringEscape:
                    if (c == 'u') { _lex = Lex.StringUnicode; _unicode = 0; return true; }
                    if ("\"\\/bfnrt".IndexOf(c) >= 0) { _lex = Lex.String; return true; }
                    return false;
                case Lex.StringUnicode:
                    if (!Uri.IsHexDigit(c)) return false;
                    if (++_unicode == 4) _lex = Lex.String;
                    return true;
                case Lex.Literal:
                    if (c != _literal[_litPos]) return false;
                    if (++_litPos == _literal.Length) { _lex = Lex.None; ValueEnded(); }
                    return true;
                case Lex.Number:
                    if (char.IsDigit(c))
                    {
                        if (_numLeadingZero && !_numHasDot && !_numHasExp) return false;
                        if (_numHasExp) _numExpDigits = true; else _numHasDigits = true;
                        return true;
                    }
                    if (c == '.' && !_numHasDot && !_numHasExp && _numHasDigits) { _numHasDot = true; _numLeadingZero = false; _numHasDigits = true; return true; }
                    if ((c == 'e' || c == 'E') && !_numHasExp && _numHasDigits) { _numHasExp = true; return true; }
                    if ((c == '+' || c == '-') && _numHasExp && !_numExpDigits) return true;
                    if (!_numHasDigits || (_numHasExp && !_numExpDigits)) return false;
                    _lex = Lex.None;
                    ValueEnded();
                    if (Done) return char.IsWhiteSpace(c);
                    return Feed(c);
            }

            if (char.IsWhiteSpace(c)) return ++_wsRun <= 48; // bounded whitespace (prevents endless blank output)
            _wsRun = 0;
            if (Done) return false;
            switch (_expect)
            {
                case Expect.Value:
                case Expect.ArrValueOrEnd:
                    if (_expect == Expect.ArrValueOrEnd && c == ']') { _depth--; ValueEnded(); return true; }
                    if (_depth == 0 && _requireObject && c != '{') return false;
                    _started = true;
                    switch (c)
                    {
                        case '{': Push('o'); _expect = Expect.ObjKeyOrEnd; return true;
                        case '[': Push('a'); _expect = Expect.ArrValueOrEnd; return true;
                        case '"': _lex = Lex.String; _stringIsKey = false; return true;
                        case 't': _lex = Lex.Literal; _literal = "true"; _litPos = 1; return true;
                        case 'f': _lex = Lex.Literal; _literal = "false"; _litPos = 1; return true;
                        case 'n': _lex = Lex.Literal; _literal = "null"; _litPos = 1; return true;
                        case '-':
                            _lex = Lex.Number; _numHasDigits = _numHasDot = _numHasExp = _numExpDigits = _numLeadingZero = false; return true;
                        default:
                            if (!char.IsDigit(c)) return false;
                            _lex = Lex.Number; _numHasDot = _numHasExp = _numExpDigits = false; _numHasDigits = true; _numLeadingZero = c == '0';
                            return true;
                    }
                case Expect.ObjKeyOrEnd:
                    if (c == '}') { _depth--; ValueEnded(); return true; }
                    goto case Expect.ObjKey;
                case Expect.ObjKey:
                    if (c != '"') return false;
                    _lex = Lex.String; _stringIsKey = true; return true;
                case Expect.Colon:
                    if (c != ':') return false;
                    _expect = Expect.Value; return true;
                case Expect.CommaOrEnd:
                {
                    char top = _stack[_depth - 1];
                    if (c == ',') { _expect = top == 'o' ? Expect.ObjKey : Expect.Value; return true; }
                    if ((c == '}' && top == 'o') || (c == ']' && top == 'a')) { _depth--; ValueEnded(); return true; }
                    return false;
                }
                default: return false;
            }
        }
    }
}
