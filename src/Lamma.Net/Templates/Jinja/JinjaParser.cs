using System.Globalization;

namespace LammaNet.Templates.Jinja;

/// <summary>Recursive-descent parser producing the template AST (Jinja2 grammar and precedence).</summary>
internal sealed class JinjaParser
{
    private readonly List<Tok> _t;
    private int _i;

    public JinjaParser(List<Tok> tokens) => _t = tokens;

    private Tok Peek(int o = 0) => _t[Math.Min(_i + o, _t.Count - 1)];
    private Tok Next() => _t[_i++];
    private bool IsOp(string op, int o = 0) => Peek(o).Kind == TokKind.Op && Peek(o).Value == op;
    private bool IsName(string n, int o = 0) => Peek(o).Kind == TokKind.Name && Peek(o).Value == n;

    private void ExpectOp(string op)
    {
        if (!IsOp(op)) throw Error($"expected '{op}'");
        _i++;
    }

    private string ExpectName()
    {
        if (Peek().Kind != TokKind.Name) throw Error("expected name");
        return Next().Value;
    }

    private void ExpectBlockEnd()
    {
        if (Peek().Kind != TokKind.BlockEnd) throw Error("expected end of block tag");
        _i++;
    }

    private JinjaException Error(string msg) => new($"Template syntax error at {Peek().Pos}: {msg} (got {Peek().Kind} '{Peek().Value}')");

    public List<Node> ParseTemplate()
    {
        var body = ParseBody(out var end);
        if (end != null) throw Error($"unexpected '{end}'");
        return body;
    }

    /// <summary>Parses nodes until an end tag (returned in <paramref name="endTag"/>) or EOF.</summary>
    private List<Node> ParseBody(out string? endTag, params string[] terminators)
    {
        var nodes = new List<Node>();
        endTag = null;
        while (true)
        {
            var t = Peek();
            switch (t.Kind)
            {
                case TokKind.Eof:
                    if (terminators.Length > 0) throw Error($"unexpected end of template, expected one of: {string.Join(", ", terminators)}");
                    return nodes;
                case TokKind.Text:
                    nodes.Add(new TextNode(Next().Value));
                    break;
                case TokKind.VarStart:
                    _i++;
                    nodes.Add(new OutputNode(ParseExpression()));
                    if (Peek().Kind != TokKind.VarEnd) throw Error("expected '}}'");
                    _i++;
                    break;
                case TokKind.BlockStart:
                {
                    var kw = Peek(1);
                    if (kw.Kind == TokKind.Name && terminators.Contains(kw.Value))
                    {
                        _i += 2;
                        endTag = kw.Value;
                        return nodes;
                    }
                    _i++;
                    nodes.Add(ParseStatement());
                    break;
                }
                default:
                    throw Error("unexpected token");
            }
        }
    }

    private Node ParseStatement()
    {
        string kw = ExpectName();
        switch (kw)
        {
            case "if": return ParseIf();
            case "for": return ParseFor();
            case "set": return ParseSet();
            case "macro": return ParseMacro();
            case "filter":
            {
                var filters = ParseFilterChain(first: true);
                ExpectBlockEnd();
                var body = ParseBody(out _, "endfilter");
                ExpectBlockEnd();
                return new FilterBlockNode(filters, body);
            }
            case "do":
            {
                var e = ParseExpression();
                ExpectBlockEnd();
                return new DoNode(e);
            }
            case "break": ExpectBlockEnd(); return new BreakNode();
            case "continue": ExpectBlockEnd(); return new ContinueNode();
            case "generation":
            {
                ExpectBlockEnd();
                var body = ParseBody(out _, "endgeneration");
                ExpectBlockEnd();
                return new IfNode([(new ConstExpr(true), body)], null);
            }
            default: throw Error($"unknown statement '{kw}'");
        }
    }

    private Node ParseIf()
    {
        var branches = new List<(Expr, List<Node>)>();
        var cond = ParseExpression();
        ExpectBlockEnd();
        List<Node>? elseBody = null;
        while (true)
        {
            var body = ParseBody(out var end, "elif", "else", "endif");
            branches.Add((cond, body));
            if (end == "elif") { cond = ParseExpression(); ExpectBlockEnd(); continue; }
            if (end == "else")
            {
                ExpectBlockEnd();
                elseBody = ParseBody(out _, "endif");
            }
            ExpectBlockEnd();
            break;
        }
        return new IfNode(branches, elseBody);
    }

    private Node ParseFor()
    {
        var targets = new List<string>();
        bool paren = IsOp("(");
        if (paren) _i++;
        targets.Add(ExpectName());
        while (IsOp(",")) { _i++; targets.Add(ExpectName()); }
        if (paren) ExpectOp(")");
        if (!IsName("in")) throw Error("expected 'in'");
        _i++;
        var iter = ParseExpression(noCondExpr: true);
        Expr? filter = null;
        if (IsName("if")) { _i++; filter = ParseExpression(noCondExpr: true); }
        if (IsName("recursive")) _i++;
        ExpectBlockEnd();
        var body = ParseBody(out var end, "else", "endfor");
        List<Node>? elseBody = null;
        if (end == "else")
        {
            ExpectBlockEnd();
            elseBody = ParseBody(out _, "endfor");
        }
        ExpectBlockEnd();
        return new ForNode(targets, iter, filter, body, elseBody);
    }

    private Node ParseSet()
    {
        var targets = new List<Expr> { ParsePrimaryTarget() };
        while (IsOp(",")) { _i++; targets.Add(ParsePrimaryTarget()); }
        if (IsOp("="))
        {
            _i++;
            var value = ParseExpression();
            if (IsOp(","))
            {
                var items = new List<Expr> { value };
                while (IsOp(",")) { _i++; items.Add(ParseExpression()); }
                value = new TupleExpr(items);
            }
            ExpectBlockEnd();
            return new SetNode(targets, value);
        }
        var filters = IsOp("|") ? ParseFilterChain(first: false) : new List<(string, List<Expr>)>();
        ExpectBlockEnd();
        var body = ParseBody(out _, "endset");
        ExpectBlockEnd();
        return new SetBlockNode(targets[0], body, filters);
    }

    private Expr ParsePrimaryTarget()
    {
        Expr e = new NameExpr(ExpectName());
        while (IsOp("."))
        {
            _i++;
            e = new AttrExpr(e, ExpectName());
        }
        return e;
    }

    private Node ParseMacro()
    {
        string name = ExpectName();
        ExpectOp("(");
        var ps = new List<(string, Expr?)>();
        while (!IsOp(")"))
        {
            string pn = ExpectName();
            Expr? def = null;
            if (IsOp("=")) { _i++; def = ParseExpression(); }
            ps.Add((pn, def));
            if (IsOp(",")) _i++;
        }
        ExpectOp(")");
        ExpectBlockEnd();
        var body = ParseBody(out _, "endmacro");
        ExpectBlockEnd();
        return new MacroNode(name, ps, body);
    }

    private List<(string, List<Expr>)> ParseFilterChain(bool first)
    {
        var res = new List<(string, List<Expr>)>();
        bool needPipe = !first;
        while (true)
        {
            if (needPipe) { if (!IsOp("|")) break; _i++; }
            needPipe = true;
            string name = ExpectName();
            var args = new List<Expr>();
            if (IsOp("("))
            {
                _i++;
                while (!IsOp(")")) { args.Add(ParseExpression()); if (IsOp(",")) _i++; }
                _i++;
            }
            res.Add((name, args));
        }
        return res;
    }

    // ------------------------------------------------------------------ expressions

    public Expr ParseExpression(bool noCondExpr = false)
    {
        var e = ParseOr();
        if (noCondExpr) return e;
        while (IsName("if"))
        {
            _i++;
            var test = ParseOr();
            Expr? els = null;
            if (IsName("else")) { _i++; els = ParseExpression(); }
            e = new CondExpr(test, e, els);
        }
        return e;
    }

    private Expr ParseOr()
    {
        var l = ParseAnd();
        while (IsName("or")) { _i++; l = new BinaryExpr("or", l, ParseAnd()); }
        return l;
    }

    private Expr ParseAnd()
    {
        var l = ParseNot();
        while (IsName("and")) { _i++; l = new BinaryExpr("and", l, ParseNot()); }
        return l;
    }

    private Expr ParseNot()
    {
        if (IsName("not")) { _i++; return new UnaryExpr("not", ParseNot()); }
        return ParseCompare();
    }

    private static readonly HashSet<string> CompareOps = ["==", "!=", "<", ">", "<=", ">="];

    private Expr ParseCompare()
    {
        var l = ParseConcat();
        while (true)
        {
            if (Peek().Kind == TokKind.Op && CompareOps.Contains(Peek().Value))
            {
                string op = Next().Value;
                l = new BinaryExpr(op, l, ParseConcat());
            }
            else if (IsName("in")) { _i++; l = new BinaryExpr("in", l, ParseConcat()); }
            else if (IsName("not") && IsName("in", 1)) { _i += 2; l = new BinaryExpr("notin", l, ParseConcat()); }
            else break;
        }
        return l;
    }

    private Expr ParseConcat()
    {
        var l = ParseMath1();
        while (IsOp("~")) { _i++; l = new BinaryExpr("~", l, ParseMath1()); }
        return l;
    }

    private Expr ParseMath1()
    {
        var l = ParseMath2();
        while (IsOp("+") || IsOp("-")) { string op = Next().Value; l = new BinaryExpr(op, l, ParseMath2()); }
        return l;
    }

    private Expr ParseMath2()
    {
        var l = ParsePow();
        while (IsOp("*") || IsOp("/") || IsOp("//") || IsOp("%")) { string op = Next().Value; l = new BinaryExpr(op, l, ParsePow()); }
        return l;
    }

    private Expr ParsePow()
    {
        var l = ParseUnary();
        while (IsOp("**")) { _i++; l = new BinaryExpr("**", l, ParseUnary()); }
        return l;
    }

    private Expr ParseUnary()
    {
        if (IsOp("-")) { _i++; return ParseFilterTest(new UnaryExpr("-", ParseUnary())); }
        if (IsOp("+")) { _i++; return ParseUnary(); }
        return ParseFilterTest(ParsePostfix(ParsePrimary()));
    }

    private Expr ParsePrimary()
    {
        var t = Next();
        switch (t.Kind)
        {
            case TokKind.String:
            {
                string s = t.Value;
                while (Peek().Kind == TokKind.String) s += Next().Value; // implicit concatenation
                return new ConstExpr(s);
            }
            case TokKind.Int: return new ConstExpr(long.Parse(t.Value, CultureInfo.InvariantCulture));
            case TokKind.Float: return new ConstExpr(double.Parse(t.Value, CultureInfo.InvariantCulture));
            case TokKind.Name:
                return t.Value switch
                {
                    "true" or "True" => new ConstExpr(true),
                    "false" or "False" => new ConstExpr(false),
                    "none" or "None" => new ConstExpr(null),
                    _ => new NameExpr(t.Value),
                };
            case TokKind.Op when t.Value == "(":
            {
                if (IsOp(")")) { _i++; return new TupleExpr([]); }
                var e = ParseExpression();
                if (IsOp(","))
                {
                    var items = new List<Expr> { e };
                    while (IsOp(",")) { _i++; if (IsOp(")")) break; items.Add(ParseExpression()); }
                    ExpectOp(")");
                    return new TupleExpr(items);
                }
                ExpectOp(")");
                return e;
            }
            case TokKind.Op when t.Value == "[":
            {
                var items = new List<Expr>();
                while (!IsOp("]")) { items.Add(ParseExpression()); if (IsOp(",")) _i++; }
                _i++;
                return new ListExpr(items);
            }
            case TokKind.Op when t.Value == "{":
            {
                var items = new List<(Expr, Expr)>();
                while (!IsOp("}"))
                {
                    var k = ParseExpression();
                    ExpectOp(":");
                    items.Add((k, ParseExpression()));
                    if (IsOp(",")) _i++;
                }
                _i++;
                return new DictExpr(items);
            }
        }
        _i--;
        throw Error("unexpected token in expression");
    }

    private Expr ParsePostfix(Expr e)
    {
        while (true)
        {
            if (IsOp("."))
            {
                _i++;
                var t = Next();
                if (t.Kind == TokKind.Int) e = new ItemExpr(e, new ConstExpr(long.Parse(t.Value)));
                else e = new AttrExpr(e, t.Value);
            }
            else if (IsOp("["))
            {
                _i++;
                Expr? a = null, b = null, c = null;
                bool slice = false;
                if (!IsOp(":")) a = ParseExpression();
                if (IsOp(":"))
                {
                    slice = true; _i++;
                    if (!IsOp("]") && !IsOp(":")) b = ParseExpression();
                    if (IsOp(":")) { _i++; if (!IsOp("]")) c = ParseExpression(); }
                }
                ExpectOp("]");
                e = slice ? new SliceExpr(e, a, b, c) : new ItemExpr(e, a!);
            }
            else if (IsOp("("))
            {
                var (args, kwargs) = ParseCallArgs();
                e = new CallExpr(e, args, kwargs);
            }
            else return e;
        }
    }

    private (List<Expr>, List<(string, Expr)>) ParseCallArgs()
    {
        ExpectOp("(");
        var args = new List<Expr>();
        var kwargs = new List<(string, Expr)>();
        while (!IsOp(")"))
        {
            if (Peek().Kind == TokKind.Name && IsOp("=", 1))
            {
                string n = Next().Value; _i++;
                kwargs.Add((n, ParseExpression()));
            }
            else args.Add(ParseExpression());
            if (IsOp(",")) _i++;
            else if (!IsOp(")")) throw Error("expected ',' or ')'");
        }
        _i++;
        return (args, kwargs);
    }

    private static readonly HashSet<string> NonArgNames = ["and", "or", "else", "not", "in", "is", "if"];

    private Expr ParseFilterTest(Expr e)
    {
        while (true)
        {
            if (IsOp("|"))
            {
                _i++;
                string name = ExpectName();
                while (IsOp(".")) { _i++; name += "." + ExpectName(); }
                var args = new List<Expr>();
                var kwargs = new List<(string, Expr)>();
                if (IsOp("(")) (args, kwargs) = ParseCallArgs();
                e = new FilterExpr(e, name, args, kwargs);
            }
            else if (IsName("is"))
            {
                _i++;
                bool neg = false;
                if (IsName("not")) { _i++; neg = true; }
                string name = Peek().Kind == TokKind.Name ? Next().Value : Next().Value switch { _ => throw Error("expected test name") };
                var args = new List<Expr>();
                if (IsOp("(")) args = ParseCallArgs().Item1;
                else if (Peek().Kind is TokKind.String or TokKind.Int or TokKind.Float
                         || (Peek().Kind == TokKind.Name && !NonArgNames.Contains(Peek().Value)))
                    args.Add(ParsePostfix(ParsePrimary()));
                e = new TestExpr(e, name, args, neg);
            }
            else return e;
        }
    }
}
