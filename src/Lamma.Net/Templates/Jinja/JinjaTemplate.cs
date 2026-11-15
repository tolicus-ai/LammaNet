using System.Collections;
using System.Globalization;
using System.Text;

namespace LammaNet.Templates.Jinja;

/// <summary>
/// A compiled Jinja2 template with the semantics used by Hugging Face chat templates
/// (trim_blocks, lstrip_blocks, loop controls, Python-compatible values, <c>tojson</c>, <c>raise_exception</c>).
/// </summary>
public sealed class JinjaTemplate
{
    private readonly List<Node> _nodes;

    public JinjaTemplate(string source)
    {
        var tokens = new JinjaLexer(source).Lex();
        _nodes = new JinjaParser(tokens).ParseTemplate();
    }

    /// <summary>Renders the template with the given variables (CLR objects are converted to Jinja values).</summary>
    /// <param name="variables">Template variables.</param>
    /// <param name="now">Clock used by <c>strftime_now</c> (defaults to the current local time).</param>
    public string Render(IDictionary<string, object?> variables, DateTime? now = null)
    {
        var root = new Scope(null);
        Builtins.Register(root, now);
        foreach (var (k, v) in variables) root.Set(k, JinjaValue.FromClr(v));
        var sb = new StringBuilder();
        new Interpreter(root).Exec(_nodes, root, sb);
        return sb.ToString();
    }

    internal sealed class Scope(Scope? parent)
    {
        private readonly Dictionary<string, object?> _vars = new(StringComparer.Ordinal);
        public Scope? Parent { get; } = parent;

        public bool TryGet(string name, out object? value)
        {
            for (var s = this; s != null; s = s.Parent)
                if (s._vars.TryGetValue(name, out value)) return true;
            value = null;
            return false;
        }

        public void Set(string name, object? value) => _vars[name] = value;
    }

    private enum Flow { Normal, Break, Continue }

    internal sealed class LoopInfo
    {
        public long Index0;
        public long Length;
        public object? Prev, Next;
    }

    private sealed class Interpreter(Scope root)
    {
        private readonly Scope _root = root;

        public Flow Exec(List<Node> nodes, Scope scope, StringBuilder sb)
        {
            foreach (var n in nodes)
            {
                var f = ExecNode(n, scope, sb);
                if (f != Flow.Normal) return f;
            }
            return Flow.Normal;
        }

        private Flow ExecNode(Node node, Scope scope, StringBuilder sb)
        {
            switch (node)
            {
                case TextNode t: sb.Append(t.Text); return Flow.Normal;
                case OutputNode o: sb.Append(JinjaValue.ToStr(Eval(o.Expr, scope))); return Flow.Normal;
                case IfNode iff:
                    foreach (var (cond, body) in iff.Branches)
                        if (JinjaValue.Truthy(Eval(cond, scope))) return Exec(body, scope, sb);
                    return iff.Else != null ? Exec(iff.Else, scope, sb) : Flow.Normal;
                case ForNode f: return ExecFor(f, scope, sb);
                case SetNode s:
                {
                    var val = Eval(s.Value, scope);
                    if (s.Targets.Count == 1) Assign(s.Targets[0], val, scope);
                    else
                    {
                        var items = JinjaValue.Iterate(val).ToList();
                        if (items.Count != s.Targets.Count) throw new JinjaException("cannot unpack: wrong number of values");
                        for (int i = 0; i < items.Count; i++) Assign(s.Targets[i], items[i], scope);
                    }
                    return Flow.Normal;
                }
                case SetBlockNode sb2:
                {
                    var inner = new StringBuilder();
                    Exec(sb2.Body, scope, inner);
                    object? val = inner.ToString();
                    foreach (var (name, args) in sb2.Filters) val = Builtins.ApplyFilter(this, name, val, args.Select(a => Eval(a, scope)).ToList(), new(), scope);
                    Assign(sb2.Target, val, scope);
                    return Flow.Normal;
                }
                case MacroNode m:
                    scope.Set(m.Name, MakeMacro(m, scope));
                    return Flow.Normal;
                case FilterBlockNode fb:
                {
                    var inner = new StringBuilder();
                    Exec(fb.Body, scope, inner);
                    object? val = inner.ToString();
                    foreach (var (name, args) in fb.Filters) val = Builtins.ApplyFilter(this, name, val, args.Select(a => Eval(a, scope)).ToList(), new(), scope);
                    sb.Append(JinjaValue.ToStr(val));
                    return Flow.Normal;
                }
                case DoNode d: Eval(d.Expr, scope); return Flow.Normal;
                case BreakNode: return Flow.Break;
                case ContinueNode: return Flow.Continue;
                default: throw new JinjaException($"Unsupported node {node.GetType().Name}");
            }
        }

        private void Assign(Expr target, object? value, Scope scope)
        {
            switch (target)
            {
                case NameExpr n: scope.Set(n.Name, value); break;
                case AttrExpr a:
                {
                    var obj = Eval(a.Target, scope);
                    if (obj is JinjaNamespace ns) ns.Values[a.Name] = value;
                    else if (obj is JinjaDict d) d[a.Name] = value;
                    else throw new JinjaException($"cannot assign attribute on {JinjaValue.TypeName(obj)}");
                    break;
                }
                default: throw new JinjaException("invalid assignment target");
            }
        }

        private Flow ExecFor(ForNode f, Scope scope, StringBuilder sb)
        {
            var iterVal = Eval(f.Iter, scope);
            List<object?> items = iterVal is JinjaDict dict && f.Targets.Count == 2
                ? dict.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList()
                : JinjaValue.Iterate(iterVal).ToList();
            if (f.Filter != null)
            {
                var filtered = new List<object?>();
                foreach (var it in items)
                {
                    var probe = new Scope(scope);
                    BindTargets(f.Targets, it, probe);
                    if (JinjaValue.Truthy(Eval(f.Filter, probe))) filtered.Add(it);
                }
                items = filtered;
            }
            if (items.Count == 0)
                return f.Else != null ? Exec(f.Else, scope, sb) : Flow.Normal;

            var info = new LoopInfo { Length = items.Count };
            var loopObj = MakeLoopObject(info);
            for (int i = 0; i < items.Count; i++)
            {
                info.Index0 = i;
                info.Prev = i > 0 ? items[i - 1] : Undefined.Instance;
                info.Next = i + 1 < items.Count ? items[i + 1] : Undefined.Instance;
                var body = new Scope(scope);
                BindTargets(f.Targets, items[i], body);
                body.Set("loop", loopObj);
                var flow = Exec(f.Body, body, sb);
                if (flow == Flow.Break) break;
            }
            return Flow.Normal;
        }

        private static void BindTargets(List<string> targets, object? item, Scope s)
        {
            if (targets.Count == 1) { s.Set(targets[0], item); return; }
            var parts = JinjaValue.Iterate(item).ToList();
            if (parts.Count != targets.Count) throw new JinjaException($"cannot unpack {parts.Count} values into {targets.Count} targets");
            for (int i = 0; i < targets.Count; i++) s.Set(targets[i], parts[i]);
        }

        private static LoopDict MakeLoopObject(LoopInfo info) => new(info);

        private object MakeMacro(MacroNode m, Scope defScope)
        {
            return new JinjaCallable(m.Name, (args, kwargs) =>
            {
                var s = new Scope(defScope);
                for (int i = 0; i < m.Params.Count; i++)
                {
                    var (name, def) = m.Params[i];
                    object? v;
                    if (i < args.Count) v = args[i];
                    else if (kwargs.TryGetValue(name, out var kv)) v = kv;
                    else v = def != null ? Eval(def, defScope) : Undefined.Instance;
                    s.Set(name, v);
                }
                s.Set("varargs", args.Skip(m.Params.Count).ToList());
                var sb = new StringBuilder();
                Exec(m.Body, s, sb);
                return sb.ToString();
            });
        }

        // ------------------------------------------------------------------ expressions

        public object? Eval(Expr e, Scope s)
        {
            switch (e)
            {
                case ConstExpr c: return c.Value;
                case NameExpr n: return s.TryGet(n.Name, out var v) ? v : new Undefined(n.Name);
                case ListExpr l: return l.Items.Select(x => Eval(x, s)).ToList();
                case TupleExpr t: return t.Items.Select(x => Eval(x, s)).ToList();
                case DictExpr d:
                {
                    var r = new JinjaDict();
                    foreach (var (k, v2) in d.Items) r[JinjaValue.ToStr(Eval(k, s))] = Eval(v2, s);
                    return r;
                }
                case AttrExpr a: return GetAttr(Eval(a.Target, s), a.Name);
                case ItemExpr it: return GetItem(Eval(it.Target, s), Eval(it.Index, s));
                case SliceExpr sl: return Slice(Eval(sl.Target, s), sl.Start == null ? null : Eval(sl.Start, s), sl.Stop == null ? null : Eval(sl.Stop, s), sl.Step == null ? null : Eval(sl.Step, s));
                case CallExpr call: return EvalCall(call, s);
                case FilterExpr fe:
                {
                    var target = Eval(fe.Target, s);
                    var args = fe.Args.Select(x => Eval(x, s)).ToList();
                    var kw = fe.Kwargs.ToDictionary(k => k.Name, k => Eval(k.Value, s));
                    return Builtins.ApplyFilter(this, fe.Name, target, args, kw, s);
                }
                case TestExpr te:
                {
                    bool r = Builtins.ApplyTest(this, te.Name, Eval(te.Target, s), te.Args.Select(x => Eval(x, s)).ToList(), s);
                    return te.Negated ? !r : r;
                }
                case UnaryExpr u:
                {
                    var v2 = Eval(u.Operand, s);
                    return u.Op switch
                    {
                        "not" => !JinjaValue.Truthy(v2),
                        "-" => v2 is long l ? -l : (object)(-JinjaValue.ToDouble(v2)),
                        _ => throw new JinjaException($"unknown unary {u.Op}"),
                    };
                }
                case BinaryExpr b: return EvalBinary(b, s);
                case CondExpr c2:
                    return JinjaValue.Truthy(Eval(c2.Test, s)) ? Eval(c2.Then, s) : c2.Else != null ? Eval(c2.Else, s) : Undefined.Instance;
                default: throw new JinjaException($"Unsupported expression {e.GetType().Name}");
            }
        }

        private object? EvalBinary(BinaryExpr b, Scope s)
        {
            if (b.Op == "and")
            {
                var l = Eval(b.Left, s);
                return JinjaValue.Truthy(l) ? Eval(b.Right, s) : l;
            }
            if (b.Op == "or")
            {
                var l = Eval(b.Left, s);
                return JinjaValue.Truthy(l) ? l : Eval(b.Right, s);
            }
            var a = Eval(b.Left, s);
            var c = Eval(b.Right, s);
            switch (b.Op)
            {
                case "==": return JinjaValue.Equal(a, c);
                case "!=": return !JinjaValue.Equal(a, c);
                case "<": return JinjaValue.Compare(a, c) < 0;
                case ">": return JinjaValue.Compare(a, c) > 0;
                case "<=": return JinjaValue.Compare(a, c) <= 0;
                case ">=": return JinjaValue.Compare(a, c) >= 0;
                case "in": return JinjaValue.Contains(c, a);
                case "notin": return !JinjaValue.Contains(c, a);
                case "~": return JinjaValue.ToStr(a) + JinjaValue.ToStr(c);
                case "+":
                    if (a is string sa && c is string sc) return sa + sc;
                    if (a is IList la && c is IList lc) return la.Cast<object?>().Concat(lc.Cast<object?>()).ToList();
                    if (a is string || c is string) throw new JinjaException($"cannot add {JinjaValue.TypeName(a)} and {JinjaValue.TypeName(c)}");
                    return Arith(a, c, (x, y) => x + y, (x, y) => x + y);
                case "-": return Arith(a, c, (x, y) => x - y, (x, y) => x - y);
                case "*":
                    if (a is string rs && JinjaValue.IsInteger(c)) return string.Concat(Enumerable.Repeat(rs, (int)Math.Max(0, JinjaValue.ToLong(c))));
                    if (a is IList rl && JinjaValue.IsInteger(c)) return Enumerable.Repeat(rl.Cast<object?>(), (int)Math.Max(0, JinjaValue.ToLong(c))).SelectMany(x => x).ToList();
                    return Arith(a, c, (x, y) => x * y, (x, y) => x * y);
                case "/": return JinjaValue.ToDouble(a) / JinjaValue.ToDouble(c);
                case "//":
                    if (JinjaValue.IsInteger(a) && JinjaValue.IsInteger(c)) return (long)Math.Floor((double)JinjaValue.ToLong(a) / JinjaValue.ToLong(c));
                    return Math.Floor(JinjaValue.ToDouble(a) / JinjaValue.ToDouble(c));
                case "%":
                    if (a is string fmt) return PyFormat(fmt, c);
                    if (JinjaValue.IsInteger(a) && JinjaValue.IsInteger(c))
                    {
                        long x = JinjaValue.ToLong(a), y = JinjaValue.ToLong(c);
                        long m = x % y;
                        return m != 0 && (m < 0) != (y < 0) ? m + y : m;
                    }
                    return JinjaValue.ToDouble(a) % JinjaValue.ToDouble(c);
                case "**":
                    if (JinjaValue.IsInteger(a) && JinjaValue.IsInteger(c) && JinjaValue.ToLong(c) >= 0) return (long)Math.Pow(JinjaValue.ToLong(a), JinjaValue.ToLong(c));
                    return Math.Pow(JinjaValue.ToDouble(a), JinjaValue.ToDouble(c));
                default: throw new JinjaException($"unknown operator {b.Op}");
            }
        }

        private static object Arith(object? a, object? b, Func<long, long, long> li, Func<double, double, double> d)
        {
            if (a is bool ba) a = ba ? 1L : 0L;
            if (b is bool bb) b = bb ? 1L : 0L;
            if (JinjaValue.IsInteger(a) && JinjaValue.IsInteger(b)) return li(JinjaValue.ToLong(a), JinjaValue.ToLong(b));
            return d(JinjaValue.ToDouble(a), JinjaValue.ToDouble(b));
        }

        private static string PyFormat(string fmt, object? args)
        {
            var list = args is IList l ? l.Cast<object?>().ToList() : [args];
            var sb = new StringBuilder();
            int ai = 0;
            for (int i = 0; i < fmt.Length; i++)
            {
                if (fmt[i] != '%' || i + 1 >= fmt.Length) { sb.Append(fmt[i]); continue; }
                char c = fmt[++i];
                if (c == '%') { sb.Append('%'); continue; }
                var v = ai < list.Count ? list[ai++] : null;
                sb.Append(c switch
                {
                    'd' or 'i' => JinjaValue.ToLong(v).ToString(CultureInfo.InvariantCulture),
                    'r' => JinjaValue.Repr(v),
                    _ => JinjaValue.ToStr(v),
                });
            }
            return sb.ToString();
        }

        public object? GetAttr(object? obj, string name)
        {
            switch (obj)
            {
                case JinjaNamespace ns: return ns.Values.TryGetValue(name, out var v) ? v : new Undefined(name);
                case JinjaDict d:
                    if (d.TryGetValue(name, out var dv)) return dv;
                    return (object?)Builtins.Method(this, obj, name) ?? new Undefined(name);
                case LoopDict ld: return ld.Get(name);
                case null or Undefined: return new Undefined(name);
                default: return (object?)Builtins.Method(this, obj, name) ?? new Undefined(name);
            }
        }

        public object? GetItem(object? obj, object? key)
        {
            switch (obj)
            {
                case JinjaDict d:
                    return d.TryGetValue(JinjaValue.ToStr(key), out var v) ? v : new Undefined(JinjaValue.ToStr(key));
                case JinjaNamespace ns: return GetAttr(ns, JinjaValue.ToStr(key));
                case LoopDict ld: return ld.Get(JinjaValue.ToStr(key));
                case IList l:
                {
                    if (!JinjaValue.IsInteger(key)) return Undefined.Instance;
                    long i = JinjaValue.ToLong(key);
                    if (i < 0) i += l.Count;
                    return i >= 0 && i < l.Count ? l[(int)i] : Undefined.Instance;
                }
                case string s:
                {
                    if (!JinjaValue.IsInteger(key)) return Undefined.Instance;
                    long i = JinjaValue.ToLong(key);
                    if (i < 0) i += s.Length;
                    return i >= 0 && i < s.Length ? s[(int)i].ToString() : Undefined.Instance;
                }
                default: return key is string k ? GetAttr(obj, k) : Undefined.Instance;
            }
        }

        private static object? Slice(object? obj, object? start, object? stop, object? step)
        {
            int len = obj switch { string s => s.Length, IList l => l.Count, _ => throw new JinjaException($"cannot slice {JinjaValue.TypeName(obj)}") };
            long st = step == null ? 1 : JinjaValue.ToLong(step);
            if (st == 0) throw new JinjaException("slice step cannot be zero");
            long Norm(object? v, long def)
            {
                if (v == null || v is Undefined) return def;
                long x = JinjaValue.ToLong(v);
                if (x < 0) x += len;
                return st > 0 ? Math.Clamp(x, 0, len) : Math.Clamp(x, -1, len - 1);
            }
            long a = Norm(start, st > 0 ? 0 : len - 1);
            long b = Norm(stop, st > 0 ? len : -1);
            var idx = new List<int>();
            if (st > 0) for (long i = a; i < b; i += st) idx.Add((int)i);
            else for (long i = a; i > b; i += st) idx.Add((int)i);
            if (obj is string str) return new string(idx.Select(i => str[i]).ToArray());
            var list = (IList)obj!;
            return idx.Select(i => list[i]).ToList();
        }

        private object? EvalCall(CallExpr call, Scope s)
        {
            var args = call.Args.Select(a => Eval(a, s)).ToList();
            var kwargs = call.Kwargs.ToDictionary(k => k.Name, k => Eval(k.Value, s));
            // method call on an object: obj.method(...)
            if (call.Func is AttrExpr ae)
            {
                var target = Eval(ae.Target, s);
                if (target is not JinjaNamespace && !(target is JinjaDict d && d.ContainsKey(ae.Name) && d[ae.Name] is JinjaCallable))
                {
                    var r = Builtins.CallMethod(this, target, ae.Name, args, kwargs);
                    if (r is not NoMethod) return r;
                }
            }
            var fn = Eval(call.Func, s);
            if (fn is JinjaCallable c) return c.Invoke(args, kwargs);
            throw new JinjaException($"'{(call.Func is NameExpr n ? n.Name : JinjaValue.TypeName(fn))}' is not callable");
        }
    }

    /// <summary>Sentinel returned when a method name is not supported for a value.</summary>
    internal sealed class NoMethod { public static readonly NoMethod Instance = new(); }

    /// <summary>The <c>loop</c> variable of a for-loop.</summary>
    internal sealed class LoopDict
    {
        private readonly LoopInfo _info;
        public LoopDict(LoopInfo info) => _info = info;

        public object? Get(string name)
        {
            var i = _info;
            long idx0 = i.Index0, len = i.Length;
            return name switch
            {
                "index0" => idx0,
                "index" => idx0 + 1,
                "revindex" => len - idx0,
                "revindex0" => len - idx0 - 1,
                "first" => idx0 == 0,
                "last" => idx0 == len - 1,
                "length" => len,
                "previtem" => i.Prev,
                "nextitem" => i.Next,
                "depth" => 1L,
                "depth0" => 0L,
                "cycle" => new JinjaCallable("cycle", (a, _) => a.Count == 0 ? null : a[(int)(idx0 % a.Count)]),
                _ => new Undefined(name),
            };
        }
    }
}
