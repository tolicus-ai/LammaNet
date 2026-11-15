using System.Collections;
using System.Globalization;
using System.Text;

namespace LammaNet.Templates.Jinja;

/// <summary>Built-in globals, filters, tests and Python-style methods.</summary>
internal static class Builtins
{
    private static void LogDefault(Exception e, string filter) =>
        Diagnostics.Log.TemplateConversionDefault(Diagnostics.LammaLogging.Static("LammaNet.Templates.Jinja"), e, filter);

    private static object? Arg(List<object?> a, Dictionary<string, object?> kw, int i, string name, object? def = null)
    {
        if (i < a.Count) return a[i];
        if (kw.TryGetValue(name, out var v)) return v;
        return def;
    }

    public static void Register(JinjaTemplate.Scope root, DateTime? now = null)
    {
        root.Set("range", new JinjaCallable("range", (a, _) =>
        {
            long start = 0, stop, step = 1;
            if (a.Count == 1) stop = JinjaValue.ToLong(a[0]);
            else { start = JinjaValue.ToLong(a[0]); stop = JinjaValue.ToLong(a[1]); if (a.Count > 2) step = JinjaValue.ToLong(a[2]); }
            if (step == 0) throw new JinjaException("range() step must not be zero");
            var r = new List<object?>();
            if (step > 0) for (long i = start; i < stop; i += step) r.Add(i);
            else for (long i = start; i > stop; i += step) r.Add(i);
            return r;
        }));
        root.Set("namespace", new JinjaCallable("namespace", (a, kw) =>
        {
            var ns = new JinjaNamespace();
            if (a.Count > 0 && a[0] is JinjaDict d) foreach (var kv in d) ns.Values[kv.Key] = kv.Value;
            foreach (var (k, v) in kw) ns.Values[k] = v;
            return ns;
        }));
        root.Set("dict", new JinjaCallable("dict", (a, kw) =>
        {
            var d = new JinjaDict();
            if (a.Count > 0 && a[0] is JinjaDict src) foreach (var kv in src) d[kv.Key] = kv.Value;
            foreach (var (k, v) in kw) d[k] = v;
            return d;
        }));
        root.Set("raise_exception", new JinjaCallable("raise_exception", (a, _) =>
            throw new JinjaException(a.Count > 0 ? JinjaValue.ToStr(a[0]) : "raise_exception")));
        root.Set("strftime_now", new JinjaCallable("strftime_now", (a, _) => Strftime(now ?? DateTime.Now, a.Count > 0 ? JinjaValue.ToStr(a[0]) : "%Y-%m-%d")));
        root.Set("cycler", new JinjaCallable("cycler", (a, _) =>
        {
            var items = a.ToList();
            int idx = 0;
            var c = new JinjaDict();
            c["next"] = new JinjaCallable("next", (_, _) => items.Count == 0 ? null : items[idx++ % items.Count]);
            return c;
        }));
        root.Set("joiner", new JinjaCallable("joiner", (a, _) =>
        {
            string sep = a.Count > 0 ? JinjaValue.ToStr(a[0]) : ", ";
            bool first = true;
            return new JinjaCallable("joiner", (_, _) => { if (first) { first = false; return ""; } return sep; });
        }));
    }

    public static string Strftime(DateTime t, string fmt)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < fmt.Length; i++)
        {
            if (fmt[i] != '%' || i + 1 >= fmt.Length) { sb.Append(fmt[i]); continue; }
            char c = fmt[++i];
            sb.Append(c switch
            {
                'Y' => t.Year.ToString("0000"),
                'y' => (t.Year % 100).ToString("00"),
                'm' => t.Month.ToString("00"),
                'd' => t.Day.ToString("00"),
                '-' when i + 1 < fmt.Length && fmt[i + 1] == 'd' => t.Day.ToString(CultureInfo.InvariantCulture).Also(() => i++),
                'H' => t.Hour.ToString("00"),
                'I' => ((t.Hour + 11) % 12 + 1).ToString("00"),
                'M' => t.Minute.ToString("00"),
                'S' => t.Second.ToString("00"),
                'p' => t.Hour < 12 ? "AM" : "PM",
                'B' => t.ToString("MMMM", CultureInfo.InvariantCulture),
                'b' => t.ToString("MMM", CultureInfo.InvariantCulture),
                'A' => t.ToString("dddd", CultureInfo.InvariantCulture),
                'a' => t.ToString("ddd", CultureInfo.InvariantCulture),
                'j' => t.DayOfYear.ToString("000"),
                '%' => "%",
                _ => "%" + c,
            });
        }
        return sb.ToString();
    }

    private static string Also(this string s, Action a) { a(); return s; }

    // ------------------------------------------------------------------ filters

    public static object? ApplyFilter(object interp, string name, object? v, List<object?> a, Dictionary<string, object?> kw, JinjaTemplate.Scope scope)
    {
        switch (name)
        {
            case "safe" or "e" or "escape" or "forceescape" when name == "safe": return v;
            case "e" or "escape" or "forceescape":
                return JinjaValue.ToStr(v).Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&#34;").Replace("'", "&#39;");
            case "tojson":
            {
                var ind = Arg(a, kw, 0, "indent");
                int? indent = ind is null or Undefined ? null : (int)JinjaValue.ToLong(ind);
                bool sortKeys = JinjaValue.Truthy(kw.GetValueOrDefault("sort_keys"));
                bool ascii = JinjaValue.Truthy(kw.GetValueOrDefault("ensure_ascii"));
                string? itemSep = null, keySep = null;
                if (kw.TryGetValue("separators", out var seps) && seps is IList sl && sl.Count == 2)
                { itemSep = JinjaValue.ToStr(sl[0]); keySep = JinjaValue.ToStr(sl[1]); }
                return JinjaValue.ToJson(v, indent, sortKeys, itemSep, keySep, ascii);
            }
            case "string": return JinjaValue.ToStr(v);
            case "int":
                try { return JinjaValue.ToLong(v); }
                catch (Exception e) { LogDefault(e, "int"); return a.Count > 0 ? a[0] : 0L; }   // Jinja semantics: the default on failure
            case "float":
                try { return JinjaValue.ToDouble(v); }
                catch (Exception e) { LogDefault(e, "float"); return a.Count > 0 ? a[0] : 0.0; }
            case "bool": return JinjaValue.Truthy(v);
            case "abs": return v is long al ? Math.Abs(al) : (object)Math.Abs(JinjaValue.ToDouble(v));
            case "round":
            {
                int prec = (int)JinjaValue.ToLong(Arg(a, kw, 0, "precision", 0L));
                string method = JinjaValue.ToStr(Arg(a, kw, 1, "method", "common"));
                double d = JinjaValue.ToDouble(v), f = Math.Pow(10, prec);
                return method switch
                {
                    "floor" => Math.Floor(d * f) / f,
                    "ceil" => Math.Ceiling(d * f) / f,
                    _ => Math.Round(d * f, MidpointRounding.AwayFromZero) / f,
                };
            }
            case "trim" or "strip":
            {
                string s = JinjaValue.ToStr(v);
                var chars = Arg(a, kw, 0, "chars");
                return chars is string cs ? s.Trim(cs.ToCharArray()) : s.Trim();
            }
            case "upper": return JinjaValue.ToStr(v).ToUpperInvariant();
            case "lower": return JinjaValue.ToStr(v).ToLowerInvariant();
            case "capitalize": { var s = JinjaValue.ToStr(v); return s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant(); }
            case "title": return TitleCase(JinjaValue.ToStr(v));
            case "length" or "count":
                return v switch
                {
                    string s => (long)s.Length,
                    JinjaDict d => (long)d.Count,
                    IList l => (long)l.Count,
                    null or Undefined => 0L,
                    _ => (long)JinjaValue.Iterate(v).Count(),
                };
            case "wordcount": return (long)JinjaValue.ToStr(v).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
            case "default" or "d":
            {
                var def = Arg(a, kw, 0, "default_value", "");
                bool boolean = JinjaValue.Truthy(Arg(a, kw, 1, "boolean", false));
                if (v is Undefined || (boolean && !JinjaValue.Truthy(v))) return def;
                return v;
            }
            case "first": return JinjaValue.Iterate(v).FirstOrDefault() ?? (JinjaValue.Iterate(v).Any() ? null : Undefined.Instance);
            case "last": { var ll = JinjaValue.Iterate(v).ToList(); return ll.Count > 0 ? ll[^1] : Undefined.Instance; }
            case "list": return JinjaValue.Iterate(v).ToList();
            case "join":
            {
                string sep = JinjaValue.ToStr(Arg(a, kw, 0, "d", ""));
                var attr = Arg(a, kw, 1, "attribute");
                var items = JinjaValue.Iterate(v);
                if (attr is string at) items = items.Select(x => GetAttrStatic(interp, x, at));
                return string.Join(sep, items.Select(JinjaValue.ToStr));
            }
            case "reverse":
                if (v is string rs) { var arr = rs.ToCharArray(); Array.Reverse(arr); return new string(arr); }
                return JinjaValue.Iterate(v).Reverse().ToList();
            case "unique":
            {
                var res = new List<object?>();
                foreach (var x in JinjaValue.Iterate(v)) if (!res.Any(y => JinjaValue.Equal(x, y))) res.Add(x);
                return res;
            }
            case "sort":
            {
                bool rev = JinjaValue.Truthy(Arg(a, kw, 0, "reverse", false));
                var attr = Arg(a, kw, 2, "attribute");
                var items = JinjaValue.Iterate(v).ToList();
                Func<object?, object?> key = attr is string at ? x => GetAttrStatic(interp, x, at) : x => x;
                items.Sort((x, y) => JinjaValue.Compare(key(x), key(y)));
                if (rev) items.Reverse();
                return items;
            }
            case "dictsort":
            {
                if (v is not JinjaDict d) return new List<object?>();
                bool byValue = JinjaValue.ToStr(Arg(a, kw, 1, "by", "key")) == "value";
                bool rev = JinjaValue.Truthy(Arg(a, kw, 2, "reverse", false));
                var items = d.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList();
                items.Sort((x, y) => JinjaValue.Compare(((IList)x!)[byValue ? 1 : 0], ((IList)y!)[byValue ? 1 : 0]));
                if (rev) items.Reverse();
                return items;
            }
            case "items": return v is JinjaDict di ? di.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList() : new List<object?>();
            case "map":
            {
                var items = JinjaValue.Iterate(v);
                if (kw.TryGetValue("attribute", out var attrName))
                {
                    var def = kw.GetValueOrDefault("default");
                    return items.Select(x =>
                    {
                        var r = GetAttrStatic(interp, x, JinjaValue.ToStr(attrName));
                        return r is Undefined && def != null ? def : r;
                    }).ToList();
                }
                if (a.Count == 0) return items.ToList();
                string filter = JinjaValue.ToStr(a[0]);
                var rest = a.Skip(1).ToList();
                return items.Select(x => ApplyFilter(interp, filter, x, rest, new(), scope)).ToList();
            }
            case "select" or "reject":
            {
                bool want = name == "select";
                var items = JinjaValue.Iterate(v);
                if (a.Count == 0) return items.Where(x => JinjaValue.Truthy(x) == want).ToList();
                string test = JinjaValue.ToStr(a[0]);
                var rest = a.Skip(1).ToList();
                return items.Where(x => ApplyTest(interp, test, x, rest, scope) == want).ToList();
            }
            case "selectattr" or "rejectattr":
            {
                bool want = name == "selectattr";
                string attr = JinjaValue.ToStr(a[0]);
                var items = JinjaValue.Iterate(v);
                if (a.Count == 1) return items.Where(x => JinjaValue.Truthy(GetAttrStatic(interp, x, attr)) == want).ToList();
                string test = JinjaValue.ToStr(a[1]);
                var rest = a.Skip(2).ToList();
                return items.Where(x => ApplyTest(interp, test, GetAttrStatic(interp, x, attr), rest, scope) == want).ToList();
            }
            case "replace":
            {
                string s = JinjaValue.ToStr(v), o = JinjaValue.ToStr(a[0]), n = JinjaValue.ToStr(a[1]);
                if (a.Count > 2) return ReplaceCount(s, o, n, (int)JinjaValue.ToLong(a[2]));
                return o.Length == 0 ? s : s.Replace(o, n, StringComparison.Ordinal);
            }
            case "indent":
            {
                string s = JinjaValue.ToStr(v);
                var w = Arg(a, kw, 0, "width", 4L);
                string pad = w is string ps ? ps : new string(' ', (int)JinjaValue.ToLong(w));
                bool first = JinjaValue.Truthy(Arg(a, kw, 1, "first", false));
                bool blank = JinjaValue.Truthy(Arg(a, kw, 2, "blank", false));
                var lines = s.Split('\n');
                for (int i = 0; i < lines.Length; i++)
                {
                    if (i == 0 && !first) continue;
                    if (lines[i].Length == 0 && !blank) continue;
                    lines[i] = pad + lines[i];
                }
                return string.Join("\n", lines);
            }
            case "center":
            {
                string s = JinjaValue.ToStr(v);
                int width = (int)JinjaValue.ToLong(Arg(a, kw, 0, "width", 80L));
                if (s.Length >= width) return s;
                int total = width - s.Length, left = total / 2;
                return new string(' ', left) + s + new string(' ', total - left);
            }
            case "min" or "max":
            {
                var items = JinjaValue.Iterate(v).ToList();
                if (items.Count == 0) return Undefined.Instance;
                var best = items[0];
                foreach (var x in items.Skip(1))
                {
                    int c = JinjaValue.Compare(x, best);
                    if (name == "min" ? c < 0 : c > 0) best = x;
                }
                return best;
            }
            case "sum":
            {
                object? acc = Arg(a, kw, 1, "start", 0L);
                var attr = Arg(a, kw, 0, "attribute");
                foreach (var x in JinjaValue.Iterate(v))
                {
                    var val = attr is string at ? GetAttrStatic(interp, x, at) : x;
                    acc = JinjaValue.IsInteger(acc) && JinjaValue.IsInteger(val) ? JinjaValue.ToLong(acc) + JinjaValue.ToLong(val) : (object)(JinjaValue.ToDouble(acc) + JinjaValue.ToDouble(val));
                }
                return acc;
            }
            case "batch":
            {
                int size = (int)JinjaValue.ToLong(a[0]);
                var items = JinjaValue.Iterate(v).ToList();
                var res = new List<object?>();
                for (int i = 0; i < items.Count; i += size) res.Add(items.Skip(i).Take(size).ToList());
                return res;
            }
            case "truncate":
            {
                string s = JinjaValue.ToStr(v);
                int len = (int)JinjaValue.ToLong(Arg(a, kw, 0, "length", 255L));
                return s.Length <= len ? s : s[..Math.Max(0, len - 3)] + "...";
            }
            case "pprint": return JinjaValue.Repr(v);
            case "attr": return GetAttrStatic(interp, v, JinjaValue.ToStr(a[0]));
            default:
                throw new JinjaException($"Unknown filter '{name}'");
        }
    }

    private static string ReplaceCount(string s, string o, string n, int count)
    {
        var sb = new StringBuilder();
        int start = 0, done = 0;
        while (done < count)
        {
            int idx = s.IndexOf(o, start, StringComparison.Ordinal);
            if (idx < 0) break;
            sb.Append(s, start, idx - start).Append(n);
            start = idx + o.Length;
            done++;
        }
        sb.Append(s, start, s.Length - start);
        return sb.ToString();
    }

    private static string TitleCase(string s)
    {
        var sb = new StringBuilder(s.Length);
        bool prevLetter = false;
        foreach (char c in s)
        {
            sb.Append(prevLetter ? char.ToLowerInvariant(c) : char.ToUpperInvariant(c));
            prevLetter = char.IsLetter(c);
        }
        return sb.ToString();
    }

    private static object? GetAttrStatic(object interp, object? obj, string name)
    {
        if (obj is JinjaDict d) return d.TryGetValue(name, out var v) ? v : Undefined.Instance;
        if (obj is JinjaNamespace ns) return ns.Values.TryGetValue(name, out var v2) ? v2 : Undefined.Instance;
        if (obj is IList l && int.TryParse(name, out int idx)) return idx < l.Count ? l[idx] : Undefined.Instance;
        return Undefined.Instance;
    }

    // ------------------------------------------------------------------ tests

    public static bool ApplyTest(object interp, string name, object? v, List<object?> a, JinjaTemplate.Scope scope)
    {
        switch (name)
        {
            case "defined": return v is not Undefined;
            case "undefined": return v is Undefined;
            case "none": return v is null;
            case "string": return v is string;
            case "number": return v is long or int or double;
            case "integer": return v is long or int;
            case "float": return v is double;
            case "boolean": return v is bool;
            case "true": return v is true;
            case "false": return v is false;
            case "mapping": return v is JinjaDict or JinjaNamespace;
            case "sequence": return v is string or IList or JinjaDict;
            case "iterable": return v is string or IList or JinjaDict;
            case "callable": return v is JinjaCallable;
            case "odd": return JinjaValue.ToLong(v) % 2 != 0;
            case "even": return JinjaValue.ToLong(v) % 2 == 0;
            case "divisibleby": return JinjaValue.ToLong(v) % JinjaValue.ToLong(a[0]) == 0;
            case "eq" or "equalto" or "==": return JinjaValue.Equal(v, a[0]);
            case "ne" or "!=": return !JinjaValue.Equal(v, a[0]);
            case "lt" or "lessthan" or "<": return JinjaValue.Compare(v, a[0]) < 0;
            case "le" or "<=": return JinjaValue.Compare(v, a[0]) <= 0;
            case "gt" or "greaterthan" or ">": return JinjaValue.Compare(v, a[0]) > 0;
            case "ge" or ">=": return JinjaValue.Compare(v, a[0]) >= 0;
            case "in": return JinjaValue.Contains(a[0], v);
            case "sameas": return ReferenceEquals(v, a[0]) || (v is null && a[0] is null) || (v is bool b1 && a[0] is bool b2 && b1 == b2);
            case "lower": return v is string ls && ls == ls.ToLowerInvariant();
            case "upper": return v is string us && us == us.ToUpperInvariant();
            case "startingwith": return v is string sw && sw.StartsWith(JinjaValue.ToStr(a[0]), StringComparison.Ordinal);
            default: throw new JinjaException($"Unknown test '{name}'");
        }
    }

    // ------------------------------------------------------------------ methods

    public static JinjaCallable? Method(object interp, object? obj, string name)
    {
        object? r = CallMethodProbe(obj, name);
        if (r == null) return null;
        return new JinjaCallable(name, (a, kw) => CallMethod(interp, obj, name, a, kw));
    }

    private static object? CallMethodProbe(object? obj, string name) => obj switch
    {
        string when StringMethods.Contains(name) => true,
        JinjaDict when DictMethods.Contains(name) => true,
        IList when ListMethods.Contains(name) => true,
        _ => null,
    };

    private static readonly HashSet<string> StringMethods = ["strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "startswith", "endswith", "upper", "lower", "title", "capitalize", "replace", "find", "rfind", "index", "count", "join", "format", "isdigit", "isalpha", "isalnum", "isspace", "isupper", "islower", "zfill", "removeprefix", "removesuffix", "center", "ljust", "rjust"];
    private static readonly HashSet<string> DictMethods = ["get", "items", "keys", "values", "update", "pop", "setdefault", "copy"];
    private static readonly HashSet<string> ListMethods = ["append", "extend", "pop", "insert", "index", "count", "remove", "copy", "reverse", "sort"];

    public static object? CallMethod(object interp, object? obj, string name, List<object?> a, Dictionary<string, object?> kw)
    {
        switch (obj)
        {
            case string s:
                switch (name)
                {
                    case "strip": return a.Count > 0 && a[0] is string c1 ? s.Trim(c1.ToCharArray()) : s.Trim();
                    case "lstrip": return a.Count > 0 && a[0] is string c2 ? s.TrimStart(c2.ToCharArray()) : s.TrimStart();
                    case "rstrip": return a.Count > 0 && a[0] is string c3 ? s.TrimEnd(c3.ToCharArray()) : s.TrimEnd();
                    case "split":
                    {
                        var sep = Arg(a, kw, 0, "sep");
                        int max = (int)JinjaValue.ToLong(Arg(a, kw, 1, "maxsplit", -1L));
                        if (sep is null or Undefined)
                            return s.Split((char[]?)null, max < 0 ? int.MaxValue : max + 1, StringSplitOptions.RemoveEmptyEntries).Select(x => (object?)x).ToList();
                        return s.Split(JinjaValue.ToStr(sep), max < 0 ? int.MaxValue : max + 1).Select(x => (object?)x).ToList();
                    }
                    case "rsplit":
                    {
                        var sep = JinjaValue.ToStr(Arg(a, kw, 0, "sep", " "));
                        int max = (int)JinjaValue.ToLong(Arg(a, kw, 1, "maxsplit", -1L));
                        var parts = s.Split(sep).ToList();
                        if (max >= 0 && parts.Count > max + 1)
                        {
                            int keep = parts.Count - max;
                            var head = string.Join(sep, parts.Take(keep));
                            parts = new[] { head }.Concat(parts.Skip(keep)).ToList();
                        }
                        return parts.Select(x => (object?)x).ToList();
                    }
                    case "splitlines": return s.Split('\n').Select(x => (object?)x.TrimEnd('\r')).ToList();
                    case "startswith": return a[0] is IList pl ? pl.Cast<object?>().Any(p => s.StartsWith(JinjaValue.ToStr(p), StringComparison.Ordinal)) : s.StartsWith(JinjaValue.ToStr(a[0]), StringComparison.Ordinal);
                    case "endswith": return a[0] is IList el ? el.Cast<object?>().Any(p => s.EndsWith(JinjaValue.ToStr(p), StringComparison.Ordinal)) : s.EndsWith(JinjaValue.ToStr(a[0]), StringComparison.Ordinal);
                    case "upper": return s.ToUpperInvariant();
                    case "lower": return s.ToLowerInvariant();
                    case "title": return TitleCase(s);
                    case "capitalize": return s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant();
                    case "replace": return a.Count > 2 ? ReplaceCount(s, JinjaValue.ToStr(a[0]), JinjaValue.ToStr(a[1]), (int)JinjaValue.ToLong(a[2])) : s.Replace(JinjaValue.ToStr(a[0]), JinjaValue.ToStr(a[1]), StringComparison.Ordinal);
                    case "find": return (long)s.IndexOf(JinjaValue.ToStr(a[0]), StringComparison.Ordinal);
                    case "rfind": return (long)s.LastIndexOf(JinjaValue.ToStr(a[0]), StringComparison.Ordinal);
                    case "index":
                    {
                        int i = s.IndexOf(JinjaValue.ToStr(a[0]), StringComparison.Ordinal);
                        if (i < 0) throw new JinjaException("substring not found");
                        return (long)i;
                    }
                    case "count":
                    {
                        string sub = JinjaValue.ToStr(a[0]);
                        if (sub.Length == 0) return (long)(s.Length + 1);
                        long n = 0; int p = 0;
                        while ((p = s.IndexOf(sub, p, StringComparison.Ordinal)) >= 0) { n++; p += sub.Length; }
                        return n;
                    }
                    case "join": return string.Join(s, JinjaValue.Iterate(a[0]).Select(JinjaValue.ToStr));
                    case "format":
                    {
                        var sb = new StringBuilder();
                        int ai = 0;
                        for (int i = 0; i < s.Length; i++)
                        {
                            if (s[i] == '{' && i + 1 < s.Length && s[i + 1] == '{') { sb.Append('{'); i++; continue; }
                            if (s[i] == '}' && i + 1 < s.Length && s[i + 1] == '}') { sb.Append('}'); i++; continue; }
                            if (s[i] == '{')
                            {
                                int close = s.IndexOf('}', i);
                                string key = s[(i + 1)..close];
                                object? val = key.Length == 0 ? (ai < a.Count ? a[ai++] : null)
                                    : int.TryParse(key, out int idx) ? a[idx] : kw.GetValueOrDefault(key);
                                sb.Append(JinjaValue.ToStr(val));
                                i = close;
                                continue;
                            }
                            sb.Append(s[i]);
                        }
                        return sb.ToString();
                    }
                    case "isdigit": return s.Length > 0 && s.All(char.IsDigit);
                    case "isalpha": return s.Length > 0 && s.All(char.IsLetter);
                    case "isalnum": return s.Length > 0 && s.All(char.IsLetterOrDigit);
                    case "isspace": return s.Length > 0 && s.All(char.IsWhiteSpace);
                    case "isupper": return s.Any(char.IsLetter) && s == s.ToUpperInvariant();
                    case "islower": return s.Any(char.IsLetter) && s == s.ToLowerInvariant();
                    case "zfill": return s.PadLeft((int)JinjaValue.ToLong(a[0]), '0');
                    case "removeprefix": { var p = JinjaValue.ToStr(a[0]); return s.StartsWith(p, StringComparison.Ordinal) ? s[p.Length..] : s; }
                    case "removesuffix": { var p = JinjaValue.ToStr(a[0]); return p.Length > 0 && s.EndsWith(p, StringComparison.Ordinal) ? s[..^p.Length] : s; }
                    case "ljust": return s.PadRight((int)JinjaValue.ToLong(a[0]));
                    case "rjust": return s.PadLeft((int)JinjaValue.ToLong(a[0]));
                    case "center": return ApplyFilter(interp, "center", s, a, kw, null!);
                }
                break;
            case JinjaDict d:
                switch (name)
                {
                    case "get":
                    {
                        string key = JinjaValue.ToStr(a[0]);
                        return d.TryGetValue(key, out var v) ? v : (a.Count > 1 ? a[1] : null);
                    }
                    case "items": return d.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList();
                    case "keys": return d.Keys.Select(k => (object?)k).ToList();
                    case "values": return d.Values.ToList();
                    case "copy": { var c = new JinjaDict(); foreach (var kv in d) c[kv.Key] = kv.Value; return c; }
                    case "update":
                        if (a.Count > 0 && a[0] is JinjaDict src) foreach (var kv in src) d[kv.Key] = kv.Value;
                        foreach (var (k, v) in kw) d[k] = v;
                        return null;
                    case "pop":
                    {
                        string key = JinjaValue.ToStr(a[0]);
                        if (d.TryGetValue(key, out var v)) { d.Remove(key); return v; }
                        return a.Count > 1 ? a[1] : throw new JinjaException($"KeyError: {key}");
                    }
                    case "setdefault":
                    {
                        string key = JinjaValue.ToStr(a[0]);
                        if (!d.TryGetValue(key, out var v)) { v = a.Count > 1 ? a[1] : null; d[key] = v; }
                        return v;
                    }
                }
                break;
            case IList l:
                switch (name)
                {
                    case "append": l.Add(a[0]); return null;
                    case "extend": foreach (var x in JinjaValue.Iterate(a[0])) l.Add(x); return null;
                    case "insert": l.Insert((int)JinjaValue.ToLong(a[0]), a[1]); return null;
                    case "pop":
                    {
                        int i = a.Count > 0 ? (int)JinjaValue.ToLong(a[0]) : l.Count - 1;
                        if (i < 0) i += l.Count;
                        var v = l[i]; l.RemoveAt(i); return v;
                    }
                    case "remove":
                        for (int i = 0; i < l.Count; i++) if (JinjaValue.Equal(l[i], a[0])) { l.RemoveAt(i); break; }
                        return null;
                    case "index":
                        for (int i = 0; i < l.Count; i++) if (JinjaValue.Equal(l[i], a[0])) return (long)i;
                        throw new JinjaException("value not in list");
                    case "count": return (long)l.Cast<object?>().Count(x => JinjaValue.Equal(x, a[0]));
                    case "copy": return l.Cast<object?>().ToList();
                    case "reverse":
                    {
                        var tmp = l.Cast<object?>().Reverse().ToList();
                        for (int i = 0; i < tmp.Count; i++) l[i] = tmp[i];
                        return null;
                    }
                    case "sort":
                    {
                        var tmp = l.Cast<object?>().ToList();
                        tmp.Sort(JinjaValue.Compare);
                        for (int i = 0; i < tmp.Count; i++) l[i] = tmp[i];
                        return null;
                    }
                }
                break;
        }
        return JinjaTemplate.NoMethod.Instance;
    }
}
