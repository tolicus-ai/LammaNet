using System.Collections;
using System.Globalization;
using System.Text;

namespace LammaNet.Templates.Jinja;

/// <summary>The Jinja "undefined" value: falsy, renders as empty, iterates as empty.</summary>
public sealed class Undefined
{
    public static readonly Undefined Instance = new();
    public string? Name { get; init; }
    private Undefined() { }
    public Undefined(string name) => Name = name;
    public override string ToString() => "";
}

/// <summary>Mutable attribute bag created by <c>namespace(...)</c>.</summary>
public sealed class JinjaNamespace
{
    public Dictionary<string, object?> Values { get; } = new(StringComparer.Ordinal);
}

/// <summary>A callable value (macro or built-in function).</summary>
public sealed class JinjaCallable(string name, Func<List<object?>, Dictionary<string, object?>, object?> invoke)
{
    public string Name { get; } = name;
    public object? Invoke(List<object?> args, Dictionary<string, object?> kwargs) => invoke(args, kwargs);
}

/// <summary>Ordered string-keyed dictionary used for Jinja mappings (insertion order matters for tojson).</summary>
public sealed class JinjaDict : IEnumerable<KeyValuePair<string, object?>>
{
    private readonly List<string> _keys = new();
    private readonly Dictionary<string, object?> _map = new(StringComparer.Ordinal);

    public int Count => _keys.Count;
    public IReadOnlyList<string> Keys => _keys;
    public IEnumerable<object?> Values => _keys.Select(k => _map[k]);

    public object? this[string key]
    {
        get => _map[key];
        set
        {
            if (!_map.ContainsKey(key)) _keys.Add(key);
            _map[key] = value;
        }
    }

    public bool ContainsKey(string key) => _map.ContainsKey(key);
    public bool TryGetValue(string key, out object? value) => _map.TryGetValue(key, out value);

    public bool Remove(string key)
    {
        if (!_map.Remove(key)) return false;
        _keys.Remove(key);
        return true;
    }

    public IEnumerator<KeyValuePair<string, object?>> GetEnumerator()
    {
        foreach (var k in _keys) yield return new(k, _map[k]);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>Python-like semantics for Jinja values.</summary>
public static class JinjaValue
{
    public static bool IsUndefined(object? v) => v is Undefined;

    public static bool Truthy(object? v) => v switch
    {
        null or Undefined => false,
        bool b => b,
        long l => l != 0,
        int i => i != 0,
        double d => d != 0,
        string s => s.Length > 0,
        JinjaDict d => d.Count > 0,
        IList l => l.Count > 0,
        _ => true,
    };

    public static bool IsNumber(object? v) => v is long or int or double;
    public static bool IsInteger(object? v) => v is long or int;

    public static double ToDouble(object? v) => v switch
    {
        long l => l,
        int i => i,
        double d => d,
        bool b => b ? 1 : 0,
        string s when double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) => d,
        _ => throw new JinjaException($"Cannot convert {TypeName(v)} to number"),
    };

    public static long ToLong(object? v) => v switch
    {
        long l => l,
        int i => i,
        double d => (long)d,
        bool b => b ? 1 : 0,
        string s when long.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var l) => l,
        string s when double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var d) => (long)d,
        _ => throw new JinjaException($"Cannot convert {TypeName(v)} to integer"),
    };

    public static string TypeName(object? v) => v switch
    {
        null => "NoneType",
        Undefined => "Undefined",
        bool => "bool",
        long or int => "int",
        double => "float",
        string => "str",
        JinjaDict => "dict",
        IList => "list",
        _ => v.GetType().Name,
    };

    /// <summary>Python <c>str()</c>.</summary>
    public static string ToStr(object? v) => v switch
    {
        null => "None",
        Undefined => "",
        string s => s,
        bool b => b ? "True" : "False",
        long or int => Convert.ToString(v, CultureInfo.InvariantCulture)!,
        double d => FormatFloat(d),
        _ => Repr(v),
    };

    /// <summary>Python <c>repr()</c>.</summary>
    public static string Repr(object? v)
    {
        switch (v)
        {
            case string s:
                return "'" + s.Replace("\\", "\\\\").Replace("'", "\\'").Replace("\n", "\\n") + "'";
            case JinjaDict d:
                return "{" + string.Join(", ", d.Select(kv => Repr(kv.Key) + ": " + Repr(kv.Value))) + "}";
            case IList l:
                return "[" + string.Join(", ", l.Cast<object?>().Select(Repr)) + "]";
            default:
                return ToStr(v);
        }
    }

    public static string FormatFloat(double d)
    {
        if (double.IsNaN(d)) return "nan";
        if (double.IsPositiveInfinity(d)) return "inf";
        if (double.IsNegativeInfinity(d)) return "-inf";
        string r = d.ToString("R", CultureInfo.InvariantCulture);
        if (r.Contains('E'))
        {
            // python style: 1e+20, 1.5e-07
            var parts = r.Split('E');
            int exp = int.Parse(parts[1], CultureInfo.InvariantCulture);
            return parts[0] + "e" + (exp < 0 ? "-" : "+") + Math.Abs(exp).ToString("00", CultureInfo.InvariantCulture);
        }
        if (!r.Contains('.')) r += ".0";
        return r;
    }

    public static bool Equal(object? a, object? b)
    {
        if (a is Undefined && b is Undefined) return true;
        if (a is null || b is null) return a is null && b is null;
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) == ToDouble(b);
        if (a is bool ba && IsNumber(b)) return (ba ? 1 : 0) == ToDouble(b);
        if (b is bool bb && IsNumber(a)) return (bb ? 1 : 0) == ToDouble(a);
        if (a is string sa && b is string sb) return string.Equals(sa, sb, StringComparison.Ordinal);
        if (a is IList la && b is IList lb)
        {
            if (la.Count != lb.Count) return false;
            for (int i = 0; i < la.Count; i++) if (!Equal(la[i], lb[i])) return false;
            return true;
        }
        if (a is JinjaDict da && b is JinjaDict db)
        {
            if (da.Count != db.Count) return false;
            foreach (var kv in da) if (!db.TryGetValue(kv.Key, out var o) || !Equal(kv.Value, o)) return false;
            return true;
        }
        return Equals(a, b);
    }

    public static int Compare(object? a, object? b)
    {
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a).CompareTo(ToDouble(b));
        if (a is string sa && b is string sb) return string.CompareOrdinal(sa, sb);
        if (a is bool x && b is bool y) return x.CompareTo(y);
        if (a is IList la && b is IList lb)
        {
            for (int i = 0; i < Math.Min(la.Count, lb.Count); i++)
            {
                int c = Compare(la[i], lb[i]);
                if (c != 0) return c;
            }
            return la.Count.CompareTo(lb.Count);
        }
        if (a is null && b is null) return 0;
        throw new JinjaException($"'<' not supported between {TypeName(a)} and {TypeName(b)}");
    }

    /// <summary>Enumerates a value like Python iteration (dict → keys, string → chars).</summary>
    public static IEnumerable<object?> Iterate(object? v) => v switch
    {
        null or Undefined => [],
        string s => s.Select(c => (object?)c.ToString()),
        JinjaDict d => d.Keys.Cast<object?>(),
        IList l => l.Cast<object?>(),
        IEnumerable<object?> e => e,
        _ => throw new JinjaException($"'{TypeName(v)}' object is not iterable"),
    };

    public static bool Contains(object? container, object? item) => container switch
    {
        string s => item is string i && s.Contains(i, StringComparison.Ordinal),
        JinjaDict d => item is string k && d.ContainsKey(k),
        IList l => l.Cast<object?>().Any(x => Equal(x, item)),
        null or Undefined => false,
        _ => throw new JinjaException($"argument of type '{TypeName(container)}' is not iterable"),
    };

    /// <summary>Converts CLR data (dictionaries, lists, primitives, JSON) into Jinja values.</summary>
    public static object? FromClr(object? v)
    {
        switch (v)
        {
            case null: return null;
            case string or bool or long or double or Undefined or JinjaDict or JinjaNamespace or JinjaCallable: return v;
            case int i: return (long)i;
            case short s: return (long)s;
            case byte b: return (long)b;
            case uint ui: return (long)ui;
            case float f: return (double)f;
            case decimal m: return (double)m;
            case System.Text.Json.JsonElement je: return FromJson(je);
            case System.Text.Json.Nodes.JsonNode jn: return FromJson(System.Text.Json.JsonSerializer.SerializeToElement(jn));
            case IDictionary<string, object?> d:
            {
                var r = new JinjaDict();
                foreach (var kv in d) r[kv.Key] = FromClr(kv.Value);
                return r;
            }
            case IDictionary d:
            {
                var r = new JinjaDict();
                foreach (DictionaryEntry kv in d) r[kv.Key.ToString()!] = FromClr(kv.Value);
                return r;
            }
            case IList<object?> l when l is List<object?> lo && lo.All(x => x is null or string or bool or long or double or JinjaDict): return lo;
            case IEnumerable e:
            {
                var r = new List<object?>();
                foreach (var x in e) r.Add(FromClr(x));
                return r;
            }
            default: return v.ToString();
        }
    }

    public static object? FromJson(System.Text.Json.JsonElement e)
    {
        switch (e.ValueKind)
        {
            case System.Text.Json.JsonValueKind.Object:
            {
                var d = new JinjaDict();
                foreach (var p in e.EnumerateObject()) d[p.Name] = FromJson(p.Value);
                return d;
            }
            case System.Text.Json.JsonValueKind.Array: return e.EnumerateArray().Select(FromJson).ToList();
            case System.Text.Json.JsonValueKind.String: return e.GetString();
            case System.Text.Json.JsonValueKind.Number: return e.TryGetInt64(out var l) ? l : e.GetDouble();
            case System.Text.Json.JsonValueKind.True: return true;
            case System.Text.Json.JsonValueKind.False: return false;
            default: return null;
        }
    }

    /// <summary>Python <c>json.dumps</c> (ensure_ascii=False by default, as in HF chat templates).</summary>
    public static string ToJson(object? v, int? indent = null, bool sortKeys = false, string? itemSep = null, string? keySep = null, bool ensureAscii = false)
    {
        var sb = new StringBuilder();
        itemSep ??= indent == null ? ", " : ",";
        keySep ??= ": ";
        WriteJson(sb, v, indent, 0, sortKeys, itemSep, keySep, ensureAscii);
        return sb.ToString();
    }

    private static void WriteJson(StringBuilder sb, object? v, int? indent, int level, bool sortKeys, string itemSep, string keySep, bool ascii)
    {
        switch (v)
        {
            case null or Undefined: sb.Append("null"); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case long or int: sb.Append(Convert.ToString(v, CultureInfo.InvariantCulture)); break;
            case double d:
                sb.Append(double.IsNaN(d) ? "NaN" : double.IsInfinity(d) ? (d > 0 ? "Infinity" : "-Infinity") : FormatFloat(d));
                break;
            case string s: WriteJsonString(sb, s, ascii); break;
            case JinjaDict dict:
            {
                if (dict.Count == 0) { sb.Append("{}"); break; }
                sb.Append('{');
                var keys = sortKeys ? dict.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList() : dict.Keys.ToList();
                for (int i = 0; i < keys.Count; i++)
                {
                    if (i > 0) sb.Append(itemSep);
                    Newline(sb, indent, level + 1);
                    WriteJsonString(sb, keys[i], ascii);
                    sb.Append(keySep);
                    WriteJson(sb, dict[keys[i]], indent, level + 1, sortKeys, itemSep, keySep, ascii);
                }
                Newline(sb, indent, level);
                sb.Append('}');
                break;
            }
            case IList list:
            {
                if (list.Count == 0) { sb.Append("[]"); break; }
                sb.Append('[');
                for (int i = 0; i < list.Count; i++)
                {
                    if (i > 0) sb.Append(itemSep);
                    Newline(sb, indent, level + 1);
                    WriteJson(sb, list[i], indent, level + 1, sortKeys, itemSep, keySep, ascii);
                }
                Newline(sb, indent, level);
                sb.Append(']');
                break;
            }
            default: WriteJsonString(sb, v.ToString() ?? "", ascii); break;
        }
    }

    private static void Newline(StringBuilder sb, int? indent, int level)
    {
        if (indent == null) return;
        sb.Append('\n').Append(' ', indent.Value * level);
    }

    private static void WriteJsonString(StringBuilder sb, string s, bool ascii)
    {
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20 || (ascii && c > 0x7e)) sb.Append("\\u").Append(((int)c).ToString("x4"));
                    else sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }
}
