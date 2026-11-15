using System.Collections;
using System.Diagnostics.CodeAnalysis;

namespace LammaNet.Gguf;

/// <summary>
/// Placeholder for a metadata array whose elements were not loaded (e.g. the vocabulary when
/// <c>ModelInfoOptions.LoadVocabulary</c> is false): only its element type and length are known.
/// </summary>
public sealed record GgufArraySummary(GgufValueType ElementType, long Length)
{
    public override string ToString() => $"[{ElementType} × {Length}]";
}

/// <summary>
/// The key/value metadata of a GGUF file with typed accessors. Scalars are stored as their CLR type
/// (uint, int, float, bool, string, ...), arrays as CLR arrays (or <see cref="GgufArraySummary"/> when not loaded).
/// </summary>
public sealed class GgufMetadata : IReadOnlyDictionary<string, object>
{
    private readonly Dictionary<string, object> _values;

    public GgufMetadata() : this(new Dictionary<string, object>(StringComparer.Ordinal)) { }

    internal GgufMetadata(Dictionary<string, object> values) => _values = values;

    internal void Set(string key, object value) => _values[key] = value;

    public bool Has(string key) => _values.ContainsKey(key);

    public T Get<T>(string key, T defaultValue)
    {
        if (!_values.TryGetValue(key, out var v)) return defaultValue;
        return ConvertValue<T>(key, v);
    }

    public T Get<T>(string key)
    {
        if (!_values.TryGetValue(key, out var v)) throw new KeyNotFoundException($"Missing GGUF key '{key}'");
        return ConvertValue<T>(key, v);
    }

    public bool TryGet<T>(string key, out T value)
    {
        if (_values.TryGetValue(key, out var v)) { value = ConvertValue<T>(key, v); return true; }
        value = default!;
        return false;
    }

    /// <summary>Gets a key that may be either a scalar or a per-layer array; result has <paramref name="n"/> entries.</summary>
    public int[]? GetIntOrArray(string key, int n)
    {
        if (!_values.TryGetValue(key, out var v)) return null;
        var res = new int[n];
        if (v is Array arr)
        {
            for (int i = 0; i < n; i++) res[i] = i < arr.Length ? Convert.ToInt32(arr.GetValue(i)) : 0;
        }
        else
        {
            int s = Convert.ToInt32(v);
            Array.Fill(res, s);
        }
        return res;
    }

    /// <summary>Number of elements of an array value (also for arrays that were not loaded), or -1.</summary>
    public long ArrayLength(string key) => _values.TryGetValue(key, out var v) ? v switch
    {
        Array a => a.Length,
        GgufArraySummary s => s.Length,
        _ => -1,
    } : -1;

    private static T ConvertValue<T>(string key, object v)
    {
        if (v is T t) return t;
        if (v is GgufArraySummary s) throw new InvalidOperationException($"GGUF array '{key}' ({s}) was not loaded");
        var target = typeof(T);
        if (target == typeof(string[]) && v is Array sa) return (T)(object)sa.Cast<object>().Select(o => o.ToString()!).ToArray();
        if (target == typeof(int[]) && v is Array ia) return (T)(object)ia.Cast<object>().Select(Convert.ToInt32).ToArray();
        if (target == typeof(float[]) && v is Array fa) return (T)(object)fa.Cast<object>().Select(Convert.ToSingle).ToArray();
        if (target == typeof(bool[]) && v is Array ba) return (T)(object)ba.Cast<object>().Select(Convert.ToBoolean).ToArray();
        var underlying = Nullable.GetUnderlyingType(target) ?? target;
        return (T)Convert.ChangeType(v, underlying, System.Globalization.CultureInfo.InvariantCulture);
    }

    // ---------------- IReadOnlyDictionary

    public object this[string key] => _values[key];
    public IEnumerable<string> Keys => _values.Keys;
    public IEnumerable<object> Values => _values.Values;
    public int Count => _values.Count;
    public bool ContainsKey(string key) => _values.ContainsKey(key);
    public bool TryGetValue(string key, [MaybeNullWhen(false)] out object value) => _values.TryGetValue(key, out value);
    public IEnumerator<KeyValuePair<string, object>> GetEnumerator() => _values.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
