using System.Text.RegularExpressions;

namespace LammaNet.Models;

/// <summary>
/// A GGUF model in a Hugging Face repository. Without <see cref="File"/>, the file is chosen by <see cref="Quantization"/>
/// (like llama.cpp's <c>-hf owner/repo:Q4_K_M</c>; default Q4_K_M, then Q8_0, then the first model file).
/// <code>
/// new HuggingFaceModel("unsloth/Qwen3-0.6B-GGUF") { Quantization = "Q8_0" }
/// HuggingFaceModel.Parse("unsloth/Qwen3-0.6B-GGUF:Q8_0")
/// HuggingFaceModel.Parse("hf://unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf")
/// </code>
/// </summary>
public sealed partial record HuggingFaceModel(string Repository)
{
    /// <summary>Path of the GGUF file in the repository (overrides <see cref="Quantization"/>).</summary>
    public string? File { get; init; }
    /// <summary>Quantization tag matched against file names (e.g. <c>Q4_K_M</c>, <c>UD-Q4_K_XL</c>, <c>BF16</c>).</summary>
    public string? Quantization { get; init; }
    /// <summary>Branch, tag or commit hash.</summary>
    public string Revision { get; init; } = "main";
    /// <summary>Also fetch the multimodal projector (mmproj) that best matches the model, when the repository has one.</summary>
    public bool IncludeMmproj { get; init; } = true;
    /// <summary>Path of the mmproj file in the repository (overrides the automatic choice).</summary>
    public string? MmprojFile { get; init; }

    /// <summary>
    /// Parses <c>owner/repo[@revision][:QUANT]</c>, <c>owner/repo[@revision]/path/file.gguf</c> (optionally prefixed with
    /// <c>hf://</c> or <c>hf:</c>) and <c>https://huggingface.co/owner/repo/resolve/revision/path/file.gguf</c> URLs.
    /// </summary>
    public static HuggingFaceModel Parse(string reference) =>
        TryParse(reference, out var m) ? m! : throw new FormatException($"'{reference}' is not a Hugging Face model reference (owner/repo[:QUANT] or owner/repo/file.gguf)");

    public static bool TryParse(string reference, out HuggingFaceModel? model)
    {
        model = null;
        string s = reference.Trim();
        if (Uri.TryCreate(s, UriKind.Absolute, out var uri) && (uri.Scheme == Uri.UriSchemeHttps || uri.Scheme == Uri.UriSchemeHttp))
            return TryFromUrl(uri, out model);
        if (s.StartsWith("hf://", StringComparison.OrdinalIgnoreCase)) s = s[5..];
        else if (s.StartsWith("hf:", StringComparison.OrdinalIgnoreCase)) s = s[3..];

        var parts = s.Split('/');
        if (parts.Length < 2) return false;
        string owner = parts[0], repo = parts[1], revision = "main";
        string? quant = null, file = null;
        if (parts.Length == 2)
        {
            int colon = repo.IndexOf(':');
            if (colon >= 0) { quant = repo[(colon + 1)..]; repo = repo[..colon]; if (quant.Length == 0) return false; }
        }
        else file = string.Join('/', parts[2..]);
        int at = repo.IndexOf('@');
        if (at >= 0) { revision = repo[(at + 1)..]; repo = repo[..at]; if (revision.Length == 0) return false; }
        string id = $"{owner}/{repo}";
        if (!IsValidRepositoryId(id) || file is { Length: 0 }) return false;
        model = new HuggingFaceModel(id) { File = file, Quantization = quant, Revision = revision };
        return true;
    }

    /// <summary><c>https://huggingface.co/owner/repo/resolve|blob/revision/path</c> → the file in that repository.</summary>
    internal static bool TryFromUrl(Uri uri, out HuggingFaceModel? model, string? endpointHost = null)
    {
        model = null;
        if (!string.Equals(uri.Host, endpointHost ?? "huggingface.co", StringComparison.OrdinalIgnoreCase)) return false;
        var seg = uri.AbsolutePath.Trim('/').Split('/').Select(Uri.UnescapeDataString).ToArray();
        if (seg.Length < 5 || (seg[2] != "resolve" && seg[2] != "blob")) return false;
        string id = $"{seg[0]}/{seg[1]}";
        if (!IsValidRepositoryId(id)) return false;
        model = new HuggingFaceModel(id) { Revision = seg[3], File = string.Join('/', seg[4..]), IncludeMmproj = false };
        return true;
    }

    /// <summary>Hugging Face repository id rules (as llama.cpp checks them): one '/', [A-Za-z0-9_] plus '.', '-' between them.</summary>
    public static bool IsValidRepositoryId(string id) => id.Length is > 0 and <= 256 && RepoIdRegex().IsMatch(id);

    [GeneratedRegex(@"^[A-Za-z0-9_]+([.-][A-Za-z0-9_]+)*/[A-Za-z0-9_]+([.-][A-Za-z0-9_]+)*$")]
    private static partial Regex RepoIdRegex();

    public override string ToString() =>
        $"{Repository}{(Revision != "main" ? "@" + Revision : "")}{(File != null ? "/" + File : Quantization != null ? ":" + Quantization : "")}";
}
