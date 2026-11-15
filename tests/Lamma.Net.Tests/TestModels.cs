namespace LammaNet.Tests;

/// <summary>Locates test GGUF models (set LAMMA_MODELS to override the default ~/models directory).</summary>
public static class TestModels
{
    public static string Dir => Environment.GetEnvironmentVariable("LAMMA_MODELS")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    public static string? Find(string fileName)
    {
        var p = Path.Combine(Dir, fileName);
        return File.Exists(p) ? p : null;
    }

    public const string Qwen3Q4 = "Qwen3-0.6B-Q4_K_M.gguf";
    public const string Qwen3Q8 = "Qwen3-0.6B-Q8_0.gguf";
    public const string Llama32 = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    public const string Qwen35 = "Qwen3.5-0.8B-Q4_K_M.gguf";
    public const string Granite = "granite-4.0-micro-Q4_K_M.gguf";
    public const string GraniteH = "granite-4.0-h-micro-Q4_K_M.gguf";
    public const string Gemma4 = "gemma-4-E2B-it-Q4_K_M.gguf";
    public const string EmbeddingGemma2 = "embeddinggemma-2-UD-Q4_K_XL.gguf";
    public const string Lfm2Q4 = "LFM2.5-230M-Q4_K_M.gguf";
    public const string Lfm2Bf16 = "LFM2.5-230M-BF16.gguf";
}
