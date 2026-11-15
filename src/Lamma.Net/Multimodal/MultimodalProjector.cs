using LammaNet.Chat;
using LammaNet.Gguf;
using LammaNet.Models;

namespace LammaNet.Multimodal;

/// <summary>Loads multimodal projector (mmproj) GGUF files.</summary>
public static class MultimodalProjector
{
    /// <summary>Projector types that can be loaded.</summary>
    public static IReadOnlyList<string> SupportedTypes { get; } = ["gemma4v", "gemma4a", "qwen3vl_merger", "idefics3"];

    /// <summary>
    /// Loads an mmproj GGUF for the engine's model; the vision encoder runs on the engine's accelerator (GPU offload)
    /// when it has one, unless <see cref="ProjectorOptions.Accelerator"/> is set.
    /// </summary>
    public static IMediaEncoder Load(string mmprojPath, Inference.InferenceEngine engine, ProjectorOptions? options = null)
    {
        options ??= new ProjectorOptions();
        if (options.Accelerator == null && engine.Offload is IVisionAccelerator acc) options = options with { Accelerator = acc };
        if (options.LoggerFactory == null) options = options with { LoggerFactory = engine.LoggerFactory };
        if (options.Resources == null) options = options with { Resources = engine.Resources };
        return Load(mmprojPath, engine.Model, options);
    }

    /// <summary>Loads an mmproj GGUF for <paramref name="model"/> and returns a media encoder.</summary>
    public static IMediaEncoder Load(string mmprojPath, Model model, ProjectorOptions? options = null)
    {
        string vision, audio;
        using (var f = new GgufFile(mmprojPath))
        {
            if (f.Get("general.architecture", "") != "clip")
                throw new InvalidDataException($"'{mmprojPath}' is not a multimodal projector (general.architecture != clip)");
            vision = f.Get("clip.vision.projector_type", f.Get("clip.projector_type", ""));
            audio = f.Get("clip.audio.projector_type", "");
        }
        var log = Diagnostics.LammaLogging.Create(options?.LoggerFactory, "LammaNet.Multimodal.MultimodalProjector");
        string type = audio == "gemma4a" && vision != "gemma4v" ? audio : vision;
        try
        {
            switch (type)
            {
                case "gemma4v" or "gemma4a":
                    var g = new Gemma4Projector(mmprojPath, model, options);
                    Diagnostics.Log.ProjectorLoaded(log, mmprojPath, type, options?.Accelerator != null ? "the accelerator when they fit (created on first use)" : "the CPU");
                    return g;
                case "qwen3vl_merger":
                    var q = new Qwen3VLProjector(mmprojPath, model, options);
                    Diagnostics.Log.ProjectorLoaded(log, mmprojPath, type, q.EncoderDevice);
                    return q;
                case "idefics3":
                    var i = new Idefics3Projector(mmprojPath, model, options);
                    Diagnostics.Log.ProjectorLoaded(log, mmprojPath, type, i.EncoderDevice);
                    return i;
            }
            throw new NotSupportedException($"Multimodal projector type not supported yet (vision='{vision}', audio='{audio}'). Supported: {string.Join(", ", SupportedTypes)}");
        }
        catch (Exception e)
        {
            Diagnostics.Log.ProjectorLoadFailed(log, e, mmprojPath);
            throw;
        }
    }
}
