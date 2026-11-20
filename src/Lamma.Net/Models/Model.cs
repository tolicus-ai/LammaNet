using LammaNet.Diagnostics;
using LammaNet.Gguf;
using LammaNet.Tensors;
using LammaNet.Tokenization;

namespace LammaNet.Models;

/// <summary>Options controlling how a model is loaded.</summary>
public sealed class ModelLoadOptions
{
    /// <summary>Lock weights into RAM (pre-fault all pages) to avoid page faults on first use.</summary>
    public bool Prefetch { get; init; } = true;
    /// <summary>Logging of the load (default: <see cref="LammaLogging.LoggerFactory"/>).</summary>
    public Microsoft.Extensions.Logging.ILoggerFactory? LoggerFactory { get; init; }
}

/// <summary>
/// An immutable, thread-safe loaded GGUF model: configuration, tokenizer and weights (zero-copy from the mapping).
/// Create any number of <see cref="Inference.InferenceEngine"/> / contexts on top of one model.
/// </summary>
public sealed unsafe class Model : IDisposable
{
    public GgufFile File { get; }
    public ModelConfig Config { get; }
    public Vocabulary Vocab { get; }
    public WeightTensor TokenEmbedding { get; }
    public WeightTensor Output { get; }
    public float[] OutputNorm { get; }
    public float[]? OutputNormBias { get; private init; }
    public float[]? OutputBias { get; private init; }
    public LayerWeights[] Layers { get; }

    // Gemma 3n/4 per-layer embeddings
    public WeightTensor? PerLayerTokenEmbedding { get; }
    public WeightTensor? PerLayerModelProj { get; }
    public float[]? PerLayerProjNorm { get; }

    /// <summary>The raw chat template stored in the GGUF (Jinja), if any.</summary>
    public string? ChatTemplate { get; }

    private ModelInfo? _info;
    /// <summary>Metadata of the GGUF file: provenance, hyper-parameters, tokenizer, tensors and quantization.</summary>
    public ModelInfo Info => _info ??= ModelInfo.FromGguf(File);

    private Model(GgufFile file, ModelConfig config, Vocabulary vocab, WeightTensor tokEmbd, WeightTensor output,
        float[] outputNorm, LayerWeights[] layers, WeightTensor? pleTok, WeightTensor? pleProj, float[]? pleNorm)
    {
        File = file; Config = config; Vocab = vocab; TokenEmbedding = tokEmbd; Output = output; OutputNorm = outputNorm;
        Layers = layers; PerLayerTokenEmbedding = pleTok; PerLayerModelProj = pleProj; PerLayerProjNorm = pleNorm;
        ChatTemplate = file.Get<string?>("tokenizer.chat_template", null);
    }

    public static Model Load(string path, ModelLoadOptions? options = null)
    {
        options ??= new ModelLoadOptions();
        var log = LammaLogging.Create<Model>(options.LoggerFactory);
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        GgufFile f;
        try { f = new GgufFile(path); }
        catch (Exception e)
        {
            Log.ModelLoadFailed(log, e, path);
            throw;
        }
        try
        {
            var cfg = ModelConfig.FromGguf(f);
            var vocab = Vocabulary.FromGguf(f);
            if (cfg.NVocab == 0) cfg.NVocab = vocab.Count;

            WeightTensor T(string name) => WeightTensor.FromGguf(f, f.GetTensor(name));
            WeightTensor? TOpt(string name) => f.TryGetTensor(name, out var i) ? WeightTensor.FromGguf(f, i) : null;
            float[]? FOpt(string name) => TOpt(name)?.ToFloatArray();

            var tok = T("token_embd.weight");
            var output = TOpt("output.weight") ?? tok;
            // LFM2 stores the final norm as token_embd_norm
            var outNorm = (TOpt("output_norm.weight") ?? T("token_embd_norm.weight")).ToFloatArray();

            var layers = new LayerWeights[cfg.NLayer];
            bool qwen35 = cfg.Traits.FfnNormNamedPostAttention;
            for (int i = 0; i < cfg.NLayer; i++)
            {
                string p = $"blk.{i}.";
                var lw = new LayerWeights { Index = i, Kind = cfg.Kinds[i] };
                lw.AttnNorm = FOpt(p + "attn_norm.weight");
                if (cfg.Traits.Phi2)
                {
                    lw.AttnNormBias = T(p + "attn_norm.bias").ToFloatArray();
                    lw.Bqkv = FOpt(p + "attn_qkv.bias");
                }
                if (qwen35) lw.FfnNorm = FOpt(p + "post_attention_norm.weight");
                else
                {
                    lw.AttnPostNorm = FOpt(p + "post_attention_norm.weight");
                    lw.FfnNorm = FOpt(p + "ffn_norm.weight");
                }
                lw.FfnPostNorm = FOpt(p + "post_ffw_norm.weight");

                switch (cfg.Kinds[i])
                {
                    case LayerKind.Attention:
                        lw.Wq = TOpt(p + "attn_q.weight");
                        lw.Wk = TOpt(p + "attn_k.weight");
                        lw.Wv = TOpt(p + "attn_v.weight");
                        lw.Wqkv = TOpt(p + "attn_qkv.weight");
                        lw.Wo = T(p + "attn_output.weight");
                        lw.Bq = FOpt(p + "attn_q.bias");
                        lw.Bk = FOpt(p + "attn_k.bias");
                        lw.Bv = FOpt(p + "attn_v.bias");
                        lw.Bo = FOpt(p + "attn_output.bias");
                        lw.QNorm = FOpt(p + "attn_q_norm.weight");
                        lw.KNorm = FOpt(p + "attn_k_norm.weight");
                        break;
                    case LayerKind.GatedDeltaNet:
                        lw.SsmInProjQkv = T(p + "attn_qkv.weight");
                        lw.SsmGate = T(p + "attn_gate.weight");
                        lw.SsmBeta = T(p + "ssm_beta.weight");
                        lw.SsmAlpha = T(p + "ssm_alpha.weight");
                        lw.SsmOut = T(p + "ssm_out.weight");
                        lw.SsmConv1d = T(p + "ssm_conv1d.weight").ToFloatArray();
                        lw.SsmDtBias = FOpt(p + "ssm_dt.bias");
                        lw.SsmA = FOpt(p + "ssm_a");
                        lw.SsmNorm = FOpt(p + "ssm_norm.weight");
                        break;
                    case LayerKind.ShortConv:
                        lw.SsmIn = T(p + "shortconv.in_proj.weight");
                        lw.SsmOut = T(p + "shortconv.out_proj.weight");
                        lw.SsmConv1d = T(p + "shortconv.conv.weight").ToFloatArray();
                        break;
                    case LayerKind.Mamba2:
                        lw.SsmIn = T(p + "ssm_in.weight");
                        lw.SsmOut = T(p + "ssm_out.weight");
                        lw.SsmConv1d = T(p + "ssm_conv1d.weight").ToFloatArray();
                        lw.SsmConv1dBias = FOpt(p + "ssm_conv1d.bias");
                        lw.SsmDtBias = FOpt(p + "ssm_dt.bias");
                        lw.SsmA = FOpt(p + "ssm_a");
                        lw.SsmD = FOpt(p + "ssm_d");
                        lw.SsmNorm = FOpt(p + "ssm_norm.weight");
                        break;
                }

                lw.FfnGate = TOpt(p + "ffn_gate.weight");
                lw.FfnUp = TOpt(p + "ffn_up.weight");
                lw.FfnDown = TOpt(p + "ffn_down.weight");
                lw.FfnGateBias = FOpt(p + "ffn_gate.bias");
                lw.FfnUpBias = FOpt(p + "ffn_up.bias");
                lw.FfnDownBias = FOpt(p + "ffn_down.bias");
                var os = FOpt(p + "layer_output_scale.weight");
                if (os != null) lw.OutScale = os[0];
                lw.PleInpGate = TOpt(p + "inp_gate.weight");
                lw.PleProj = TOpt(p + "proj.weight");
                lw.PlePostNorm = FOpt(p + "post_norm.weight");
                if (lw.FfnGate == null && lw.FfnUp == null && f.Has($"{cfg.Architecture}.expert_count") && f.Get($"{cfg.Architecture}.expert_count", 0) > 0)
                    throw new NotSupportedException("Mixture-of-experts layers are not supported yet");
                layers[i] = lw;
            }

            var pleTok = f.TryGetTensor("per_layer_token_embd.weight", out var pi) ? WeightTensor.FromGguf(f, pi) : null;
            var pleProj = f.TryGetTensor("per_layer_model_proj.weight", out var pj) ? WeightTensor.FromGguf(f, pj) : null;
            var pleNorm = f.TryGetTensor("per_layer_proj_norm.weight", out var pn) ? WeightTensor.FromGguf(f, pn).ToFloatArray() : null;

            var model = new Model(f, cfg, vocab, tok, output, outNorm, layers, pleTok, pleProj, pleNorm)
            {
                OutputNormBias = cfg.Traits.Phi2 ? T("output_norm.bias").ToFloatArray() : null,
                OutputBias = cfg.Traits.Phi2 ? T("output.bias").ToFloatArray() : null,
            };
            if (options.Prefetch) model.Prefetch();
            Log.ModelLoaded(log, path, cfg.Name, cfg.Architecture, cfg.NLayer, f.FileSize >> 20,
                System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            return model;
        }
        catch (Exception e)
        {
            Log.ModelLoadFailed(log, e, path);
            f.Dispose();
            throw;
        }
    }

    /// <summary>Touches every weight page so the first inference does not stall on page faults.</summary>
    public void Prefetch()
    {
        var tensors = File.Tensors.ToArray();
        Parallel.ForEach(tensors, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount / 2) }, t =>
        {
            byte* p = File.GetTensorData(t);
            long n = t.ByteSize;
            long sum = 0;
            for (long i = 0; i < n; i += 4096) sum += p[i];
            if (sum == long.MinValue) Console.Write(""); // keep the loop alive
        });
    }

    public long WeightBytes => File.Tensors.Sum(t => t.ByteSize);

    public void Dispose() => File.Dispose();

    public override string ToString() => Config.ToString();
}
