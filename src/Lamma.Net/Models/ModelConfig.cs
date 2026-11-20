using LammaNet.Compute;
using LammaNet.Gguf;

namespace LammaNet.Models;

/// <summary>Kind of sequence mixer used by a layer.</summary>
public enum LayerKind
{
    /// <summary>Softmax self-attention with a KV cache.</summary>
    Attention,
    /// <summary>Gated DeltaNet linear attention (Qwen3-Next / Qwen3.5 / Qwen3.6).</summary>
    GatedDeltaNet,
    /// <summary>Mamba-2 selective state space layer (Granite 4 hybrid, Falcon-H1, ...).</summary>
    Mamba2,
    /// <summary>
    /// Gated short convolution (LFM2): in_proj → (b, c, x); y = c · conv(b · x) over a SsmConvKernel-wide causal
    /// window; out_proj. The only state is the conv window.
    /// </summary>
    ShortConv,
}

/// <summary>Feed-forward activation.</summary>
public enum FfnActivation { SwiGlu, GeGlu }

/// <summary>
/// Hyper-parameters of a decoder model, read from GGUF metadata. Per-layer values are expanded
/// into arrays so heterogeneous architectures (SWA, hybrid recurrent, per-layer FFN widths) share one code path.
/// </summary>
public sealed class ModelConfig
{
    public required string Architecture { get; init; }
    public string Name { get; init; } = "";
    public int NLayer { get; init; }
    public int NEmbd { get; init; }
    public int NVocab { get; set; }
    public int NCtxTrain { get; init; }
    public float RmsEps { get; init; } = 1e-5f;

    public int[] NHead { get; init; } = [];
    public int[] NHeadKv { get; init; } = [];
    public int[] NFf { get; init; } = [];
    public int[] HeadDimK { get; init; } = [];
    public int[] HeadDimV { get; init; } = [];
    public LayerKind[] Kinds { get; init; } = [];
    public bool[] IsSwa { get; init; } = [];
    /// <summary>For each layer, the layer whose KV cache it reads (itself unless KV is shared).</summary>
    public int[] KvSourceLayer { get; init; } = [];
    public RopeConfig?[] Rope { get; init; } = [];

    public int SlidingWindow { get; init; }
    public FfnActivation Activation { get; init; } = FfnActivation.SwiGlu;

    // scales & caps
    /// <summary>Attention score scale; 0 means 1/sqrt(head_dim).</summary>
    public float AttentionScale { get; init; }
    public float EmbeddingScale { get; init; } = 1f;
    public float ResidualScale { get; init; } = 1f;
    /// <summary>Logits are divided by this value (Granite).</summary>
    public float LogitScale { get; init; } = 1f;
    public float AttentionTemperatureScale { get; init; }
    public float FinalLogitSoftcap { get; init; }
    public float AttnLogitSoftcap { get; init; }

    // Gemma 3n/4 per-layer embeddings
    public int NEmbdPerLayer { get; init; }

    // SSM / linear attention
    public int SsmConvKernel { get; init; }
    public int SsmStateSize { get; init; }
    public int SsmGroupCount { get; init; }
    public int SsmTimeStepRank { get; init; }
    public int SsmInnerSize { get; init; }

    // flags
    public bool QkNormFromRms { get; init; }
    /// <summary>Gemma 4 normalizes V with a weight-less RMS norm.</summary>
    public bool VNorm { get; init; }
    /// <summary>Qwen3.5 attention: q_proj outputs [q, gate] per head; output gated with sigmoid(gate).</summary>
    public bool GatedAttention { get; init; }
    public bool AddBos { get; init; }

    /// <summary>False for bidirectional (encoder / embedding) models: each token attends to the whole sequence.</summary>
    public bool Causal { get; init; } = true;
    /// <summary>Non-causal sliding windows are symmetric: |Δpos| ≤ SlidingWindow/2.</summary>
    public bool SymmetricSwa { get; init; }
    /// <summary>GGUF pooling_type: -1 unspecified, 0 none, 1 mean, 2 CLS, 3 last, 4 rank.</summary>
    public int PoolingType { get; init; } = -1;
    /// <summary>Width of the embedding output projection (output.weight of embedding models), or 0.</summary>
    public int NEmbdOut { get; init; }
    /// <summary>True for embedding models (bidirectional or with a pooling type): no LM head.</summary>
    public bool IsEmbeddingModel => !Causal || PoolingType > 0;
    /// <summary>Architecture-specific behavior (see <see cref="ArchitectureTraits"/>).</summary>
    public ArchitectureTraits Traits { get; init; } = new();

    public bool HasRecurrentLayers => Kinds.Any(k => k != LayerKind.Attention);
    public int NAttentionLayers => Kinds.Count(k => k == LayerKind.Attention);

    public static ModelConfig FromGguf(GgufFile f)
    {
        float[]? freqFactors = null;
        if (f.TryGetTensor("rope_freqs.weight", out var rfInfo))
            freqFactors = Tensors.WeightTensor.FromGguf(f, rfInfo).ToFloatArray();
        return FromMetadata(f.Metadata, freqFactors);
    }

    /// <summary>
    /// Derives the configuration from GGUF metadata alone. <paramref name="ropeFreqFactors"/> is the content of the
    /// <c>rope_freqs.weight</c> tensor, if the model has one (it is not part of the metadata).
    /// </summary>
    public static ModelConfig FromMetadata(GgufMetadata f, float[]? ropeFreqFactors = null)
    {
        string arch = f.Get<string>("general.architecture");
        var traits = ArchitectureTraits.For(arch);
        string P(string k) => $"{arch}.{k}";
        int nLayer = f.Get<int>(P("block_count"));
        int nEmbd = f.Get<int>(P("embedding_length"));
        int nCtx = f.Get(P("context_length"), 4096);
        var nHead = f.GetIntOrArray(P("attention.head_count"), nLayer) ?? new int[nLayer];
        var nHeadKv = f.GetIntOrArray(P("attention.head_count_kv"), nLayer) ?? (int[])nHead.Clone();
        var nFf = f.GetIntOrArray(P("feed_forward_length"), nLayer) ?? new int[nLayer];
        int maxHead = nHead.Max();
        int hdK = f.Get(P("attention.key_length"), maxHead > 0 ? nEmbd / maxHead : 0);
        int hdV = f.Get(P("attention.value_length"), hdK);
        int hdKSwa = f.Get(P("attention.key_length_swa"), hdK);
        int hdVSwa = f.Get(P("attention.value_length_swa"), hdV);
        int nRot = f.Get(P("rope.dimension_count"), hdK);
        int nRotSwa = f.Get(P("rope.dimension_count_swa"), nRot);
        float ropeBase = f.Get(P("rope.freq_base"), 10000f);
        float ropeBaseSwa = f.Get(P("rope.freq_base_swa"), ropeBase);
        float eps = f.Get(P("attention.layer_norm_rms_epsilon"), f.Get(P("attention.layer_norm_epsilon"), 1e-5f));
        int swa = f.Get(P("attention.sliding_window"), 0);

        // ---------------- layer kinds
        var kinds = new LayerKind[nLayer];
        if (traits.DeltaNetHybrid)
        {
            int interval = f.Get(P("full_attention_interval"), 4);
            bool[]? recr = f.Has(P("attention.recurrent_layers")) ? f.Get<bool[]>(P("attention.recurrent_layers")) : null;
            for (int i = 0; i < nLayer; i++)
                kinds[i] = (recr != null ? recr[i] : (i + 1) % interval != 0) ? LayerKind.GatedDeltaNet : LayerKind.Attention;
        }
        else if (traits.MambaHybrid)
            for (int i = 0; i < nLayer; i++) kinds[i] = nHeadKv[i] == 0 ? LayerKind.Mamba2 : LayerKind.Attention;
        else if (traits.ShortConvHybrid)
            for (int i = 0; i < nLayer; i++) kinds[i] = nHeadKv[i] == 0 ? LayerKind.ShortConv : LayerKind.Attention;

        // ---------------- sliding window pattern
        var isSwa = new bool[nLayer];
        if (f.TryGetValue(P("attention.sliding_window_pattern"), out var pat))
        {
            if (pat is bool[] bp) for (int i = 0; i < nLayer; i++) isSwa[i] = i < bp.Length && bp[i];
            else
            {
                int n = Convert.ToInt32(pat);
                // ggml: dense (non-SWA) layer every n-th layer
                for (int i = 0; i < nLayer; i++) isSwa[i] = n > 0 && (i % n) < n - 1;
            }
        }
        else if (swa > 0 && traits.SwaPeriod > 0)
            for (int i = 0; i < nLayer; i++) isSwa[i] = i % traits.SwaPeriod < traits.SwaPeriod - 1;

        // ---------------- kv sharing (Gemma 3n / 4)
        var kvSrc = new int[nLayer];
        for (int i = 0; i < nLayer; i++) kvSrc[i] = i;
        int nShared = f.Get(P("attention.shared_kv_layers"), 0);
        if (nShared > 0)
        {
            int kvFromStart = nLayer - nShared;
            for (int i = kvFromStart; i < nLayer; i++)
            {
                // reuse the last KV layer of the same type (SWA vs full)
                int src = kvFromStart - 1;
                while (src >= 0 && isSwa[src] != isSwa[i]) src--;
                kvSrc[i] = src;
            }
        }

        // ---------------- rope
        var ropeMode = RopeModeFor(arch);
        float freqScale = 1f;
        float extFactor = 0f;
        string scalingType = f.Get(P("rope.scaling.type"), "linear");
        float factor = f.Get(P("rope.scaling.factor"), f.Get(P("rope.scale_linear"), 0f));
        if (factor != 0f) freqScale = 1f / factor;
        if (scalingType == "yarn") extFactor = 1f;
        int nCtxOrig = f.Get(P("rope.scaling.original_context_length"), nCtx);
        float attnFactor = f.Get(P("rope.scaling.attn_factor"), 1f);
        float tempScale = arch == "mistral3" ? f.Get(P("attention.temperature_scale"), 0f) : 0;
        if (tempScale != 0 && nCtxOrig <= 0) throw new InvalidDataException("Attention temperature scaling requires an original context length");
        if (arch == "mistral3" && scalingType == "yarn" && factor > 1)
            attnFactor /= 1 + 0.1f * f.Get(P("rope.scaling.yarn_log_multiplier"), 0f) * MathF.Log(factor);
        int[] sections = f.Has(P("rope.dimension_sections")) ? f.Get<int[]>(P("rope.dimension_sections")) : [0, 0, 0, 0];
        if (sections.Length < 4) sections = [.. sections, .. new int[4 - sections.Length]];
        float[]? freqFactors = ropeFreqFactors;

        RopeConfig MakeRope(bool swaLayer)
        {
            return new RopeConfig
            {
                Mode = ropeMode,
                Dims = swaLayer ? nRotSwa : nRot,
                FreqBase = swaLayer ? ropeBaseSwa : ropeBase,
                // gemma3: linear scaling applies only to global layers; SWA layers use scale 1
                FreqScale = swaLayer && traits.Gemma ? 1f : freqScale,
                // gemma4 uses rope_freqs only on full-attention layers ("proportional rope")
                FreqFactors = traits.ProportionalRope && swaLayer ? null : freqFactors,
                ExtFactor = extFactor,
                AttnFactor = attnFactor,
                NCtxOrig = nCtxOrig,
                BetaFast = f.Get(P("rope.scaling.yarn_beta_fast"), 32f),
                BetaSlow = f.Get(P("rope.scaling.yarn_beta_slow"), 1f),
                Sections = sections,
            };
        }
        var ropeFull = MakeRope(false);
        var ropeSwa = MakeRope(true);
        // Granite uses rope.scaling.finetuned as a switch: false => no positional encoding (NoPE, e.g. Granite 4.0-H)
        bool useRope = !(traits.RopeSwitchByFinetuned && !f.Get(P("rope.scaling.finetuned"), true));
        var ropes = new RopeConfig?[nLayer];
        for (int i = 0; i < nLayer; i++) ropes[i] = useRope && kinds[i] == LayerKind.Attention ? (isSwa[i] ? ropeSwa : ropeFull) : null;

        var hk = new int[nLayer];
        var hv = new int[nLayer];
        for (int i = 0; i < nLayer; i++) { hk[i] = isSwa[i] ? hdKSwa : hdK; hv[i] = isSwa[i] ? hdVSwa : hdV; }

        bool gemma = traits.Gemma;
        float attnScale = f.Get(P("attention.scale"), 0f);
        if (traits.AttentionScaleOne) attnScale = 1f;

        float embScale = f.Get(P("embedding_scale"), 0f);
        if (embScale == 0f) embScale = gemma ? MathF.Sqrt(nEmbd) : 1f;

        int nVocab = f.Get(P("vocab_size"), 0);
        if (nVocab == 0) nVocab = (int)Math.Max(0, f.ArrayLength("tokenizer.ggml.tokens"));

        bool addBos = f.Get("tokenizer.ggml.add_bos_token", traits.AddBosDefault);

        return new ModelConfig
        {
            Architecture = arch,
            Name = f.Get("general.name", arch),
            NLayer = nLayer,
            NEmbd = nEmbd,
            NVocab = nVocab,
            NCtxTrain = nCtx,
            RmsEps = eps,
            NHead = nHead,
            NHeadKv = nHeadKv,
            NFf = nFf,
            HeadDimK = hk,
            HeadDimV = hv,
            Kinds = kinds,
            IsSwa = isSwa,
            KvSourceLayer = kvSrc,
            Rope = ropes,
            SlidingWindow = swa,
            Activation = gemma ? FfnActivation.GeGlu : FfnActivation.SwiGlu,
            AttentionScale = attnScale,
            AttentionTemperatureScale = tempScale,
            EmbeddingScale = embScale,
            ResidualScale = f.Get(P("residual_scale"), 1f),
            LogitScale = f.Get(P("logit_scale"), 1f),
            FinalLogitSoftcap = f.Get(P("final_logit_softcapping"), 0f),
            AttnLogitSoftcap = f.Get(P("attn_logit_softcapping"), 0f),
            NEmbdPerLayer = f.Get(P("embedding_length_per_layer_input"), 0),
            SsmConvKernel = f.Get(P("ssm.conv_kernel"), f.Get(P("shortconv.l_cache"), 0)),
            SsmStateSize = f.Get(P("ssm.state_size"), 0),
            SsmGroupCount = f.Get(P("ssm.group_count"), 0),
            SsmTimeStepRank = f.Get(P("ssm.time_step_rank"), 0),
            SsmInnerSize = f.Get(P("ssm.inner_size"), 0),
            VNorm = traits.VNorm,
            GatedAttention = traits.GatedAttention,
            AddBos = addBos,
            Causal = f.Get(P("attention.causal"), true),
            SymmetricSwa = traits.SymmetricSwa,
            PoolingType = f.Get(P("pooling_type"), -1),
            NEmbdOut = f.Get(P("embedding_length_out"), 0),
            Traits = traits,
        };
    }

    public static RopeMode RopeModeFor(string arch) => arch switch
    {
        "llama" or "granite" or "granitehybrid" or "granitemoe" or "mistral" or "mistral3" or "baichuan" or "deci" or "llama4" => RopeMode.Normal,
        "qwen35" or "qwen35moe" or "qwen3vl" or "qwen3vlmoe" => RopeMode.IMRope,
        "qwen2vl" => RopeMode.MRope,
        _ => RopeMode.Neox,
    };

    public override string ToString() =>
        $"{Architecture} '{Name}': layers={NLayer} embd={NEmbd} vocab={NVocab} heads={NHead.Max()}/{NHeadKv.Max()} " +
        $"ff={NFf.Max()} ctx={NCtxTrain} kinds=[{string.Join("", Kinds.Select(k => k switch { LayerKind.Attention => 'A', LayerKind.GatedDeltaNet => 'D', LayerKind.ShortConv => 'C', _ => 'M' }))}]";
}
