using LammaNet.Compute;
using LammaNet.Gguf;
using LammaNet.Tensors;

namespace LammaNet.Models;

/// <summary>One layer of an <see cref="MtpDrafterModel"/>: Q-only attention over the target's KV cache, then a GELU FFN, Gemma-style sandwich norms.</summary>
public sealed class MtpDrafterLayer
{
    public required float[] AttnNorm { get; init; }
    public required WeightTensor Wq { get; init; }
    public required float[] QNorm { get; init; }
    public required WeightTensor Wo { get; init; }
    public required float[] AttnPostNorm { get; init; }
    public required float[] FfnNorm { get; init; }
    public required WeightTensor Gate { get; init; }
    public required WeightTensor Up { get; init; }
    public required WeightTensor Down { get; init; }
    public required float[] FfnPostNorm { get; init; }
    public required float OutScale { get; init; }
    public required bool IsSwa { get; init; }
    public required int HeadDim { get; init; }
}

/// <summary>
/// A multi-token-prediction drafter for speculative decoding: the Gemma 4 "assistant" (GGUF architecture
/// <c>gemma4-assistant</c>, published next to the model, e.g. <c>mtp-gemma-4-E2B-it.gguf</c>). It predicts the token after
/// the one just sampled from (target embedding of that token × √d_target, the target's LM-head input - its normalized last
/// hidden state - at the row that produced it): a projection to
/// its own width, four small layers whose attention reads the target's KV cache (sliding layers the target's
/// second-to-last layer, the full layer its last layer; no KV of their own), its own tied 256-wide output, and a
/// projection back to the target width that seeds the next draft step. All draft steps of a round use the position of
/// the sampled token.
/// </summary>
public sealed class MtpDrafterModel : IDisposable
{
    public GgufFile File { get; }
    public string Name { get; }
    /// <summary>Width of the drafter (256 for the E2B assistant).</summary>
    public int NEmbd { get; }
    /// <summary>Hidden width of the target model it drafts for.</summary>
    public int NEmbdTarget { get; }
    public int NHead { get; }
    public int NHeadKv { get; }
    public int NFf { get; }
    public int SlidingWindow { get; }
    public float Eps { get; }
    public int NVocab { get; }
    public RopeConfig RopeFull { get; }
    public RopeConfig RopeSwa { get; }
    /// <summary>Token embedding, also the (tied) output projection: [NVocab, NEmbd].</summary>
    public WeightTensor TokenEmbedding { get; }
    public float[] OutputNorm { get; }
    /// <summary>[2·NEmbdTarget → NEmbd] input projection of (target embedding, target hidden state).</summary>
    public WeightTensor PreProjection { get; }
    /// <summary>[NEmbd → NEmbdTarget] projection of the normalized output, the hidden state of the next draft step.</summary>
    public WeightTensor PostProjection { get; }
    public MtpDrafterLayer[] Layers { get; }

    private MtpDrafterModel(GgufFile f)
    {
        File = f;
        const string a = "gemma4-assistant.";
        Name = f.Get("general.name", "MTP drafter");
        NEmbd = f.Get<int>(a + "embedding_length");
        NEmbdTarget = f.Get<int>(a + "embedding_length_out");
        int nLayer = f.Get<int>(a + "block_count");
        NHead = f.Get<int>(a + "attention.head_count");
        NHeadKv = f.Get(a + "attention.head_count_kv", 1);
        NFf = f.Get<int>(a + "feed_forward_length");
        SlidingWindow = f.Get(a + "attention.sliding_window", 512);
        Eps = f.Get(a + "attention.layer_norm_rms_epsilon", 1e-6f);
        int hdFull = f.Get<int>(a + "attention.key_length"), hdSwa = f.Get(a + "attention.key_length_swa", hdFull);
        var isSwa = new bool[nLayer];
        if (f.TryGet<bool[]>(a + "attention.sliding_window_pattern", out var pat))
            for (int i = 0; i < nLayer; i++) isSwa[i] = i < pat.Length && pat[i];

        float[]? freqFactors = f.TryGetTensor("rope_freqs.weight", out var rf) ? WeightTensor.FromGguf(f, rf).ToFloatArray() : null;
        RopeFull = new RopeConfig { Mode = RopeMode.Neox, Dims = f.Get(a + "rope.dimension_count", hdFull), FreqBase = f.Get(a + "rope.freq_base", 1e6f), FreqFactors = freqFactors };
        RopeSwa = new RopeConfig { Mode = RopeMode.Neox, Dims = f.Get(a + "rope.dimension_count_swa", hdSwa), FreqBase = f.Get(a + "rope.freq_base_swa", 1e4f) };

        WeightTensor T(string n) => WeightTensor.FromGguf(f, f.TryGetTensor(n, out var i) ? i : throw new InvalidDataException($"MTP drafter tensor {n} missing"));
        float[] V(string n) => T(n).ToFloatArray();
        TokenEmbedding = T("token_embd.weight");
        NVocab = TokenEmbedding.Rows;
        OutputNorm = V("output_norm.weight");
        PreProjection = T("nextn.pre_projection.weight");
        PostProjection = T("nextn.post_projection.weight");
        Layers = new MtpDrafterLayer[nLayer];
        for (int il = 0; il < nLayer; il++)
        {
            string p = $"blk.{il}.";
            Layers[il] = new MtpDrafterLayer
            {
                AttnNorm = V(p + "attn_norm.weight"), Wq = T(p + "attn_q.weight"), QNorm = V(p + "attn_q_norm.weight"), Wo = T(p + "attn_output.weight"),
                AttnPostNorm = V(p + "post_attention_norm.weight"), FfnNorm = V(p + "ffn_norm.weight"),
                Gate = T(p + "ffn_gate.weight"), Up = T(p + "ffn_up.weight"), Down = T(p + "ffn_down.weight"), FfnPostNorm = V(p + "post_ffw_norm.weight"),
                OutScale = V(p + "layer_output_scale.weight")[0], IsSwa = isSwa[il], HeadDim = isSwa[il] ? hdSwa : hdFull,
            };
        }
        if (PreProjection.Cols != 2 * NEmbdTarget || PreProjection.Rows != NEmbd || PostProjection.Cols != NEmbd || PostProjection.Rows != NEmbdTarget)
            throw new InvalidDataException("MTP drafter projections do not match its widths");
    }

    /// <summary>Loads a drafter GGUF (architecture gemma4-assistant).</summary>
    public static MtpDrafterModel Load(string path)
    {
        var f = new GgufFile(path);
        try
        {
            string arch = f.Get("general.architecture", "");
            if (arch != "gemma4-assistant")
                throw new NotSupportedException($"'{path}' is not a supported MTP drafter (architecture '{arch}'; supported: gemma4-assistant)");
            return new MtpDrafterModel(f);
        }
        catch
        {
            f.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The target layer whose KV cache drafter layer <paramref name="il"/> reads: sliding layers the target's
    /// second-to-last layer, the full layer its last layer (each through the target's KV sharing).
    /// </summary>
    public int TargetKvLayer(int il, ModelConfig target) => target.KvSourceLayer[Layers[il].IsSwa ? target.NLayer - 2 : target.NLayer - 1];

    /// <summary>Throws when this drafter cannot draft for <paramref name="target"/>.</summary>
    public void Validate(Model target)
    {
        var c = target.Config;
        if (c.Architecture != "gemma4") throw new NotSupportedException($"MTP drafter '{Name}' drafts for Gemma 4, not {c.Architecture}");
        if (c.NEmbd != NEmbdTarget) throw new NotSupportedException($"MTP drafter '{Name}' expects a target width of {NEmbdTarget}, the model has {c.NEmbd}");
        if (c.NVocab != NVocab && target.Vocab.Count != NVocab) throw new NotSupportedException($"MTP drafter '{Name}' has {NVocab} tokens, the model {target.Vocab.Count}");
        for (int il = 0; il < Layers.Length; il++)
        {
            int src = TargetKvLayer(il, c);
            if (c.NHeadKv[src] != NHeadKv || c.HeadDimK[src] != Layers[il].HeadDim || c.HeadDimV[src] != Layers[il].HeadDim || c.IsSwa[src] != Layers[il].IsSwa)
                throw new NotSupportedException($"MTP drafter '{Name}' layer {il} does not match the KV layout of target layer {src}");
        }
    }

    public void Dispose() => File.Dispose();
}
