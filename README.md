# Lamma.Net

Pure C# (.NET 10) in-process inference engine for GGUF models, with Microsoft.Extensions.AI / Agent Framework bindings.

## Status
- **Models (verified against llama.cpp / exact float64 reference):** Llama 3.x, Qwen3, Qwen3.5/3.6 family (Gated DeltaNet hybrid), Granite 4 (dense + Mamba2 hybrid), Gemma 2/3/4, LFM2/LFM2.5 (short-conv hybrid), EmbeddingGemma 2 (bidirectional embeddings), granite-docling (document pages → DocTags).
- **Quantization:** F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K–Q6_K, IQ4_NL/XS. AVX2/FMA (+AVX-VNNI) int8 kernels; packed float GEMM for prefill.
- **Engine:** paged KV cache (F16/F32/Q8_0) with prefix reuse, recurrent-state snapshots for hybrid models, continuous batching, chunked prefill, flash-decoding attention, sliding window, shared KV (Gemma 4), speculative decoding (prompt lookup), context shift (opt-in).
- **GPU (Lamma.Net.Gpu, NVIDIA/CUDA via ILGPU — kernels written in C#):** attention, Gated DeltaNet and Mamba2 layers, LM head and Gemma per-layer inputs. Supports FP16 and Q8_0 paged KV caches, recurrent states in device slots, CUDA graphs for decode, and tensor-core GEMM and flash attention for prompts on sm_80+. The engine fits the CPU/GPU split to free VRAM. Used automatically when the package is referenced and a CUDA device is present (`LAMMA_DEVICE=cpu` disables).
- **Chat:** Jinja2 chat templates from GGUF (conformance-tested against Python jinja2), reasoning + tool-call parsing (Hermes JSON, Qwen XML, Llama JSON, Gemma 4), JSON-constrained decoding.
- **Multimodal:** Gemma 4 vision and audio encoders, Qwen3.5/3.6 vision (Qwen3-VL merger), Idefics3 / granite-docling (SigLIP, tiled pages) — mmproj GGUF; the encoder transformers run on the GPU when the model does. Image preprocessing reproduces Pillow/llama.cpp exactly.
- **Microsoft.Extensions.AI:** `IChatClient`, `IEmbeddingGenerator`, `ISpeechToTextClient`, `IRealtimeClient` (full duplex, VAD, barge-in), DI helpers; works with `Microsoft.Agents.AI`.
- **Not done yet:** ARM NEON kernels, OpenCL/AMD GPUs, NuGet packaging, split GGUF files.

### GPU performance (RTX 3070, Q4_K_M, tokens/s; llama.cpp CUDA via llama-server for reference)
| Model | prompt (pp512) | generation (tg128) | llama.cpp pp / tg |
|---|---|---|---|
| Qwen3-0.6B | 13,800 | 436 | ~17,000 / 348 |
| Qwen3.5-0.8B | 8,200 | 388 | ~10,000 / 273 * |
| Llama-3.2-1B | 10,000 | 310 | ~14,800 / 326 |
| Granite-4.0-micro | 3,250 | 123 | ~4,800 / 128 * |
| Granite-4.0-H-micro | 2,550 | 104 | ~3,300 / 134 * |
| Gemma-4-E2B | 4,350 | 156 | ~5,850 / 149 |

llama.cpp with a ~460-token prompt (\* ~700-token prompt, earlier run). Measured ceilings of this card (`lamma gpu-peak`): fp16→fp32 tensor cores 41.7 TFLOPS, int8 170 TOPS.
Projections of normalized inputs (q/k/v, FFN gate/up, LM head) use 8-bit integer activations: dp4a matrix-vector kernels
for generation and an int8 tensor-core GEMM (Q4_K, Q4_0, Q8_0) for prompts, 21-33 TOPS effective vs 17-24 TFLOPS for
the fp16 GEMM on the same shapes, which the remaining projections use. `GpuOffloadOptions.Int8` defaults to `Auto`.
It selects float activations for Gemma 3 with Q4_0 query weights and normalized int8 projections elsewhere. Set
`Int8` explicitly or use `LAMMA_GPU_Q8=off|normalized|all` to override the activation choice. On the CPU (8-core i7-9700K) prompt processing is at 90-100% of llama.cpp and
generation on par.

### Choosing KV cache capacity

FP16 is the default. For a longer context or more layers on the GPU, pass `--compact-kv` to `ask`, `bench`, or `gpucheck`. This command requests a 16,384-token context with a Q8_0 cache.

```powershell
lamma ask D:\Models\granite-4.2-8b-Q4_K_M.gguf "Summarize the following text..." --ctx 16384 --compact-kv
```

In the .NET API, set `EngineOptions.KvCacheType = KvCacheType.Q8_0` and set `ContextSize` to the per-sequence limit. Both host and CUDA layers use 34 bytes per 32 cached values, compared with 64 bytes for FP16. Host layers with incompatible head widths fall back to FP16. Model weights keep their original precision. CUDA can fit more layers with the smaller cache, though reading compressed values adds compute and quantization can change the next token. Use `gpucheck` with the model and cache mode you intend to run.

The one-shot `ask` command reserves one context of cache, rounded up to a page. The multi-request .NET engine defaults to two contexts of pooled cache and accepts an explicit `KvCacheTokens` budget.

Granite 4.2 8B Q4_K_M on an RTX 3070 with 7,114 MiB free produced the same 119-token completion in each run below. The prompt occupied 34 tokens. Context is the reserved capacity. Device memory counts engine buffers at startup.

| Context | KV cache | GPU placement | Device MiB | Decode tokens/s |
|---:|---|---|---:|---:|
| 8,192 | FP16 | 40/40 layers + output head | 6,287 | 55.7 |
| 8,192 | Q8_0 | 40/40 layers + output head | 5,689 | 54.3 |
| 16,384 | FP16 | 36/40 layers | 6,503 | 10.8 |
| 16,384 | Q8_0 | 40/40 layers + output head | 6,369 | 54.6 |

On Gemma 3 4B Q4_0, prefer FP16 KV for the closest match to the CPU reference. In a 32-position comparison, default GPU prefill matched the top token 32 times with FP16 and 31 times with Q8_0. The maximum logit differences were 0.018 and 0.301. Phi-2 uses CPU inference and its 80-wide heads use FP16 in both modes.

<details>
<summary>RTX 3070 cache benchmarks across 11 GGUF files</summary>

Median tokens/s from three runs per mode, one sequence, a 512-token prompt and 32 decode steps. The weight format in the first column is the format on disk. For reproduction, set `$env:LAMMA_GPU = '-1:cuda'` in PowerShell and run `lamma bench <file> 512 32`, then repeat with `--compact-kv`. Automatic layer placement depends on available VRAM.

| Model and weights | FP16 prompt | Q8 cache prompt | FP16 decode | Q8 cache decode |
|---|---:|---:|---:|---:|
| Gemma 3 270M Q4_K_M | 26,831 | 23,765 | 540.1 | 446.2 |
| Gemma 3 4B Q4_0 | 2,842 | 2,758 | 77.9 | 76.7 |
| Granite 4.2 3B Q4_K_M | 3,163 | 3,064 | 120.0 | 112.0 |
| Granite 4.2 3B Q8_0 | 3,108 | 3,036 | 83.7 | 81.4 |
| Granite 4.2 8B Q4_K_M | 1,502 | 1,448 | 56.1 | 53.9 |
| Granite 4.2 8B Q8_0 (CPU/GPU split) | 156 | 135 | 5.6 | 6.6 |
| Granite Vision 4.1 4B Q4_K_M (text) | 3,126 | 2,978 | 117.7 | 109.3 |
| Granite Vision 4.1 4B Q8_0 (text) | 3,131 | 2,962 | 85.5 | 79.1 |
| Ministral 3 3B Q4_K_M | 3,352 | 3,184 | 124.8 | 118.8 |
| Phi-2 Q4_K_M (CPU, FP16 cache in both runs) | 101 | 90 | 12.0 | 11.4 |
| Qwen3 8B Q4_K_M | 1,687 | 1,648 | 60.4 | 59.5 |

</details>

## Quick start
```csharp
using var client = LammaChatClient.Load("gemma-4-E2B-it-Q4_K_M.gguf", new EngineOptions { ContextSize = 8192 },
    new LammaChatClientOptions { MmprojPath = "mmproj-gemma-4-E2B-it-Q8_0.gguf" });
IChatClient chat = new ChatClientBuilder(client).UseFunctionInvocation().Build();
var agent = chat.AsAIAgent(instructions: "You are helpful.", tools: [AIFunctionFactory.Create(MyTool)]);
Console.WriteLine(await agent.RunAsync("Hello!"));
```
Reference `Lamma.Net.Gpu` to run on an NVIDIA GPU automatically; for explicit control pass
`Offload = GpuLayerOffload.Factory(new GpuOffloadOptions { Layers = 20 })` (or `AutoAccelerator = false` for CPU only).

Document conversion with granite-docling: its DocTags markup is made of special tokens, so keep them in the output
(`ChatRequestOptions.RenderSpecialTokens = true`, M.E.AI `AdditionalProperties["render_special_tokens"] = true`, CLI
`lamma ask granite-docling-258M-BF16.gguf "Convert this page to docling." --mmproj mmproj-granite-docling-258M-f16.gguf --image page.png --special`).
A page is encoded in ~0.3 s on an RTX 3070 (~10 s on 8 CPU cores): the tiles' pixels go to the GPU, which does
everything from patch embedding to the projection.

Media encoders, RTX 3070 vs 8 CPU cores (`lamma encode model.gguf mmproj.gguf image|audio.wav` times both and compares them):

| Encoder | input | CPU | GPU |
|---|---|---|---|
| Gemma 4 vision | photo, 130 tokens | 1.3 s | 0.08 s |
| Gemma 4 vision | document page, 266 tokens | 3.5 s | 0.16 s |
| Qwen3.5 vision | photo, 300 tokens | 0.68 s | 0.03 s |
| Gemma 4 audio | 38 s clip, 963 tokens | 1.7 s | 0.25 s |
| granite-docling | document page, 17 tiles | 9.9 s | 0.30 s |

## Models from Hugging Face or a URL (model cache)
```csharp
// owner/repo[:QUANT] like llama.cpp's -hf (default Q4_K_M, then Q8_0); the matching mmproj comes along when the repo has one
using var chat = await LammaChatClient.LoadAsync(HuggingFaceModel.Parse("unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"));
using var fromUrl = LammaChatClient.Load(new Uri("https://example.com/models/my-model-Q4_K_M.gguf"));

// lower level: resolve / fetch, then load anything (InferenceEngine, embeddings, ...) from the cached path
var cache = new ModelCache(new ModelCacheOptions { Progress = new Progress<ModelDownloadProgress>(p => Console.Write($"\r{p.BytesDownloaded >> 20} MB")) });
ModelResolution files = await cache.GetAsync(new HuggingFaceModel("unsloth/Qwen3-0.6B-GGUF") { Quantization = "Q8_0" });
using var engine = InferenceEngine.Load(files.ModelPath);
```
The cache first resolves the model version (the repository commit; a URL's ETag) and uses the cached copy when it has
it; otherwise it downloads into the cache (resumable, SHA-256 verified, a stalled transfer is retried) and loads from there.
- Hugging Face files use the hub cache layout (`HF_HUB_CACHE`, default `~/.cache/huggingface/hub`), shared with
  huggingface_hub and llama.cpp: content lives in `blobs/{sha256}`, linked into `snapshots/{commit}/` (symlink, else hard
  link), so a commit that leaves the file unchanged is not downloaded again; copies made by other tools are reused after a
  hash check. `HF_TOKEN` (or the token saved by `hf auth login`) unlocks gated repos; `HF_HUB_OFFLINE=1` uses the cache only,
  which also happens automatically when the hub cannot be reached. A pinned commit (`Revision = "<sha>"`) that is cached
  needs no network at all.
- URL downloads go to `~/.cache/lamma.net/downloads`, one directory per server version, so a new
  version never overwrites a file that is still in use.
- `LAMMA_CACHE` moves both (hub layout in it, URL downloads in its `downloads` folder); `ModelCacheOptions` sets each explicitly.
- Split GGUFs (`-00001-of-0000N`) are not supported yet.
CLI: `lamma pull unsloth/LFM2.5-230M-GGUF:Q4_K_M [--dry-run]`, and `hf:owner/repo[:QUANT]` or an `https://` URL in place of a
model path for `ask` and `info`.

## Inspecting a GGUF without loading it
`ModelInfo` reads only the header of a GGUF file (no weights are loaded or touched) and describes the model:
provenance (`general.*`), hyper-parameters exactly as the engine derives them (context length, embedding width,
per-layer head counts and head dims, layer kinds, sliding windows, RoPE), the tokenizer (type, pre-tokenizer, vocabulary,
merges, special tokens, chat templates), the tensor table (names, shapes, types) with a quantization summary, and mmproj
encoder details.
```csharp
var info = ModelInfo.Read("Qwen3-0.6B-Q4_K_M.gguf");          // or Read(stream) / ReadAsync(stream): any readable stream
Console.WriteLine(info);                                       // multi-line summary
Console.WriteLine($"{info.Architecture} {info.Weights.FileTypeName} quantized={info.Weights.IsQuantized}");
Console.WriteLine($"ctx={info.ContextLength} embd={info.EmbeddingLength} heads={info.HeadCount}/{info.HeadCountKv} rope={info.Rope?.FreqBase}");
foreach (var t in info.Tensors.Take(5)) Console.WriteLine($"{t.Name} {t.Type} [{string.Join(", ", t.Dims)}]");
string? template = info.Tokenizer?.ChatTemplate;
var raw = info.Metadata.Get<float>("qwen3.rope.freq_base");    // any key, typed
```
`ModelInfoOptions.LoadVocabulary = false` keeps only the vocabulary size; `IsComplete` tells whether a file holds all
its tensor data (e.g. an interrupted download); a loaded `Model` exposes the same object as `model.Info`.
CLI: `lamma info model.gguf [--tensors] [--keys] [--vocab N]`.

## Speculative decoding with an MTP drafter
Gemma 4 ships multi-token-prediction drafters (`gemma4-assistant` GGUFs such as `mtp-gemma-4-E2B-it.gguf`, published next
to the model). With one, the engine drafts tokens with the small model and verifies them in one pass of the big one:
```csharp
using var engine = InferenceEngine.Load("gemma-4-E2B-it-Q4_K_M.gguf", new EngineOptions { DraftModelPath = "mtp-gemma-4-E2B-it.gguf" });
```
The drafter reads the target's KV cache and its LM-head input (no cache of its own), on the CPU or on the GPU next to the
target. On the CPU it drafts 2 tokens per step by default: Gemma-4-E2B generates 1.5x faster (20 -> 31 tokens/s on 8
cores, ~60% of drafts accepted). On the GPU it is off by default for now - verifying several tokens there still costs
nearly as much as generating them one by one - and can be switched on with `SpeculativeDraftTokens`. Without a drafter,
CPU engines draft from the prompt itself (prompt lookup). `lamma spec <model> <drafter> [tokens] [--gpu] [--drafts n] [--pmin p]`
compares speed, acceptance and output.

## Sharing the machine (resource limits)
A `ResourceLimiter` (LammaNet.Compute) bounds what Lamma.Net takes from the machine it shares with the application.
Every engine and media encoder follows `ResourceLimiter.Default` unless `EngineOptions.Resources` /
`ProjectorOptions.Resources` name another, and every limit can change while engines run (they follow within a step):
```csharp
var limits = ResourceLimiter.Default;
limits.MaxThreads = 4;            // compute threads per parallel step; the others sleep (no spinning)
limits.MaxUtilization = 0.5;      // engines compute at most half the time (pause after each step): CPU and GPU
limits.MaxConcurrentSteps = 1;    // several engines (agents on different models) take turns
limits.Adaptive = true;           // watch the machine's CPU load; fewer threads while other processes need the cores
limits.ThreadPriority = ThreadPriority.BelowNormal;   // default Normal (was AboveNormal)
limits.MaxDeviceMemoryBytes = 2L << 30;               // GPU offload plans with at most 2 GB (engines created later)
```
Environment defaults: `LAMMA_MAX_THREADS`, `LAMMA_MAX_UTILIZATION`, `LAMMA_MAX_STEPS`, `LAMMA_ADAPTIVE=1`,
`LAMMA_THREAD_PRIORITY`, `LAMMA_SPIN`, `LAMMA_MAX_DEVICE_MB`.

## Logging
Lamma.Net logs through `Microsoft.Extensions.Logging` (source-generated messages: no cost when a level is off).
```csharp
using var engine = InferenceEngine.Load("model.gguf", new EngineOptions { LoggerFactory = loggerFactory });
LammaLogging.LoggerFactory = loggerFactory;   // default for everything created without one (LammaNet.Diagnostics)
services.AddLogging().AddLammaChatClient("model.gguf");   // DI: the engine uses the container's ILoggerFactory
```
An engine passes its factory to the model load, the accelerator, projectors and chat clients; `ModelCacheOptions` and
`ProjectorOptions` take one too. Every caught exception and degraded path is logged with the exception and stable event
ids — 1xxx engine (started/stopped, request rejected 1011, a request's sampling failed 1018 (only that request fails),
a step failed 1020, KV cache full 1014, scheduler crash 1003), 2xxx accelerator (device and layer plan, CPU fallback
reasons), 3xxx model loading, 4xxx model cache (retries 4002, corrupt download 4004, offline fallback 4005), 5xxx
encoders falling back to the CPU, 6xxx prompt building and malformed tool calls, 7xxx Microsoft.Extensions.AI adapters
and realtime sessions. Requests carry an id (`GenerationHandle.RequestId`, engines `InferenceEngine.Id`) in every
message; prompt and output text is never logged and URL queries are stripped. A failing log provider never interrupts
inference. CLI: logs go to stderr, `LAMMA_LOG=debug|information|warning|error|none` (default warning).

CLI demos: `lamma info|pull|ask|agent|structured|embed|vision|stt|realtime|bench <model.gguf> ...` (see `src/Lamma.Net.Cli`).
GPU tools: `gpucheck` (GPU vs CPU logits), `enginecheck` (engine paths on the GPU), `bench-gpu`, `gpu-kstats`; `LAMMA_GPU=<layers>` for `bench`/`logits`, `LAMMA_GPU_PROFILE=1` for a per-kernel device profile.
