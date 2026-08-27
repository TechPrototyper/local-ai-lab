# The sm121 node: DGX Spark (GB10)

The lab's capacity tier and research testbed in one box. When it is not
running experiments, this is what it does for a living.

## Hardware & OS

- **GB10** (Grace-Blackwell, sm121): 20-core Arm + Blackwell GPU,
  **128 GB unified memory** (~121 GB usable), ~273 GB/s memory bandwidth.
- The bandwidth is the defining constraint: decode speed for a dense 27B
  lives around 10 tok/s per stream, so the box is configured for **long
  context and many parallel sessions**, not single-stream speed
  ([framing](../README.md#the-framing)).
- DGX OS (Ubuntu-based), desktop enabled deliberately — it doubles as a
  workstation. Services run as a plain **Docker fleet**, not Kubernetes:
  one box, explicit memory budgeting, no scheduler that could move things.

### The driver story (so nobody repeats it)

NVIDIA's DGX OS OTA channel ships the **580 series** (currently
580.173.02, which brought OOM-handling fixes). The **595 series** exists
in Ubuntu's channels, and we tried it — without success; community reports
show sideloaded 590/595 drivers have bricked Sparks outright. Conclusion,
learned cheaply where others learned it expensively: **on this device,
stay on the OTA path.** The 595 features will arrive when DGX OS ships
them.

## Serving stack

Custom vLLM build for sm12x (consumer/embedded Blackwell), carrying the
NVFP4-KV enablement line — FlashInfer kernels
([flashinfer#3684](https://github.com/flashinfer-ai/flashinfer/pull/3684),
merged) plus [vllm#46329](https://github.com/vllm-project/vllm/pull/46329)
and local patches; see [`recipes/build-stack.md`](../recipes/build-stack.md).
Weights are quantized with the PrismaQuant pipeline — **mixed precision**
(AQUA: KL-Fisher allocated, NVFP4 + FP8, ~5.5 bpp), not uniform NVFP4;
the KV cache is served uncalibrated
([why](../README.md#findings-so-far)).

| Service | Model | Precision | Why this one |
|---|---|---|---|
| Chat/agent LLM | Qwen3.8-27B (AQUA quant) | mixed-precision weights (NVFP4+FP8, ~5.5 bpp) + fp8 KV + **DFlash2 speculation** (draft length 7) | Best dense quality that fits; since 2026-08-21 with block-diffusion speculation in production — ~4× single-stream on reasoning traffic, up to ~227 tok/s aggregate, quality verdict-equal. fp8 KV was DFlash2's price (~443k tokens at 20 GiB vs ~966k at NVFP4); since 2026-08-22 the drafter is fp8-quantized too — its −1.6 GB went back into the pool: **21.6 GiB = 478k tokens**, single-stream ~45 tok/s; prefix caching on. **Since 2026-08-27 the fp8-KV price is optional:** DFlash2 + NVFP4-KV served end-to-end on this box for the first time (paired battery: prose +72%, count-200 5.9× vs no-spec; [`notes/dflash2-nvfp4-sm120-spec-serves.md`](../notes/dflash2-nvfp4-sm120-spec-serves.md)) — a production cutover (≈2× KV pool) waits on a verdict-tier quality gate |
| Embedder | Qwen3-Embedding 8B | FP8 | Big-model retrieval quality at half the bytes; embedding is prefill-only, so the bandwidth ceiling barely hurts. **Currently paused** — see note ¹ |
| Reranker | Qwen3-Reranker | FP8 | Served as a **real rerank API** (`POST /rerank`, server-side score template) — clients send query + documents, get scores, no prompt assembly on the client |
| Reranker (light) | bge-reranker-v2-m3 | — | Cheap multilingual second stage when the big reranker is overkill |
| Audio | Whisper large-v3 + turbo | — | Transcription tier: large for quality, turbo for fast/medium-quality passes |

¹ The quality embedder (Qwen3-Embedding 8B) is paused for memory
budgeting while the LLM's KV pool and the small services share the box;
a lighter FP8 embedding fallback carries retrieval in the meantime.

Configuration is measurement-derived; the recipes are in
[`recipes/dgx-spark-sm121.md`](../recipes/dgx-spark-sm121.md). The
unified-memory gotchas that shaped them: explicit
`--kv-cache-memory-bytes` (profiling subtracts resident neighbors),
capped CUDA-graph capture sizes for the small services (default costs
~5 GiB *per service*), small-services-first start order, ≥10 GB OS
reserve in steady state, and since 2026-08-20 a **post-boot warmup
request** ([`notes/gdn-first-step-crash.md`](../notes/gdn-first-step-crash.md)).

## What it actually does (production use)

- **Agentic AI** — the LLM serves coding/ops agents through several
  harnesses, including a multi-agent, multi-user setup on the Kubernetes
  side (per-user profiles; that part is cluster functionality — the
  Hermes agent gateway — this box just serves the tokens). Long contexts
  plus prefix caching are what make multi-turn agents affordable here.
- **Overflow tier** — the gateway in front of the lab's RTX 5090 node
  routes sessions here when the latency tier is saturated
  ([`nodes/rtx-5090.md`](rtx-5090.md)).
- **Retrieval** — the embedder feeds a document management system and
  general semantic search; embedder + reranker together do two-stage
  retrieval for legal research (the reranker sees ~20k requests from the
  cluster pipelines in its current log window).
- **Audio** — meeting transcription and summarization (minutes from
  recordings), and transcription/translation of incoming calls so
  agents can process them as text.

## Research on the same box

The same node hosts the experiment line documented across this repo
(NVFP4-KV calibration, quantization runtime, DFlash2 speculation).
Experiments run in separate containers against the same model files,
with scripted rollback to the production fleet — the notes tagged GB10
in the [findings table](../README.md#findings-so-far) all came from here.
