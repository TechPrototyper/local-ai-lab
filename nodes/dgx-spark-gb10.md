# The sm121 node: DGX Spark (GB10)

The lab's German-language model, retrieval and transcription box, and its
research testbed. When it is not running experiments, this is what it does
for a living.

## Hardware & OS

- **GB10** (Grace-Blackwell, sm121): 20-core Arm + Blackwell GPU,
  **128 GB unified memory** (~121 GB usable), ~273 GB/s memory bandwidth.
- The bandwidth is the defining constraint: a dense 31B decodes at ~8 tok/s
  per stream without speculation, so speed comes from MTP. The other
  constraint is that the main LLM shares its 128 GB with a whole fleet
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

## Serving stack (since 2026-10-11)

Stock vLLM nightly + FlashInfer 0.7.1 + [vllm#46558](https://github.com/vllm-project/vllm/pull/46558)
for Gemma-4 image spans on NVFP4 KV
([image and serve command](../containers/stock-nightly/)). The August
carry line ([`recipes/build-stack.md`](../recipes/build-stack.md)) is
history.

| Service | Model | Precision | Why this one |
|---|---|---|---|
| German LLM | **Gemma-4-31B-IT, PrismaQuant 6-bit, abliterated in format** ([HF](https://huggingface.co/TechPrototyper/Gemma4-31B-IT-PrismaQuant-6bit-abliterated-vllm)) + MTP with Google's [gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant), n=4 | mixed 6-bit weights (attention quantized too), NVFP4 KV | Best German prose we measured at this size, with vision and tool-calling. 26.8 GiB weights. ~17 tok/s per stream; prefill 1.2k tok/s at 8k, 271 tok/s at 240k. KV pinned with `--kv-cache-memory`: target 2 × 256k = 524,903 tokens, **currently 398,934** so that both transcribers fit next to it ([why](../notes/vllm-stock-nightly-migration-2026-10.md#what-runs-now-2026-10-11-evening)) |
| Embedder | Qwen3-Embedding 8B | FP8 | Big-model retrieval quality at half the bytes; embedding is prefill-only, so the bandwidth ceiling barely hurts |
| Embedder (second) | Harrier embedding model | — | Second vector space for a document pipeline |
| Reranker | Qwen3-Reranker 8B | FP8 | Served as a **real rerank API** (`POST /rerank`, server-side score template) — clients send query + documents, get scores |
| Reranker (light) | bge-reranker-v2-m3 | — | Cheap multilingual second stage when the big reranker is overkill |
| Audio | Whisper large-v3 + large-v3-turbo | — | Both needed: large for quality, turbo for fast passes |

### Memory discipline on unified memory

What the box taught us, condensed. Each point has cost at least one outage.
The details are in the [September](../notes/making-it-boring-sep-2026.md) and
[October](../notes/vllm-stock-nightly-migration-2026-10.md) notes.

- **Pin every pool.** `gpu-memory-utilization` sizes against whatever else
  happens to be resident. Use `--kv-cache-memory` for the LLM and
  `--kv-cache-memory-bytes` for the small services.
- **CUDA allocates from *free* memory, not *available*.** A lazily loaded kernel
  or workspace at the first image request fails with an internal cuBLAS or
  CUTLASS error when free memory is low. `vm.min_free_kbytes` at 3 GiB keeps
  that headroom real.
- **cgroup caps don't see GPU memory.** They limit host-side RAM only, so a
  container's cap says nothing about what it costs the box.
- **earlyoom must be told to ignore swap** (`-s 100,100`). With low
  swappiness the swap never fills, and the default gate never opens.
- **No Docker autostart for the main LLM.** A watchdog starts it after boot or
  crash: staggered, memory-gated, at most three times an hour. Experiment
  containers never get a restart policy.
- **One start script per service, and a drift checker** that compares the
  running fleet against intent (flags, pools, earlyoom mode, restart
  policies, reserve).
- **Never two heavy loads at once.** A build next to a stress test, or two
  agents each "just quickly" using the box, is how it wedged.

## What it actually does (production use)

- **German prose** — minutes, letters, reading and writing documents; the
  gateway routes German-writing jobs here by model, while agents and code go
  to the RTX 5090 ([`nodes/rtx-5090.md`](rtx-5090.md)). Two 256k sessions
  can share the pool, and the prefix cache keeps a second long session fast.
- **Retrieval** — the embedder feeds a document management system and
  general semantic search; embedder + reranker together do two-stage
  retrieval for legal research (the reranker sees ~20k requests from the
  cluster pipelines in its current log window).
- **Audio** — meeting transcription and summarization (minutes from
  recordings), and transcription/translation of incoming calls so
  agents can process them as text.

## Research on the same box

The same node hosts the experiment line documented across this repo
(NVFP4-KV calibration, quantization runtime, DFlash2 speculation,
in-format abliteration, the stock-nightly migration).
Experiments run in separate containers against the same model files,
with scripted rollback to the production fleet — the notes tagged GB10
in the [findings table](../README.md#findings-so-far) all came from here.
