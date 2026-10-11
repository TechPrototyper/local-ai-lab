# Making it boring: GridBook on the 5090, Gemma-4 on the GB10 (September 2026)

*Covers 2026-09-04 → 2026-10-03. The August chapter ended with measurements
("the triad composes", "the 13 GB line holds at verdict tier"). September was
about the other half of the job: turning those measurements into two serves
that stay up for weeks, under real agent traffic, without anyone watching.*

## Where we stood on 2026-08-31

- **RTX 5090:** Qwen3.8-27B as the 13 GB GridBook artifact
  ([rdtand](https://huggingface.co/rdtand)'s
  [Qwen3.8-27B-PrismaAQUA-gridbook-13GB](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm)),
  NVFP4 KV, native MTP@2 — [cutover note](rtx-gridbook-mtp-cutover-2026-08-31.md).
- **DGX Spark (GB10):** still the Qwen3.8 AQUA + DFlash2 stack from August;
  the GridBook evaluation there came out ~35 % slower and was dropped
  ([note](spark-gridbook-eval-2026-08-31.md)).

## The RTX: from "boots" to "serves eight agents"

| Date | What happened | Lesson |
|---|---|---|
| 09-04 | GridBook + MTP start arguments moved into Git (they had only lived in an imperative deployment). `gpu-memory-utilization 0.97` with MTP → CUDA OOM and a crash loop; back to 0.92. | A config that only exists in a running object does not exist. |
| 09-10 | **Phase 1: MTP off**, `max-num-seqs` 5, util 0.97 → **778,645-token KV pool** (the 898k "config M" from August is the no-spec, no-vision upper bound). Draft weights and draft activations come out of the KV budget; without MTP, greedy parity and tool-calling under load were better. Speed is secondary to quality on this box. | On a 32 GB card speculation competes with context, not just with compute. |
| 09-15 | **Vision is mandatory**, not optional. The GridBook checkpoint ships without a vision tower (HTTP 400 on image requests). util 0.98 for the extra headroom → OOM with 50 MiB free; reverted. | |
| 09-16 | **`gridbook-13gb-vision` built**: an index file plus symlinks — the language model stays the GridBook artifact (1,363 tensors), the 333 `model.visual.*` tensors (0.86 GiB) come from the AQUA export and stay BF16 (they are on the quantization ignore list, which also keeps them off the FP4-GEMM path). Two fixes were needed on the way: a `lm_head` namespace fix (regex on the raw serving prefix; `lm_head` stays FP8) and a shard repair. About 7 h of crash loop (82 restarts) before it came up. Acceptance: **KV 760,477 tokens (2.90× at 262k)**, a 960×686 image in 7.0 s, German probe 9/12. | Composing a checkpoint from two exports is cheap on disk and expensive in naming. |
| 09-22 | MTP stays blocked on this line regardless: NVFP4 KV reports `UNIFORM_SINGLE_TOKEN_DECODE`, which the speculative verify capture rejects — and the vision-tower state does not load the MTP heads anyway. | |
| 10-08/09 | `max-num-seqs` 8 for eight parallel agents (contexts mostly < 100k; larger simultaneous sessions preempt instead of failing). | |

## The GB10: a German model, and the memory discipline it forced

The GB10's role changed in September: it became the box for **German prose**
(minutes, letters, reading and writing documents), next to a fleet of
embedders, rerankers and two Whisper transcribers — all on 128 GB of
*unified* memory.

| Date | What happened | Number |
|---|---|---|
| 09-10 | Gemma-4-31B-IT (nvidia/modelopt NVFP4) arrives as the German model. | |
| 09-11 | MTP with Google's official drafter [`gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant), n=4, NVFP4 KV: **6.9 → 15.7 tok/s** prose, 27.6 structured. DFlash for Gemma rejected (acceptance only 14–23 % against the NVFP4 target; n=8 gave 5.4 tok/s structured). 262k context, tool-calling on. | 2.3× prose |
| 09-13 | **Fleet incident** (below). | |
| 09-13 | **KV pinning:** the pool is pinned to *exactly* 2 sessions × 256Ki = 524,288 tokens via `--kv-cache-memory 20132659200` (18.75 GiB) → 524,903 tokens measured. `gpu-memory-utilization` on unified memory sizes the pool against whatever else happens to be resident, so it drifts with start order ([Aug finding](../README.md#findings-so-far)). | 524,903 |
| 09-13 | **Weights cutover** from the modelopt export to [rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm](https://huggingface.co/rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm). The modelopt artifact was 64 % BF16 — its quantization config puts every `self_attn` module on the ignore list. PrismaQuant 6-bit quantizes attention: **26.8 vs 31.85 GiB** loaded, same KV pool, decode 18.2 vs 15.7 tok/s, determinism 5/5 vs 4/5, German 9/12 = 9/12, vision 3/3, tool suite pass. | −5 GiB, +16 % decode |
| 09-14 | A 5.5-bit sibling disqualified: tool suite failed, determinism 1/5. | |
| 09-16 | **A single 960×654 painting killed the engine** in the FP4 GEMM path (`[FP4 gemm Runner StreamK] Failed to initialize cutlass FP4 gemm on sm120/sm121`), ~6 min outage. The vision gate grew three guards (local downscale to 640 px, refusal to run against production ports, abort on any 5xx). Root cause stayed open — see the [October note](vllm-stock-nightly-migration-2026-10.md), where it finally got one. | |
| 09-29 | Gemma *without* the drafter produced garbage on the fork's sm121 build (the NVFP4 single-token decode path was broken there). Production depended on the drafter, which in turn means greedy output is not bit-exact. A standing "health probe" was added. | |

### The fleet incident (2026-09-13) and what it left behind

Two failures at once. The Qwen3 embedder and reranker crash-looped on a
**cgroup OOM while loading** (7.36 of an 8 GiB cap) — invisible in
`docker inspect` (`OOMKilled=false`; only the kernel log shows it, because
only the engine *child* dies). The same bug had happened in August and had
only been fixed at runtime, never in the start script. In parallel, Gemma had
been started by hand — without MTP, without the tool and reasoning parsers —
and served 6.9 instead of 15.7 tok/s plus HTTP 400 on `tool_choice=auto`.

What came out of it, and is still in force:

- **One start script per service, `MODE=prod` as default.** No hand starts.
- **A drift checker** (`verify-spark-stack`) that compares the running
  containers against the intended flags, caps, KV pool and memory reserve.
- **Caps sized to the load peak**, not to the steady state.
- **The stagger doctrine** (since August): the main LLM first and
  health-gated, then the fleet one by one, each waiting for physically free
  memory — CUDA allocates from *free* memory and does not trigger page-cache
  reclaim, so a box with plenty of "available" memory can still OOM a loader.
- **earlyoom** as the last line, swappiness 10, `oom_score_adj` for the
  daemons ([August hardening](gb10-oom-hardening.md)). It did fire on our own
  night jobs (a 52 GB abliteration run, a 50 GiB quantization pre-load) —
  exactly what it is for. Read the [October note](vllm-stock-nightly-migration-2026-10.md)
  for why it then *didn't* fire when it mattered most.

## What "boring" meant in numbers by early October

| Box | Model | KV pool | Speed | Gates |
|---|---|---|---|---|
| RTX 5090 | Qwen3.8-27B GridBook-13GB + BF16 vision tower | 760,477 tokens, NVFP4, 262k, 8 seqs | ~58 tok/s single, ~152 at 4 streams | vision, tools 5/5, needle, determinism 5/5 |
| GB10 | Gemma-4-31B PrismaQuant 6-bit + MTP n=4 | 524,903 tokens, NVFP4, 2 × 256k | ~18 tok/s prose | German 9/12, vision 3/3, tools, needle 3/3, determinism 5/5 |

Both were then abliterated in place — [next note](abliteration-in-format-2026-10.md).
