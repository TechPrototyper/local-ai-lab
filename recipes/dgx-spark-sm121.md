# DGX Spark / GB10 (sm_121) — quality → speed → memory

Same 27B as the RTX recipe — opposite dials, and since 2026-08-21 a
different KV dtype: fp8, the price of DFlash2 speculation (the RTX keeps
NVFP4 KV and no speculation — memory-bound boxes trade the other way).
Here memory is abundant (128 GB unified), so it is spent on long context
and many parallel sessions ([framing](../README.md#the-framing)). If you
have a GB10, this is the path: get the stack, pull the model, serve, warm
up, verify.

## 1. What you need

- **DGX Spark / GB10** (sm_121, 128 GB unified, ~273 GB/s). The bandwidth
  is the defining constraint — dense-27B decode lives around 10 tok/s per
  stream — so the box is configured for context and concurrency, not
  single-stream speed.
- **Stay on the DGX OS OTA driver path (580 series, 580.173.02).** The 595
  series exists in Ubuntu's channels; sideloaded 590/595 have bricked
  Sparks outright. Learned cheaply where others learned it expensively
  ([node profile](../nodes/dgx-spark-gb10.md#the-driver-story-so-nobody-repeats-it)).

## 2. Get the stack

Custom sm12x vLLM build carrying the consumer-Blackwell NVFP4-KV line
(vllm#46329 + flashinfer#3684). Image tag `vllm-sm121:f4c27c0da`, the
8-commit stack, the sm121 FlashInfer pin (**0.6.15**), the source-mount
contract, and the local build patch are documented in
[`build-stack.md`](build-stack.md).

> Containers: see [vllm-sm12x](https://github.com/TechPrototyper/vllm-sm12x)
> — Dockerfiles, entrypoints (first-boot warmup built in) and
> docker/podman quickstarts are up; prebuilt images follow after the
> first promoted build. Until then, build from `build-stack.md`.

## 3. Get the model

```
rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm
```

Rob Tand's ([@RobTand](https://github.com/RobTand)) PrismaQuant **AQUA**
mixed-precision export of Qwen3.8-27B: NVFP4 weights + FP8 attention,
~5.5 bpp, `compressed-tensors`, Apache-2.0. **~24 GB on disk** (5
safetensors shards, ~22 GiB); vLLM downloads it on the first
`vllm serve <hf-id>`, not gated. This lab reproduced the AURA line at
full-split parity on this box (n=1319, McNemar p=0.883—
[note](../notes/qwen36-aura-head-to-head.md)); the AQUA export is the
current production checkpoint.

## 4. Serve

```
vllm serve rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm \
  --served-model-name qwen3.8-27b \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"dflash","model":"<dflash2-drafter-path>","num_speculative_tokens":7}' \
  --enable-prefix-caching \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.44 \
  --kv-cache-memory-bytes 21474836480 \
  --enable-chunked-prefill --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Container-level cap (`docker run`): `--memory=48g --memory-swap=48g` —
the DFlash2 config needs more than the old 32g: spec decode widens the
CUDA-graph capture palette (max size 64 → 512) and the transient compile
peak (~20 GiB observed) plus drafter weights killed the engine core
*silently* under tighter caps (a SIGKILL leaves no Python traceback —
if the core dies with zero ERROR lines, suspect the cgroup first).

Key decisions, each measured:

- **Size the KV pool explicitly with `--kv-cache-memory-bytes`, not
  `--gpu-memory-utilization`.** On unified memory, `gpu-memory-utilization`
  sizes the pool against *every other resident process*, so identical
  values yield wildly different pools depending on service start order
  (README finding 2026-08-06). At 20 GiB explicit (21474836480 bytes) the
  NVFP4 KV pool is **~966k tokens ≈ 3.7 × 262k-token sessions in parallel**
  ([node profile](../nodes/dgx-spark-gb10.md#serving-stack)).
- **`gpu-memory-utilization 0.44` is a startup check, not the pool** — that
  fraction of total memory must be free at launch or vLLM refuses to start.
  Size it to actual need (weights + pool + activations).
- **Serve uncalibrated (k/v scale = 1.0)** — same rationale as the RTX
  recipe: baked per-tensor amax KV scales cost accuracy at 4 bit (README
  finding 2026-08-01). The AQUA checkpoint carries no baked KV scales.
- **Prefix caching on** — agents re-send the full context every turn.
- **Parsers:** `--reasoning-parser qwen3` +
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- **Start order:** bring the small services (embedder/reranker/audio) up
  *first* and cap their CUDA-graph capture sizes — default capture costs
  ~5 GiB *per service* — so the LLM's explicit pool math holds
  ([node profile](../nodes/dgx-spark-gb10.md#serving-stack)).
- **Speculative decoding is DFlash2, draft length 7** (adopted
  2026-08-21, replacing "MTP off" as the box's speculation stance): every
  gate passed at verdict level — GSM8K n=1319 paired p=0.122, tool-calling
  byte-identical to the non-speculative canon, faster at every measured
  concurrency ≤24 with n=7 (single ~42 tok/s on reasoning-heavy prompts,
  ~20 on free prose — acceptance is content-dependent; aggregate up to
  ~227 tok/s) ([verdict](../notes/dflash2-full-split-verdict.md) ·
  [draft-length map](../notes/dflash2-draft-length-map.md)). **Never raise
  the draft length for agent traffic** — n=15 plateaus and inverts against
  baseline around c≈20; n=7 never inverted in the measured range.
- **The fp8 KV cache is the price of DFlash2** — the drafter's non-causal
  attention cannot read NVFP4 KV on sm12x. Quality cost: none detected at
  verdict level. Capacity cost: the 20 GiB pool drops from ~966k to
  **~443k tokens ≈ 1.7 × 262k sessions** (drafter layers share the pool).
  The old MTP config stays documented for the record
  ([note](../notes/mtp-tool-calling.md)) — its tool-calling failure was
  parser-path-specific, not "speculation" (README finding 2026-08-20).
- **Requires the DFlash2 source line** until vLLM
  [#52816](https://github.com/vllm-project/vllm/pull/52816)/[#52883](https://github.com/vllm-project/vllm/pull/52883)
  merge: the sm12x build plus 10 cherry-picked commits — branch
  [`dflash2-sm121`](https://github.com/TechPrototyper/vllm/tree/dflash2-sm121)
  on this lab's vLLM fork.

## 5. First boot discipline

The 27B is a hybrid linear-attention model, and **≥2 concurrent prefills in
the very first engine step after boot** can crash the engine
(`cudaErrorNotPermitted` in the GDN state write) — and a restart policy
turns that into a crash-loop hazard on any reboot into live traffic
([note](../notes/gdn-first-step-crash.md)).

Every boot path therefore ends the same way: **wait for the health
endpoint, send exactly one small completion request, then open the door.**
Cost: one request; it closes the first-step window deterministically.

## 6. Verify

Smoke test once healthy and warmed:

```
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role":"user","content":"What is 84 * 3 / 2?"}],
  "temperature": 0
}'
```

Expect **126**. Screening quality on this box: **GSM8K n=250 = 96.8%**
(baseline NVFP4-KV, paired gate 2026-08-20;
[note](../notes/dflash2-n250-gate-2026-08-20.md)); full-split parity vs the
reference confirmed at 95.00% (n=1319).

Confirm the KV pool at startup in vLLM's log: `GPU KV cache size: … tokens`
should report on the order of **443k tokens** (fp8 + drafter; the old
NVFP4-only config reported ~966k). Speculation health: after warm traffic,
`grep SpecDecoding` should show mean acceptance length ≈5 on
reasoning-heavy prompts. If the pool is far smaller, KV is not actually
NVFP4 or a resident neighbor ate the pool (see the `--kv-cache-memory-bytes`
note above).

Single-stream decode is ~10 tok/s (bandwidth-bound). The historical
aggregate figures (~20 tok/s per session, 135 tok/s at 8 concurrent
sessions, TTFT 0.3–0.7 s) were captured 2026-08-06 under the
since-corrected MTP-on config and have **not** been re-benchmarked without
MTP — treat as historical. Prefill is unaffected by MTP and still holds:
~1,200 tok/s to 32k context, ~510 tok/s at 229k.

## 7. Known limits

- **DFlash2 speculative decoding is an adoption candidate here, not the
  default.** The paired n=250 quality gate **passed** (equal quality,
  ~2.5–3× single-stream), and the controlled batch sweep has it winning at
  **every tier through c=8** (48.3→134.9 agg tok/s vs 10.7→70.9 baseline).
  Beyond c=8 / mixed load, measure first — first light saw the gain invert
  there. It requires fp8 KV (affordable on this box: ~966k → ~550k tokens),
  and the n=1319 full-split verdict is still pending before verdict-level
  language ([gate](../notes/dflash2-n250-gate-2026-08-20.md) ·
  [batch sweep](../notes/dflash2-batch-sweep-and-skiplayers.md)).
- The decode-throughput aggregate numbers above are historical (MTP-on) and
  await a re-benchmark on the current config.
