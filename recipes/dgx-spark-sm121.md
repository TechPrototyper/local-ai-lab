# DGX Spark / GB10 (sm_121) — quality → speed → memory

Same 27B as the RTX recipe — opposite dials, and since 2026-08-21 a
different KV dtype: fp8, historically the price of DFlash2 speculation.
(Since 2026-08-27 that price is optional — spec now serves over NVFP4 KV
on this box too ([note](../notes/dflash2-nvfp4-sm120-spec-serves.md));
this recipe stays on the verdict-validated fp8 config until the NVFP4
variant passes its own verdict-tier gate. The RTX keeps NVFP4 KV and no
speculation — its agent traffic lives on prefix caching, see below.)
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
> **Prebuilt, self-contained image (2026-08-28):**
> `ghcr.io/techprototyper/vllm-sm12x:sm121-dflash2-pc50897-dd02ed4d` —
> DFlash2 + FlashInfer + warm caches baked, **includes the vllm#50897
> prefix-cache-under-speculation fix** (90% replay reuse in production
> here). Run it exactly like the production serve below, minus every
> `-v .../vllm-src` and `-v .../fi` mount.
>
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
  --speculative-config '{"method":"dflash","model":"<dflash2-drafter-fp8-path>","quantization":"compressed-tensors","num_speculative_tokens":7}' \
  --enable-prefix-caching \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.44 \
  --kv-cache-memory-bytes 23192823808 \
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
- **Prefix caching on — with an honest caveat.** Agents re-send the full
  context every turn, which is what the cache is for. But under
  speculation on hybrid-GDN models, prefix-cache hits currently appear to
  drop to zero (mechanism tracked upstream as
  [vllm#52244](https://github.com/vllm-project/vllm/pull/52244);
  live-confirmed on the lab's sm120 box with the same model family and
  drafter). The flag stays on deliberately — it is harmless, and cache
  benefits return automatically when the upstream fix lands.
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
- **The fp8 KV cache *was* the price of DFlash2** — the drafter's
  non-causal verify could not read NVFP4 KV on sm12x. That gap is closed:
  since 2026-08-27 DFlash2 + NVFP4-KV serves end-to-end on this box
  (paired battery: prose +72%, count-200 5.9× vs no-spec; filed upstream
  as [#53977](https://github.com/vllm-project/vllm/pull/53977)/[#53978](https://github.com/vllm-project/vllm/pull/53978)/[#53979](https://github.com/vllm-project/vllm/pull/53979);
  [note](../notes/dflash2-nvfp4-sm120-spec-serves.md)). This recipe stays
  on fp8 until the NVFP4 variant passes a verdict-tier quality gate —
  the prize is ≈2× pool (~966k vs ~443k tokens at 20 GiB). Current
  production pool: **21.6 GiB fp8 = 478,334 tokens** — funded by
  quantizing the drafter (below). Quality cost of fp8: none detected at
  verdict level. The old MTP config stays documented for the record
  ([note](../notes/mtp-tool-calling.md)) — its tool-calling failure was
  parser-path-specific, not "speculation" (README finding 2026-08-20).
- **Requires the DFlash2 source line, now mostly upstream:**
  [#52816](https://github.com/vllm-project/vllm/pull/52816) (the DFlash2
  core) **merged 2026-08-24**; still open are
  [#52883](https://github.com/vllm-project/vllm/pull/52883) (LM-head
  loading) and the lab's warmup fixes
  [#53977](https://github.com/vllm-project/vllm/pull/53977)/[#53978](https://github.com/vllm-project/vllm/pull/53978).
  Until those land, the working line is branch
  [`dflash2-sm121`](https://github.com/TechPrototyper/vllm/tree/dflash2-sm121)
  on this lab's vLLM fork (or the
  [portable container](../notes/dflash2-gb10-portable-container.md)).

- **The drafter itself is fp8-quantized** (since 2026-08-22): all decoder
  projections, **per-channel weights** + dynamic activations,
  compressed-tensors, declared via `"quantization":
  "compressed-tensors"` in the speculative config. Acceptance-neutral
  (5.05–5.60 vs 5.11 bf16), slightly *faster* (bandwidth-bound drafter),
  and −1.6 GB — which this recipe hands to the KV pool. Two hard
  requirements: (1) the fused-scale loading fixes from
  [vllm#53122](https://github.com/vllm-project/vllm/pull/53122) (carried
  on the lab's [`dflash2-sm121`](https://github.com/TechPrototyper/vllm/tree/dflash2-sm121)
  branch), and (2) **per-channel scales** — per-tensor checkpoints fail
  in the context-KV dequant (scalar-per-shard scale cannot map onto the
  K/V slices). The path here: graph-free tensor-level RTN, no
  calibration — drafter quality can only affect speed, never output
  ([`notes/dflash2-drafter-fp8-quant.md`](../notes/dflash2-drafter-fp8-quant.md)).

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
should report on the order of **478k tokens** (21.6 GiB fp8, fp8 drafter;
the old NVFP4-only config reported ~966k at 20 GiB). Speculation health: after warm traffic,
`grep SpecDecoding` should show mean acceptance length ≈5 on
reasoning-heavy prompts. If the pool is far smaller, KV is not actually
the dtype you set or a resident neighbor ate the pool (see the
`--kv-cache-memory-bytes` note above).

Single-stream decode is ~10 tok/s (bandwidth-bound). The historical
aggregate figures (~20 tok/s per session, 135 tok/s at 8 concurrent
sessions, TTFT 0.3–0.7 s) were captured 2026-08-06 under the
since-corrected MTP-on config and have **not** been re-benchmarked without
MTP — treat as historical. Prefill is unaffected by MTP and still holds:
~1,200 tok/s to 32k context, ~510 tok/s at 229k.

## 7. Known limits

- **DFlash2 beyond the measured range: measure first.** The full-split
  verdict passed (n=1319 paired, p=0.122 — hence adoption, see §4), and
  draft length n=7 never inverted against baseline in the measured range
  (≤c=24). Beyond that, and under mixed prefill-heavy load, measure before
  trusting the gain — early n=15 runs inverted around c≈20
  ([verdict](../notes/dflash2-full-split-verdict.md) ·
  [draft-length map](../notes/dflash2-draft-length-map.md)).
- **NVFP4-KV under speculation is validated but not yet promoted** —
  the ≈2× pool upgrade waits on its own verdict-tier gate
  ([note](../notes/dflash2-nvfp4-sm120-spec-serves.md)).
- The decode-throughput aggregate numbers above are historical (MTP-on) and
  await a re-benchmark on the current config.
