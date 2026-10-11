# RTX Cutover: MTP-on-GridBook (drafter-free) — Build, Serving & Tests

**As of 2026-08-31 ~07:45 UTC. Prod cutover completed and verified.**
The RTX 5090 (sm120) now serves prod with **MTP-on-GridBook** — drafter-free,
~624k KV pool. This is the complete documentation of the build, serving configuration and
all tests that led to the switch.

---

## 1. Decision & Result (summary)

Switched: RTX from research → **prod-primary** with `gridbook-13gb-mtpfix` + MTP@2.
Spark remains the overflow fallback (still AQUA + DFlash2). Reason: MTP-on-GridBook is
quality-equivalent to no-spec (McNemar p=0.83), passes the full tool-calling
battery, is drafter-free (no external drafter model → more VRAM/KV) and delivers
a large context pool (624k tokens).

Decision register (as of cutover):
- **RTX base:** GridBook-13GB (vs AQUA-23.6GB) — smaller → more KV. **switched.**
- **RTX spec:** MTP (DFlash2 does not load at full max-len). **switched.**
- **Sampling/thinking:** thinking-ON stays (sweep: C1/thinking-on best;
  `-tools`/thinking-off counterproductive). **no change.**
- **Spark:** AQUA + DFlash2 stays (cross-arch: GridBook-on-Spark needs build match/
  prefix strip; separately in progress for Whisper headroom).

---

## 2. Build — how the vLLM is patched together

**Serving image:** `<registry>/inference/vllm-nvfp4:sm120-pc50897-gridbook-088`
- vLLM base: sm120-NVFP4 line (V1 engine, `v0.1.dev1+g2cf8b8ae0`).
- **`pc50897`** = vllm#50897 pre-integrated: lookahead-aware prefix-cache hashing
  for EAGLE/spec drafters (restores prefix-cache hits under speculation; addresses
  vllm#52244). Ported to sm120 + validated.
- **`gridbook-088`** = GridBook plugin 0.8.8 (out-of-tree vLLM plugin via
  `vllm.general_plugins` entry point) — provides `quantization=gridbook`
  (NVFP4-CB / FP8-CB product-codebook dequant) built into the image.
- **The "vLLM upstream package" seam** (our contribution, cherry-picks): #53977
  (VocabParallelEmbedding tp==1 OOB mask), #53978 (DFlash2 survives spec warmup with
  unfilled draft buffers), #53979 (NVFP4-KV: non-causal FA2 prefill wrapper for
  the DFlash family + SWA guard). These harden the non-causal NVFP4 seam; not all
  load-bearing for MTP-on-GridBook, but part of the sm12x serving line.
- FlashInfer backend (`VLLM_ATTENTION_BACKEND=FLASHINFER`), warm JIT/autotune cache.

**The checkpoint artifact `gridbook-13gb-mtpfix`** (the actual model fix):
- Base: rdtand's GridBook-13GB (`Qwen3.8-27B-PrismaAQUA-gridbook-13GB`, NVFP4-CB,
  ~13 GB, quant_method `gridbook`). Keeps the 15 native MTP head tensors.
- **Problem:** GridBook (from 0.8.7) quantizes the shared body embedding
  `model.language_model.embed_tokens` (weight_packed/scale/global_scale). vLLM's
  MTP predictor shares the target embedding with the draft → the packed embedding
  breaks the shared draft (`weight_global_scale` load error).
- **Fix (bit-exact, self-serve, NO Rob re-export):** drop the 3 packed embed tensors
  and insert the **BF16 `model.language_model.embed_tokens.weight` from AQUA**
  (`qwen3.8-27b-prismaaqua55`, identical Qwen3.8 base) verbatim;
  `quant_config.json`: remove the `quantized_embedding` key + put the unit in `ignore`.
  Tensor count 1365→1363. Script: `s24/surgical_embed_swap.py`.
- **Why bit-exact is correct:** MTP speculation is lossless; the byte-identity
  gate was unsuitable (FP non-associativity batched-verify vs. incremental-decode;
  even native AQUA-MTP "fails" it) → correctness proven via the n=1319 verdict,
  not via byte equality.

---

## 3. Serving Configuration (live)

**k8s deployment `vllm-qwen38-mtp-gridbook`** (namespace inference, node <rtx-node>/RTX 5090),
behind service `llm-active` (selector match), litellm routes `qwen3.8-27b` → llm-active:8000.

vLLM serve (core flags):
```
--model /cache/models/gridbook-13gb-mtpfix
--served-model-name qwen3.8-27b
--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":2}
--kv-cache-dtype nvfp4
--gpu-memory-utilization 0.90          # STABLE — 0.97 crashed (see §5)
--max-model-len 262144
--reasoning-parser qwen3
--tool-call-parser qwen3_coder  --enable-auto-tool-choice
--async-scheduling  --enable-prefix-caching  --enable-chunked-prefill
```
Engine log confirms: `quantization=gridbook`, `speculative_config method='mtp'
num_spec_tokens=2`, `Detected MTP model. Sharing target embedding + lm_head weights`,
`served_model_name=qwen3.8-27b`.

**Measured KV pool:** **624,152 tokens** (2.38× concurrency @ 262,144/request),
exclusive GPU. Important: with a GPU split (s24 simultaneously) only 534k → the RTX must belong to
serving exclusively (no split), otherwise vLLM re-profiles against less memory.

**Routing:** litellm semaphores SPARK_DOWN=0 / RTX_DOWN=0 → RTX primary, Spark overflow
(>3 concurrent local sessions → Spark). Fallback on RTX failure: Spark (AQUA+DFlash2).

---

## 4. Tests & Results

### 4.1 Quality (the release gates)
| Test | Result |
|---|---|
| **n=1319 McNemar** (MTP@2 vs no-spec, GridBook, greedy, paired) | Acc_MTP 0.9704 · Acc_nospec 0.9719 · **b=10 c=12 p=0.8318** → statistically indistinguishable |
| **Tool-calling battery** (on mtpfix-MTP, util 0.90) | **Tools 6/6** · GSM8K-250 **0.9840** (0 errors) · **Needle 12k/24k/64k 9/9** · **Determinism 5/5** |
| Live spot check (after cutover) | E2E 200, tool call `get_weather{"city":"Paris"}` clean |

### 4.2 Speed / KV (context)
| Metric | Value |
|---|---|
| KV pool (stable, util 0.90, exclusive) | **624,152 tokens** (2.38× @ 262k) |
| MTP-Turbo (measured on AQUA, transfers) | Prose +49% @ 54.3% acceptance; count200 +112%; JSON +72%; retrieval +84% |
| MTP vs DFlash2 (head-to-head, sm120) | MTP retains 82.9% of the no-spec pool; DFlash2 does not load at full max-len at all |

### 4.3 Sampling/Thinking Sweep (loop behavior, sm120 AND sm121)
5 configs × 4 tasks × K6. **C1 (thinking-ON, current baseline) best on both arches**
(highest success, 0% loops); thinking-OFF (`-tools` profile, pp1.5) worse, on sm121
even loop-inducing. → Sampling stays thinking-on; the tool-loop fix is NOT in
sampling but in Hermes (inert guardrails + steering-in-tool-content). Caveat:
the harness barely reproduces the real Hermes loops → directional signal, not final.

### 4.4 Cross-Arch sm121 (Spark) — Feasibility
MTP drain is easier on the pool-rich sm121 (affordable); MTP+tool-calling desync (old
08-09 blocker) no longer reproduces on the new build. GridBook-on-Spark does not
load yet (rdtand-3.8-gridbook mis-exported: `model.language_model.*` prefix vs.
text-only CausalLM; Spark build f4c27c0da rejects it — the RTX build accepts it).
Fix in progress: `gridbook-13gb-fixed` (prefix strip) for Whisper headroom.

---

## 5. Known Points / Lessons

- **util 0.97 crashes reproducibly** (fragmentation OOM under needle 12k–64k load:
  vLLM's static KV reservation eats the slack that long-context activations +
  allocator fragmentation need beyond the profiling estimate). **util 0.90
  is stable** (battery incl. needle 9/9). Open optimization lever:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` might make higher util stable
  → pool recovery. Not yet tested.
- **No GPU split:** <rtx-node>/RTX is splittable (allocatable 2), but we do not split —
  the prod pod needs the GPU exclusively (otherwise vLLM re-profiles against less memory,
  534k instead of 624k).
- **Verdict robustness:** the n=1319 run was lost twice to ephemerality
  (Mac orchestrator death; pod `sleep 86400` death). Lesson: make long runs pod-death-proof +
  outputs on PVC (`/cache`), never ephemeral `/tmp`.

## 6. Artifacts / Sources
- Serving: deployment `vllm-qwen38-mtp-gridbook`, image `...gridbook-088`.
- Checkpoint: `/cache/models/gridbook-13gb-mtpfix` (PVC).
- Result JSONs: `s24/RESULT_verdict_mtpfix_n1319.json`, `RESULT_mtpfix-mtp-full-battery.json`,
  `RESULT_phase4_sweep.json`, `RESULT_headtohead_aqua_prod.json`, `RESULT_kv_drain_diagnosis_aqua.json`.
- Swap script: `s24/surgical_embed_swap.py`.
