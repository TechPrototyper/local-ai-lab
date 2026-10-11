# Recon: NVFP4-KV + non-causal DFlash2 drafter attention on sm120/sm121

**Status:** DRAFT — recon spike (Kanban #29). No GPU run, pure code/architecture archaeology.
**Date:** 2026-08-20 (night). **Author:** Session-B agent.
**Sources:** local fork checkout `~/Projects/vllm-upstream-work` (branch
`jethac/spark/hijinks-020-aeon-qwen-dflash-sm121a`, HEAD `e2a8197a9` — the `e24d26638`
named in the assignment is not present locally, the branch contents are equivalent),
flashinfer PR **#3684** (MERGED, via `gh`, read-only), vLLM PRs **#52816/#52883** (both OPEN),
InferenceDream (`custom_kv_kernel.cu`, `kdev.py`, `RESUME_NIGHT_KERNEL_OPT.md`).

Line numbers refer to the branch state above.

---

## 1. The gap (summary)

DFlash/DFlash2 verify needs attention in which a small **query block** of
`1 + num_speculative_tokens` tokens (for `num_speculative_tokens=15` → **16 query tokens/req**)
**sees the entire committed prefix in full AND is bidirectional within the draft block**
(block diffusion). The prefix K/V is present as an NVFP4 page pool.

This combination — **non-causal query block over NVFP4-KV pages** — exists **nowhere**
in the stack:

- **flashinfer:** The NVFP4 paged-prefill path (FA2, sm120/121) from #3684 is wired in the vLLM call
  effectively **causal-only**; whether the kernel itself correctly carries `mask_mode ≠ causal` / `custom_mask`
  over NVFP4 pages is **untested** (see §4 Option A).
- **vLLM:** Two hard guards (backend selection + hard-coded `causal=True` in the build)
  prevent the non-causal drafter from reaching the FlashInfer NVFP4 path at all. The
  drafter is forced onto a non-causal-capable backend (FLASH_ATTN) whose KV pages are **not
  NVFP4** (bf16/fp16; fp8 only via `kv_cache_dtype_skip_layers`). Result: separate/padded
  page pool for the drafter layers → page-padding blowup (~41 GiB @32k). The skip-layer escape route is
  exactly this blowup and is therefore dead.

**Payoff when resolved:** speculation **and** 4-bit KV simultaneously on every memory-bound
card — the central speed↔kv-max conflict is resolved instead of managed.

---

## 2. Where exactly it lives (files + lines)

### 2a. Guard #1 — backend selection excludes FlashInfer for non-causal
- `vllm/v1/attention/backend.py:229-237` — `AttentionBackend.supports_non_causal()` returns
  **`False`** by default. **FlashInferBackend does NOT override this** (no override in
  `vllm/v1/attention/backends/flashinfer.py`).
- `vllm/v1/attention/backend.py:323-324` — `validate_configuration(...)`: for
  `use_non_causal and not cls.supports_non_causal()` → invalid reason
  `"non-causal attention not supported"`. This drops FlashInfer for the drafter.
- `vllm/v1/attention/backends/flash_attn.py:113-114` — only **FLASH_ATTN** (plus rocm_attn,
  rocm_aiter_fa, flex_attention) returns `supports_non_causal() == True`.
- `vllm/v1/attention/backends/flash_attn.py:70-74` — FLASH_ATTN
  `supported_kv_cache_dtypes = ["auto","float16","bfloat16"]` → **no fp8, no nvfp4**.
- `vllm/config/speculative.py:114` — the comment says it explicitly:
  *"DFlash needs a non-causal-capable backend like FLASH_ATTN"*.
- Selection/init path: `vllm/v1/spec_decode/dflash.py:73-82` (`_create_draft_vllm_config` sets
  `use_non_causal=not self.dflash_causal`), assertion in `dflash.py:286-293`
  (every layer must satisfy `attn_metadata.causal is False`), drafter backend init in
  `vllm/v1/worker/gpu_model_runner.py:6816-6819`.

**Net:** non-causal ⇒ only FLASH_ATTN qualifies ⇒ drafter KV cannot be NVFP4.

### 2b. Guard #2 — FlashInfer build hard-wires `causal=True`
Even if Guard #1 were bypassed:
- `vllm/v1/attention/backends/flashinfer.py:1445-1463` — `prefill_wrapper.plan(..., causal=True, ...)`
  **hard-coded**; `common_attn_metadata.causal` is ignored, **no `custom_mask`/`packed_custom_mask`
  plumbing**.
- Likewise `flashinfer.py:1350` (cascade) and the trtllm prefill paths — `causal=True` everywhere.
- The only non-causal prefill in the backend is the **DCP context run**
  (`flashinfer.py:450 causal=False` / `:466 causal=True` = two-pass context+new tokens with
  LSE merge) — but that is **explicitly blocked** for the FA2-NVFP4 path
  (`flashinfer.py:877-880`: `use_fa2_nvfp4_kv and use_dcp` → `NotImplementedError`).

### 2c. The NVFP4-FA2 path itself (sm120/121)
- `vllm/v1/attention/backends/flashinfer.py:815-835` — `is_kvcache_nvfp4` +
  `use_fa2_nvfp4_kv = is_device_capability_family(120)`; sm100 → trtllm-gen, otherwise error.
- `flashinfer.py:995-1005` — prefill wrapper `BatchPrefillWithPagedKVCacheWrapper(..., backend="fa2")`
  for the NVFP4-KV case; `flashinfer.py:1024-1042` analogous for decode.
- `flashinfer.py:1145-1146` — for FA2-NVFP4, `prefill_use_trtllm=False` is forced (i.e. the
  native FA2 prefill path, which in principle knows masks).
- V-scale deswizzle flag: `flashinfer.py:82, 273-278, 881-886`
  (`-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`).
- NVFP4 page geometry: `vllm/utils/torch_utils.py:415 nvfp4_kv_cache_full_dim`,
  `:472 nvfp4_kv_cache_split_views`.

### 2d. Skip-layer escape route (the dead path)
- `vllm/config/cache.py:115 kv_cache_dtype_skip_layers`,
  `vllm/model_executor/layers/attention/attention.py:253-265` (layer → native dtype),
  `vllm/platforms/interface.py:596`. These layers get a **different page geometry** →
  padding to the larger page width → the aforementioned ~41 GiB blowup @32k.

### 2e. DFlash2 delta (upstream, OPEN, cherry-picked into the fork)
- vLLM **#52816** (SubSir): `DFlash2DraftModel` = DFlash + grouped dynamic depthwise conv +
  candidate selector; V2 runner: `model_executor/models/qwen3_dflash2.py`,
  `v1/worker/gpu/spec_decode/dflash2/speculator.py`, `.../__init__.py`, `registry.py`.
- vLLM **#52883** (oceanplexian, stacked on #52816): LM-head guard fix (unquantized head with
  quantized body).
- **Important:** conv + selector are additional sublayers; DFlash2 inherits the **attention-mask semantics**
  unchanged from the DFlash proposer (§3). For this kernel problem, DFlash == DFlash2.

---

## 3. Required mask semantics (precise)

From `vllm/v1/spec_decode/dflash.py`:
- **Geometry:** `num_query_per_req = 1 + num_speculative_tokens` (`dflash.py:104`). Bonus token +
  `num_speculative_tokens` mask/proposal tokens are the **only queries**; the entire
  context (accepted prefix) is **K/V** and is written into the cache in advance via `precompute_and_store_context_kv`
  (`dflash.py:238, 265-269`). With `num_speculative_tokens=15` → **16 queries/req**.
- **Mask:** `new_cad.causal = self.dflash_causal` (default **False**, `dflash.py:71, 189`) and
  draft config `use_non_causal = not dflash_causal` (`dflash.py:80`). The proposer **asserts**
  `causal is False` for all draft layers (`dflash.py:286-293`).
- **Effectively:** fully non-causal over `[prefix | block]`. Since the block is appended **after** the entire
  prefix, "fully non-causal" is **identical** to the assignment's formulation
  *"causal to the prefix + bidirectional within the draft block"*:
  each of the 16 query positions sees (a) **all** prefix columns (rectangular full view) and
  (b) **every** other block position (non-triangular `16×16` submatrix). No sliding window,
  no soft-cap requirement.
- **Consequence for the kernel API:** The pure block-diffusion case needs **no** bespoke
  `custom_mask` — an FA2 paged-append-prefill `plan(causal=False)` delivers exactly these
  semantics (query = the last `qo_len` positions of the KV sequence, non-causal ⇒ every query sees
  every KV position). A **packed custom_mask** would only be needed if DFlash2 later introduces
  a stricter intra-block structure (e.g. tree/causal sub-mask).

---

## 4. Solution architectures A / B / C

### Option A — Mask-mode extension of the existing NVFP4-FA2 kernel family in flashinfer
**Idea:** Make the FA2-NVFP4 paged prefill (#3684, `include/flashinfer/attention/prefill.cuh`) usable for
`mask_mode = kNone` (non-causal) or `kCustom` over NVFP4 pages; wire up vLLM so
that the drafter prefill plans with `causal=False` (instead of hard `causal=True`).

**Evidence for:**
- #3684 touches **exactly this kernel** (prefill.cuh +99/-41, scheduler.cuh, page.cuh,
  `flashinfer/prefill.py` +132/-19) and already contains a **custom-mask fix**:
  *"plan(): move `mask_indptr` to the custom mask's device before `segment_packbits`"* — i.e.
  the **mask plumbing of the paged/ragged prefill wrapper exists and was even repaired**.
- The FA2 prefill is the generic FA2 path that in principle knows `custom_mask`/`packed_custom_mask`
  (standard in the fp16/fp8 case).

**Evidence against / open points:**
- #3684 numerically validates only **causal** ("multi-tile seq 512/1024 + causal all sane") and
  symmetric shapes; **non-causal / custom_mask over the NVFP4 V-load path is not
  demonstrated**. Risk: the FP4 V-load + `[all-data|all-SF]` SF layout + asymmetric offsets
  interact with mask iteration/tile skipping.
- **Split-KV is hard-disabled for NVFP4** (#3684 additional fix: mid-block split violates the
  16-element FP8 block alignment → retrieval cliff). Drafter verify is **exactly** the
  extend geometry (`qo_len=16 ≪ kv_len`) that would otherwise drive the scheduler into split-KV → the path
  runs **single-split** (correct, but without flash-decoding parallelism). Perf footnote, not a blocker.
- Where causality is hard-wired: `CTA_TILE_Q`/schedule dispatch (`scheduler.cuh`) + the tile iteration
  in `prefill.cuh` (mask-dependent KV tile skipping). With non-causal the skipping is dropped, i.e.
  potentially **more** KV tiles per query tile — negligible with 16 queries.
- vLLM side: Guard #1 (`supports_non_causal` override for FlashInfer, gated on `use_fa2_nvfp4_kv`)
  **and** Guard #2 (`causal` from `common_attn_metadata` instead of hard `True`, + optional custom_mask
  plumbing through `plan()`).

**Scope:** **medium–large.** flashinfer kernel verification/adaptation (unknown depth) + two
vLLM guards + metadata plumbing. Perf expectation: **high** (native FA2, close to tensor cores) once
non-causal is correct.

### Option B — On-the-fly dequant shim (NVFP4 pages → bf16/fp8 only for drafter attention)
**Idea:** Only for the (few) drafter attention calls, dequantize the relevant NVFP4 prefix pages into
a **scratch buffer** to bf16 (or fp8) and run the **existing non-causal
FLASH_ATTN kernel** on that. The persistent KV pool stays NVFP4 throughout (no
page padding, no second pool).

**Evidence/infra:**
- Dequant building blocks already exist: `nvfp4_kv_cache_split_views` (`torch_utils.py:472`) and the
  Triton dequant `trtllm_prefill_attn_kvfp8_dequant` (`flashinfer.py:292-394`) as a template for an
  NVFP4→bf16 scratch dequant.
- FLASH_ATTN supports non-causal (`flash_attn.py:113`) and bf16 KV (`:70-74`) — fits directly.

**Memory/latency estimate (drafter, per layer, per req):**
- Context to dequantize ≈ `seq_len × num_kv_heads × head_size` bf16. Example Gemma full
  attention (HS=256, NKV small): @32k, HS=256, NKV=2, bf16 = `32768·2·256·2 B ≈ 32 MiB/layer/req`.
  As a **one-time** scratch (reused across layers/reqs, not persistent) this is a small
  fixed reserve, **not** the ~41 GiB padding — the key advantage over the skip-layer escape route.
- Latency: additional dequant read/write of the prefix per drafter forward. Since the drafter has only
  16 queries but reads the **full** prefix as K/V, the dequant is memory-bound ~ the
  attention itself — roughly **~2× bandwidth** of the drafter attention share. Drafter attention is
  a small fraction of the total step → moderate, predictable overhead; **no** new MMA kernels.

**Scope:** **small–medium.** A dequant shim + scratch allocator + drafter attention routing.
Perf expectation: **moderate** (bandwidth overhead), **correctness trivially verifiable**
(dequant→SDPA is the reference itself). **Best correctness-first starting point.**

### Option C — Custom non-causal NVFP4-KV kernel from scratch
**Only if A fails structurally** (kernel does not carry non-causal) **and B is not sufficient performance-wise.**
**Important:** such a kernel already exists substantially in InferenceDream —
`custom_kv_kernel.cu` (870 lines): NVFP4-KV attention, modes 0 (K+V NVFP4) / 1 (K fp8 / V NVFP4),
HS=128/256 templates, stream fix, **MTP-compatible**, **bit-exact parity** vs. stock verified.
Causality sits in **one** place: `custom_kv_kernel.cu:216`
`int num_keys = seq_lens[s_idx] - q_len + q_offset + 1;` (upper key bound per query = own position).
For block diffusion: `num_keys` such that each query sees the **full** prefix + **all** block positions
(prefix length + full block view instead of `+ q_offset + 1`). The split-K variant (`:395ff`) analogous.

**Scope:** **medium** (not large, because kernel + split-K skeleton + parity are already there), but
**low perf ceiling** (scalar software dequant, **no** tensor cores — documented in
`RESUME_NIGHT_KERNEL_OPT.md`). For 16 drafter queries over a long prefix it is nevertheless acceptable, since
memory-bound. Perf expectation: **low–moderate**; advantage = full control + existing parity.

---

## 5. Recommended path

**Two-stage: B first (correctness gate + immediate benefit), probe A in parallel as the perf target.**

1. **B as a correctness-first prototype** — resolves the 41 GiB blowup immediately (NVFP4 pool throughout),
   correctness is trivially verifiable against dequant-SDPA, no new MMA kernels, small scope.
   Delivers the first "speculation + 4-bit KV on one card" result.
2. **A as the perf upgrade** — in parallel a **micro-probe** (no vLLM needed): test the flashinfer FA2
   NVFP4 paged prefill directly with `mask_mode=kNone`/`custom_mask` against a bf16 reference.
   If the kernel carries non-causal correctly, A is the end state (native speed) and B falls back
   to being the fallback. If not → A = kernel patch in prefill.cuh, or C.
3. **C only** if the A probe is red **and** B perf is insufficient — with the InferenceDream kernel as the base.

---

## 6. Phased plan (correctness gate before perf phase)

**Phase 0 — Reference & fixtures (no GPU needed).**
- Extend the `kdev.py` oracle (InferenceDream) with a **non-causal / block-diffusion reference path**:
  `ref_attention(..., causal=False)` or a `[prefix full | block full]` mask; gate **cos > 0.99**
  against dequant-KV SDPA, batch case with variable `seq_lens`/`q_lens=16` (`kdev.py:139-166`).
- DFlash2 mask fixture: `qo_len=16`, long prefix, NVFP4 pages, against bf16 ground truth.

**Phase 1 — Correctness prototype (Option B).**
- Dequant shim NVFP4→bf16 scratch (template `flashinfer.py:292-394`), drafter attention on
  FLASH_ATTN non-causal. **Gate:** near-bit parity (cos > 0.99) drafter output vs. full-bf16 drafter,
  acceptance rate unchanged vs. fp8/bf16 baseline. **Only after a green gate** → perf.

**Phase 1' (parallel) — A feasibility probe.**
- Standalone flashinfer test: FA2-NVFP4 paged prefill with non-causal/custom_mask, `max_abs_err`
  against bf16 (analogous to the #3684 validation `~0.0047`). Red/green decides A vs. C.

**Phase 2 — Perf & integration.**
- If A is green: vLLM Guard #1 (`supports_non_causal` override gated on `use_fa2_nvfp4_kv`) +
  Guard #2 (`causal` from metadata, optional custom_mask plumbing through `plan()`); E2E against bf16
  facts (retrieval needle as in #3684, radix/prefix cache **on**).
- If B is the end state: scratch allocator tuning, overhead measurement, possibly dequant fusion.

**Risks (largest first):**
- **A:** The FA2-NVFP4 kernel may not carry non-causal/custom_mask correctly (FP4 V-load × mask-tile
  iteration untested) → fallback B/C. The **split-KV gate** makes drafter verify single-split (perf footnote).
- **B:** Dequant overhead per drafter forward (memory-bound, ~2×); scratch lifetime/CUDA-graph
  stability (keep buffer addresses stable — cf. DFlash context buffer pattern `dflash.py:42-62`).
- **Fork drift:** DFlash2 (#52816/#52883) is **OPEN** upstream; the mask semantics may still change with
  selector/conv (currently it inherits unchanged from DFlash).
- **Timing bet:** post the upstream issue only after the prototype decision (assignment).
