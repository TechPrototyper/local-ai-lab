# AURA (PrismaQuant KL-Fisher) vs NVIDIA Model-Optimizer AutoQuantize — paired head-to-head design (DRAFT, do not commit)

**Status:** design document, Kanban #26. Desk work only — nothing executed, no
GPU touched, nothing posted upstream. Builds directly on the closed spike
(`notes/modelopt-autoquant-feasibility.md`, Kanban #13/#24,
`results/RESULT_modelopt_autoquant_spike.json`,
`results/RESULT_modelopt_uniform_fp8_serving.json`) and this lab's existing
paired methodology (`notes/qwen36-aura-head-to-head.md`, `README.md#method`).

## Why this document exists

The feasibility spike closed Q1–Q3 (serving, install, format bridge) but
explicitly flagged two confounds that would make a naive "AURA vs
AutoQuantize at 5.5 bits" claim wrong: (B2) the two tools define
**"effective bits" differently** and (B3) they disagree on whether
**KV-cache quantization** is part of the budget. This document reconciles
both, names the remaining confounds, and gives a concrete run plan. Read
`modelopt-autoquant-feasibility.md` first for the "how we got here."

---

## 1. Common bit-accounting metric

### 1.1 What each tool actually computes today (verified in source, not assumed)

**ModelOpt (`NVIDIA/Model-Optimizer`, commit `913f5e2`,
`modelopt/torch/quantization/_auto_quantize_cost.py` +
`algorithms.py`):**

- Default cost model = `WeightCostModel`: counts **only quantizable weight
  elements** (`module.weight.numel()`, or `gate_up_proj + down_proj` for
  fused MoE containers). Activations, biases, embeddings/lm_head (unless
  explicitly in the search space) and **KV cache are not counted** — there
  is no `kv_cache` term anywhere in the cost-accounting code, and our own
  export confirms it (`hf_quant_config.json → kv_cache_quant_algo: null`).
- Per-format nominal cost resolves in this priority order
  (`estimate_quant_compression`, `algorithms.py:203-260`):
  1. an explicit `effective_bits` field on the format config (library
     default for NVFP4 = **4.5**, documented in
     `modelopt_recipes/configs/numerics/nvfp4.yaml`: *"4 value bits + an
     FP8 (8-bit) scale per 16-element block = 4.5 bits/element"* — i.e.
     `4 + 8/16 = 4.5`);
  2. else the raw `num_bits` heuristic: `num_bits` directly for int
     formats (INT4 → 4.0, INT8 → 8.0), or `sum(num_bits) + 1` for FP
     tuple formats (FP8 E4M3 → `4+3+1 = 8.0`).
- The reported search-time `effective_bits` (`algorithms.py:1321`,
  `:1423`) is:

  ```
  effective_bits = (Σ_i n_params_i · bits_i) / (Σ_i n_params_i) 
  ```

  i.e. a **parameter-count-weighted average of per-format nominal bits**,
  normalized against 16 (the bf16 baseline) only in the constraint-solving
  step (`weight_compression = effective_bits / 16`), not in the reported
  number itself. `constraints={"effective_bits": 5.5}` means exactly this
  weighted average must land at 5.5.

**AURA (`RobTand/prismaquant`, `prismaquant/format_registry.py` +
`allocator_solver.py`):**

- `FormatSpec.effective_bits` (`format_registry.py:119-132`): **"Average
  bits per parameter accounting for scales"** —
  `weight_bits + scale_bits / group_size` for grouped formats (its NVFP4
  entry: `weight_bits=4, scale_bits=8, group_size=16` → **4.5**, byte-for-byte
  the same derivation as ModelOpt's), or `weight_bits + 0.02` (a small
  fixed per-tensor-scale fudge) for ungrouped/per-tensor formats.
- `allocator_solver.solve_allocation()` optimizes a **multi-choice knapsack
  in "average-bits-per-parameter" units**: `target_bits` is compared
  against `Σ_i n_params_i · bits_per_param_i / Σ_i n_params_i` — the
  identical parameter-count-weighted-average shape as ModelOpt's
  `effective_bits`.
- No KV term anywhere in `format_registry.py`/`allocator_solver.py`'s
  bit-accounting classes/functions — KV precision is a **separate, serving-side
  decision** for AURA too (confirmed operationally: the sm120 production
  config note treats `--kv-cache-dtype nvfp4` as a vLLM serving flag layered
  on top of the already-fixed weight allocation, not something the
  allocator solves for).

### 1.2 Conclusion: no bridge needs inventing — the two tools already agree on the formula shape

Both tools compute the **same quantity** — a parameter-count-weighted
average of per-format nominal bits/weight-element, weight-only, KV-excluded
— they just expose it through different knobs (`constraints={"effective_bits": X}`
vs `target_bits=X`) and different per-format nominal-bit tables. The common
metric for this H2H is therefore defined, not designed from scratch:

> **bpp (bits per weight-parameter)** =
> `Σ_i n_params_i · bits_per_param(format_i) / Σ_i n_params_i`,
> where `bits_per_param(format) = weight_bits + scale_bits / group_size`
> for block/group-quantized formats (NVFP4, INT4-AWQ-g128, …) and
> `weight_bits` alone for ungrouped/per-tensor formats (FP8) — **weights
> only, no activations, no biases/embeddings unless explicitly included in
> both allocators' search spaces, no KV cache.**

### 1.3 Calibrating both tools to the same number

Both tools already read this parameter directly:

- AURA: `target_bits=5.5` (as used for the existing `prismaaura55` export —
  confirms the reference allocation NVFP4:264/BF16:187/FP8:163 modules
  already sits at this budget).
- ModelOpt: `mtq.auto_quantize(constraints={"effective_bits": 5.5}, quantization_formats=[FP8_DEFAULT_CFG, INT4_AWQ], …)`.

No conversion factor is needed *if* the per-format nominal-bit tables are
verified equal for the formats actually in play. Before trusting a "5.5 vs
5.5" claim, do a cheap arithmetic check (no GPU): confirm ModelOpt's
`INT4_AWQ` config resolves to `4 + 8/128 = 4.0625` (group_size 128, FP8
scale) or whatever its actual `group_size`/`scale_bits` are, and diff that
against AURA's own INT4/AWQ format entry in `format_registry.py`. If they
differ (e.g. different group_size or scale dtype), state the delta
explicitly in the published bit-maps rather than silently averaging over
it — a 4.06 vs 4.5 nominal-bit mismatch on the INT4 arm would quietly skew
which layers each allocator is willing to demote.

**One real accounting gap to flag, not hide:** neither tool's declared
`effective_bits`/`bits_per_param` accounts for the **pre-quant AWQ
scale/zero-point tensors** (`pre_quant_scale`, `has_zero_point` in
ModelOpt's export) or compressed-tensors' `weight_zero_point` beyond the
group-scale term already in the formula. These are small relative to the
group-scale term (typically ≤0.1 bit/param) but are asymmetric between
formats (AWQ carries a zero-point, plain NVFP4 doesn't) — note this as a
second-order, sub-0.1-bit systematic bias in the published write-up rather
than claiming exact 5.500 == 5.500 parity.

---

## 2. KV-cache configuration decision

**Confound:** AURA's *production* Qwen3.x-27B config pins NVFP4-KV
(`qwen36-prismaaura-target-configs.md`: sm120 target = plain aura55 +
`--kv-cache-dtype nvfp4` scale 1.0, 94.92% GSM8K n=1319, +32% KV pool vs
fp8). ModelOpt's export leaves `kv_cache_quant_algo: null` — i.e. it makes
no KV-quant decision at all; that is a pure vLLM serving-flag choice
downstream of the checkpoint.

**Decision: serve *both* arms with unquantized KV cache
(`--kv-cache-dtype auto`, fp16/bf16), identically.**

**Why not NVFP4-KV for both, and why not "each tool's own default":**

1. KV-cache quantization is, by §1's own finding, **out of both tools'
   budget accounting** — it is a variable neither allocator optimizes for
   or is even aware of. The instruction to "hold KV-quant strictly out of
   the comparison" is best satisfied by literally not quantizing it in
   either arm, not by equalizing both arms at some quantized setting
   (equalizing at NVFP4-KV would still leave "is the already-known small
   NVFP4-KV quality cost applied identically to both arms' very different
   attention/logit distributions" as an unverified assumption).
2. fp16/bf16 KV is the **zero-extra-engineering choice for the ModelOpt
   arm** (it's already what `kv_cache_quant_algo: null` gives you) and a
   **one-flag change for the AURA arm** (drop `--kv-cache-dtype nvfp4`,
   which the existing recipe already documents as an override, not an
   export-baked property).
3. AURA's own NVFP4-KV cost/benefit is **already separately measured and
   published** (`sm120-nvfp4kv-quality-deficit` memory line: baked
   auto-scale NVFP4-KV loses 1.67pp vs the pinned plain+explicit-scale
   config) — this H2H does not need to re-litigate that axis; isolating
   weight-allocation quality is the whole point of #26.

**Caveats to state explicitly in the published write-up:**

- The H2H numbers describe **weight-allocation quality only**, not AURA's
  *as-deployed* production pipeline (which does run NVFP4-KV). Cite the
  existing 94.92%-with-NVFP4-KV number from
  `qwen36-prismaaura-target-configs.md` as the separate "real deployment"
  reference point, not as part of this comparison.
- fp16/bf16 KV means both arms get a **smaller KV pool / lower max
  concurrency** than AURA's real deployment (no +32% pool expansion). The
  tok/s numbers from this H2H are therefore **not representative of either
  method's deployed throughput** — label them "weight-format tok/s at
  fp16-KV" in any published table, not bare "tok/s".

---

## 3. Remaining confounds and the mixed-vs-uniform fairness question

### 3.1 Confounds to hold identical (checklist, from the spike + this design)

| # | Confound | How to hold it constant |
|---|---|---|
| 1 | **Calibration data** | Same file, same `nsamples`, same `seqlen` passed to *both* allocators (AURA's KL-Fisher probe and ModelOpt's AWQ/Hessian search are both calibration-data-sensitive; different corpora would confound "allocation policy" with "calibration draw"). Use the same calibration set already pinned for the existing `prismaaura55` export if AURA's pipeline used a fixed/versioned one; otherwise freeze one file and use it for both. |
| 2 | **Bit-budget definition** | §1 — verified-equal formula, per-format nominal-bit tables diffed and any residual gap published, not hidden. |
| 3 | **KV-cache dtype** | §2 — fp16/bf16, identical, both arms. |
| 4 | **Serving config** | Same parity vLLM build/fingerprint, same `max_model_len`, same `gpu_memory_utilization`, same sampling (greedy, temp=0), same GSM8K parser — reuse the `COMPARE_ref_1319` harness verbatim (per the spike's own stated design). |
| 5 | **Sequential, not co-tenant** | One physical card, arms run one after another, never time-sliced concurrently (repo convention: results-mode benches run sequentially). |
| 6 | **Node** | Both arms served from the same node in a given comparison run, to remove node-to-node variance from the quality numbers (quantization *production* can happen on different boxes — see §4 — but serving/measurement should not). |
| 7 | **transformers/torch drift** | ModelOpt's install bumped `transformers` to 5.14.1 in the spike; verify the production/measurement environment is pinned and identical for both arms' *serving* image (the isolated ModelOpt venv from B2 only needs to touch the *quantization* step, not the parity serving image). |

### 3.2 The uniform-vs-mixed asymmetry (the actual open fairness question)

The spike's B1 blocker is still live: vLLM's `--quantization modelopt`
loader crashes (`load_merged_column_weight`/`narrow` overrun) on
ModelOpt's genuine **per-layer MIXED_PRECISION** export (FP8 + W4A16_AWQ
split within a fused gate/up column) — only ModelOpt's **uniform-FP8**
export is currently servable. AURA's mixed compressed-tensors export
already serves fine today (it's the live production checkpoint). So the
three options from the task brief, evaluated:

- **(i) Both uniform FP8.** Removes the allocation-policy variable
  entirely — this measures only "does 8-bit rounding hurt Qwen3.x-27B",
  which is not what either tool's differentiator is. Low value; agree with
  the brief's own framing.
- **(ii) AURA mixed vs ModelOpt uniform FP8.** Honest and **runnable
  today** — every precondition (install, uniform-FP8 export, uniform-FP8
  serving) is already positively closed in the spike. But it is not
  "AutoQuantize vs AURA" — it is **"AURA's KL-Fisher allocation vs a
  same-budget naive-uniform baseline."** That is a real, useful,
  well-defined question (does per-layer allocation beat uniform at equal
  average bits at all?) and a *necessary* prerequisite before the harder
  question ("whose allocation search is better") is even meaningful — but
  it must be labeled exactly that, not oversold as testing AutoQuantize's
  mixed-precision search.
- **(iii) Wait for vLLM's mixed-modelopt loader fix.** The "real"
  comparison, but open-ended: this is a vLLM engine limitation (the
  modelopt loader's merged-column path assumes uniform width across the
  fused gate/up projection), not something under this lab's control on a
  timeline. No fix is in flight that we know of.

**Recommendation: run (ii) now under #26, explicitly labeled as an
allocation-vs-uniform ablation (not AutoQuantize's real output), and open a
*separate*, blocked-on-upstream follow-up ticket for (iii).**

Rationale: (ii) is the only option that produces a result inside this
sprint's GPU-time budget without depending on an external fix, and it is
not a wasted result — "does smart allocation beat uniform at the same
budget" is the load-bearing question a reader needs answered *before*
"AURA vs AutoQuantize search quality" is even interesting. Framing it
honestly (title, changelog entry, and any published table say "AURA
mixed-precision vs ModelOpt uniform-FP8 baseline, equal 5.5 bpp" — never
"AURA vs AutoQuantize") avoids the credibility risk of implicitly crediting
or blaming AutoQuantize's *search* for a result that is actually just
"uniform vs non-uniform."

**One cheap mitigation worth a 15-minute try before committing to (ii) as
strictly uniform:** add the fused gate/up-proj modules to ModelOpt's
`excluded_module_name_patterns` (keep them uniform-FP8) while letting
`auto_quantize` search mixed precision freely across the *other*
quantizable modules (attention q/k/v/o, `down_proj`, which are not
column-fused and shouldn't trip the vLLM narrow bug). This would recover a
**partial** mixed-precision ModelOpt arm without touching vLLM — still not
equivalent to AURA's fully-free search space (introduces its own
"constrained search space" confound, which must then be published too),
but worth a quick static check against the vLLM loader code before
defaulting straight to uniform. If it works, relabel as "(ii')
AURA mixed vs ModelOpt partially-mixed (gate/up excluded)" and note the
search-space asymmetry explicitly.

---

## 4. Run plan

**Model:** Qwen3.8-27B BF16 (per task instruction). Note: earlier notes in
this repo label the same artifact family "Qwen3.6-27B"
(`notes/qwen36-aura-head-to-head.md`) while the actual served model root in
this spike's own JSON artifacts is `qwen3.8-27b-prismaaqua55`
(`RESULT_modelopt_uniform_fp8_serving.json`). Confirm the exact HF model ID
with Tim before scheduling GPU time — this may just be version-label drift
on the same checkpoint family, but a 3.6-vs-3.8 mismatch would invalidate
any calibration-data/base-model parity claim.

### 4.1 Quantization runs

| | AURA arm | ModelOpt arm |
|---|---|---|
| **Where** | GB10 (sm121) — matches the precedent in `notes/quant-runtime-probe.md`; AURA's layer-streaming forward (the PR #80 fix path) is what makes 27B dense tractable under constrained VRAM, and that path is already validated on GB10. | neo26 (sm120), **isolated modelopt venv/build layer** (B2 from the spike — must not touch the parity serving image's pinned `transformers`/`torch`). |
| **Method/budget** | `target_bits=5.5` per §1's common metric, same calibration file/nsamples/seqlen as the ModelOpt arm (confound #1). | `mtq.auto_quantize(constraints={"effective_bits": 5.5}, quantization_formats=[FP8_DEFAULT_CFG, INT4_AWQ_CFG], data_loader=…)` on the same calibration file/nsamples/seqlen. |
| **Duration** | ~1.1–3.4 h, **already measured** and extrapolated from a 4B pipeline run (`notes/quant-runtime-probe.md`, 2026-08-11) — reuse, do not re-measure unless the calibration set changes materially. | **Not yet measured at 27B.** The spike only timed a 4B model (71.8 s mixed / ~13.5 s uniform-FP8, on neo26/sm120). Extrapolating from FLOPs alone suggests low tens of minutes for 27B, but this is a guess, not a measurement — **run a short timing-only dry run before committing an overnight window.** |
| **Memory** | Fits GB10 today (that's why the layer-streaming fix mattered). | **Open question, must be verified before scheduling:** the spike's `auto_quantize` ran on a fully-resident 4B BF16 model on a 32 GB sm120 card; a 27B BF16 model (~54 GB of weights alone) will not fit resident on the same card. Check whether `nvidia-modelopt`'s calibration path supports `device_map="auto"`/CPU-offload (see `examples/hf_ptq/hf_ptq.py` in the ModelOpt repo before scheduling) — if not, the ModelOpt arm may need to move to GB10 too, which reopens the sm120-vs-sm121 build-variant question for the *quantization* step (not the serving step, which is unaffected — export artifacts are sm-generic per the spike's Q3 findings). |

### 4.2 Serving (both arms)

- Same parity vLLM build/fingerprint as the spike (`gf4c27c0da` line or
  whatever is current at run time — pin and record it).
- Same `max_model_len`, same `gpu_memory_utilization`.
- **KV:** `--kv-cache-dtype auto` (fp16/bf16), identical — §2.
- **Sampling:** greedy, temperature 0, identical stop conditions.
- **Parser/harness:** reuse `COMPARE_ref_1319` verbatim (same GSM8K
  extraction regex, same needle-test harness) so any quality delta is
  attributable to the model/quant arm, not to harness drift.
- Sequential on one physical card, one arm at a time, same node for both
  arms within a given comparison run (confounds #5–#6).

### 4.3 Measurement

1. **n=250 GSM8K screen** (triage only, ±2.7 pp per `README.md#method`) →
   go/no-go gate before spending the full-n window.
2. **n=1319 GSM8K confirm** (verdict-level), McNemar exact test on the
   discordant pairs, reusing the exact paired-battery structure from
   `notes/qwen36-aura-head-to-head.md`.
3. **Needle** (long-context recall), same needle count/positions as the
   existing harness.
4. **Determinism**: 5 reruns per arm, identical greedy output required.
5. **tok/s**: single-stream and batch-8, **reported as a separate table
   from quality** (never blended into one score) — per this repo's
   established method and the spike's own stated design; labeled
   "fp16-KV, not representative of deployed throughput" per §2.

### 4.4 Publication

- Publish **both arms' full per-layer bit-maps**: AURA's module→format
  allocation table (as already done for `prismaaura55`:
  NVFP4/BF16/FP8 module counts) and ModelOpt's `hf_quant_config.json`
  `quantized_layers` map (or, for the uniform-FP8 fallback arm, state
  explicitly "uniform, no per-layer map — see §3.2 for why").
- Publish the §1.3 per-format nominal-bit diff table (any residual gap
  between the two tools' declared `bits_per_param` for the formats
  actually used), not just the headline "5.5 vs 5.5."
- Title/label everything per §3.2's fairness ruling — "AURA mixed vs
  ModelOpt uniform-FP8 baseline," never "AURA vs AutoQuantize," until/unless
  the (iii) mixed-vs-mixed follow-up actually runs.
- Follow the repo's changelog discipline (`README.md` changelog table gets
  a row; the "Consequence/go decision" convention from
  `modelopt-autoquant-feasibility.md` gets mirrored in whatever result note
  this run produces).

---

## Open items / follow-ups

1. **Cheap, no-GPU (do before scheduling):** diff ModelOpt's actual
   `INT4_AWQ`/`FP8` config `group_size`/`scale_bits` against AURA's
   `format_registry.py` entries for the same nominal formats (§1.3) —
   confirms or corrects the "5.5 == 5.5" assumption before it's load-bearing.
2. **Cheap, no-GPU:** check `nvidia-modelopt`'s `examples/hf_ptq/hf_ptq.py`
   for offload/`device_map` support at 27B before scheduling the ModelOpt
   quant run (§4.1 memory question).
3. **Cheap, ~15 min GPU:** try the `excluded_module_name_patterns` partial
   mixed-precision workaround (§3.2 mitigation) before defaulting the
   ModelOpt arm to fully uniform.
4. **Blocked on upstream, separate ticket:** vLLM `--quantization modelopt`
   loader's merged-column narrow crash on split-precision gate/up — files
   as the precondition for a future true mixed-vs-mixed H2H ((iii)).
5. **Confirm with Tim before scheduling:** exact base-model HF ID
   (Qwen3.6 vs Qwen3.8 naming drift, §4) and whether AURA's calibration
   corpus for the existing `prismaaura55` export is fixed/versioned and
   reusable as-is for the ModelOpt arm's calibration (confound #1).
