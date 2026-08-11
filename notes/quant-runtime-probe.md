# Quantization runtime probe — 4B measured, 27B extrapolated

**Status:** measured (2026-08-11). Planning input for the next-model
readiness pipeline: how long does a full PrismaQuant aura run take on a
27B-class dense model, on hardware we own?

## Why

The quantization pipeline (probe → cost → allocator → export,
[PrismaQuant](https://github.com/RobTand/prismaquant), `COST_MODE=aura`)
had no published runtime numbers for large models. That single unknown
dominated the question of whether big quant runs need external GPUs or
fit into local maintenance windows. So: measure a small model end to end,
count what actually scales, extrapolate with the uncertainty stated.

## Measurement

Full pipeline run on **Qwen3-4B (bf16 source)** on GB10 (DGX Spark,
121 GiB unified memory), box to itself, quick-calibration setting
(`NSAMPLES=4`, `SEQLEN=256`, the pipeline's documented smoke-run minimum;
`TARGET_BITS=5.5`, `COST_MODE=aura`, `SELECTION_MODE=validated-surrogate`,
`compressed-tensors==0.15.0.1`). Exit 0, export artifact 2.91 GB.

| Phase | Wall time |
|---|---|
| Probe (Fisher-KL sweep) | 20 s |
| Baseline cost | 28 s |
| **Production cache (GPTQ + static_act_order + joint_scale_opt)** | **504 s** |
| **AURA cost (KL-adjoint, fp32-resident)** | **661 s** |
| Allocator (knapsack DP, CPU) | 11 s |
| Frontier-KL + select/refit | 95 s + 73 s |
| Export (compressed-tensors) | 57 s |
| **Total** | **24 m 26 s** |

The two cost phases dominate (~80% of wall time) and both scale with the
number of linear modules, not raw parameter count. Memory: 79.6 GB peak
(transient load spike at the start of the GPTQ phase), ~35 GB steady
state during the GPU-heavy phases — no ceiling risk on 121 GiB unified,
and none projected for 27B.

## Extrapolation method: per-linear, not per-parameter

The 27B target (dense — no MoE) is a **hybrid linear-attention
architecture**: 48 of 64 layers run `linear_attention` with a different
linear-module set than the 16 `full_attention` layers. The 4B probe model
has no counterpart for those modules, which makes per-parameter scaling
unreliable. Counted directly from the safetensors index / production
manifest instead: 4B = **252** body linears, 27B = **496** (614 including
MTP/passthrough assignment entries) → ratio **1.97–2.44×**.

Calibration rescaling to production settings (`NSAMPLES=32`,
`SEQLEN=1024`): the probe phase scales linearly (confirmed by the
pipeline author's own comments); the expensive GPTQ/refit phases cap
activation rows at `--max-act-rows=512`, which both the quick and the
production calibration already exceed — so their runtime is plausibly
calibration-invariant. That cap assumption is the largest remaining
uncertainty and is stated as such.

## Result

- **27B dense, full pipeline, production calibration: 1.1–3.4 h**
  (per-linear ratio, calibration rescaling per phase).
- **Conservative worst case: ~12–15 h** if the max-act-rows-cap
  assumption does not hold — still a single overnight window.
- Cheap follow-up: instrument the cap assumption during the first real
  27B run; one measurement kills the residual band.

## Consequence

**Serial-local quantization is viable; external GPUs are not on the
critical path** for producing a deployable quantized checkpoint. The
actual time sink in a promotion cycle is validation, not quantization:
a full GSM8K n=1319 arm runs ~8 h. External capacity would buy parallel
validation arms, not a faster quant.
