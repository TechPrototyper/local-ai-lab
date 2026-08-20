# NVFP4 KV: does GPTQ or a bigger Hadamard rescue amax calibration?

**Status:** measured (2026-08-20, RTX 5090 / sm120, vLLM
`0.1.dev1+gf4c27c0da` parity build, torch 2.13.0+cu130). Two model-free
`reshape_and_cache_flash` round-trip probes, run in a batch Job on the
freed GPU (production `vllm-qwen38` scaled 1→0 for a ~4-min window, then
restored and `/health` 200 verified). Answers the two follow-up questions
in llm-compressor#2936 (Dellabetta: "larger hadamard size or some other
modifier like GPTQ might be needed?").

## What this is

Both probes extend `nvfp4_hadamard_probe.py`: same real NVFP4-KV writer,
same independent linear-scale dequant, same sink provenance (2 outlier
channels at the measured magnitudes 42k/125k), same metrics computed in
the original space (bulk rel-L2, bulk exact-zero fraction, outlier
reconstruction). Prior result being extended: an R2-style per-head
(d=128) Hadamard does **not** rescue per-tensor amax calibration but
gives 5.7× better outlier preservation under scale-1.0 serving.

- **Probe A — GPTQ ablation** (`nvfp4_gptq_ablation_probe.py`): the value
  bulk is now produced by a real projection V = X·Wᵥᵀ, and Wᵥ is
  optionally quantized with a genuine GPTQ pass (Hessian H = XᵀX,
  Cholesky inverse, left→right column error propagation, symmetric int4).
  Full matrix {scale 1.0, amax/6} × {no rot, R2} × {no GPTQ, GPTQ}.
- **Probe B — online Hadamard > head_dim**
  (`nvfp4_online_hadamard_probe.py`): a non-foldable runtime rotation of
  size B ∈ {1, 128, 256, 512, 1024} across the flattened kv-head axis
  (B>128 crosses head boundaries → cannot fold into Wᵥ/W_o; applied at
  write, inverted at read). The amax observer sees the post-rotation
  space, i.e. what a real KV amax observer would bake.

## Measurements

**Probe A — GPTQ does not touch the KV failure.** Under amax/6, the bulk
is 100% erased (rel-L2 1.0, ~100% exact zeros) in **all four**
{rot × gptq} cells, at both 42k and 125k, and the baked per-tensor
v_scale is *identical* with and without GPTQ (6997 unrot / 1237 R2 at
42k) — sinks dominate amax, GPTQ never sees them. GPTQ is doing real work
on its own axis (W rel-err 0.093 vs round-to-nearest 0.128, ~27%
better); it just adds a small, separate bulk cost (unit no-sink bulk
0.096 → 0.108) and leaves the rotation/outlier behaviour unchanged.

**Probe B — block size is a strong but magnitude-bounded lever.** Under
amax/6, bulk exact-zero fraction vs online block size B:

| sink | B=1 | 128 | 256 | 512 | 1024 | v_scale @1024 |
|---|---|---|---|---|---|---|
| 42k  | 100% | 99.9% | 99.95% | 87.8% | **0.9%** | 437 |
| 125k | 100% | 99.9% | 100%   | 99.6% | **100%** | 1301 |

The e4m3 block-scale subnormal floor is 2⁻⁹ ≈ 1.95e-3; bulk survival
needs the baked v_scale ≲ ~340. A non-foldable B=1024 rotation drops the
42k sink's v_scale to 437 (near the floor → bulk essentially recovered,
0.9% zeros, rel-L2 0.89) but only to 1301 for the 125k sink (far above
the floor → still 100% erased). Under scale-1.0, the same online rotation
improves outlier reconstruction monotonically (42k: 2688-clip → 15.2k →
21.5k → 30.4k → 43.0k ≈ exact at B=1024) for a modest bulk cost
(0.096 → 0.130).

## Read carefully

- These are single model-free probes, not a checkpoint eval. Probe A's
  bulk is a synthetic projection with sink-source rows; that inflates the
  unit-scale R2 bulk cost (0.27 here vs 0.10 in the pure-Gaussian probe,
  because rotation folds sink energy into the sink tokens' own bulk
  channels) — a nuance the earlier pure-Gaussian probe understated, not a
  new deployable number.
- The ~340 survival threshold is approximate (bulk block-amax varies);
  treat 42k@B=1024 as "at the floor," not "safely under."
- Nothing here was serving-validated end-to-end. The uncalibrated
  scale-1.0 config remains at parity with fp8-KV (94.92% full-split
  GSM8K), so there is still no measured deficit to close.

## Why it matters

Answers the maintainer directly. **GPTQ:** orthogonal axis — it
compensates weight-quant error, not the KV-cache per-tensor-scale
underflow, so it cannot rescue amax KV (bulk erased with or without it).
**Larger Hadamard:** block size genuinely helps, and a non-foldable
online rotation > head_dim rescues a *moderate* (42k) sink under amax —
but the max-magnitude 125k layers, which are the ones that actually drove
the GSM8K loss, stay 100% erased even at B=1024 (required size ~1e4,
impractical). So for a checkpoint whose worst sink layers reach ~125k,
neither a GPTQ modifier nor a practical online Hadamard changes the
verdict: per-tensor amax observers are the wrong objective for 4-bit KV,
and scale-1.0 serving remains the fallback. Bigger H is a real complement
to scale-1.0 outlier preservation, not a fix for the amax path.

Artifacts: `results/RESULT_nvfp4_gptq_ablation_probe.json`,
`results/RESULT_nvfp4_online_hadamard_probe.json`;
scripts in `probes/`.
