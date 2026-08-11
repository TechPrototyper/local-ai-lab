# Upstream engagement

Neutral record of interactions with upstream projects this lab's findings
touch: what was posted, when, and the distilled substance. Facts, recipes,
and outcomes only — no evaluation of the projects or their maintainers.

## llm-compressor#2936 — KV-cache calibration finding

- **2026-08-02** — comment posted distilling the calibration finding
  (README, 2026-08-01): baked per-tensor amax KV-cache scales measurably
  reduce NVFP4 accuracy (GSM8K full 1319-item split: 93.25% calibrated vs.
  94.92% uncalibrated, McNemar exact p=0.013), with the mechanism —
  fp8-e4m3 block-scale underflow on massive-activation/sink layers —
  confirmed by a model-free round-trip probe through
  `reshape_and_cache_flash`
  ([`probes/nvfp4_calib_scale_study.py`](../probes/nvfp4_calib_scale_study.py)).
- **2026-08-06** — follow-up posted (reply to a SpinQuant/R2-Hadamard-
  rotation suggestion raised in the thread): an R2-style Hadamard rotation
  does **not** rescue per-tensor amax calibration at 4-bit KV — real sink
  tokens carry multiple outlier channels whose rotated contributions still
  push the block scale below the e4m3 subnormal floor (99.8% bulk erasure
  measured even at a moderate 42k sink amax). Rotation *combined with*
  scale-1.0 serving does preserve outliers substantially better (5.7×
  improved sink reconstruction, at ~8% relative bulk-error cost) — a
  technique worth keeping in reserve, though with the uncalibrated config
  already at GSM8K/needle parity there is currently nothing it needs to
  repair
  ([`probes/nvfp4_hadamard_probe.py`](../probes/nvfp4_hadamard_probe.py)).

## vLLM#46329 — calibration FYI draft

- **2026-08-02** — a short FYI comment distilling the same calibration
  finding for the consumer-Blackwell NVFP4-KV PR thread was drafted and
  reviewed. Parked, not posted — held back pending further review.

## vLLM#50288 — sm121/GB10 GDN prefill datapoint

- **2026-08-11** — comment posted with a GB10 (sm121) prefill-throughput
  datapoint for the GDN (Gated-DeltaNet FlashInfer kernel vs. Triton/FLA)
  prefill path: measured on one box with the fleet down, identical config
  across arms apart from the GDN gate; kernel engagement confirmed per arm
  in the serving log (`FlashInfer GDN prefill kernel` vs. `Triton/FLA GDN
  prefill kernel`). Prefill throughput (prompt_tokens / TTFT at
  `max_tokens=1`, unique-prefix prompts):

  | Prompt tokens | Triton/FLA | FlashInfer GDN | Δ |
  |---|---|---|---|
  | ~29k | 1360.2 tok/s | 1445.9 tok/s | +6.3% |
  | ~111k | 867.4 tok/s | 899.6 tok/s | +3.7% |
  | ~208k | 609.2 tok/s | 625.8 tok/s | +2.7% |

  Real but modest, shrinking with context length; output stayed coherent on
  both arms, no correctness regression. This is a datapoint, not itself a
  production decision — the gain was measured as one candidate (a 9th
  commit on top of the adopted 8-commit ch2lab stack), evaluated separately
  for production adoption.
