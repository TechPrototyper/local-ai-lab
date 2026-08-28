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

## flashinfer#3684 — sm120 validation of the NVFP4 VO-split prefill kernel

- **2026-08-06** — comment posted with an independent sm120 (RTX 5090)
  validation of the asymmetric VO-split NVFP4 paged-prefill PR, after the
  project's internal CI lost both 5090 rows to infrastructure failures:
  117,829 attention tests, 8,453 fp4 gemm tests, 125 fp4 moe tests — zero
  failures — plus a `scipy` collection-time gap in `tests/attention` worth
  flagging upstream.
- **2026-08-12** — the run above was taken at commit `dd25a783`; the PR
  head had meanwhile moved to `00054844`, and the author re-ran the suite
  at the current head on a rented 5090 himself (green; two torch versions
  now covered between the two runs). Lesson recorded for this lab's
  process: validation posts pin the current PR head, name the hash, and
  re-check for head movement before posting.
- **2026-08-13** — **PR merged** (`8f9ad200`). With the kernel side
  upstream in flashinfer main, [vLLM#46329](https://github.com/vllm-project/vllm/pull/46329)
  remains the single outstanding piece of the consumer-Blackwell
  (sm120/sm121) NVFP4-KV serving line — the stack this lab has been running
  in production on both arches. Credit where due: the line is
  [@jethac](https://github.com/jethac)'s design and persistence end to end;
  this lab contributed validation datapoints on 5090 and GB10 along the way.

## vLLM#46329 — production datapoint

- **2026-08-11** — comment posted with a production datapoint in support of
  the PR: the stack has been serving Qwen3.6-27B at 262k context with
  `--kv-cache-dtype nvfp4` on an RTX 5090 (sm120, weeks) and a DGX Spark
  GB10 (sm121, since early August) under sustained agentic load. GSM8K
  94.9% (sm120, n=1319, scale 1.0) / 96.4% (sm121, n=250) — no quality
  regression attributable to the 4-bit KV path. PR currently awaiting a
  rebase (merge conflicts since 2026-08-07).

## PrismaQuant PR #80 — hybrid linear-attention masking fix

- **2026-08-12/13** — full cycle from crash to **approved for production**
  in two review rounds; substance and the two generalized lessons (API
  existence is not a compatibility gate; screening greenness only certifies
  exercised paths) recorded in
  [`prismaquant-pr80-review-cycle.md`](prismaquant-pr80-review-cycle.md).

## vLLM #52816 — quantized DFlash2 drafters: fused-scale loading gap

- **2026-08-22** — reported ([comment](https://github.com/vllm-project/vllm/pull/52816#issuecomment-5376688332))
  after the drafter-quant experiment: quantized draft checkpoints load
  only when the quantized layers are unfused; fused qkv/gate_up
  `weight_scale` loading resolves the module instead of the parameter.
  Includes the acceptance-neutral motivation (drafter quantization is
  quality-risk-free by construction) and an sm121 test offer. Backing
  measurement: [`dflash2-drafter-fp8-quant.md`](dflash2-drafter-fp8-quant.md).

## vLLM #53334 — TurboQuant KV: two observations

- **2026-08-22** — filed ([issue](https://github.com/vllm-project/vllm/issues/53334)):
  (1) hybrid linear-attention models fail engine init
  (`Unknown TurboQuant cache dtype: 'auto'`) — asked whether unsupported
  by intent; (2) the value path's fp16 zero-point silently overflows to ∞
  on |min| > 65504 (probe-level, deterministic repro; measured real-world
  sinks reach ~125k). Suggested fp32 scale/zero (+4 B/vector) or a
  clamp-with-warning. Backing probe:
  [`turboquant-vs-nvfp4-kv-value-probe.md`](turboquant-vs-nvfp4-kv-value-probe.md).

## vLLM #53977 / #53978 / #53979 — the non-causal-NVFP4 spec-decode package

- **2026-08-27** — filed as three reviewable PRs after a paired cross-arch
  revalidation on both boxes (fresh sm120 image = main ∪ #46329@`7a5cf14` ∪
  the fixes; sm121 on the production lineage):
  [#53977](https://github.com/vllm-project/vllm/pull/53977)
  (`VocabParallelEmbedding` tp==1 OOB mask + regression test — hits every
  single-GPU spec config, independent of KV dtype),
  [#53978](https://github.com/vllm-project/vllm/pull/53978) (DFlash2 warmup
  OOBs: embed-outside-compile + candidate-selector gather clamp, against the
  merged #52816 line),
  [#53979](https://github.com/vllm-project/vllm/pull/53979) (the non-causal
  FA2-NVFP4 prefill seam, **stacked on #46329** with an explicit fold-in
  offer to its author). Linking datapoint comment in
  [#46329](https://github.com/vllm-project/vllm/pull/46329#issuecomment-5433626787).
  Backing measurement:
  [`../results/RESULT_nvfp4_spec_crossarch_revalidation.json`](../results/RESULT_nvfp4_spec_crossarch_revalidation.json)
  · [`dflash2-nvfp4-sm120-spec-serves.md`](dflash2-nvfp4-sm120-spec-serves.md).

## vLLM #53122 — quantized-drafter fixes: production datapoint + test artifact

- **2026-08-27** — [datapoint posted](https://github.com/vllm-project/vllm/pull/53122#issuecomment-5442968584):
  both walls the PR addresses were hit independently here while quantizing a
  DFlash2 drafter to fp8; with the PR (cherry-picked 2026-08-22) the fully
  quantized drafter serves in production on a GB10 — acceptance-neutral,
  −1.6 GB into the KV pool. Adds fp8 e4m3 per-channel to the thread's
  verified-format matrix.
  [Addendum](https://github.com/vllm-project/vllm/pull/53122#issuecomment-5442978186):
  the checkpoint behind the datapoint is public
  ([`TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm`](https://huggingface.co/TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm))
  as a test artifact for the format class.

## vLLM #53334 (follow-up) — contributor onboarding

- **2026-08-27** — a first-time contributor claimed the issue;
  [pointers posted](https://github.com/vllm-project/vllm/issues/53334#issuecomment-5439861335):
  the two findings are separable, code entry points for each, both
  reproducible without special hardware, plus an sm120/sm121 validation
  offer once a branch exists.

## vLLM #52883 — the fix is needed one file over

- **2026-08-28** — [analysis posted](https://github.com/vllm-project/vllm/pull/52883#issuecomment-5446093771):
  the guard this PR fixes was removed by the merged #52816 refactor (the
  PR's crash scenario passes on current `main` — verified), but the same
  `UnquantizedEmbeddingMethod`-vs-`UnquantizedLinearMethod` confusion
  survives in `LogitsProcessor._apply_head`'s `head_dtype` branch, which
  falsely rejects the same scenario. Retarget recommendation with a
  runnable four-scenario probe
  ([`../probes/lmhead_quant_method_guard_probe.py`](../probes/lmhead_quant_method_guard_probe.py))
  and a re-validation offer.

## vLLM #50897 — cross-arch validation of the prefix-cache×spec fix

- **2026-08-28** — [validation posted](https://github.com/vllm-project/vllm/pull/50897#issuecomment-5448682111):
  the PR ported onto this lab's DFlash2×NVFP4 line (main-derived base:
  ~15 conflicts, 13 mechanical, one semantic seam — resolution described
  in the comment for whoever rebases); exact-replay cache hits
  0 → 11,648/12,812 (90.9%) on sm120 *and* sm121, byte-identical
  completions on every arm, clean n=250 screens. Notably outside the
  MTP/EAGLE family the PR targets (block-diffusion drafter, fp8-quantized
  on sm121). Port branch shared
  (`TechPrototyper/vllm@integrate/dflash2-nvfp4-v4-pc50897`); raw:
  [`../results/RESULT_pc50897_replay_probe.json`](../results/RESULT_pc50897_replay_probe.json).
