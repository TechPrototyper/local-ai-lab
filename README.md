# local-ai-lab

My personal lab notebook: making local AI **measurably** excellent on hardware
people actually own. Spare-time research on my own machines — this repo tracks
what I tried, what I measured, and where it led.

## The framing

Every serving system has one bottleneck that matters most, and the art is to
optimize *that* — not the technology you happen to like:

| Box | Bottleneck order | Consequence |
|---|---|---|
| **RTX 5090** (32 GB consumer GPU) | quality → **memory** → speed | 4-bit KV cache (NVFP4) buys ~4× context; speculative decoding stays *off* (it costs KV) |
| **DGX Spark / GB10** (128 GB unified) | quality → **speed** → memory | memory is abundant, so it is spent on long context and throughput (NVFP4 KV, prefix caching, 262k); MTP speculative decoding is *off* — it broke tool-calling on sm121 ([`notes/mtp-tool-calling.md`](notes/mtp-tool-calling.md)) |

Same model, same quantization stack — opposite settings, each derived from
measurement rather than habit.

## Findings so far

Newest first.

| Date | Finding | Evidence |
|---|---|---|
| 2026-08-11 | **Quantization runtime measured, not guessed.** Full PrismaQuant aura pipeline on a 4B model: 24 m 26 s on GB10 (GPTQ production-cache 504 s + KL-adjoint cost 661 s dominate; both scale with linear-module count). Per-linear extrapolation — required because the 27B target is hybrid linear-attention (48/64 layers), which breaks per-parameter scaling — puts a **27B dense run at 1.1–3.4 h** (worst case 12–15 h if the max-act-rows-cap assumption fails). Serial-local quantization is viable; external GPUs are not on the critical path — validation (~8 h per full GSM8K arm) is the real time sink. | [`notes/quant-runtime-probe.md`](notes/quant-runtime-probe.md) |
| 2026-08-06 | **Hadamard rotation does not rescue amax calibration.** Real sink tokens carry *multiple* outlier channels; their rotated contributions add up and block scales still underflow (99.8% bulk erasure even at moderate sinks). Rotation + scale 1.0 preserves outliers 5.7× better at +8% relative bulk error — a tool worth keeping, but with quality already at parity there is nothing to repair. | [`probes/nvfp4_hadamard_probe.py`](probes/nvfp4_hadamard_probe.py) · [`results/RESULT_nvfp4_hadamard_probe.json`](results/RESULT_nvfp4_hadamard_probe.json) |
| 2026-08-06 | **GB10 production numbers** (27B model, NVFP4 KV, MTP): ~20 tok/s per session, **135 tok/s aggregate at 8 concurrent sessions** with sub-second first-token latency on short prompts; prefill ~1,200 tok/s up to 32k context, degrading to ~510 tok/s at 229k (first token after ~7.5 min at full context — physics, not misconfiguration). | [`benchmarks/spark_bench.py`](benchmarks/spark_bench.py) · [`results/`](results/) |
| 2026-08-06 | **Unified-memory gotcha:** on GB10, vLLM's `gpu-memory-utilization` sizes the KV pool against whatever else is resident — same value, wildly different pools depending on service start order. Use `--kv-cache-memory-bytes` for reproducible sizing (and expect ~10% loss to block rounding + speculative-decoding layers). | [`recipes/`](recipes/) |
| 2026-08-01 | **KV-cache calibration hurts at 4 bit.** Per-tensor amax scales, best practice at fp8, erase the value-cache *bulk* of attention-sink layers at NVFP4 (block scales underflow fp8-e4m3): calibrated 93.25% vs uncalibrated **94.92%** GSM8K, full 1319-item split, McNemar p=0.013. Serve uncalibrated; let scale 1.0 clip the rare outliers — needle retrieval to 240k tokens shows no damage. | [`probes/nvfp4_calib_scale_study.py`](probes/nvfp4_calib_scale_study.py) · [`results/`](results/) · [discussion](https://github.com/vllm-project/llm-compressor/issues/2936) |

## Layout

- **`probes/`** — model-free kernel round-trip studies (run against an installed
  vLLM binding in minutes; deterministic, self-contained)
- **`results/`** — raw JSON exactly as measured, provenance included
- **`benchmarks/`** — end-to-end serving benchmarks (TTFT, tok/s, concurrency)
- **`recipes/`** — the serving configurations the measurements led to
- **`notes/`** — internal findings and decisions that aren't a probe/
  benchmark/recipe on their own (production decisions, incident write-ups,
  cross-references, the upstream-engagement record)

## Method

Measure, then believe. Screening at n=250 is triage only (±2.7 pp on GSM8K);
verdicts take the full n=1319 split and paired statistics. Mechanisms get
model-free kernel round-trips so they stand independent of any benchmark.
When a result kills a nice idea, the idea stays dead and the numbers stay up.

## Changelog

Retroactive back to the campaign start; going forward this table gets a row
per change to this repo. Newest first.

| Date | Change | Why | Detail |
|---|---|---|---|
| 2026-08-12 | Findings and changelog re-sorted newest-first; changelog "Detail" column linkified to match the findings table | surface recent changes faster; consistent linking across both tables | none |
| 2026-08-11 | Quantization runtime probe: 4B pipeline run measured on GB10, extrapolated to 27B dense (1.1–3.4 h) | kill the last runtime unknown before scheduling next-model quant windows; decides that external GPUs are off the critical path | [`notes/quant-runtime-probe.md`](notes/quant-runtime-probe.md) |
| 2026-08-11 | `recipes/build-stack.md` mount paths genericized to `$WORK_DIR` | keep host-specific paths out of the public notebook | [`recipes/build-stack.md`](recipes/build-stack.md) |
| 2026-08-11 | Framing table (Spark row) rewritten to lead with the current config; MTP-off demoted from headline to a linked reference | the front table should show the current state, not a disabled-feature narrative | [`notes/mtp-tool-calling.md`](notes/mtp-tool-calling.md) |
| 2026-08-11 | Changelog table added to this README | establish changelog discipline going forward | none |
| 2026-08-11 | Build stack documented (image line, 8-commit ch2lab stack, FlashInfer pins, local patch) | capture the current source-of-truth for the sm12x custom build | [`recipes/build-stack.md`](recipes/build-stack.md) |
| 2026-08-11 | GDN sm121 prefill datapoint measured, posted on vLLM#50288, discarded for production | +2.7–6.3% prefill, shrinking with context — not worth a 9th ch2lab commit | [`notes/gdn-prod-decision.md`](notes/gdn-prod-decision.md) |
| 2026-08-11 | Overflow banner removed from the LiteLLM callback (cluster-side) | banner text looping through replayed context caused a multi-hour generation runaway | [`notes/banner-runaway.md`](notes/banner-runaway.md) |
| 2026-08-11 | GB10 OOM-hardening (swappiness=10, earlyoom, per-container caps, oom_score_adj) | repeated thrash-wedge required physical reboots | [`notes/gb10-oom-hardening.md`](notes/gb10-oom-hardening.md) |
| 2026-08-11 | `recipes/dgx-spark-sm121.md` corrected to match the on-box script (MTP off, current flags) | recipe had drifted from source-of-truth | [`recipes/dgx-spark-sm121.md`](recipes/dgx-spark-sm121.md) |
| 2026-08-09 | Prefix caching enabled on sm121 | avoid re-prefilling the full agent context every turn | [`recipes/dgx-spark-sm121.md`](recipes/dgx-spark-sm121.md) |
| 2026-08-09 | MTP speculative decoding disabled on sm121 (Spark) | `qwen3_5_mtp` × tool-calling caused empty / aborting tool-call chains | [`notes/mtp-tool-calling.md`](notes/mtp-tool-calling.md) |
| 2026-08-06 | GB10 production benchmark captured (NVFP4-KV + MTP, 262k context) | validate the sm121 target config before rollout | [`benchmarks/spark_bench.py`](benchmarks/spark_bench.py) |
| 2026-08-06 | Hadamard/R2-rotation probe run; follow-up comment posted on llm-compressor#2936 | test a SpinQuant/R2 suggestion against the calibration finding | [`probes/nvfp4_hadamard_probe.py`](probes/nvfp4_hadamard_probe.py) |
| 2026-08-02 | FYI draft for vLLM#46329 written and reviewed, parked (not posted) | same finding, PR context — held back pending further review | [`notes/upstream-contributions.md`](notes/upstream-contributions.md) |
| 2026-08-02 | Calibration finding posted as a comment on llm-compressor#2936 | share a measured KV-cache-side NVFP4 accuracy contributor | [`notes/upstream-contributions.md`](notes/upstream-contributions.md) |
| 2026-08-01 | NVFP4 KV-cache calibration campaign run; baked amax scales found to hurt at 4-bit | investigate a reported quality deficit on calibrated NVFP4-KV | [`probes/nvfp4_calib_scale_study.py`](probes/nvfp4_calib_scale_study.py) |

## Context

Kernel and serving work this builds on: FlashInfer sm12x kernels
([flashinfer#3684](https://github.com/flashinfer-ai/flashinfer/pull/3684)),
vLLM consumer-Blackwell NVFP4-KV enablement
([vllm#46329](https://github.com/vllm-project/vllm/pull/46329)), and the
calibration discussion in
[llm-compressor#2936](https://github.com/vllm-project/llm-compressor/issues/2936).

---

*Personal project, run on my own time and hardware; unrelated to my day job.
Licensed Apache-2.0.*
