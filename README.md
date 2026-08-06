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
| **DGX Spark / GB10** (128 GB unified) | quality → **speed** → memory | MTP speculative decoding (+69% tok/s); memory is the abundant resource |

Same model, same quantization stack — opposite settings, each derived from
measurement rather than habit.

## Findings so far

| Date | Finding | Evidence |
|---|---|---|
| 2026-08-01 | **KV-cache calibration hurts at 4 bit.** Per-tensor amax scales, best practice at fp8, erase the value-cache *bulk* of attention-sink layers at NVFP4 (block scales underflow fp8-e4m3): calibrated 93.25% vs uncalibrated **94.92%** GSM8K, full 1319-item split, McNemar p=0.013. Serve uncalibrated; let scale 1.0 clip the rare outliers — needle retrieval to 240k tokens shows no damage. | [`probes/nvfp4_calib_scale_study.py`](probes/nvfp4_calib_scale_study.py) · [`results/`](results/) · [discussion](https://github.com/vllm-project/llm-compressor/issues/2936) |
| 2026-08-06 | **Hadamard rotation does not rescue amax calibration.** Real sink tokens carry *multiple* outlier channels; their rotated contributions add up and block scales still underflow (99.8% bulk erasure even at moderate sinks). Rotation + scale 1.0 preserves outliers 5.7× better at +8% relative bulk error — a tool worth keeping, but with quality already at parity there is nothing to repair. | [`probes/nvfp4_hadamard_probe.py`](probes/nvfp4_hadamard_probe.py) · [`results/RESULT_nvfp4_hadamard_probe.json`](results/RESULT_nvfp4_hadamard_probe.json) |
| 2026-08-06 | **GB10 production numbers** (27B model, NVFP4 KV, MTP): ~20 tok/s per session, **135 tok/s aggregate at 8 concurrent sessions** with sub-second first-token latency on short prompts; prefill ~1,200 tok/s up to 32k context, degrading to ~510 tok/s at 229k (first token after ~7.5 min at full context — physics, not misconfiguration). | [`benchmarks/spark_bench.py`](benchmarks/spark_bench.py) · [`results/`](results/) |
| 2026-08-06 | **Unified-memory gotcha:** on GB10, vLLM's `gpu-memory-utilization` sizes the KV pool against whatever else is resident — same value, wildly different pools depending on service start order. Use `--kv-cache-memory-bytes` for reproducible sizing (and expect ~10% loss to block rounding + speculative-decoding layers). | [`recipes/`](recipes/) |

## Layout

- **`probes/`** — model-free kernel round-trip studies (run against an installed
  vLLM binding in minutes; deterministic, self-contained)
- **`results/`** — raw JSON exactly as measured, provenance included
- **`benchmarks/`** — end-to-end serving benchmarks (TTFT, tok/s, concurrency)
- **`recipes/`** — the serving configurations the measurements led to

## Method

Measure, then believe. Screening at n=250 is triage only (±2.7 pp on GSM8K);
verdicts take the full n=1319 split and paired statistics. Mechanisms get
model-free kernel round-trips so they stand independent of any benchmark.
When a result kills a nice idea, the idea stays dead and the numbers stay up.

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
