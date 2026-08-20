# DFlash2: the batch sweep reverses a finding; the skip-layers hybrid dies honestly (2026-08-20, evening)

Two results from the evening window, one correction and one closed door.

## 1. Correction: DFlash2 wins at every measured concurrency

[First light](dflash2-sm121-first-light.md) concluded "a latency tool, not
an aggregate tool" from one concurrent battery run. The controlled sweep
(GSM prompts, greedy, identical prompt sets both arms, fresh boot + warmup
each arm, c×3 requests per tier) says otherwise:

| Concurrency | DFlash2 fp8-KV (agg tok/s) | Baseline NVFP4-KV | Factor |
|---|---|---|---|
| 1 | **48.3** | 10.7 | 4.5× |
| 2 | 68.6 | 20.3 | 3.4× |
| 4 | 103.4 | 38.0 | 2.7× |
| 8 | **134.9** | 70.9 | 1.9× |

Per-session at c=8: 19.4 vs 10.0 tok/s. The gain shrinks with load but
never inverts through c=8 — 134.9 tok/s aggregate is the multi-agent
range this box was hoped to reach. The first-light inversion (acceptance
~2.7, slower than baseline) came from a mixed battery at higher effective
concurrency on a long-lived engine; it stands as a caveat for c>8 and
mixed workloads, not as the headline. **Revised: DFlash2 is a latency
tool *and* an aggregate tool up to at least 8 concurrent sessions;
beyond that, measure first.** Also note single-stream 48.3 tok/s on
GSM-style prompts — higher than first light's 23–31 on a fresh,
warmed engine.

Raw: [`results/RESULT_sweep_dflash2.json`](../results/RESULT_sweep_dflash2.json),
[`results/RESULT_sweep_base.json`](../results/RESULT_sweep_base.json).

## 2. Closed door: the skip-layers hybrid, three revisions deep

The idea: `--kv-cache-dtype-skip-layers 0 1 2 3 4` exempts the five
drafter layers (plus, collaterally, one target layer) from NVFP4 — target
KV stays 4-bit, the drafter's non-causal attention reads bf16. If it
worked, the 32 GB card would get DFlash2 without the fp8-KV context
penalty.

What happened, in order:

- **REV A (262k context):** the `FlashInfer non-causal attention is not
  supported with NVFP4 KV cache` error is **gone** — the mechanism works.
  But the engine demands **57.3 GiB** of KV for 262k tokens (pool: 15).
- **REV B (32k context):** still demands **41.5 GiB** — not remotely
  proportional (expected ~1/8 of REV A). Estimated max length: 5,504
  tokens. Something fixed-cost dominates.
- **REV C (32k, prefix caching off):** 40.9 GiB, max length 6,112 —
  prefix caching (and its block-alignment mode) is *not* the driver.

Diagnosis: on this hybrid linear-attention model, pages of layers skipped
from KV quantization are padded to align with the state-layer page size
(the hybrid's block granularity is thousands of tokens). The padding —
not the bf16 bytes — eats the pool: ~41 GiB for 32k regardless of
settings. On a 32 GB card the approach disqualifies itself; on the 128 GB
box it loses to plain fp8-KV on every axis.

**Verdict: skip-layers is not the RTX path for hybrid models.** What
remains for the 32 GB card, in order of appeal: NVFP4 non-causal kernel
support (upstream work — the skip-layers result sharpens the case for
it), or context capping with length-based routing of long jobs to the
big-memory box. The Spark adoption question is unaffected: there, fp8-KV
is affordable and the sweep above is the argument.
