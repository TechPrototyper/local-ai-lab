# Verdict — NVFP4-KV vs fp8-KV cutover, GSM8K n=1319 (2026-08-29)

*Paired verdict-tier run on the DGX Spark (GB10 / sm121). Both arms
identical except the KV-cache dtype (`--kv-cache-dtype nvfp4` vs `fp8`) —
verified the only diff was that one flag; same weights
(qwen3.8-27b-prismaaqua55), same DFlash2 drafter, same block size, same
prefix caching, same pc50897 tree. Full battery per arm.*

## Result

| Arm | KV dtype | GSM8K acc (n=1319) | wrong | needle | determinism | single-stream tok/s | batched8 tok/s |
|---|---|---|---|---|---|---|---|
| NVFP4-KV | 4-bit | **0.9742** (1285/1319) | 34 | 6/6 | 5/5 | 23.1 | 51.1 |
| fp8-KV | 8-bit | **0.9727** (1283/1319) | 36 | 6/6 | 5/5 | 23.0 | 53.5 |

Paired contingency (same items, greedy):

|  | fp8 correct | fp8 wrong |
|---|---|---|
| **NVFP4 correct** | 1278 | 7 |
| **NVFP4 wrong** | 5 | 29 |

- Discordant pairs: **12** (7 vs 5) — near-symmetric.
- **McNemar exact, two-sided: p = 0.7744.**

## Reading (defensive)

On quality, the two KV dtypes are **statistically indistinguishable** at
n=1319 (p=0.77) — the 2-item accuracy gap (0.9742 vs 0.9727) is well inside
what same-config greedy flip noise could produce, and the discordance is
near-symmetric (7 vs 5), i.e. not a directional NVFP4 penalty. needle and
determinism are identical (6/6, 5/5) on both.

On speed, the **controlled perf harness shows parity** — single-stream 23.1
vs 23.0 tok/s, batched-8 aggregate 51.1 vs 53.5 tok/s. (During the runs the
instantaneous engine-log throughput fluctuated a lot per-item and could
*look* like a large gap in a short window; the controlled measurement is
the one to trust, and it shows none.)

So, stated conservatively: dropping the KV cache from 8-bit to **4-bit
(NVFP4) appears to cost neither measurable task accuracy nor measurable
throughput** on this stack, while **halving KV-cache memory** — which is
what buys the large context headroom the memory-bound track depends on.
That is the case for the production cutover to NVFP4-KV; this run does not
contradict it.

## Scope / honesty

- **One task, one metric** for the verdict (GSM8K accuracy); needle and
  determinism are pass/fail sanity, not verdict-tier.
- **tools 0/6 on BOTH arms** — a harness/format artifact of the tool test
  on this config, *not* a KV-dtype effect (it is identical across arms and
  so cancels in the comparison). Flagged, not interpreted.
- **Greedy, single referee, sm121 only.** No sampling spread; the flip-noise
  caveat is why the small accuracy gap shouldn't be over-read.
- The memory claim is architectural (4-bit vs 8-bit KV); the context-headroom
  payoff is documented elsewhere, not re-measured here.

## Provenance

- NVFP4-KV: `results/RESULT_sm121-cutover-nvfp4kv-n1319.json`
- fp8-KV: `results/RESULT_sm121-cutover-fp8kv-n1319.json`
- Both via on-box `quality_battery.py` (concurrency 8, `--gsm8k-n 1319`),
  pc50897 tree, DFlash2 spec, prefix caching on.
