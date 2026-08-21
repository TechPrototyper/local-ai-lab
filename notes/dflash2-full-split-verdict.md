# DFlash2 full-split verdict: n=1319, PASS — direction favors the drafter (2026-08-21)

The verdict-level gate per this lab's [method](../README.md#method):
GSM8K **full test split, n=1319 paired**, greedy, both arms the complete
production parser config, same runner, same concurrency (4), sequential
overnight on the GB10 with a fresh boot + warmup per arm.

| Arm | Accuracy | Wall-clock |
|---|---|---|
| Baseline (NVFP4 KV, no speculation) | 1255/1319 = **95.15%** | 16,214 s |
| DFlash2 (fp8 KV, spec n=15) | 1264/1319 = **95.83%** | **5,893 s (2.75×)** |

Discordant pairs: 9 where only the baseline was right, **18 where only
DFlash2 was right** — McNemar exact p = 0.122. No significant difference;
what direction there is favors the speculative arm. (No mystery: greedy
tie-flips land on both sides; verified-lossless speculation cannot
systematically improve output — this is the batch-numerics class again,
now with n large enough to say so.) Mean acceptance length over the run:
4.34 at concurrency 4.

With this, every gate this lab set for DFlash2 on the GB10 has been
passed at verdict level:

- **Quality:** n=250 screening p=0.375, n=1319 full split p=0.122 —
  equal within paired statistics both times.
- **Tool-calling:** conformance suite passed, byte-identical to the
  non-speculative canon in 4/5 cases; the ngram parser corruption that
  poisoned earlier speculation attempts is absent.
- **Throughput:** faster at every measured concurrency ≤8 (sweep:
  4.5× at c=1 down to 1.9× at c=8, 134.9 tok/s aggregate).

The cost stays what it was: **fp8 KV instead of NVFP4** (the non-causal
drafter cannot read NVFP4 KV on sm12x), roughly halving KV token
capacity — affordable on the 128 GB box, disqualifying on the 32 GB
card. Production adoption on the GB10 is now an operator decision, not a
measurement question. Remaining measurement items: the c>8 inversion
boundary, and the NVFP4 non-causal kernel question for the RTX.

Raw: [`results/RESULT_full_base1319.json`](../results/RESULT_full_base1319.json) ·
[`results/RESULT_full_dflash1319.json`](../results/RESULT_full_dflash1319.json) ·
[`results/FULL_VERDICT.json`](../results/FULL_VERDICT.json)
