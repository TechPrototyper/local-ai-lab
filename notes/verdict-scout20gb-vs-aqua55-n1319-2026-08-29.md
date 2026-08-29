# Verdict — Scout-20GB vs AQUA-5.5, GSM8K n=1319 (2026-08-29)

*Paired, verdict-tier follow-up to the 08-28 H2H triage
([`night-2026-08-28-pc50897-scout-h2h.md`](night-2026-08-28-pc50897-scout-h2h.md),
where the pool-level p=1.0 / +39.4% first appeared). Same greedy config,
same 1319 GSM8K items, run back-to-back on the same RTX 5090 (sm120)
through the same service, weights the only variable.*

## Result

| Arm | Weights | GSM8K acc (n=1319) | wrong | truncated | errors |
|---|---|---|---|---|---|
| Scout-20GB | ~20 GB | **0.9757** (1287/1319) | 32 | 11 | 0 |
| AQUA-5.5 | ~23 GB | **0.9757** (1287/1319) | 32 | 11 | 0 |

Paired contingency (same items, greedy, c=1 referee):

|  | AQUA correct | AQUA wrong |
|---|---|---|
| **Scout correct** | 1280 | 7 |
| **Scout wrong** | 7 | 25 |

- Discordant pairs: **14** (7 Scout-only misses, 7 AQUA-only misses) —
  **symmetric**.
- **McNemar exact, two-sided: p = 1.0000.**
- Shared misses: 25 items both arms get wrong.

## Reading (defensive)

At n=1319 the two arms are **statistically indistinguishable** — the test
could not detect any accuracy difference (p=1.0), and the disagreement it
did find is symmetric (7↔7), which is what same-config greedy **bistable
flip noise** looks like rather than a directional quality change. The
seven items each arm uniquely misses look like the coin-flip tail on
borderline problems, not a systematic regression on either side (a
same-config repeat control would be the clean way to bound that noise
floor; the symmetry already argues for it).

So, stated conservatively: the ~3 GB the Scout build saves (20 vs 23 GB)
**does not appear to cost measurable task accuracy** at this tier. The
degradation this run was designed to catch did not appear.

## Scope / honesty

- **One task, one metric.** GSM8K accuracy only; tool-calling, long-context
  needle, and other axes are not in this verdict (they were clean at the
  n=250 triage tier but are not re-measured here).
- **Greedy, single referee.** No sampling spread; the flip-noise caveat
  above is the reason the symmetric discordance shouldn't be over-read.
- **sm120 only.** Not a cross-arch claim.
- This is an accuracy verdict, **not** a speed claim. The +39.4% decode-pool
  figure lives in the 08-28 night note and is a separate axis.

## Provenance

- Scout: `results/RESULT_sm120-scout20gb-n1319.json`
- AQUA: `results/RESULT_sm120-aqua55-n1319.json`
- Both via the cluster-internal `quality_battery.py` (concurrency 4,
  `--gsm8k-n 1319`), v4 image, KV NVFP4 pinned, no spec, greedy.
