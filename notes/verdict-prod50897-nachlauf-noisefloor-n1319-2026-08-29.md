# Verdict — prod-50897 nachlauf + noise floor, GSM8K n=1319 (2026-08-29)

*Third n=1319 run of the night, on the DGX Spark (GB10 / sm121). It is a
**fresh, independent run of the exact production config** deployed 08-28
(NVFP4-KV + DFlash2 + prefix caching on the #50897 tree,
`spark_pc50897_arm.sh`). It serves two purposes: (a) confirm the deployed
#50897 prefix-cache fix holds at verdict tier, and (b) act as a
**same-config repeat control** — the run-to-run noise floor against which
the two equivalence verdicts of the night should be read.*

## Deployment confirmation

| Axis | Result |
|---|---|
| GSM8K acc (n=1319) | **0.9704** (1280/1319), 0 errors |
| Needle (12k/24k words × depths 0.1/0.5/0.9) | **6/6** — context retrieval intact under prefix caching |
| Determinism (5 prompts × 3 repeats) | **5/5** — identical output hashes every repeat |
| perf | single-stream 23.3 tok/s, batched-8 51.6 tok/s |
| tools | 0/6 — **harness artifact**, `400 Bad Request` before the model runs (see below) |

The production config holds: greedy output is deterministic across repeats,
long-context needle retrieval is exact under prefix caching, and accuracy
sits where the other NVFP4-KV arm sits. Nothing here contradicts the 08-28
rollout.

## The noise floor (why this run matters)

This nachlauf and the cutover NVFP4-KV arm are the **same config run twice**.
Paired McNemar between them:

| | nvfp4-arm correct | nvfp4-arm wrong |
|---|---|---|
| **nachlauf correct** | 1275 | 5 |
| **nachlauf wrong** | 10 | 29 |

- acc 0.9742 (arm) vs 0.9704 (nachlauf); **15 discordant** (10 vs 5);
  McNemar exact **p = 0.3018**.

Read that against the night's two real comparisons:

| Comparison | discordant | asymmetry | McNemar p |
|---|---|---|---|
| **same config, twice** (this control) | 15 | 10:5 | **0.30** |
| Scout-20GB vs AQUA-5.5 | 14 | 7:7 | 1.00 |
| NVFP4-KV vs fp8-KV | 12 | 7:5 | 0.77 |

**Two byte-identical configs disagree on 15 items and swing ~0.4 pp — and
that repeat produces a *lower* p (more apparent divergence) than either
weights- or KV-dtype comparison did.** So the "differences" in the two
equivalence verdicts are smaller than what pure run-to-run greedy
flip-noise generates. The noise floor is ~15 discordant items / ~0.4 pp /
p≈0.3; anything inside that is not signal. Both verdicts (p=1.0, p=0.77)
sit comfortably inside it.

## On the tools 0/6 (all three arms)

Every arm scored tools 0/6 with detail `400 Client Error: Bad Request` on
`/v1/chat/completions` — the request is rejected at the API layer before
the model generates. It is a harness/serve-config mismatch on the
tool-call schema for this deployment, **identical across arms**, and so
cancels out of every paired comparison. Flagged, not interpreted as a
quality signal.

## Provenance

- `results/RESULT_sm121-prod50897-nachlauf-n1319.json`
- vs `results/RESULT_sm121-cutover-nvfp4kv-n1319.json` (the same-config pair)
- On-box `quality_battery.py`, full suite, concurrency 8, `--gsm8k-n 1319`,
  `spark_pc50897_arm.sh` (NVFP4-KV + DFlash2 + prefix caching, pc50897 tree).
