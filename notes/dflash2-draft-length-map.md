# DFlash2: the draft-length × concurrency map — n=7 dissolves the inversion (2026-08-21)

Morning window, three questions closed: where does the n=15 gain invert,
what does draft length do under load, and does the GDN first-step crash
reproduce.

## The map (aggregate tok/s, GSM prompts, greedy, fresh boot + warmup per arm)

| Concurrency | Baseline | DFlash2 **n=7** | DFlash2 n=15 | DFlash2 n=23 |
|---|---|---|---|---|
| 1 | 10.7 | 38.1 | 48.3 | 40.8 |
| 8 | 60.4–70.9 | **179.1** | 128.0–134.9 | 78.2 |
| 12 | 82.7 | ~226 ¹ | 130.0 | — |
| 16 | 111.7 | **226.9** | 134.5 | — |
| 24 | 140.7 | **215.6** | 132.8 | — |

¹ 19 requests instead of 24 (an offset mistake exhausted the question
pool; the number is real but thinner —
[`results/RESULT_sweep_nspec7_hi.json`](../results/RESULT_sweep_nspec7_hi.json)).

Readings:

- **n=15 plateaus at ~130–135 from c=8 on and is overtaken by the
  baseline between c=16 and c=24.** That was the inversion first light
  saw. It is not a hardware ceiling —
- **n=7 blows through it: peak ~227 tok/s at c=16, still 215.6 at c=24
  (1.5× baseline), no inversion anywhere in the measured range.** Long
  drafts are the waste: at high concurrency the drafter's speculative
  tokens compete with batch decode, and 8 fewer of them per step turn
  the plateau into headroom. Acceptance at n=7 stays high (mean length
  5.11, 58.7% draft acceptance vs ~29% at n=15).
- **n=23 is strictly worse under load** (78.2 at c=8) and no better
  single-stream — draft length has an interior optimum, and it sits low.
- The single-stream cost of n=7 vs n=15 (38 vs 48 tok/s) is the only
  trade — and both are 3.5–4.5× the baseline's 10.7.

**Config recommendation for the GB10:** `num_speculative_tokens: 7` —
agents run concurrent far more than solo, and n=7 is the arm that never
loses. (A per-load adaptive draft length would beat both; that's an
upstream feature, not a config.) Quality is unaffected by draft length
(verified-lossless speculation; the n=1319 verdict was measured at n=15
and transfers).

## GDN first-step crash: repro attempt negative

Two controlled cycles on the production engine
([`results/RESULT_gdn_repro.json`](../results/RESULT_gdn_repro.json)):
restart **without** warmup + immediately 2 concurrent prefills → **no
crash**; restart **with** warmup + same load → stable. So "first step ×
concurrent prefills" alone does not reproduce the 08-20 incident (repro
count stays n=1). The suspect list narrows toward a co-factor the
incident had and the repro lacked: a **co-resident GPU process**
(a short-lived container with device access ran during the original
first step). The warmup mitigation stays deployed — one request, zero
cost, closes the window whatever the trigger combination is. Upstream
report stays on hold until a positive repro exists.
