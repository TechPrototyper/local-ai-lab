# The workload acceptance test that almost fired a false rollback (2026-08-21/22)

The DFlash2 adoption had passed every generic gate (GSM8K n=1319 paired,
tool-calling canon, batch sweeps). History says that's not enough: an
earlier speculation config broke *real agent chains* while every generic
benchmark stayed green. So the final tier was a **workload acceptance
test**: 88 tasks distilled from this lab's real Kanban board (goal
triage, resource assignment, status classification, decomposition —
ground truth = the actual labels), both configs paired at temperature 0.

## Act one: a significant regression (p = 0.039)

| Arm | Config | exact set-match |
|---|---|---|
| A | new prod (fp8 KV + DFlash2 n=7) | 28/82 |
| B | old baseline (NVFP4, no spec) | 35/82 |

Paired discordance 1:8 for the old config, McNemar exact **p = 0.039**.
A significant quality regression on the real workload — rollback case,
apparently.

## Act two: the decomposition weakens it

Arm C (fp8 KV, *no* speculation) splits the difference: C-vs-B isolates
the KV dtype (1:5, p = 0.22), C-vs-A isolates speculation (4:1 to
no-spec, p = 0.38). Two same-direction sub-significant effects — 
suspicious, but not yet damning.

## Act three: the control kills it

**Arm A2 — the identical production config, run a second time — flips
24 of 82 answers against itself** (28 → 32 correct; discordance 3:7,
p = 0.34). Against this repeat, the "significant" arm B is
indistinguishable (5:2, p = 0.45). The task class — long reasoning
chains ending in short bistable decisions — is so trajectory-unstable
under concurrent batching (c=4) that **any single run against any other
single run flips ~25–30% of answers, same config or not.** Act one's
p = 0.039 was one weak draw from a noisy distribution.

## Act four: determinism as the referee

At c=1 sequential, greedy is deterministic — differences between arms
are then *real* config effects. Result: the configs genuinely diverge on
**28 of 82 trajectories**, but the quality delta is 5 discordant cases,
all in the old config's favor, **p = 0.0625** — a directional trend
below significance at n=88. Meanwhile c1-vs-c4 of the *same* config
flips 25 answers with perfectly symmetric quality (4:3, p = 1.0):
batching moves trajectories just as much, without moving quality.

## Verdict and lessons

**Adoption stands.** Quality-equal within measurement power on the real
workload (weak 0:5 trend filed as a watch item for a larger task set),
quality-favoring-DFlash2 on GSM n=1319, 2.75–4× the speed.

The transferable lessons outrank the verdict:

1. **A significant McNemar between two configs is uninterpretable
   without a same-config repeat control** when tasks are bistable and
   serving is batched. Our control was the single most important run of
   the evening — it converted a would-be rollback into a noise
   diagnosis.
2. **Bistable judgment tasks amplify batch numerics** far beyond what
   token-level benchmarks suggest: ~25–30% answer-flip rate between any
   two batched runs (byte-identical tool-call canons and needle tests
   coexist with this — different failure surface).
3. **c=1 sequential is the referee** for config comparisons on such
   tasks: it removes the noise floor entirely and shows the real (here:
   small) effect.

Raw: [`results/RESULT_kanban_ab_summary.json`](../results/RESULT_kanban_ab_summary.json)
(per-arm aggregates + all pairwise discordance tables + per-item
prediction/ground-truth label sets). The full per-item transcripts stay
internal — the tasks are distilled from a real project board and the
answers quote its content.
