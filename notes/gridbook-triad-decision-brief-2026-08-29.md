# GridBook triad — decision brief (2026-08-29)

*One-page prep for a Tim decision. Not a published claim; defensive throughout.*

## The question on the table

Is the **RTX "Endgegner"** now runnable on the agent tier — GridBook 13 GB
weights + NVFP4 KV cache + DFlash2 speculation, *with prefix caching live*
— given that the one thing that forced the 08-24 rollback may have been
fixed today?

## What changed today that touches this

The 08-24 triad rollback had a single, precise cause (not the weights, not
the KV, not the composition): **prefix caching returned 0% hits under
DFlash2 speculation on hybrid-GDN models**, tracked upstream as
[vllm#52244](https://github.com/vllm-project/vllm/pull/52244). On the
agent tier, where harnesses re-send large contexts every turn, cache beats
speculation — so production chose cache, no-spec.

Today we validated and shipped **[vllm#50897](https://github.com/vllm-project/vllm/pull/50897)**
(successor-aware / EAGLE-style cache hashing) to Spark prod. #50897
addresses the *same* prefix-cache-under-speculation failure that #52244
was chartered for, and on our replay probe it restored **90.9% replay
hits under spec on both arches** (up from the 0% that caused the
rollback). In other words: the specific gate that sent the triad back to
AQUA on 08-24 **could now be lifted** — if #50897 carries onto an sm120
build and the hit-rate holds there under the full triad.

## Honest status of each triad leg

| Leg | Where it stands | Confidence |
|---|---|---|
| GridBook 13 GB **quality** | n=250 triage: task-equivalent to the 24 GB prod quant across GSM/tool-call/needle; +4.56% intrinsic PPL that **does not** surface at task level. Measured on **sm121/GB10**. | triage only (±2.7pp) — **no n=1319 verdict yet** |
| **Composes** (all three boot+serve) | Ran together on sm120 on 08-24 (877k-token KV config); spec×NVFP4-KV pair since validated paired + cross-arch (#53977/8/9). | high — it served |
| Prefix cache under spec | 0% → 90.9% replay hits with #50897, both arches, today. | high on sm121; **unproven on sm120 under the full triad** |
| DFlash2 **acceptance** vs GridBook target | Drafter trained on BF16/AQUA distribution; greedy quality unaffected by construction, but acceptance (= speed) could shift against the codebook-quantised target. | **unmeasured** |

## Why it matters (the upside, stated in the conditional)

If GridBook holds at verdict tier **and** #50897's cache fix holds on
sm120 under the triad, then a single RTX 5090 could serve near-BF16
quality at ~13 GB weights, a ~1M-token NVFP4 context, speculation *and*
prefix caching simultaneously — which would raise what a consumer 32 GB
card is worth for agent workloads overnight. That is the whole thesis of
the memory-bound track landing at once. None of it is claimable until the
two open measurements below exist.

## What it would take to decide (smallest path)

1. **sm120 build carrying #50897** on top of the #46329 NVFP4-KV line
   (the Spark build already carries it at `dd02ed4d`; the RTX line does
   not yet). One parametrized `Dockerfile.fullstand`-style build.
2. **Re-run the 08-24 triad config on that build** and confirm prefix
   cache hit-rate under spec on sm120 (the one number that gated
   rollback). A short probe, not a night.
3. **GridBook n=1319** greedy verdict vs the 24 GB prod quant (paired
   McNemar) — the quality tier the triage doesn't reach. This is a GPU
   night, and it is a natural **fourth arm** to fold into the verdict
   batch already structured for Scout/AQUA/NVFP4-cutover.
4. *(diagnostic, not gating)* DFlash2 acceptance-rate against the
   GridBook target, to know whether the speed leg carries or the drafter
   needs a retarget.

Steps 1–2 are cheap and answer "is the gate really lifted on RTX?".
Step 3 answers "is 13 GB really lossless at verdict tier?". Only both
green would justify promoting the triad to the agent tier.

## Recommendation (for Tim, not acted on)

Fold **GridBook n=1319** into the next verdict batch as a fourth arm
(cheap incremental — the harness is already standing), and schedule the
**sm120 #50897 cache-probe** as a short daytime task rather than a night.
Hold triad promotion until both read green. Everything here stays internal
until the numbers exist; if they do, publish notebook-first, conditional
voice, then link.

## Pointers

- Triad ops log + rollback rationale:
  [`gridbook-nvfp4-dflash2-rtx-triad.md`](gridbook-nvfp4-dflash2-rtx-triad.md)
- GridBook quality triage:
  [`gridbook-13gb-quality-holds.md`](gridbook-13gb-quality-holds.md)
- #50897 validation + Spark rollout:
  [`night-2026-08-28-pc50897-scout-h2h.md`](night-2026-08-28-pc50897-scout-h2h.md)
