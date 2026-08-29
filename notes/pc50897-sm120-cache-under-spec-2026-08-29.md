# #50897 on sm120 — prefix caching hits under speculation on the RTX 5090 (2026-08-29)

*The sm120 counterpart to the 08-28 sm121 validation. Confirms vllm#50897
lifts the prefix-cache-under-speculation gate on the consumer RTX 5090,
not just the GB10.*

## Setup

- **Image:** overlay-built in-cluster (kaniko) — vLLM v4@`2cf8b8a`
  (#46329 NVFP4-KV + #53977/8/9 non-causal spec seam) **∪ vllm#50897**
  (EAGLE/successor-aware prefix-cache hashing) **∪ the jethac SWA guard**
  (`61f980bb5`). Python-only overlay onto the compiled v4 base — no kernel
  recompile. Tag `sm120-v4-pc50897-guard-61f980bb-v2`.
- **Serve (neo26 / RTX 5090):** `scout-aqua-20gb` target + `qwen3.8-27b-dflash2`
  drafter (DFlash2 spec, 7 tokens), **fp8 KV**, prefix caching on, 8k ctx.
- **Probe:** the same long (~250-fact) prefix sent repeatedly; read the
  engine's prefix-cache hit rate and spec metrics.

## Result

| Metric | Value |
|---|---|
| **Prefix cache hit rate under spec** | **37.8% → 48.7%** (climbs with each replay of the shared prefix) |
| Spec mean acceptance length | 7.50 → 8.00 |
| Draft acceptance rate | 92.9% → 100.0% |
| Per-position acceptance | mostly 1.000 |
| Requests | 200 OK, fp8-KV + DFlash2 serves clean on sm120 |

Pre-#50897 this was **0% under spec** (the #52244 gate). With #50897 the
cache **hits and climbs** while speculation runs healthy — the two compose.

## Reading (defensive)

On the RTX 5090, prefix caching now **appears to hit under DFlash2
speculation** — the hit rate is >0 and rising toward the shared-prefix
fraction as replays accumulate (37.8→48.7% over a handful of requests on a
cold cache), with spec acceptance unaffected (>90%). This mirrors the
sm121 result (0 → 90.9% replay hits) and suggests the agent-tier
"Entscheidung B" tradeoff — cache **or** spec — **could now be cache
*and* spec** on the 5090. Full production adoption still wants a longer
warm-cache run and the verdict-tier quality gate; this is the enabling
datapoint, not the adoption decision.

## Side-finding (guard validated)

Running the same config with **NVFP4-KV** raised the new SWA guard exactly
as designed: `NotImplementedError: FA2 NVFP4-KV non-causal prefill with a
sliding window ...`. The `qwen3.8-27b-dflash2` drafter is a **sliding-window
(SWA) DFlash model** (`window_left >= 0`), so NVFP4-KV + DFlash2 spec on
sm120 hits jethac's unvalidated non-causal+SWA path — which the guard
correctly blocks. So the fp8-KV run above is the guard-free path for the
cache test; NVFP4-KV + this drafter needs either a full-attention drafter
or the Option-2 symmetric-mask fix before it can serve.

## Provenance

- Metrics: `scratchpad/night-results/sm120_pc50897_metrics.txt`
- Image build: kaniko overlay on `sm120-v4-2cf8b8a` base, in-cluster (neo26).
- Companion sm121 validation: [`night-2026-08-28-pc50897-scout-h2h.md`](night-2026-08-28-pc50897-scout-h2h.md)
