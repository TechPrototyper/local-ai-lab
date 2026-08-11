# GDN prefill kernel — discarded for production

**Status:** measured, discarded for production (2026-08-11).

## Measurement

GDN (Gated-DeltaNet) prefill-kernel enablement for Qwen3.6's linear-
attention layers, tested as a candidate 9th commit on top of the adopted
8-commit ch2lab stack (see
[`recipes/build-stack.md`](../recipes/build-stack.md)). Measured on GB10
(sm121), fleet down, box to itself, identical config across arms (util
0.40, NVFP4-KV, prefix caching, MTP off) — only the GDN gate flipped.
Kernel engagement confirmed per arm in the serving log:
`FlashInfer GDN prefill kernel` vs. `Triton/FLA GDN prefill kernel`.
Prefill throughput = prompt_tokens / TTFT at `max_tokens=1`, unique-prefix
prompts (no prefix-cache hit):

| Prompt tokens | Triton/FLA | FlashInfer GDN | Δ |
|---|---|---|---|
| ~29k | 1360.2 tok/s | 1445.9 tok/s | +6.3% |
| ~111k | 867.4 tok/s | 899.6 tok/s | +3.7% |
| ~208k | 609.2 tok/s | 625.8 tok/s | +2.7% |

Output stayed coherent on both arms; no correctness regression.

## Decision

**Discarded for production.** The gain is real but modest and shrinks with
context length; not worth carrying a 9th commit (and its maintenance
surface) on top of the 8-commit ch2lab stack for a low-single-digit-percent
win at the context lengths this lab actually serves (up to 262k). Posted
as a datapoint on vLLM#50288 — see
[`upstream-contributions.md`](upstream-contributions.md).
