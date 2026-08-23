# GridBook 13 GB holds quality against our 24 GB production quant (2026-08-23)

The question: RobTand's GridBook codebook quant compresses Qwen3.8-27B to
**13 GB** (FP8-CB product-codebook), vs our production PrismaAQUA at
**24 GB** (NVFP4 weights + FP8 attention, ~5.5 bpp). Does the smaller
model hold quality? If yes, it roughly doubles the KV headroom on a 32 GB
consumer card — a real step for the RTX 5090.

## Method (honest, leverage-published-then-fill-the-gap)

- **Intrinsic quality: already published by Rob**, on the exact served
  artifact — so we did not re-measure it: KL vs BF16 = **0.0917**
  (top-K K=1024, 98.7% coverage), WikiText-2 perplexity **9.792 vs 9.365
  BF16 = +4.56%**. A modest, real distributional degradation.
- **Task-level quality: nobody had measured it** for GridBook. That is
  the gap we produced. Three axes, both models served **identically**
  (fp8 KV, greedy T=0, no speculation, same items), paired:
  GSM8K n=250 (reasoning), tool-calling conformance (agentic), needle
  recall at ~32k (long-context).
- Reference level: Qwen's official Qwen3.8-27B BF16 sits at ~96% GSM8K.

## Result — indistinguishable on every axis

| Axis | GridBook 13 GB | PrismaAQUA 24 GB | Verdict |
|---|---|---|---|
| GSM8K n=250 | 240/250 = **96.0%** | 241/250 = **96.4%** | McNemar exact **p = 1.0** (discordant 2/3) — no difference |
| Tool-calling conformance | PASS | PASS | equal |
| Needle @ ~32k | recalled | recalled | equal |

The +4.56% intrinsic PPL degradation **does not manifest at task level**:
one-item difference on GSM (noise at n=250), tool-calling clean, needle
exact. The 13 GB model is task-equivalent to the 24 GB production quant.

## Honest scope

- **n=250 is triage** (±2.7 pp), not the n=1319 verdict tier — but the
  triage signal is consistent across three independent axes.
- Measured on **GB10 / sm121**. The RTX payoff (does GridBook run on
  sm120, and does it **compose with NVFP4 KV** for the ~1M-token context
  on a 32 GB card) is a separate technical validation, not covered here.
- One process note: the needle test first "failed" for GridBook — a
  harness artifact (32 max-tokens starved a reasoning model of its
  answer). With `enable_thinking:false` + 256 tokens both models recall
  exactly. Fixed before drawing any conclusion.

## Where this sits

GridBook is served via its out-of-tree vLLM plugin plus a **one-file
cherry-pick** on our sm12x build (the dev693 `qwen3_5.py` that strips the
VL-text-tower `model.language_model.` prefix) — coexists with DFlash2 /
NVFP4, on a dedicated tree so it never touches production's.

Raw: [`results/RESULT_qual_gridbook_gsm250.json`](../results/RESULT_qual_gridbook_gsm250.json) ·
[`results/RESULT_qual_aqua_gsm250.json`](../results/RESULT_qual_aqua_gsm250.json)
