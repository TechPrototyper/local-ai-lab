# Qwen3.6-27B: self-produced AURA quant vs. the reference — screening

**2026-08-12** · Status: screening passed; full-split verdict queued

## What this is

First end-to-end dense 27B run of [PrismaQuant](https://github.com/RobTand/prismaquant)'s AURA pipeline (KL-Fisher bit allocation — Rob Tandler's method) produced on this lab's own hardware: mixed-format NVFP4-MLP + FP8-attention + NVFP4-KV at ~5.5 bpp, allocation NVFP4:264 / BF16:187 / FP8:163 modules. The question: does a self-produced quant reproduce the quality of the maintainer's own reference (`prismaaura55`)?

Producing it required fixing two bugs in PrismaQuant's layer-streaming forward that surface on transformers >= 5.15 with hybrid linear-attention models (dense causal mask routed into `linear_attention` layers; `.layer_type` → `.block_type` rename from transformers#47630 breaking layer-type resolution). Fix submitted upstream: **[PR #80](https://github.com/RobTand/prismaquant/pull/80)**. PrismaQuant's own lockfile pins transformers 5.8.0, so stock environments never see this — any environment resolving a current transformers does.

## Measurements

Paired battery, candidate vs. reference, served **sequentially** on the same GB10 under identical serving config (never co-tenant). Raw JSON: [`results/RESULT_cand-export27b.json`](../results/RESULT_cand-export27b.json), [`results/RESULT_ref-prismaaura55.json`](../results/RESULT_ref-prismaaura55.json), [`results/COMPARE_cand_vs_prismaaura55.json`](../results/COMPARE_cand_vs_prismaaura55.json).

| Metric | Candidate (self-produced) | Reference (`prismaaura55`) |
|---|---|---|
| GSM8K, n=250 screening | **97.2%** | 95.6% |
| McNemar (exact) | p=0.34, discordant 10 (+7/−3) — no detected difference | — |
| Needle (long context) | 9/9 | 9/9 |
| Determinism (5 reruns) | 5/5 | 5/5 |
| Single-stream | 10.7 tok/s | 11.5 tok/s |
| Batch-8 aggregate | 79.2 tok/s | 85.3 tok/s |

## Read carefully

- Per this repo's [method](../README.md#method), **n=250 is triage, not a verdict** (±2.7 pp). The screening says: no red flag, direction +1.6 pp in the candidate's favor, campaign proceeds. The full n=1319 paired run is queued (~8 h per arm on GB10); this note gets updated with the verdict.
- The candidate serves **~7% slower**. That is an export-vs-export difference (different build, different allocation run), *not* a measured cost of the masking fix — attributing it would take a patch-on/off ablation on the same export, which hasn't been run.

## Why it matters

Reproducing a quantization method's reference quality from source, on independent hardware, is the strongest evidence a method can get. AURA passed at screening level on the first self-produced 27B. Credit where due: the method, the reference quant, and the pipeline are Rob Tandler's work — this lab measures it.
