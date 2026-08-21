<!--
DRAFT for Tim's review. NOT committed, NOT pushed.

Target filename on publish: notes/qwen38-pipeline-story.md (drop the
-DRAFT-EN suffix). Remove this comment block and the "Draft for Tim's
review" footer at the bottom before merging.

Suggested README.md changes:

1) Changelog row (top of table, newest-first):

   | 2026-08-21 | Consolidated 3.6→3.8 pipeline write-up added (`notes/qwen38-pipeline-story.md`): full-split verdict, the 3.8 pivot, the speculation re-evaluation, and today's per-platform recommendation in one narrative | the "Consolidated write-up" Next-up item is done — the record exists as one document instead of six | [`notes/qwen38-pipeline-story.md`](notes/qwen38-pipeline-story.md) |

2) Findings row: OPTIONAL. This note synthesizes already-published
   measurements rather than reporting a new one — every number in it has
   a Findings row of its own already. Suggest skipping a new row unless
   you want the write-up itself to be discoverable from the Findings
   table; if so:

   | 2026-08-21 | **The 3.6→3.8 pipeline story, consolidated into one record.** Full-split verdict (95.00% vs 94.84%, p=0.883), the Qwen3.8 pivot (byte-identical config, fix applies 1:1), the speculation re-evaluation (DFlash2 mechanism found, quality gate passed, batch sweep reverses first light), and the closed format question (weights: NVFP4 = INT4-Marlin; KV: neither GPTQ nor a practical Hadamard rescues amax) — now cross-referenced from one place instead of six. | [`notes/qwen38-pipeline-story.md`](notes/qwen38-pipeline-story.md) |

3) Once this lands, remove the "Consolidated write-up" bullet from
   README's "Next up" section — it's the item this note completes.
-->

# Qwen3.6 → Qwen3.8: the pipeline story, and today's recommendation

**2026-08-15 → 2026-08-20** · Status: consolidated record — every
measurement below is individually published; this note is the narrative
through-line the [README Next-up list](../README.md) asked for.

## 1. The full proof: the repaired pipeline reproduces production quality

After [PrismaQuant PR #80](https://github.com/RobTand/prismaquant/pull/80)
merged (the linear-attention masking fix, transformers ≥5.15), one
question stayed open: does a checkpoint quantized with the repaired
streaming pipeline match the maintainer's own reference export — at
verdict level, not screening? It does. Both arms on the identical
production serving config, sequential on GB10, 2,638 requests, 0 errors:

GSM8K full n=1319: **95.00% vs. 94.84%**, McNemar exact p=0.883 (46
discordant, +24/−22); needle 6/6 both arms. Full table, tools/determinism
edges, and raw JSON:
[`qwen36-aura-head-to-head.md`](qwen36-aura-head-to-head.md) ·
[`results/COMPARE_cand_vs_ref_1319.json`](../results/COMPARE_cand_vs_ref_1319.json).
Also posted on
[PR #80](https://github.com/RobTand/prismaquant/pull/80#issuecomment-5301342724).

A methodological aside worth keeping: the n=250 screening slice ran ~2×
faster per item than the full split, and the full split's own first ~60
items ran faster still than its average. Extrapolating a completion rate
from a partial run without checking the length distribution gets the ETA
wrong twice, not once.

## 2. Then Qwen3.8 dropped

Mid-verdict-weekend: **Qwen3.8-27B** (27.78B, BF16, dense) landed — config
byte-identical to 3.6 except `transformers_version`, same hybrid 48/16
schedule. The PR #80 fix applies 1:1. Community benchmarks position it as
the new reference dense model around 30B (e.g. 70.7% on CoWorkBench).
Verified on GB10 within hours (18/18 shards, CRC-checked against HF main),
and a double-quant campaign went up: **AURA 5.5** and **Gridbook-CB 5.5**,
head-to-head, winner becomes production on both sm121 and sm120.

## 3. Where that campaign stands

**Qwen3.8-AURA is in production.** The AQUA 5.5-bpp export
(`qwen3.8-27b-prismaaqua55`) has been serving on GB10 since — NVFP4-KV,
262k context, live-verified. The CB arm is still open: the clean Gridbook
quality re-run (the 08-13 number was confounded by a missing
`--reasoning-parser` flag, not a CB deficit — see the
[README Next-up list](../README.md)) and this lab's own CB re-quant (run
overnight, no cross-box transcode available) are both still outstanding.

**Speculation, re-evaluated with a mechanism instead of a hunch.** The MTP
era ended with "speculation breaks tool-calling"
([note](mtp-tool-calling.md)). The DFlash2 measurement showed that verdict
was never about speculation itself — it was the proposal/parser path: the
block-diffusion drafter runs 23.3 tok/s vs. 10.7 baseline on sm121,
tool-calling byte-identical to non-speculative in 4/5 chains, while ngram
proposals corrupt the `qwen3_coder` parser (0/5)
([first-light note](dflash2-sm121-first-light.md)). The paired n=250
quality gate then passed — 96.8% vs. 95.6%, McNemar p=0.375, no detected
quality difference, both arms byte-deterministic — and finished the
identical GSM8K workload in 40% of baseline wall-clock at concurrency 4
(1193 s vs. 3016 s)
([gate note](dflash2-n250-gate-2026-08-20.md)). The controlled batch sweep
that followed reversed the earlier "latency tool, not an aggregate tool"
read: DFlash2 wins at **every** measured concurrency through c=8 (c=1:
48.3 vs. 10.7 tok/s; c=8: 134.9 vs. 70.9 tok/s aggregate) — and in the
same window, the skip-layers hybrid that would have brought DFlash2 to the
32 GB card without the fp8-KV context penalty was ruled out: exempted
layers pad to the hybrid model's state-page granularity, ~41 GiB of KV for
a 32k window regardless of settings
([batch-sweep + skip-layers note](dflash2-batch-sweep-and-skiplayers.md)).
Still ahead before verdict-level language: the n=1319 full split and the
c>8 boundary.

**A boot crash with a lesson.** Concurrent prefills in the very first
engine step after boot can kill the hybrid linear-attention engine
(`cudaErrorNotPermitted` in the GDN state write), including crash-loop
risk under a restart policy. Mitigation on every boot path: wait for
health, send exactly one warmup request, then open the floodgates
([note](gdn-first-step-crash.md)).

**The format question, closed on both axes.** Weights: NVFP4 costs no
measurable accuracy against INT4-Marlin on a same-model comparison (117B
MoE, screening level) ([note](laguna-nvfp4-vs-int4.md)). KV cache: the
[llm-compressor#2936](https://github.com/vllm-project/llm-compressor/issues/2936)
line got two follow-up probes — GPTQ is an orthogonal axis (the baked
v_scale is identical with and without it; sinks dominate amax, GPTQ never
sees them), and Hadamard block size is a real but magnitude-bounded lever
(a non-foldable online rotation at B=1024 rescues a moderate 42k sink, but
the ~125k-magnitude layers that actually drove the GSM8K loss stay 100%
erased — the needed block size, ~1e4, is impractical). Per-tensor amax
stays the wrong objective for 4-bit KV; scale-1.0 serving stays the
fallback ([note](nvfp4-kv-gptq-online-hadamard.md)).

## 4. The recommendation, per platform

Same model family, same quant stack, opposite settings — each derived from
measurement. Full commands:
[`recipes/dgx-spark-sm121.md`](../recipes/dgx-spark-sm121.md) and
[`recipes/rtx5090-sm120.md`](../recipes/rtx5090-sm120.md).

### sm121 (DGX Spark / GB10, 128 GB unified — quality → speed → memory)

- **Model:** Qwen3.8-27B as PrismaAQUA 5.5 (the AURA pipeline behind it is
  verdict-validated on 3.6 at n=1319, p=0.883).
- **KV:** NVFP4, uncalibrated (scale 1.0) — amax calibration measurably
  hurts (n=1319, p=0.013). 262k context, prefix caching on, size the pool
  with `--kv-cache-memory-bytes` (the unified-memory gotcha, not
  `--gpu-memory-utilization`).
- **Serving:** always set both the reasoning and tool parser (otherwise
  confounded outputs); boot discipline: wait for health, one warmup
  request (the GDN first-step window).
- **Speculation:** MTP off. **DFlash2 is an adoption candidate:**
  quality-neutral at n=250, and the batch sweep has it faster on every
  measured tier through c=8 (c=1: 48.3 vs. 10.7 tok/s; c=8: 134.9 vs.
  70.9 tok/s aggregate). "Latency tool, not an aggregate tool" is
  withdrawn up to c=8. Before verdict language: n=1319, and the c>8
  boundary. CB/Gridbook: functionally proven, quality-per-bit verdict
  open — no recommendation yet.

### sm120 (RTX 5090, 32 GB — quality → memory → speed)

- Same parity build as the Spark — one vLLM stack, two cards.
- **KV:** NVFP4-KV is the whole lever (~4× context), also strictly
  uncalibrated — the two probes above close the door: neither GPTQ nor a
  practical larger Hadamard rescues the amax path.
- **Weights:** NVFP4 throughout — no measurable accuracy cost against
  INT4-Marlin (same-model, screening level).
- **Speculation: stays off.** The full fp8-KV trade would halve context,
  and the skip-layers hybrid that was meant to reconcile both is measured
  and **ruled out**: the mechanism works, but skipped-layer pages align
  to the hybrid model's state-page granularity — ~41 GiB of KV for 32k
  context, prefix caching doesn't help. Remaining paths to RTX
  speculation: NVFP4 non-causal kernel work (an upstream question) or
  context capping with length-based routing.

---

*Draft for Tim's review before publish. Translated from the internal
German diary entry; internal paths, hosts, PIDs, and session/kanban
references removed per repo convention (no internal ops detail in the
public repo).*
