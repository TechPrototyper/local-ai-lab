# Qwen3.6 → Qwen3.8: the pipeline story, and today's recommendation

**2026-08-15 → 2026-08-27** · Status: consolidated record — every
measurement below is individually published; this note is the narrative
through-line the [README Next-up list](../README.md) asked for.
Sections 1–3 are the mid-August state as it unfolded; section 4 is the
week that closed most of what section 3 left open; section 5 is the
recommendation as of 2026-08-27.

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
overnight, no cross-box transcode available) are both still outstanding
— resolved in section 4.

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
c>8 boundary — both landed within days (section 4).

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

## 4. The week that closed the chapter (2026-08-21 → 08-27)

**Speculation graduated from candidate to production — then got faster
per GB.** The n=1319 full split passed (baseline 95.15% vs DFlash2
**95.83%**, discordants 9/18 *in the drafter's favor*, p=0.122 — equal
within paired statistics, at 2.75× wall-clock), and the draft-length map
found the interior optimum: **n=7** peaks at ~227 tok/s aggregate and
never inverts in the measured range, dissolving the c>8 worry that n=15
had raised ([verdict](dflash2-full-split-verdict.md) ·
[draft-length map](dflash2-draft-length-map.md)). Adoption on the GB10
followed the same day. A day later the **drafter itself went fp8** —
acceptance-neutral by measurement, −1.6 GB straight back into the KV pool
(478k tokens at 21.6 GiB), and a fused-scale loading gap reported
upstream along the way
([drafter-fp8 note](dflash2-drafter-fp8-quant.md)). By 08-25 the whole
production stack was reproducible as a **single public container** —
one `docker run` on any GB10
([portable container](dflash2-gb10-portable-container.md)).

**The CB arm resolved.** Rob's GridBook codebook export compresses the
same model to **13 GB**, and the task battery says quality holds against
the 24 GB production quant: GSM8K n=250 96.0% vs 96.4% (McNemar p=1.0),
tool-calling and needle equal — the published +4.56% intrinsic PPL gap
does not reach task level
([gridbook note](gridbook-13gb-quality-holds.md)). On 08-24 the full
memory-track triad — GridBook + NVFP4-KV + DFlash2 — ran on the 5090
with an **877k-token context config** and was rolled back the same
morning for a reason orthogonal to the triad: prefix caching gets 0%
hits under speculation on hybrid-GDN models
([vllm#52244](https://github.com/vllm-project/vllm/pull/52244)), and the
RTX is the agent tier that lives on the cache
([triad note](gridbook-nvfp4-dflash2-rtx-triad.md)).

**And the "upstream question" from section 3 answered itself — here.**
The NVFP4 non-causal seam turned out to be one serving-layer gap plus
three spec-warmup bugs, not a kernel problem: the FA2 kernel carries
`causal=False` unchanged (Δcos ≤ 0.0013 against the validated causal
path on real pages). DFlash2 + NVFP4-KV now serves end-to-end on **both**
boxes — first on the 5090 (82.5 tok/s prose in the original run), then
cross-arch revalidated paired against no-spec at identical settings:
sm121 prose **+72%**, count-to-200 **5.9×**; sm120 count-to-200 **3.4×**,
step-by-step reasoning 135 tok/s, essay prose at parity (acceptance is
strongly content-dependent — the spectrum, 0.20 → 0.79, is the honest
number). Filed upstream as
[#53977](https://github.com/vllm-project/vllm/pull/53977) /
[#53978](https://github.com/vllm-project/vllm/pull/53978) /
[#53979](https://github.com/vllm-project/vllm/pull/53979) (the seam PR
stacked on #46329)
([seam note](dflash2-nvfp4-sm120-spec-serves.md) ·
[raw](../results/RESULT_nvfp4_spec_crossarch_revalidation.json)).

## 5. The recommendation, per platform (as of 2026-08-27)

Same model family, same quant stack, opposite settings — each derived from
measurement. Full commands:
[`recipes/dgx-spark-sm121.md`](../recipes/dgx-spark-sm121.md) and
[`recipes/rtx5090-sm120.md`](../recipes/rtx5090-sm120.md).

### sm121 (DGX Spark / GB10, 128 GB unified — quality → speed → memory)

- **Model:** Qwen3.8-27B as PrismaAQUA 5.5 (the AURA pipeline behind it is
  verdict-validated at n=1319, p=0.883).
- **Speculation: DFlash2, draft length 7, in production** — verdict-equal
  quality, ~4× single-stream on reasoning traffic, drafter fp8. Never
  raise the draft length for agent traffic.
- **KV:** fp8 at 21.6 GiB = 478k tokens — historically DFlash2's price,
  now a *pending decision*: the NVFP4 variant is battery-validated
  (≈2× pool) and waits on its verdict-tier quality gate.
- **Serving discipline unchanged:** explicit `--kv-cache-memory-bytes`,
  both parsers set, one warmup request after boot.

### sm120 (RTX 5090, 32 GB — quality → memory → speed)

- **KV:** NVFP4-KV remains the whole lever (~4× context), strictly
  uncalibrated. On current code lines set
  `use_trtllm_attention: false` (see the recipe).
- **Weights:** AQUA 24 GB in production; **GridBook 13 GB is the
  presumptive successor** (quality holds at triage level; acceptance
  under this target, a far-window pass, and n=1319 are the open gates).
- **Speculation: available, deliberately off** — no longer an
  impossibility but a tradeoff: prefix caching and speculation are
  mutually exclusive on this model family until
  [#52244](https://github.com/vllm-project/vllm/pull/52244) lands, and
  the agent tier lives on the cache. For batch/single-shot workloads,
  spec over NVFP4-KV is a live option today.

The through-line of the whole story: every one of these settings flipped
at least once — and each flip was bought with a paired measurement, not
an opinion.

