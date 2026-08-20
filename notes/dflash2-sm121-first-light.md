# DFlash2 speculative decoding on sm121 — first light (2026-08-19/20)

**TL;DR:** DFlash2 (block-diffusion drafter, 2B, bf16) runs on GB10/sm121
against this lab's production 27B NVFP4 quant: **23.3 tok/s on the
standardized 512-token decode probe vs 10.7 baseline (2.2×), with peak
single-stream windows at 26–31 tok/s** (mean acceptance length 5.2–7.1).
Tool-calling stays clean — which revises this lab's earlier "speculation
breaks tool-calling" position into something mechanistic. Two hard limits:
DFlash2's non-causal drafter attention is incompatible with NVFP4 KV cache
on the current FlashInfer sm12x path (fp8 KV required), and the gain
inverts under concurrent load. Verdict-level quality gate (paired n=250,
then n=1319 per [method](../README.md#method)) still pending.

## Why this experiment

Trigger was [ARahim3/mlx-dspark](https://github.com/ARahim3/mlx-dspark),
reporting up to ~97 tok/s on Apple Silicon. The headline number comes from
small / MoE configurations and is not comparable to a dense 27B — but the
honest core is: their **dense-27B** goes 8.4 → 30.5 tok/s (3.63×) with
DFlash2 speculation, on an M4 Pro with the **same memory-bandwidth class as
GB10 (~273 GB/s)**. This lab's baseline beats their baseline (thanks to the
NVFP4-KV recipe), but the previous speculation multiplier here (MTP, n=2)
was only ~1.5–1.75× — and was switched off anyway after it broke
tool-calling ([`mtp-tool-calling.md`](mtp-tool-calling.md)). The unrealized
lever was speculation *quality*, not hardware.

A structural constraint made classic draft-model speculation unavailable:
no small same-vocabulary sister model exists for this 27B line (the 4B
model uses a different vocabulary, 151936 vs 248320). DFlash2's trained
block-diffusion drafter sidesteps that.

## Getting it to run (dead ends included)

1. **v1 vs v2 confusion.** The v1 drafter's HF card points at vLLM PR
   #40898 — which turns out to be v1 refinements; DFlash **v1** support is
   already upstream in the build this lab runs (including the Blackwell
   fixes vllm#48167 / vllm#50065). **DFlash2** (the version with drafters
   for current models) officially targets SGLang; the vLLM side lives in
   the open PRs [vllm#52816](https://github.com/vllm-project/vllm/pull/52816)
   (local convolution + candidate selector) and
   [vllm#52883](https://github.com/vllm-project/vllm/pull/52883) (LM-head
   bugfix).
2. **Cherry-pick, don't merge.** The PR branch carries three weeks of main
   churn (~1979 files). Full merge onto the sm12x production line is not
   reviewable; instead the **10 commits exclusive to the PR** were
   cherry-picked (`git log --oneline pr-branch --not upstream-main`) plus
   one minimal backport shim for a missing interface symbol
   (`supports_multimodal_embeddings`). Conflicts were add-only or refactor
   — resolved toward the PR side.
3. **In-place-build gotcha.** Cloning a build tree does not carry the
   gitignored build artifacts (`_C*.so`, `_version.py`, `vllm_flash_attn/`,
   `third_party/`, `fla/`). List them via `git status --ignored --short`
   and rsync them across — noting that `rsync --files-from` does **not**
   imply `-r`.
4. **The real limit:** `NotImplementedError: FlashInfer non-causal
   attention is not supported with NVFP4 KV cache.` The block-diffusion
   drafter attends bidirectionally within its block; the sm12x FlashInfer
   path cannot serve that from NVFP4 KV. **Workaround: fp8 KV** for the
   whole engine. Whether `--kv-cache-dtype-skip-layers` can exempt only the
   drafter layers (target stays NVFP4) is an open probe — that would
   matter most on the 32 GB consumer box, where fp8 KV halves token
   capacity.

## Measurements (overnight 2026-08-20, production flags, greedy)

| Arm | GSM8K (n=100) | Tool gate | Decode (512-tok probe) | Notes |
|---|---|---|---|---|
| Baseline, NVFP4 KV (prod) | 96/100 | PASS (reference canon) | 10.7 tok/s | [`results/RESULT_night_base38.json`](../results/RESULT_night_base38.json) |
| Baseline, fp8 KV (control) | 96/100 | — | 10.7 tok/s | KV dtype alone changes neither quality nor decode speed measurably — [`results/RESULT_night_base38_fp8kv.json`](../results/RESULT_night_base38_fp8kv.json) |
| ngram / prompt-lookup | — | **FAIL** | — | **Parser corruption:** chat-template fragments inside tool names, 0/5 chains complete; `logit_bias` rejected with HTTP 400 under spec decode. Disqualified for agent traffic. |
| **DFlash2 (n=15 spec tokens, fp8 KV)** | 94/100 | **PASS** (4/5 byte-identical to the NVFP4 canon) | **23.3 tok/s**, peaks 26–31 | Acceptance length 5.2–7.1 single-stream — [`results/RESULT_night_dflash2_n15.json`](../results/RESULT_night_dflash2_n15.json) |

n=100 is *below* even this lab's triage tier; 94 vs 96 is two greedy flips
of the batch-numerics class (verified-lossless speculation cannot change
output distribution — the drafter only affects acceptance rate). Treated as
"no signal either way"; the paired n=250 gate and the full n=1319 verdict
are queued.

### Finding 1: "speculation breaks tool-calling" was never about speculation

The historical rule in this lab — every speculative build degraded quality
and/or tool-calling — now has a mechanism, and it is not speculation
itself:

- **ngram** proposals corrupt the `qwen3_coder` tool parser (template
  fragments end up inside tool names). Reproducible, disqualifying.
- **Sampling-parameter silent drops:** `min_p` and `logit_bias` are
  silently ignored (or rejected) under spec decode — a plausible cause of
  past "tool-calling got worse" experiences that had nothing to do with
  draft quality.
- **DFlash2** produces byte-identical greedy output in 4/5 tool cases vs.
  the non-speculative canon and passes the full conformance gate.

So the rule becomes: *judge each proposal/parser path separately;
speculation per se is quality-neutral under greedy verification.*

### Finding 2: DFlash2 is a latency tool, not an aggregate tool
*(Revised the same evening: the controlled batch sweep reversed this —
DFlash2 wins at every tier up to c=8; see
[`dflash2-batch-sweep-and-skiplayers.md`](dflash2-batch-sweep-and-skiplayers.md).
The paragraph below stands as the original observation and as a
c>8/mixed-load caveat.)*

Under concurrent load (the battery's parallel phase) mean acceptance drops
to ~2.7 and the DFlash2 engine finishes the same workload **slower** than
baseline (560 s vs 370 s wall-clock). The hoped-for "speculation raises
aggregate multi-agent throughput" thesis is refuted in this form: the
drafter's compute competes with batch decode precisely when the batch is
already keeping the GPU busy. A clean batch sweep (1/2/4/8) is queued;
adoption reasoning must be per-tier (interactive latency vs. batch
capacity), not fleet-wide.

## Where this goes

- **GB10 (capacity/overflow tier):** fp8 KV costs tolerable context here
  (20 GiB pool: ~966k tokens @NVFP4 → ~550k @fp8, still ≈ 2× 262k
  sessions). If the paired n=250 gate passes, ~2–3× single-stream is worth
  adopting for interactive traffic.
- **RTX 5090 (latency tier):** tempting, but fp8 KV halves token capacity
  — the wrong trade on a memory-bound box. Options, in order: the
  skip-layers hybrid probe, kernel work on NVFP4 non-causal attention
  (possibly worth an upstream issue), or context capping plus
  length-based routing of long-context jobs to the big-memory box.
- **Queued:** paired n=250 gate; skip-layers probe; batch sweep; spec-token
  N-sweep (7/15/23); drafter-quantization curve (acceptance-per-GB —
  drafter quality only affects speed, never output quality, so this is a
  pure speed/memory trade); upstream watch on vllm#52816/#52883 (once
  merged, the cherry-pick branch retires).
