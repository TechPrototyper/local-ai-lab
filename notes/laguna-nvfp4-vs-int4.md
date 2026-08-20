# NVFP4 vs. INT4 weights, same model: 117B MoE at ~4 bit — screening

**2026-08-01** · Status: screening passed; no full-split verdict planned (campaign answered its question)

## What this is

A same-model, same-serving-stack comparison of two ~4-bit weight formats on
`poolside/Laguna-S-2.1` (117.6B MoE, 8.5B active): **NVFP4**
(FLASHINFER_CUTLASS path) vs. **INT4** (Marlin WNA16 path), both ~67 GB on
disk, both served with **fp8 KV held constant**. Everything else identical:
vLLM v0.26.0, `gpu-memory-utilization 0.8`, `max-model-len 131072`,
`max-num-seqs 32`, greedy/deterministic decoding, served **sequentially** on
the same GB10 (sm121) — never co-tenant.

The campaign's original question was a model decision (is this MoE worth its
memory locally?). The format comparison fell out of it as a controlled
by-product and is reported here because same-model NVFP4-vs-INT4 accuracy
datapoints on large MoEs are scarce.

## Measurements

Raw JSON: [`results/RESULT_laguna_battery_{nvfp4,int4}.json`](../results/),
[`results/RESULT_laguna_agentic-hard_{nvfp4,int4}.json`](../results/),
[`results/RESULT_laguna_perf_{nvfp4,int4}.json`](../results/).

| Metric | NVFP4 | INT4 (Marlin) |
|---|---|---|
| HumanEval pass@1, n=120 | **96.7%** | 95.0% |
| Agentic easy (2 scenarios) | 2/2 all dimensions | 2/2 all dimensions |
| Agentic hard: solved / recovered | 3/3 · 1/1 | 3/3 · 1/1 |
| Agentic hard: deploy discipline / checkin | 3/3 · 3/3 | 3/3 · 3/3 |
| Agentic hard: avg turns · hygiene | 8.0 · 0.97 | 8.3 · 0.92 |
| TTFT short / ~7k prompt | 0.29 s / 2.44 s | 0.24 s / 2.80 s |
| Decode, single-stream | 19.5 tok/s | 19.6 tok/s |

The hard agentic set (lru_cache · ranges_bugfix, seeded buggy, read+fix ·
bank_transfer, atomic+rollback) was verified offline before use: the correct
solution goes green, the naive one goes red — the scenarios have teeth.

## Read carefully

- **n=120 is screening, not a verdict** (the 1.7 pp HumanEval gap is two
  items — inside noise at this n). The honest claim: *no detected accuracy
  difference between the formats; direction, if anything, favors NVFP4.*
- The non-coding battery (MMLU n=228, GSM8K n=60, needle@100k) was run on the
  **NVFP4 arm only** — that part of the campaign asked a cross-model question
  (against this lab's production 27B), not a format question. There is no
  same-model non-coding format comparison in this data.
- This is a **weights-format** comparison with KV precision pinned at fp8. It
  is complementary to this lab's NVFP4-**KV** line (calibration study,
  Hadamard probe) — different axis, same production stack.

## Why it matters

At ~4-bit weights on a large MoE, NVFP4 shows **no accuracy penalty against
INT4-Marlin at screening level** — coding one-shot, hard agentic behavior
(recovery, deploy discipline, hygiene) and decode throughput are at parity,
with slightly cleaner agentic hygiene on the NVFP4 arm. For anyone deciding
whether NVFP4 weight quantization costs accuracy relative to the INT4 status
quo: on this datapoint, it doesn't.
