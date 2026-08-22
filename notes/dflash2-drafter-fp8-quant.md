# Quantizing the DFlash2 drafter: acceptance-neutral, and a fused-scale gap (2026-08-21)

The drafter is the one model where quantization is *risk-free by
construction*: under verified-lossless speculation, drafter quality only
moves the acceptance rate (speed), never output quality. So the full
mixed-precision machinery (KL-Fisher allocation, calibration) is the
wrong tool here — plain RTN is the right one. This note documents the
graph-free path and where it hit a wall.

## The graph-free path

No quantization pipeline could load `DFlash2DraftModel` (no transformers
modeling class exists; the checkpoint's `model_type: "qwen3"` would
silently build the *wrong* graph). But RTN needs no graph: a
tensor-level script reads the safetensors, quantizes the standard
decoder projections to fp8 (per-tensor scale, symmetric, dynamic
activations — no calibration required), writes compressed-tensors format
plus `quantization_config`, and vLLM loads it as the draft model via
`"quantization": "compressed-tensors"` in the speculative config. The
DFlash2-specific modules (conv projections, candidate selector,
codebooks) are hard-wired unquantized in the model class — exactly
right.

## Result (draft length 7, GB10, vs bf16 drafter)

| | bf16 drafter | fp8 drafter (partial) |
|---|---|---|
| Single-stream (GSM) | 38–42 tok/s | 37.0 |
| c=8 aggregate | 179 | **196.9** |
| Acceptance length / draft rate | 5.11 / 58.7% | 5.08 / 58.3% |

**Acceptance-neutral**, as theory predicts. Weights: 3.85 → 3.30 GB
(partial — see below).

## The wall: fused layers cannot load quantized scales

Quantizing the *fused* projections (q/k/v → `qkv_proj`,
gate/up → `gate_up_proj`) crashes the DFlash2 weight-loading path:

```
AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
```

— the per-shard `weight_scale` loading resolves to the module instead of
the parameter (same failure for channel and per-tensor strategies; the
scheme matching itself works). The unfused projections (`o_proj`,
`down_proj`) load and run cleanly, which isolates the gap precisely.

**Update 2026-08-22 (upstream reply):** the gap was already known and
has numbers — the draft quant config never reaches
`packed_modules_mapping` ([vllm#53116](https://github.com/vllm-project/vllm/issues/53116);
the misleading AttributeError is
[#53107](https://github.com/vllm-project/vllm/issues/53107)), and a
**second wall** sits behind it that our unfused-only arm never touched:
DFlash's fused context-KV precompute applies `qkv_proj.weight` with a
bare `F.linear`, bypassing the quant method
([#51581](https://github.com/vllm-project/vllm/issues/51581)).
**[PR #53122](https://github.com/vllm-project/vllm/pull/53122) carries
fixes for both.** A fellow GB10 operator also reproduced the
acceptance-neutrality independently (BF16 4.24 / INT8 4.16 / FP8 4.24
acceptance length — with the quantized drafters slightly *faster*,
43.8 vs 41.3 tok/s: bandwidth pays). Tested the same day: **#53122 works on this line.** Full-fp8 drafter
(all 35 projections, per-channel weights): single-stream **42.5 tok/s**
(bf16: 38–42), c=8 aggregate **205.2** (bf16: 179), acceptance
5.05/57.8% (bf16: 5.11/58.7%) — acceptance-neutral, slightly faster,
**−1.6 GB**. One wrinkle reported back to the PR: the context-KV
dequant path requires **per-channel** scales; a per-tensor checkpoint
(scalar-per-shard `weight_scale (3,)` on fused qkv) fails with a clear
error. Results thread:
[#52816 comment](https://github.com/vllm-project/vllm/pull/52816#issuecomment-5378564385).

## Where this lands

- The partial fp8 drafter (10 of 35 linears) saves 0.55 GB at zero
  acceptance cost — deployable, but the full win (~1.6 GB, plus
  bandwidth) waits on the fused-scale fix.
- The fix itself is a bounded patch to the draft-model loading path (or
  an upstream report with this note as the repro).
- An NVFP4-weight variant (W4A16) would roughly double the savings, but
  inherits the same fused-scale gap *plus* an open kernel question on
  sm12x — fp8 first, then revisit.
