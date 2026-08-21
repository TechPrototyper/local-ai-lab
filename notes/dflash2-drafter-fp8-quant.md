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
`down_proj`) load and run cleanly, which isolates the gap precisely:
**the DFlash2 draft-model line has never loaded a quantized checkpoint
with fused-layer scales**. Plausibly an upstream gap in the open PR line
([vllm#52816](https://github.com/vllm-project/vllm/pull/52816)) —
quantized drafters are an obvious follow-on nobody has tested yet.

## Where this lands

- The partial fp8 drafter (10 of 35 linears) saves 0.55 GB at zero
  acceptance cost — deployable, but the full win (~1.6 GB, plus
  bandwidth) waits on the fused-scale fix.
- The fix itself is a bounded patch to the draft-model loading path (or
  an upstream report with this note as the repro).
- An NVFP4-weight variant (W4A16) would roughly double the savings, but
  inherits the same fused-scale gap *plus* an open kernel question on
  sm12x — fp8 first, then revisit.
