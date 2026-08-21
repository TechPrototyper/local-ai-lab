# TurboQuant-KV vs NVFP4-KV: value-cache fidelity probe (2026-08-21)

TurboQuant presets recently became an official 4-bit-class KV path in
vLLM (pure Triton, no SM gate — it runs on sm12x out of the box). This
lab serves uncalibrated NVFP4 KV in production, so the obvious question:
does the official path beat our recipe on the axis our findings live on —
**value-cache fidelity under attention-sink outliers**?

Model-free kernel round-trip, same methodology and the same sink
provenance (amax ≈42k and ≈125k, measured from a real checkpoint) as the
[calibration study](../probes/nvfp4_calib_scale_study.py). NVFP4 arm:
production setting (scale 1.0, group-16 fp8-e4m3 block scales) through
the installed `reshape_and_cache_flash`. TurboQuant arm: installed
`triton_turboquant_store`, reconstruction via single-key decode
(softmax over one key = 1.0 → the attention output *is* the dequantized
value; key-path error drops out). Probe:
[`probes/turboquant_vs_nvfp4_value_probe.py`](../probes/turboquant_vs_nvfp4_value_probe.py),
raw: [`results/RESULT_turboquant_vs_nvfp4_value.json`](../results/RESULT_turboquant_vs_nvfp4_value.json).

## Results (rel-L2, bulk vs outlier split; 512 tokens, 4 KV heads, d=128)

| Scenario | NVFP4 scale-1.0 | TQ k8v4 | TQ 4bit_nc | TQ 3bit_nc |
|---|---|---|---|---|
| Gaussian bulk | **0.095** | 0.100 | 0.100 | 0.214 |
| Sink 42k — bulk | **0.104** | 244.9 | 244.9 | 524.8 |
| Sink 42k — outlier | 0.936 | **0.000** | **0.000** | **0.000** |
| Sink 125k — bulk | **0.104** | ∞ | ∞ | ∞ |
| Sink 125k — outlier | 0.978 | ∞ | ∞ | ∞ |

Bytes per token per head (K+V, aligned): NVFP4 144, TQ k8v4 196,
TQ 4bit_nc 134, TQ 3bit_nc 102.

## Three readings

1. **On well-behaved data the paths are equivalent** (0.095 vs 0.100 at
   comparable bytes). Nothing to argue about.
2. **Under sinks the failure modes are complementary, and locality
   decides.** TQ's per-vector fp16 scale+zero represents the outlier
   *exactly* — and in exchange coarsens the quantization step for the
   sink token's *entire remaining vector* (bulk rel-L2 ≈245: the step
   grows to amax/15, and 126 of 128 channels drown in it). NVFP4's
   group-16 block scales do the opposite: the outlier clips at the E2M1
   ceiling (rel-L2 0.94) but the damage is confined to the outlier's own
   16-channel group — the rest of the token survives untouched. Our
   full-split results (08-01, 08-15) already showed that the clipping
   mode is quality-neutral end to end; the TQ mode (sink token's payload
   destroyed) is the one attention actually consumes when it attends to
   the sink.
3. **The fp16 zero-point is a hard ceiling.** The TQ store kernel keeps
   per-vector scale and zero-point in fp16 regardless of input dtype
   (verified with bf16 inputs). A sink of |min| > 65504 overflows the
   zero-point → the reconstruction is **Infinity** — a poisoned cache
   line, not a graceful degradation. The measured sink provenance of the
   model family this lab serves reaches ≈125k, i.e. past the ceiling on
   real data.

## Serving smoke: it doesn't get that far

The end-to-end check (boot the production 27B with
`--kv-cache-dtype turboquant_4bit_nc`) fails at engine init:

```
ValueError: Unknown TurboQuant cache dtype: 'auto'.
Valid presets: turboquant_k8v4, turboquant_4bit_nc, ...
```

The 27B is a **hybrid linear-attention model** — its state layers
resolve to cache dtype `auto`, and the TurboQuant backend currently has
no path for a mixed-layer model. So on this model family the practical
blocker sits *before* the fidelity question: the backend doesn't compose
with hybrids at all (same structural corner the skip-layers probe died
in — hybrid models keep being where KV-cache features meet reality).
The fp16-ceiling finding above therefore stays **probe-level**: real on
the kernel, not yet demonstrated end to end (that would need a
non-hybrid model with comparable sinks).

## Verdict for this lab

The production recipe stays **uncalibrated NVFP4 KV**, on three
independent grounds: equal bulk fidelity at comparable bytes; a benign,
empirically quality-neutral failure mode under sinks (clipping confined
to a 16-channel group vs. the sink token's whole payload coarsened);
and the fact that TurboQuant cannot currently serve this hybrid model
family at all. TurboQuant remains attractive where its strengths are
(3-bit capacity, non-hybrid models without extreme sinks) — this is a
*fit-for-this-model-family* verdict, not a general ranking. Two
observations look upstream-worthy once properly packaged: the hybrid
incompatibility (a clear error, arguably fine as-is) and the fp16
zero-point ceiling (silent ∞ — scale/zero in fp32 would remove it for
4 bytes per vector).
