# ModelOpt AutoQuantize — feasibility spike (ENTWURF, nicht committen)

**Status:** measured (2026-08-20, RTX 5090 / sm120, single physical card,
vLLM `0.1.dev1+gf4c27c0da` parity build, torch 2.13.0+cu130). One bounded
GPU Job on the freed card (production `vllm-qwen38` scaled 1→0 for a
~3.5-min window via merge-patch, then restored, `/health` 200 + chat smoke
verified). Kanban #13, precursor to the AutoQuantize-vs-AURA head-to-head.

**Update 2026-08-20 (B1, Kanban #24):** Q1 uniform-FP8 follow-up run — the
uniform-FP8 ModelOpt export IS accepted by the parity vLLM `modelopt` loader
(the #13 mixed-export blocker is gone). BUT the confirmatory greedy-generate
was **not** captured: during the serve run neo26 hard-froze for ~23 min in
vLLM's KV-cache memory-profiling phase, taking the co-located Talos control
plane (etcd/apiserver) down with it (~20:13→~20:32 UTC); the serve pod was
evicted before generation. Prod restored + verified (chat smoke ok,
fingerprint gf4c27c0da; Flux had reconciled `vllm-qwen38` back to replicas
1 during the outage). No reboot/talosctl (guarded). See INCIDENT below;
re-run is gated on Tim + root-causing the freeze. Local result:
`results/RESULT_modelopt_uniform_fp8_serving.json`.

## Why
Before spending an overnight H2H arm on NVIDIA Model-Optimizer AutoQuantize
vs AURA/PrismaQuant, answer four go/no-go questions: can we (1) serve a
ModelOpt checkpoint under our parity vLLM, (2) install `nvidia-modelopt`
and run AutoQuantize on our stack, (3) bridge its export to
compressed-tensors, (4) design a fair paired comparison.

## Method
Single Job in `quant-experiments` (reuses the prismaquant PVCs
`quant-models`/`quant-results`, image = parity vLLM build, nodeSelector
neo26, runtimeClass nvidia, gpu:1, activeDeadlineSeconds 5400). Job created
**suspended**, then `vllm-qwen38` merge-patched replicas 1→0 (NOT scale
subresource, NOT Flux), then Job **unsuspended** → runs alone on the idle
5090 (honours the "sequential, not GPU parallelism" rule on a single
time-sliced card). Qwen3-4B-Instruct-2507 BF16 already resident on the PVC.

## Results

**Q2 — production path: PASS.** `pip install nvidia-modelopt[hf]` rc=0 into
the parity image; **torch stayed 2.13.0+cu130 (before==after, not
clobbered)**; `modelopt_cuda_ext` + `modelopt_cuda_ext_fp8` compiled on
sm120/cu130. `mtq.auto_quantize(constraints={"effective_bits":5.5},
quantization_formats=[FP8, INT4_AWQ], data_loader=…)` ran **first try in
71.8 s** and produced a genuine per-layer mixed allocation (attn/mlp layers
split between FP8 and W4A16_AWQ g128). `export_hf_checkpoint` OK →
config.json + hf_quant_config.json + safetensors + tokenizer.
Caveat: the install bumped `transformers`→5.14.1 and pulled
nccl-cu13/datasets/deepspeed/peft/diffusers — fine in an ephemeral job pod,
but **production must isolate this in a separate venv/build layer** so it
never touches the serving image's parity pins.

**Q3 — format bridge: NOT drop-in.** ModelOpt writes an `hf_quant_config.json`
(`producer.name=modelopt 0.46.0`, `quant_algo=MIXED_PRECISION`,
`kv_cache_quant_algo=null`, a `quantized_layers` map of `<module> →
{quant_algo: FP8|W4A16_AWQ, group_size, has_zero_point, pre_quant_scale}`,
scales as `*.weight_scale`/`*.pre_quant_scale`). compressed-tensors (what
AURA emits) uses `config.json → quantization_config.config_groups`
(`weights`/`input_activations` schemes: num_bits/type/strategy/group_size +
`targets`) with a `format` (pack-quantized / float-quantized /
nvfp4-pack-quantized) and `weight_packed`/`weight_scale`/`weight_zero_point`.
Different container **and** different way of encoding mixed precision
(per-layer algo map vs multiple config_groups). vLLM reads each via its own
path (`--quantization modelopt` vs auto-detected compressed-tensors), so
**serving needs no conversion once the checkpoint is uniform/loadable**, but
the on-disk formats are not interchangeable — a real converter (or
native-per-format serving) is required for the H2H.

**Q1 — serving path: BLOCKED for the mixed export.** Booting our
AutoQuantize export under the parity vLLM with `quantization="modelopt"`
**failed at engine init**:
`RuntimeError: start (0) + length (9728) exceeds dimension size (4864)` in
`vllm/model_executor/layers/linear.py:load_merged_column_weight → narrow`.
Root cause: vLLM's modelopt loader merges the gate/up columns, but our
export has those layers at **mixed precision** (one FP8, one W4A16_AWQ-packed
→ halved dim), so the merged-column narrow overruns. i.e. **vLLM's
`--quantization modelopt` path does not accept per-layer MIXED_PRECISION
(FP8 + AWQ-int4)**. This is a positive, specific finding — not a dead end:
the uniform FP8/NVFP4 modelopt path is the intended one. Not yet positively
confirmed because I loaded our mixed export (to avoid a multi-GB download);
the cheap follow-up is to constrain auto_quantize to uniform FP8 (or NVFP4)
and re-boot, or pull a small official `nvidia/*-FP8` checkpoint. ~15 min.

**Q1 — uniform-FP8 path: LOAD/SERVE-ACCEPT CONFIRMED (2026-08-20, B1/#24).**
`mtq.auto_quantize(model, constraints={effective_bits:8.0},
quantization_formats=[FP8_DEFAULT_CFG], data_loader, forward_step, loss_func)`
ran in ~4 s and `export_hf_checkpoint` in ~9.5 s → an
`hf_quant_config.json` with top-level `quant_algo=FP8` (NOT MIXED_PRECISION),
`kv_cache_quant_algo=null`, no per-layer override map (`export_uniform=true`).
Booting it under the parity vLLM with `quantization="modelopt"` selected the
**plain `modelopt` path (not `modelopt_mixed`)**:
`Detected ModelOpt fp8 checkpoint (quant_algo=FP8)` →
`ModelOptFp8LinearMethod` + `FlashInferFP8ScaledMMLinearKernel`, weights
loaded (4.3 GiB, ~1.9 s) with **no `load_merged_column_weight`/narrow error**,
KV cache allocated (11.88 GiB → 86,512 tok), fp8_gemm autotune completed —
i.e. the engine reached serving-ready. **This positively closes Q1**: the
uniform-FP8 ModelOpt export is loadable/servable by our parity vLLM, unlike
the per-layer MIXED_PRECISION (FP8+AWQ) export. Effective-bits note: the
5.5 constraint was raised to 8.0 because a single 8-bit format can't reach
5.5 → the search assigns FP8 everywhere = uniform (documented, intended-path
adaptation of "formats=[FP8]"). **Caveat — greedy-generate NOT captured:** a
neo26 node freeze (see INCIDENT) evicted the serve pod before `llm.generate`
returned. A short confirmatory generate remains, gated on Tim + freeze RCA.

**INCIDENT — neo26 node freeze + control-plane outage (2026-08-20 ~20:13–
20:32 UTC).** The serve pod's log jumps 20:10:33 (weights loaded) → 20:33:24
(KV cache) — a ~23-min node freeze that began exactly when vLLM entered the
KV-cache memory-profiling phase (profiling forward pass + 11.88 GiB KV
reservation). The Talos control plane is co-located on neo26; apiserver
(10.1.0.138:6443) wedged (TCP open, TLS/request layer blocked on etcd) for
~19 min, so `vllm-qwen38` (scaled to 0 for the window) could not be restored
until the API returned. Self-recovered ~20:32; Flux had already reconciled
`vllm-qwen38` to replicas 1; chat smoke re-verified. No reboot/talosctl
(agent-guarded). The earlier combined job — killed before the KV-profiling
phase — did NOT freeze the node, so the phase correlation is strong. **A
re-run would re-enter the same phase and may re-freeze neo26 + the control
plane; do not repeat unprompted. Root-cause first (system-RAM/IO pressure on
the etcd-colocated GPU node; consider isolating etcd from the GPU worker).**

## Consequence / go decision
AutoQuantize is **viable on our stack** as an H2H arm: install + mixed-bit
allocation + HF export all work on sm120/cu130 in ~72 s for 4B. Two things
gate a fair comparison: (B1) produce a **uniform-FP8/NVFP4** ModelOpt export
so vLLM can serve it, and (B2) reconcile the **effective-bit accounting**
(ModelOpt `effective_bits` vs AURA weight-only `target_bits`) and the **KV
quant** (AURA NVFP4-KV vs ModelOpt kv=null) before claiming an equal 5.5-bit
budget. Paired design: same model, same budget, same parity serving config,
n=250 screen → n=1319 GSM8K confirm (reuse `COMPARE_ref_1319` harness),
publish both per-layer bit-maps and report tok/s separately from quality.

## Follow-ups (cheap)
1. Constrain AutoQuantize to uniform FP8 → confirm vLLM `--quantization
   modelopt` boot + generate (kills B1, positively closes Q1).
2. Reconcile bit-accounting + KV config across ModelOpt/AURA (kills the
   biggest H2H confound).
3. Build an isolated modelopt venv/image layer (kills B2 parity risk).

Artifacts (cluster, quant-results PVC):
`/workspace/results/modelopt-spike/{RESULT_modelopt_spike.json,env.json,
q1_serving.json,q2_autoquant.json,q3_format.json,export-autoquant/,run.log}`.
Local: `results/RESULT_modelopt_autoquant_spike.json`.
