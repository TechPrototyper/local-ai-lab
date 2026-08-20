# local-ai-lab

My personal lab notebook: making local AI **measurably** excellent on hardware
people actually own. Spare-time research on my own machines — this repo tracks
what I tried, what I measured, and where it led.

## The framing

Every serving system has one bottleneck that matters most, and the art is to
optimize *that* — not the technology you happen to like:

| Box | Bottleneck order | Consequence |
|---|---|---|
| [**RTX 5090**](nodes/rtx-5090.md) (32 GB consumer GPU) | quality → **memory** → speed | 4-bit KV cache (NVFP4) buys ~4× context; speculative decoding stays *off* for now — the one drafter that proved clean (DFlash2) currently forces fp8 KV, which would halve context ([`notes/dflash2-sm121-first-light.md`](notes/dflash2-sm121-first-light.md)) |
| [**DGX Spark / GB10**](nodes/dgx-spark-gb10.md) (128 GB unified) | quality → **speed** → memory | memory is abundant, so it is spent on long context and throughput (NVFP4 KV, prefix caching, 262k); MTP speculative decoding is *off* — it broke tool-calling on sm121 ([`notes/mtp-tool-calling.md`](notes/mtp-tool-calling.md)) — while block-diffusion speculation (DFlash2, ~2–3× single-stream, tool-calling clean) is under evaluation ([`notes/dflash2-sm121-first-light.md`](notes/dflash2-sm121-first-light.md)) |

Same model, same quantization stack — opposite settings, each derived from
measurement rather than habit. Each box name links to a node profile:
what it runs in production, why those models, and what it is used for
when it isn't running experiments. How the boxes hang together —
cluster, switch, bridge host — is sketched in
[`nodes/homelab.md`](nodes/homelab.md).

## Findings so far

Newest first.

| Date | Finding | Evidence |
|---|---|---|
| 2026-08-20 | **Concurrent prefills in the very first engine step can kill a hybrid linear-attention engine.** First traffic after a boot happened to be two parallel requests: `cudaErrorNotPermitted` in the GDN `ssm_state` write, `EngineDeadError`, container restart — and a restart policy turns that into a crash-loop hazard (boot → agents hammer → first step crashes → repeat). One single request first, and the same load is stable. Working hypothesis: a lazy capture/compile window; repro n=1 so far, controlled repro queued before an upstream report. Mitigation deployed on every boot path: wait for health, send exactly one warmup request, then open the floodgates. | [`notes/gdn-first-step-crash.md`](notes/gdn-first-step-crash.md) |
| 2026-08-20 | **"Speculation breaks tool-calling" was never about speculation — and DFlash2 runs on sm121.** The block-diffusion drafter (vLLM [#52816](https://github.com/vllm-project/vllm/pull/52816)/[#52883](https://github.com/vllm-project/vllm/pull/52883), cherry-picked onto the sm12x line) reaches **23.3 tok/s vs 10.7 baseline** on the 512-token decode probe (peaks 26–31, acceptance length 5.2–7.1) with tool-calling byte-identical to the non-speculative canon in 4/5 cases — while **ngram** proposals corrupt the `qwen3_coder` tool parser (0/5 chains) and `min_p`/`logit_bias` are silently dropped under spec decode. Mechanism found, rule revised: judge each proposal/parser path separately. Limits: non-causal drafter attention is incompatible with NVFP4 KV on sm12x (fp8 KV required), and under concurrent load acceptance falls to ~2.7 and the gain inverts — **a latency tool, not an aggregate tool**. Quality verdict pending (paired n=250, then n=1319 per [method](#method)). | [`notes/dflash2-sm121-first-light.md`](notes/dflash2-sm121-first-light.md) · [`results/`](results/) |
| 2026-08-13 | **The linear-attention masking fix is approved for production upstream.** Two review rounds on [PrismaQuant PR #80](https://github.com/RobTand/prismaquant/pull/80) turned a two-bug crash fix into something better than the original: the maintainer caught that gating on *helper import* would inherit transformers 5.13/5.14's pre-fix cache-state contract — **API existence is not a compatibility gate**. Final: local shim on every version (a booby-trapped test proves the upstream helper is never consulted), lookup-only type resolution, non-attention block dispatch, four-version test matrix green (5.8/5.13/5.14.1/5.15), fresh GB10 end-to-end through the full 64-layer hybrid stack. **Merged** 2026-08-13. | [`notes/prismaquant-pr80-review-cycle.md`](notes/prismaquant-pr80-review-cycle.md) |
| 2026-08-13 | **flashinfer#3684 merged — the kernel side of consumer-Blackwell NVFP4 KV is upstream.** The asymmetric VO-split NVFP4 paged-prefill PR ([@jethac](https://github.com/jethac)'s line end to end) landed in flashinfer main; this lab contributed sm120/sm121 validation datapoints, including a 117k-test 5090 run — taken, instructively, at a stale PR head, which the author re-ran at head himself (lesson filed: pin and re-check the head before posting). [vLLM#46329](https://github.com/vllm-project/vllm/pull/46329) is now the single outstanding piece of the sm12x NVFP4-KV serving line this lab runs in production. | [`notes/upstream-contributions.md`](notes/upstream-contributions.md) |
| 2026-08-12 | **Self-produced 27B AURA quant reproduces the maintainer's reference — at screening level.** First end-to-end dense 27B AURA run on GB10 under transformers 5.15 (took a two-bug upstream fix for hybrid linear-attention masking — [PR #80](https://github.com/RobTand/prismaquant/pull/80)). Paired vs. `prismaaura55`: GSM8K n=250 97.2% vs 95.6% (McNemar exact p=0.34 — no detected difference), needle 9/9 both, determinism 5/5 both; candidate ~7% slower (export-level, unattributed). n=250 is triage per [method](#method) — full n=1319 verdict queued. | [`notes/qwen36-aura-head-to-head.md`](notes/qwen36-aura-head-to-head.md) · [`results/`](results/) |
| 2026-08-11 | **Quantization runtime measured, not guessed.** Full PrismaQuant aura pipeline on a 4B model: 24 m 26 s on GB10 (GPTQ production-cache 504 s + KL-adjoint cost 661 s dominate; both scale with linear-module count). Per-linear extrapolation — required because the 27B target is hybrid linear-attention (48/64 layers), which breaks per-parameter scaling — puts a **27B dense run at 1.1–3.4 h** (worst case 12–15 h if the max-act-rows-cap assumption fails). Serial-local quantization is viable; external GPUs are not on the critical path — validation (~8 h per full GSM8K arm) is the real time sink. | [`notes/quant-runtime-probe.md`](notes/quant-runtime-probe.md) |
| 2026-08-06 | **Hadamard rotation does not rescue amax calibration.** Real sink tokens carry *multiple* outlier channels; their rotated contributions add up and block scales still underflow (99.8% bulk erasure even at moderate sinks). Rotation + scale 1.0 preserves outliers 5.7× better at +8% relative bulk error — a tool worth keeping, but with quality already at parity there is nothing to repair. | [`probes/nvfp4_hadamard_probe.py`](probes/nvfp4_hadamard_probe.py) · [`results/RESULT_nvfp4_hadamard_probe.json`](results/RESULT_nvfp4_hadamard_probe.json) |
| 2026-08-06 | **GB10 production numbers** (27B model, NVFP4 KV, MTP): ~20 tok/s per session, **135 tok/s aggregate at 8 concurrent sessions** with sub-second first-token latency on short prompts; prefill ~1,200 tok/s up to 32k context, degrading to ~510 tok/s at 229k (first token after ~7.5 min at full context — physics, not misconfiguration). | [`benchmarks/spark_bench.py`](benchmarks/spark_bench.py) · [`results/`](results/) |
| 2026-08-06 | **Unified-memory gotcha:** on GB10, vLLM's `gpu-memory-utilization` sizes the KV pool against whatever else is resident — same value, wildly different pools depending on service start order. Use `--kv-cache-memory-bytes` for reproducible sizing (and expect ~10% loss to block rounding + speculative-decoding layers). | [`recipes/`](recipes/) |
| 2026-08-01 | **NVFP4 weights cost no accuracy vs. INT4-Marlin — same model, 117B MoE.** Two ~4-bit weight quants of `Laguna-S-2.1` (117.6B MoE, 8.5B active), identical serving config, fp8 KV pinned, sequential on the same GB10: HumanEval pass@1 96.7% vs 95.0% (n=120, screening — inside noise), hard agentic scenarios at full parity (3/3 solved, 1/1 recovered, deploy discipline clean, NVFP4 hygiene 0.97 vs 0.92), decode 19.5 vs 19.6 tok/s. Direction, if anything, favors NVFP4. Weights axis only — complementary to the NVFP4-KV line below. | [`notes/laguna-nvfp4-vs-int4.md`](notes/laguna-nvfp4-vs-int4.md) · [`results/`](results/) |
| 2026-08-01 | **KV-cache calibration hurts at 4 bit.** Per-tensor amax scales, best practice at fp8, erase the value-cache *bulk* of attention-sink layers at NVFP4 (block scales underflow fp8-e4m3): calibrated 93.25% vs uncalibrated **94.92%** GSM8K, full 1319-item split, McNemar p=0.013. Serve uncalibrated; let scale 1.0 clip the rare outliers — needle retrieval to 240k tokens shows no damage. | [`probes/nvfp4_calib_scale_study.py`](probes/nvfp4_calib_scale_study.py) · [`results/`](results/) · [discussion](https://github.com/vllm-project/llm-compressor/issues/2936) |

## Layout

- **`nodes/`** — profiles of the lab's machines: production setup, model
  selection rationale, and what each box does besides research
- **`probes/`** — model-free kernel round-trip studies (run against an installed
  vLLM binding in minutes; deterministic, self-contained)
- **`results/`** — raw JSON exactly as measured, provenance included
- **`benchmarks/`** — end-to-end serving benchmarks (TTFT, tok/s, concurrency)
- **`recipes/`** — the serving configurations the measurements led to
- **`notes/`** — internal findings and decisions that aren't a probe/
  benchmark/recipe on their own (production decisions, incident write-ups,
  cross-references, the upstream-engagement record)

## Method

Measure, then believe. Screening at n=250 is triage only (±2.7 pp on GSM8K);
verdicts take the full n=1319 split and paired statistics. Mechanisms get
model-free kernel round-trips so they stand independent of any benchmark.
When a result kills a nice idea, the idea stays dead and the numbers stay up.

## Changelog

Retroactive back to the campaign start; going forward this table gets a row
per change to this repo. Newest first.

| Date | Change | Why | Detail |
|---|---|---|---|
| 2026-08-20 | Home-lab topology sketch added (`nodes/homelab.md`): cluster (main node + two Talos workers), GB10, bridge Mac, switch — with the role split spelled out | the node profiles needed the map they live on | [`nodes/homelab.md`](nodes/homelab.md) |
| 2026-08-20 | Node profiles added (`nodes/`): GB10 and RTX 5090 — production fleet, model rationale, driver history, use cases; framing table now links them | the notebook showed the measurements but not the machines they serve; the 595-driver dead end and the two-tier gateway architecture deserved a written home | [`nodes/dgx-spark-gb10.md`](nodes/dgx-spark-gb10.md) · [`nodes/rtx-5090.md`](nodes/rtx-5090.md) |
| 2026-08-20 | GDN first-step crash documented; warmup added to every boot path | first-traffic concurrency killed the production engine; the mitigation is one request | [`notes/gdn-first-step-crash.md`](notes/gdn-first-step-crash.md) |
| 2026-08-20 | DFlash2-on-sm121 first-light note added (port recipe, measurements, ngram parser-corruption finding, batch caveat); night-run raw JSONs added to results/; framing table updated on both rows (speculation nuance) | document the speculation re-evaluation before the verdict-level gates run; the framing table's "speculation off" claims needed mechanism-level nuance | [`notes/dflash2-sm121-first-light.md`](notes/dflash2-sm121-first-light.md) |
| 2026-08-13 | PR #80 review-cycle note added (two review rounds → merged upstream, lessons generalized); upstream-contributions extended (flashinfer#3684 validation + merge, vLLM#46329 production datapoint); two findings rows | close out the linear-attention fix story with its lessons; keep the upstream record current | [`notes/prismaquant-pr80-review-cycle.md`](notes/prismaquant-pr80-review-cycle.md) · [`notes/upstream-contributions.md`](notes/upstream-contributions.md) |
| 2026-08-12 | Qwen3.6-27B AURA head-to-head vs. reference measured (screening battery, raw JSON in results/); upstream fix PR for hybrid linear-attention masking on transformers >= 5.15 prepared | gate for the stack-recommendation lane; the fix unblocks PrismaQuant on current transformers | [`notes/qwen36-aura-head-to-head.md`](notes/qwen36-aura-head-to-head.md) |
| 2026-08-12 | Findings and changelog re-sorted newest-first; changelog "Detail" column linkified to match the findings table | surface recent changes faster; consistent linking across both tables | none |
| 2026-08-11 | Quantization runtime probe: 4B pipeline run measured on GB10, extrapolated to 27B dense (1.1–3.4 h) | kill the last runtime unknown before scheduling next-model quant windows; decides that external GPUs are off the critical path | [`notes/quant-runtime-probe.md`](notes/quant-runtime-probe.md) |
| 2026-08-11 | `recipes/build-stack.md` mount paths genericized to `$WORK_DIR` | keep host-specific paths out of the public notebook | [`recipes/build-stack.md`](recipes/build-stack.md) |
| 2026-08-11 | Framing table (Spark row) rewritten to lead with the current config; MTP-off demoted from headline to a linked reference | the front table should show the current state, not a disabled-feature narrative | [`notes/mtp-tool-calling.md`](notes/mtp-tool-calling.md) |
| 2026-08-11 | Changelog table added to this README | establish changelog discipline going forward | none |
| 2026-08-11 | Build stack documented (image line, 8-commit ch2lab stack, FlashInfer pins, local patch) | capture the current source-of-truth for the sm12x custom build | [`recipes/build-stack.md`](recipes/build-stack.md) |
| 2026-08-11 | GDN sm121 prefill datapoint measured, posted on vLLM#50288, discarded for production | +2.7–6.3% prefill, shrinking with context — not worth a 9th ch2lab commit | [`notes/gdn-prod-decision.md`](notes/gdn-prod-decision.md) |
| 2026-08-11 | Overflow banner removed from the LiteLLM callback (cluster-side) | banner text looping through replayed context caused a multi-hour generation runaway | [`notes/banner-runaway.md`](notes/banner-runaway.md) |
| 2026-08-11 | GB10 OOM-hardening (swappiness=10, earlyoom, per-container caps, oom_score_adj) | repeated thrash-wedge required physical reboots | [`notes/gb10-oom-hardening.md`](notes/gb10-oom-hardening.md) |
| 2026-08-11 | `recipes/dgx-spark-sm121.md` corrected to match the on-box script (MTP off, current flags) | recipe had drifted from source-of-truth | [`recipes/dgx-spark-sm121.md`](recipes/dgx-spark-sm121.md) |
| 2026-08-09 | Prefix caching enabled on sm121 | avoid re-prefilling the full agent context every turn | [`recipes/dgx-spark-sm121.md`](recipes/dgx-spark-sm121.md) |
| 2026-08-09 | MTP speculative decoding disabled on sm121 (Spark) | `qwen3_5_mtp` × tool-calling caused empty / aborting tool-call chains | [`notes/mtp-tool-calling.md`](notes/mtp-tool-calling.md) |
| 2026-08-06 | GB10 production benchmark captured (NVFP4-KV + MTP, 262k context) | validate the sm121 target config before rollout | [`benchmarks/spark_bench.py`](benchmarks/spark_bench.py) |
| 2026-08-06 | Hadamard/R2-rotation probe run; follow-up comment posted on llm-compressor#2936 | test a SpinQuant/R2 suggestion against the calibration finding | [`probes/nvfp4_hadamard_probe.py`](probes/nvfp4_hadamard_probe.py) |
| 2026-08-02 | FYI draft for vLLM#46329 written and reviewed, parked (not posted) | same finding, PR context — held back pending further review | [`notes/upstream-contributions.md`](notes/upstream-contributions.md) |
| 2026-08-02 | Calibration finding posted as a comment on llm-compressor#2936 | share a measured KV-cache-side NVFP4 accuracy contributor | [`notes/upstream-contributions.md`](notes/upstream-contributions.md) |
| 2026-08-01 | NVFP4 KV-cache calibration campaign run; baked amax scales found to hurt at 4-bit | investigate a reported quality deficit on calibrated NVFP4-KV | [`probes/nvfp4_calib_scale_study.py`](probes/nvfp4_calib_scale_study.py) |

## Context

Kernel and serving work this builds on: FlashInfer sm12x kernels
([flashinfer#3684](https://github.com/flashinfer-ai/flashinfer/pull/3684)),
vLLM consumer-Blackwell NVFP4-KV enablement
([vllm#46329](https://github.com/vllm-project/vllm/pull/46329)), and the
calibration discussion in
[llm-compressor#2936](https://github.com/vllm-project/llm-compressor/issues/2936).

---

*Personal project, run on my own time and hardware; unrelated to my day job.
Licensed Apache-2.0.*
