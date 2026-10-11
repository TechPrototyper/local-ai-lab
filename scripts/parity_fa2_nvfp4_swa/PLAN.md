# Kernel Parity Probe: non-causal + SWA on the FA2-NVFP4 path — SCAFFOLD

**Status: scaffold / preparation. NO verdict.** (2026-09-01, night)

## Why
The SWA guard (our contribution, #53979) keeps **NVFP4-KV + DFlash2** separate:
the DFlash2 drafter is all-layers sliding-window (`sliding_window: 2048`), and
**non-causal + window on the FA2-NVFP4 path has no kernel parity test yet**
(jethac's finding). The combination *serves* on sm121 byte-identically greedy —
strong empirical evidence — but the formal probe is the upstream commitment. Until it exists:
`turbo = fp8-KV`, `max-pool = nvfp4-KV no-spec`. **Writing this probe is
"the next work item".**

## Tooling (available, local, CPU)
- `results/byteorder_cpu_proof.py` — already mirrors vLLM's
  `dequant_nvfp4_kv_cache` (tests/kernels/quantization/nvfp4_utils.py) and proves
  shim-vs-real-page agreement WITHOUT a GPU (pure torch 2.7 CPU on the Mac).
- `results/RESULT_nvfp4_byteorder_verdict.json` — shows: vLLM stores block
  scales 4x4-SWIZZLED; read linearly ⇒ wrong drafter KV. This is the
  basis on which the SWA/non-causal parity builds.
- jethac `dequant_nvfp4_kv_cache(..., swizzle_sf=...)` @ `7a5cf143` (branch
  `nvfp4-fa2-consumer-blackwell`) — the reference reader for the GPU side.

## What the probe must show
For the non-causal + sliding-window(2048) mask on the FA2-NVFP4 path:
1. **Reference** (dequant → dense attention in fp32/bf16, with the same SWA mask
   and non-causal structure) vs.
2. **Kernel** (FA2-NVFP4 with `use_trtllm_attention:false`, as in prod) —
   outputs compared with `torch.testing.assert_close` at realistic tolerance, across
   several head sizes (128/256) and window positions (inside/beyond the 2048 boundary).

## TODO (tomorrow, supervised — GPU needed for the kernel arm)
- [ ] Flesh out `probe_skeleton.py`: reference arm (CPU, derived from byteorder_cpu_proof)
      + kernel-arm hook (GPU, jethac's reader). **Currently only the CPU reference
      + a stub for the kernel arm — deliberately no invented result.**
- [ ] Test vectors: non-causal Q/K/V, SWA mask 2048, build in the sink outlier ~125k.
- [ ] Agree on `assert_close` tolerances with jethac (review grade).
- [ ] Result → #53979 thread (ONLY after Tim's go; publishing is manual).

## Scope limits
No solo verdict. The kernel arm needs a GPU + coordination with jethac; the result
is review-relevant for upstream. The scaffold delivers the CPU reference half + the
structure, so that the supervised run tomorrow only has to attach the kernel arm.
