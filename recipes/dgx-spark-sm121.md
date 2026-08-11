# DGX Spark / GB10 (sm_121) — quality → speed → memory

Same 27B-class model and NVFP4 KV cache as the RTX recipe — opposite dials
in principle. In practice the speed lever that's currently on is chunked
prefill + prefix caching, **not** speculative decoding: MTP is off (see
below), corrected here 2026-08-11 after this recipe drifted from the
on-box script.

```
vllm serve <model-plain> \
  --served-model-name qwen3.6-27b \
  --kv-cache-dtype nvfp4 \
  --enable-prefix-caching \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.44 \
  --kv-cache-memory-bytes 21474836480 \
  --enable-chunked-prefill --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Container-level cap (`docker run`): `--memory=32g --memory-swap=32g`.
SoT: `mySpark/ops/run-vllm-aura.sh`.

Key decisions, each measured:

- **MTP speculative decoding OFF since 2026-08-09** (this recipe previously
  showed it on — that was stale). `qwen3_5_mtp` combined with tool-calling
  produced empty answers / aborting tool-call chains in agent workloads on
  this box only; RTX/sm120 is unaffected (has run without MTP since 07-12,
  which rules out MTP itself as inherently broken — the failure is specific
  to this box/build). The flag is kept reversible, commented out in the
  source script:
  `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'`.
- **Plain (uncalibrated) checkpoint**, same rationale as the RTX recipe —
  baked per-tensor amax KV scales cost accuracy at 4-bit (README finding
  2026-08-01); vLLM defaults k/v scales to 1.0.
- **Explicit KV pool via `--kv-cache-memory-bytes`** — on unified memory,
  `gpu-memory-utilization` sizes the pool against every other resident
  process, so identical values yield wildly different pools depending on
  service start order. At 20 GiB explicit (21474836480 bytes): KV pool
  ≈1.08M tokens ≈4× 262k-token sessions in parallel.
- `gpu-memory-utilization 0.44` remains as a *startup check* (that fraction
  of total memory must be free at launch or vLLM refuses to start) — size
  it to actual need (weights + pool + activations), not to the pool.
- **Prefix caching** (`--enable-prefix-caching`, added 2026-08-09) — avoids
  re-prefilling the full context on every agent turn.
- Measured: GSM8K 96.4% (n=250 — screening-level; see README methodology
  note on n=250 vs. the full 1319-item verdict split).
- Decode-throughput numbers (~20 tok/s per session, 135 tok/s aggregate at
  8 concurrent sessions, TTFT 0.3–0.7 s on short prompts) were captured
  2026-08-06 under the since-corrected MTP-on config and have not yet been
  re-benchmarked without MTP — treat as historical, not current. Prefill
  numbers are unaffected by MTP and still hold: ~1,200 tok/s to 32k
  context, ~510 tok/s at 229k.
