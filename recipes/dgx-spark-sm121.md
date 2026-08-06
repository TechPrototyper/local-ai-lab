# DGX Spark / GB10 (sm_121) — quality → speed → memory

Same 27B-class model and NVFP4 KV cache as the RTX recipe — opposite dials.

```
vllm serve <model-fp8> \
  --kv-cache-dtype nvfp4 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}' \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.44 \
  --kv-cache-memory-bytes 23622320128 \
  --enable-chunked-prefill --max-num-batched-tokens 16384
```

Key decisions, each measured:

- **MTP speculative decoding ON** (the model ships `mtp.*` weights, so it is
  flag-only): +69% single-stream (≈11 → ≈19–20 tok/s), lossless at temp 0,
  best quality scores of the whole campaign. Bandwidth is this box's
  bottleneck; the KV cost of MTP is irrelevant with 128 GB unified memory.
- **Explicit KV pool via `--kv-cache-memory-bytes`** — on unified memory,
  `gpu-memory-utilization` sizes the pool against every other resident
  process, so identical values yield wildly different pools depending on
  start order (measured: 1.84M vs 0.49M tokens at the same 0.52). With
  22 GiB explicit: 1.09M tokens ≈ 4× 262k-token sessions in parallel.
  Budget ~10% over the nominal target for block rounding + the MTP layer.
- `gpu-memory-utilization` remains as a *startup check* (that fraction of
  total memory must be free at launch or vLLM refuses to start) — size it to
  actual need (weights + pool + activations), not to the pool.
- Measured serving numbers: ~20 tok/s per session, 135 tok/s aggregate at 8
  concurrent sessions, TTFT 0.3–0.7 s on short prompts; prefill ~1,200 tok/s
  to 32k, ~510 tok/s at 229k.
