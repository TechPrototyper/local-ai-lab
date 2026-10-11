# Recipe: Qwen3.8-27B drafter-free MTP-Turbo on RTX 5090 (sm120)

> **Historical (August 2026).** Production left this stack on 2026-10-11. Current
> image, serve commands and pitfalls: [`containers/stock-nightly/`](../containers/stock-nightly/).
> Kept for reproducing the August findings.

**What:** Qwen3.8-27B as a GridBook-13GB quant, with native **MTP** (multi-token
prediction) instead of an external drafter, NVFP4-KV, prefix caching under speculation.
Result: drafter-free, **~624k token KV pool**, suitable for agentic use (tool calling +
prefix-cache hits under spec), quality-equivalent to no-spec (n=1319, p=0.83).

**Hardware:** 1× RTX 5090 (sm120, 32 GB), exclusive (no GPU split).

---

## 1. Serving image
`vllm-nvfp4:sm120-pc50897-gridbook-088` — vLLM V1 (sm120-NVFP4 line) with:
- **vllm#50897** pre-integrated (lookahead-aware prefix-cache hashing → prefix cache
  works under speculation; without it ~0% hits, vllm#52244).
- **GridBook plugin 0.8.8** (`vllm.general_plugins` entry point → `quantization=gridbook`).
- non-causal NVFP4 seam hardening (vllm#53977/#53978/#53979).
- FlashInfer backend (`VLLM_ATTENTION_BACKEND=FLASHINFER`), warm JIT/autotune cache.

## 2. Model artifact: `gridbook-13gb-mtpfix`
GridBook-13GB keeps the native MTP heads but quantizes the shared body embedding —
which breaks vLLM's MTP draft (it shares the target embedding). Fix (bit-exact, self-serve):
1. Base: `rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB` (NVFP4-CB, ~13 GB).
2. Drop the 3 packed `model.language_model.embed_tokens.{weight_packed,weight_scale,
   weight_global_scale}`; insert the **BF16 `embed_tokens.weight` from the AQUA quant**
   (`Qwen3.8-27B-PrismaAQUA-5.5bit`, identical Qwen3.8 base) verbatim.
3. `quant_config.json`: remove the `quantized_embedding` key, put the unit in `ignore`.
   (Script: `surgical_embed_swap.py` in the lab repo.)
Correctness proven via the n=1319 verdict (not via byte equality — MTP spec is
not byte-identical to no-spec due to FP effects, but is distribution-equivalent).

## 3. Serve command (the authoritative config)
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/gridbook-13gb-mtpfix \
  --served-model-name qwen3.8-27b \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}' \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --async-scheduling --enable-prefix-caching --enable-chunked-prefill
```
Env: `VLLM_ATTENTION_BACKEND=FLASHINFER`, HF cache/FlashInfer workspace mounted.

**Critical:**
- **`--gpu-memory-utilization 0.90`**, NOT 0.97. At 0.97 the engine crashes
  reproducibly (fragmentation OOM under long-context load, e.g. needle 12k–64k):
  vLLM's static KV reservation leaves too little slack for activations + allocator
  fragmentation beyond the profiling estimate. 0.90 is stable.
- **GPU exclusive** — with a shared GPU, vLLM profiles against less memory (534k instead of 624k).
- **Keep `--enable-prefix-caching`** — essential for agentic dialogues; thanks to
  #50897 it works under MTP too (measured ~77% hit rate).

## 4. Expected values (measured, RTX 5090)
| Axis | Value |
|---|---|
| KV pool | **624,152 tokens** (2.38× concurrency @ 262,144/request), util 0.90, exclusive |
| Quality (n=1319, MTP vs no-spec, greedy, paired) | Acc 0.9704 vs 0.9719 · McNemar p=0.8318 (equivalent) |
| Battery | Tools 6/6 · GSM8K-250 0.9840 · Needle 12k/24k/64k 9/9 · Determinism 5/5 |
| MTP-Turbo (AQUA-measured, transfers) | Prose +49% @ 54.3% acceptance; count200 +112%; JSON +72% |
| Prefix cache (under MTP) | ~77% hit rate (queries 887k / hits 684k) |

## 5. Known levers / open
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` might make higher util stable
  → pool recovery beyond the 624k. Not yet tested.
- Optimum sampling/thinking for agentic tool calling: thinking-ON (sweep finding),
  the loop fix lies in the agent layer (guardrails/steering), not in sampling.

See also the full cutover documentation: `notes/rtx-gridbook-mtp-cutover-2026-08-31.md`.
