# DFlash2 on GB10 as a portable container — one-command repro (2026-08-25)

**TL;DR:** The production GB10 DFlash2 stack — Qwen3.8-27B (PrismaAQUA 5.5-bit) +
DFlash2 speculative decoding (fp8 drafter, draft length 7) + fp8 KV — now runs
from a **single self-contained container** with **two public models auto-pulled
from HuggingFace**, no source-mount contract and no build. On a DGX Spark it's
one `docker run`. End-to-end verified on GB10: boot clean, quality intact,
tool-calling clean under speculation, **mean accept 87.8%**, **913,848-token KV
pool @ 262k context**, ~39–47 tok/s single-stream.

## Why

Everything else in this lab's DFlash2 line
([first-light](dflash2-sm121-first-light.md), [drafter
fp8](dflash2-drafter-fp8-quant.md), [full-split
verdict](dflash2-full-split-verdict.md)) runs from a **source-mount contract** —
vLLM + FlashInfer mounted into the base image at runtime, a drafter checkpoint
on local disk. Great for a lab that iterates on the source; useless for handing
the exact stack to someone else. This bakes the same stack into one image and
points it at public weights, so anyone with a GB10 can reproduce the production
config without cloning, building, or staging anything.

## What's in it (all public components)

- **Base:** `f4c27c0da` + the [ch2lab
  #50288](https://github.com/vllm-project/vllm/pull/50288) NVFP4-KV fix stack, on
  [flashinfer #3684](https://github.com/flashinfer-ai/flashinfer/pull/3684).
- **Speculation:** DFlash2 (the `dflash2-sm121` line — cherry-picks of
  [#52816](https://github.com/vllm-project/vllm/pull/52816) /
  [#53122](https://github.com/vllm-project/vllm/pull/53122)) — **baked**, not
  mounted. The runtime's editable-install paths (`/vllm-src`, `/fi`) are filled
  at build time, so the image needs no mounts.
- **KV:** fp8 — DFlash2's validated KV path (the non-causal drafter attention is
  still incompatible with NVFP4 KV on the current sm12x FlashInfer path; see
  [first-light](dflash2-sm121-first-light.md)).

No proprietary kernel work is in the image — it is public upstream plus this
lab's already-published DFlash2 line, assembled.

## The two public models

- **Target:**
  [`rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm)
  — PrismaQuant AQUA, ~5.5 bpp, Apache-2.0.
- **Drafter:**
  [`TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm`](https://huggingface.co/TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm)
  — this lab's fp8 (compressed-tensors, per-channel) quant of the DFlash2 drafter
  (see [drafter fp8](dflash2-drafter-fp8-quant.md)).

vLLM auto-downloads both on first serve (~27 GB); point the HF cache at a volume
so it persists.

## Quickstart (DGX Spark / GB10, arm64)

```bash
docker run -d --name vllm-spark --restart unless-stopped \
  --gpus all --ipc=host --memory=64g --memory-swap=64g -p 8000:8000 \
  -v ~/hf-cache:/root/.cache/huggingface \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python3 \
  ghcr.io/techprototyper/vllm-sm12x:sm121-dflash2-f4c27c0da \
  -m vllm.entrypoints.openai.api_server \
  --model rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm --served-model-name qwen3.8-27b \
  --trust-remote-code --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}' \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"dflash","model":"TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm","quantization":"compressed-tensors","num_speculative_tokens":7}' \
  --enable-prefix-caching --max-model-len 262144 --max-num-seqs 32 \
  --gpu-memory-utilization 0.6 --enable-chunked-prefill --max-num-batched-tokens 16384 \
  --host 0.0.0.0 --port 8000
```

First boot pulls the weights (~15–20 min). Then fire **exactly one** warmup
request before opening to traffic — the GDN first-step discipline
([why](gdn-first-step-crash.md)):

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Sag OK"}],"max_tokens":8}'
```

## Verified end-to-end (GB10, 2026-08-25)

| Check | Result |
|---|---|
| Boot (drafter loads + compiles, **no mounts**) | clean, ~16 min |
| Quality (factual + arithmetic) | correct |
| Tool-calling under speculation | clean, well-formed JSON, `finish=tool_calls` |
| Speculation | mean acceptance length 3.8–7.14, **avg accept 87.8%** |
| KV pool | **913,848 tokens** (3.49× @ 262k, fp8) |
| Single-stream | ~39–47 tok/s |
| Prefix caching | enabled, but **does not hit under speculation** on this hybrid-GDN model (Mamba `align` mode) — a known engine limit, not a correctness issue |

## Notes

- The tag `ghcr.io/techprototyper/vllm-sm12x:sm121-dflash2-f4c27c0da` is a
  **temporary convenience** and will be retired; the reproducible truth is this
  recipe plus the two public models. It sits under the `vllm-sm12x` package,
  which is otherwise the **NVFP4-KV mount-contract** line — this baked
  full-stand image is deliberately a different thing.
- `VLLM_ATTENTION_BACKEND` is a no-op in this image (FlashInfer is the default) —
  omit it.
