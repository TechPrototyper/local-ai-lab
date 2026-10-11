# Stock vLLM nightly + one cherry-pick (production since 2026-10-11)

The image both serving boxes run since the [October migration](../../notes/vllm-stock-nightly-migration-2026-10.md).
It is a thin layer over an **unmodified** vLLM nightly — no vLLM or FlashInfer
source build anymore:

| Layer | What | Why |
|---|---|---|
| `vllm/vllm-openai:nightly-7d0b4e57…` | stock nightly, torch 2.13, CUDA 13, multi-arch | carries [vllm#46963](https://github.com/vllm-project/vllm/pull/46963) — NVFP4 KV on sm12x |
| FlashInfer **0.7.1** | swapped in from the FlashInfer wheel index | `BatchPrefillWithCausalBidirectionalRangesWrapper` ([FI PR 5189](https://github.com/flashinfer-ai/flashinfer/pull/5189)); drops out once [vllm#60948](https://github.com/vllm-project/vllm/pull/60948) lands |
| [vllm#46558](https://github.com/vllm-project/vllm/pull/46558) @`070ce4baf8` | [`pr46558-070ce4baf8-on-7d0b4e57.vllm.patch`](pr46558-070ce4baf8-on-7d0b4e57.vllm.patch) (5 files, applies clean) | Gemma-4 image spans (bidirectional mm-prefix) on NVFP4 KV — the vision tower is mandatory here |
| [`shim/`](shim/) | split-KV gate lift, vLLM general plugin, opt-in | parity with the old fork line; measured neutral |
| `gridbook==0.8.8` (RTX only) | [RobTand/gridbook](https://github.com/RobTand/gridbook) vLLM plugin | serves the 13 GB GridBook checkpoint; works unchanged on stock vLLM |
| [`verify.py`](verify.py) | runs at build time | fails the build if the wrapper or the patch is missing |

```bash
# GB10 / DGX Spark (sm121, arm64)
docker build --platform linux/arm64 -t vllm-stock-nightly:gb10 .
# RTX 5090 (sm120, amd64)
docker build --platform linux/amd64 --build-arg GRIDBOOK_VERSION=0.8.8 -t vllm-stock-nightly:rtx .
```

The images are not on GHCR yet; this directory is the complete build
context. Build time is minutes, not hours.

## Serve: Gemma-4-31B on the GB10 (German, vision, MTP)

Model: [TechPrototyper/Gemma4-31B-IT-PrismaQuant-6bit-abliterated-vllm](https://huggingface.co/TechPrototyper/Gemma4-31B-IT-PrismaQuant-6bit-abliterated-vllm)
(or the non-abliterated [rdtand original](https://huggingface.co/rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm)),
drafter [google/gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant).

```bash
vllm serve <gemma4-31b-checkpoint> \
  --kv-cache-dtype nvfp4 --attention-config '{"backend":"FLASHINFER"}' \
  --kv-cache-memory 20132659200 --gpu-memory-utilization 0.44 \
  --max-model-len 262144 --max-num-seqs 2 \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":4,"audio":0}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --speculative-config '{"model":"<gemma-4-31B-it-assistant>","num_speculative_tokens":4}' \
  --generation-config vllm \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64}'
```

- **Set the attention backend explicitly.** Gemma4Config otherwise routes to
  Triton, which has no NVFP4 KV path.
- `--kv-cache-memory 20132659200` (18.75 GiB) gives exactly 2 × 256Ki tokens
  (524,903 measured). On unified memory, pin the pool. Don't let
  `gpu-memory-utilization` size it.
- vLLM's start check still requires free memory ≥ util × total, even with
  `--kv-cache-memory`, so util has to fit what is *physically free* at start.
- Keep MTP on: without it, short decode falls from 16.8 to 8.0 tok/s.

## Serve: Qwen3.8-27B GridBook-13GB + vision on the RTX 5090

Model: [TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm](https://huggingface.co/TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm).

```bash
VLLM_NVFP4_KV_SPLIT_KV=1 TORCH_CUDA_ARCH_LIST=12.0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CPATH=<dir with the missing CUDA library headers, see below> \
vllm serve <gridbook-13gb-vision-checkpoint> \
  --kv-cache-dtype nvfp4 --attention-backend FLASHINFER \
  --attention-config '{"use_trtllm_attention": false}' \
  --max-model-len 262144 --max-num-seqs 8 --block-size 32 \
  --gpu-memory-utilization 0.97 \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4]}' --async-scheduling \
  --limit-mm-per-prompt '{"image": 2, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --generation-config vllm \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}'
```

- **CUDA headers.** The nightly image ships without CUDA development headers.
  GridBook JIT-compiles a small extension at first start that needs
  `cusparse*.h`, `cublas*.h` and `cusolver*.h`.
  - Copy **only those headers** into a directory and point `CPATH` at it.
  - Don't put the whole pip `nvidia/cu13/include` on `CPATH`. Its CRT
    headers clash with nvcc's (`__cudaLaunch` macro mismatch).
  - Baking the headers into the image is on the to-do list.
- Give FlashInfer, torch extensions and the GridBook extension
  (`FLASHINFER_WORKSPACE_BASE`, `TORCH_EXTENSIONS_DIR`,
  `PRISMAQUANT_CB_EXT_DIR`) caches of their own **per vLLM build**. JIT
  artifacts from the old fork are not ABI-compatible.
- `VLLM_ATTENTION_BACKEND` and `VLLM_NVFP4_GEMM_BACKEND` are not read
  anymore. Use the CLI flags.
- Result: 765,668-token KV pool, ~59 tok/s single stream, ~153 at four
  streams ([numbers](../../notes/vllm-stock-nightly-migration-2026-10.md#results)).
