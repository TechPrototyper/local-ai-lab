# Containers

The lab's serving images, in one place. **Unofficial / experimental** — a
pre-merge carry line for consumer Blackwell (sm120 = RTX 5090, x86_64;
sm121 = DGX Spark / GB10, arm64). When a serving line merges upstream, prefer
an official image; the tags here stay pullable for reproducibility.

All images are public on GHCR:
[`ghcr.io/techprototyper/vllm-sm12x`](https://github.com/users/TechPrototyper/packages/container/package/vllm-sm12x).

## Published images (2026-08-28)

| Tag | Arch | What it is | Provenance |
|---|---|---|---|
| `sm120-nvfp4-e2446da2-prod` | sm120 (x86_64) | **RTX production** — NVFP4-KV serving, `use_trtllm_attention:false` load-bearing | vllm#46329 head `e2446da2` + flashinfer main |
| `sm120-v4-2cf8b8a-validation` | sm120 (x86_64) | The **reproduction image** for the upstream PRs — main ∪ #46329@`7a5cf14` ∪ #53977/#53978/#53979 | fork branch `integrate/dflash2-nvfp4-v4` @ `2cf8b8ae0`, flashinfer main @ `09da2e70` |
| `sm121-dflash2-f4c27c0da` | sm121 (arm64) | GB10 DFlash2 line, self-contained (tree + FlashInfer + warm caches baked) | base `f4c27c0da` |
| `sm121-dflash2-pc50897-dd02ed4d` | sm121 (arm64) | **GB10 production** — the above + #53122 fused-KV pick (`58f998f84`) + the vllm#50897 prefix-cache fix (`dd02ed4da1`) | see [`../notes/night-2026-08-28-pc50897-scout-h2h.md`](../notes/night-2026-08-28-pc50897-scout-h2h.md) |

All four are fully baked — **no source mounts needed** (the older
source-mount contract is retired). Full copy-paste serve commands, with every
flag and the model/drafter checkpoints, live in the recipes:

- **sm120 (RTX):** [`../recipes/rtx5090-sm120.md`](../recipes/rtx5090-sm120.md)
- **sm121 (Spark):** [`../recipes/dgx-spark-sm121.md`](../recipes/dgx-spark-sm121.md)

Pull:

```bash
docker pull ghcr.io/techprototyper/vllm-sm12x:sm120-nvfp4-e2446da2-prod         # RTX
docker pull ghcr.io/techprototyper/vllm-sm12x:sm121-dflash2-pc50897-dd02ed4d    # Spark
```

## Building

The sm121 images are built from [`Dockerfile.fullstand`](Dockerfile.fullstand)
— it copies the (public) DFlash2 vLLM tree, FlashInfer, and warm JIT/compile
caches onto a compiled base, so the result carries no runtime mounts. The tree
is a build-arg, so each new serving state is one build:

```bash
docker build -f Dockerfile.fullstand \
  --build-arg VLLM_TREE=vllm-dflash2-pc50897 \
  -t ghcr.io/techprototyper/vllm-sm12x:sm121-dflash2-pc50897-dd02ed4d .
```

The sm120 images are built on the RTX host from the #46329-carrying vLLM line;
image lines, pins, and provenance are in
[`../recipes/build-stack.md`](../recipes/build-stack.md). Both were mirrored to
GHCR via an in-cluster `skopeo copy` (amd64-correct, registry-to-registry).

## Sunset

Each image exists only because its serving line is not yet upstream:

| Upstream piece | State |
|---|---|
| [flashinfer#3684](https://github.com/flashinfer-ai/flashinfer/pull/3684) — sm12x NVFP4 paged-prefill kernels | **merged** (flashinfer main, 2026-08-13) |
| [vllm#46329](https://github.com/vllm-project/vllm/pull/46329) — consumer-Blackwell NVFP4-KV enablement | open — the reason this line still needs carry-builds |
| [vllm#53977/#53978/#53979](https://github.com/vllm-project/vllm/pull/53979) — the non-causal-NVFP4 spec-decode package | open |
| [vllm#50897](https://github.com/vllm-project/vllm/pull/50897) — prefix cache under speculation | open (validated here) |

When these land and an official image ships NVFP4-KV for sm12x, this line
becomes obsolete. Upstream engagement is tracked in
[`../notes/upstream-contributions.md`](../notes/upstream-contributions.md).
