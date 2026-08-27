# Build stack — sm12x custom vLLM image

Provenance for the image line both recipes serve from.

## Image

`vllm-sm121:f4c27c0da` — built against vLLM commit `f4c27c0da`. **The
image does not contain vLLM itself.** vLLM and FlashInfer are supplied via
a source-mount contract at runtime:

```
-v $WORK_DIR/vllm:/vllm-src
-v $WORK_DIR/flashinfer:/fi
```

where `$WORK_DIR` is the local checkout directory holding the `vllm` and
`flashinfer` source trees. Do not delete or move these directories while a
recipe built on this image is running.

## Commit stack

8-commit cherry-picked stack on top of vLLM#46329 (superset of the
consumer-Blackwell NVFP4-KV enablement work), **without** the GDN
(Gated-DeltaNet) prefill-kernel commit — GDN was measured as a 9th-commit
candidate and discarded for production; see
[`../notes/gdn-prod-decision.md`](../notes/gdn-prod-decision.md).

## FlashInfer pins

- **Spark (GB10, sm121):** FlashInfer 0.6.15
- **RTX 5090 (sm120):** since 2026-08-24 **FlashInfer main / 0.6.18**
  (flashinfer#3684 is merged upstream, so main carries the sm12x NVFP4
  kernels). Previous pin — 0.6.15, branch `pr3684`, commit `2ed09bd3` —
  remains the validated fallback. Note: no plain `v0.6.18` release tag
  exists (only `rc1`–`rc9`); for reproducible builds pin a main commit
  (this lab's 2026-08-27 build used `09da2e70`).

## Image lines (current, 2026-08-27)

- **sm121 (Spark):** `vllm-sm121:f4c27c0da` + source-mount contract as
  above — unchanged, serving production.
- **sm120 (RTX):** fully-baked images, no source mount:
  - `sm120-nvfp4-e2446da2-t1` — **production** since 2026-08-24
    (vllm#46329 head `e2446da2` + flashinfer main; requires the
    `use_trtllm_attention: false` flag, see the RTX recipe).
  - `sm120-v4-2cf8b8a-t2` — validation build 2026-08-27: vLLM main ∪
    #46329@`7a5cf14` ∪ the upstream package #53977/#53978/#53979
    (branch `integrate/dflash2-nvfp4-v4` @ `2cf8b8ae0` on the lab fork),
    flashinfer main @ `09da2e70`. Battery-validated (spec × NVFP4-KV);
    **not** production-promoted — that waits on a verdict-tier quality
    gate.

## Local patch

A local, uncommitted 16-file patch on top of the vLLM base commit —
build/requirements plumbing only (`CMakeLists.txt`, `pyproject.toml`,
`setup.py`, and various `requirements/*.txt`/`*.in`), no runtime-behavior
changes. Kept in the private ops repo, not mirrored here:
`mySpark/pipeline/patches/vllm-f4c27c0da-local-16files.patch` (+
`vllm-f4c27c0da-state.txt` for the exact commit/branch provenance of both
the `vllm` and `flashinfer` source trees).
