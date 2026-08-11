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
- **RTX 5090 (sm120):** FlashInfer 0.6.14

## Local patch

A local, uncommitted 16-file patch on top of the vLLM base commit —
build/requirements plumbing only (`CMakeLists.txt`, `pyproject.toml`,
`setup.py`, and various `requirements/*.txt`/`*.in`), no runtime-behavior
changes. Kept in the private ops repo, not mirrored here:
`mySpark/pipeline/patches/vllm-f4c27c0da-local-16files.patch` (+
`vllm-f4c27c0da-state.txt` for the exact commit/branch provenance of both
the `vllm` and `flashinfer` source trees).
