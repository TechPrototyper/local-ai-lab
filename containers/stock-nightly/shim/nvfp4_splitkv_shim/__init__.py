# Split-KV gate lift for NVFP4 KV prefill on sm12x, as a vLLM general plugin.
# Only active with VLLM_NVFP4_KV_SPLIT_KV=1; fails closed if the FlashInfer symbol is gone.
# Measured on the GB10 (2026-10-11): no effect (within +/-1 %); kept for parity with the old fork line.
import logging
import os

logger = logging.getLogger(__name__)


def register():
    if os.environ.get("VLLM_NVFP4_KV_SPLIT_KV") != "1":
        return
    import flashinfer.prefill as p
    if not callable(getattr(p, "_nvfp4_kv_requires_disabled_split_kv", None)):
        raise RuntimeError("FlashInfer split-KV gate symbol missing; nvfp4_splitkv_shim stale")
    p._nvfp4_kv_requires_disabled_split_kv = lambda *a, **k: False
    logger.warning("nvfp4_splitkv_shim: NVFP4 split-KV prefill gate lifted (VLLM_NVFP4_KV_SPLIT_KV=1)")
