# Build-time check: fails the build if the FlashInfer wrapper (FI PR 5189) or vllm#46558 is missing.
import importlib.metadata as m

import flashinfer
import torch
import vllm
from flashinfer import BatchPrefillWithCausalBidirectionalRangesWrapper  # noqa: F401
from vllm.v1.attention.backends import flashinfer as fib

assert hasattr(fib, "_native_mm_prefix"), "vllm#46558 not applied"
print("vllm", vllm.__version__, "flashinfer", flashinfer.__version__, "torch", torch.__version__)
for p in ("flashinfer-cubin", "flashinfer-jit-cache", "gridbook"):
    try:
        print(p, m.version(p))
    except m.PackageNotFoundError:
        print(p, "-")
print("plugins", [e.name for e in m.entry_points(group="vllm.general_plugins")])
