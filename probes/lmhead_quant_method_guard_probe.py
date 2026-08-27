"""#41 probe: LM-head quant-method dispatch in get_top_k_tokens (vllm#52883).

Runs CPU-only inside the serving image. Three checks:
  A) UnquantizedEmbeddingMethod head -> get_top_k_tokens works (control).
  B) UnquantizedLinearMethod head (jschmied's scenario: quant config that
     leaves the head unquantized) -> works on the merged #52816 path.
  C) head_dtype=float32 + UnquantizedLinearMethod -> the residual isinstance
     trap in _apply_head fires (same class confusion, surviving call site).
"""
import torch
import torch.nn as nn
from vllm.config import VllmConfig, DeviceConfig
from vllm.config.vllm import set_current_vllm_config

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

VOCAB, HID, K = 512, 64, 8


class HeadStub(nn.Module):
    """Duck-typed stand-in for ParallelLMHead: only the attributes
    get_top_k_tokens actually touches."""

    class _Idx:
        num_org_vocab_padding = 0
        org_vocab_start_index = 0

    tp_size = 1
    shard_indices = _Idx()

    def __init__(self, quant_method):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB, HID))
        self.quant_method = quant_method
        if isinstance(quant_method, UnquantizedLinearMethod):
            # UnquantizedLinearMethod.apply reads layer.weight
            pass


def run(label, quant_method, head_dtype=None, hs_dtype=torch.float32):
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device='cpu'))):
        lp = LogitsProcessor(VOCAB)
    if head_dtype is not None:
        lp.head_dtype = head_dtype
    head = HeadStub(quant_method)
    hs = torch.randn(4, HID, dtype=hs_dtype)
    try:
        ids, vals = lp.get_top_k_tokens(head, hs, K)
        print(f"{label}: OK ids={tuple(ids.shape)} vals={tuple(vals.shape)}")
        return "OK"
    except Exception as e:
        print(f"{label}: RAISED {type(e).__name__}: {e}")
        return f"RAISED {type(e).__name__}"


ref = torch.nn.functional.linear(torch.randn(1, HID), torch.randn(VOCAB, HID))
a = run("A control  UnquantizedEmbeddingMethod            ", UnquantizedEmbeddingMethod())
b = run("B jschmied UnquantizedLinearMethod               ", UnquantizedLinearMethod())
c = run("C residual UnquantizedLinearMethod + head_dtype  ", UnquantizedLinearMethod(),
        head_dtype=torch.float32, hs_dtype=torch.bfloat16)
d = run("D control  UnquantizedEmbeddingMethod + head_dtype", UnquantizedEmbeddingMethod(),
        head_dtype=torch.float32, hs_dtype=torch.bfloat16)
print("VERDICT:", {"A": a, "B": b, "C": c, "D": d})
