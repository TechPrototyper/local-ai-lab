# SPDX-License-Identifier: Apache-2.0
# TurboQuant-KV vs NVFP4-KV: value-cache fidelity, model-free round-trip.
# Same input generator as the calibration-scale study (sink-like outliers,
# measured provenance amax ~42k / ~125k). NVFP4 arm = production setting
# (scale 1.0, group-16 fp8-e4m3 block scales) via installed
# reshape_and_cache_flash + independent dequant. TurboQuant arm = installed
# triton_turboquant_store + decode with query=key (softmax over a single
# key = 1.0 -> attention output IS the dequantized value, key error drops
# out). Metric: rel-L2 split into BULK vs OUTLIER parts + bytes/token/head.
import json
import math

import torch

import vllm  # noqa: F401
from vllm._custom_ops import reshape_and_cache_flash
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import (
    triton_turboquant_store,
)

DEV = "cuda:0"
HEAD_SIZE = 128
KV_HEADS = 4
NUM_TOKENS = 512
BLOCK_SIZE = 16
DATA_DIM = HEAD_SIZE // 2
SCALE_DIM = HEAD_SIZE // 16

kE2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                     dtype=torch.float32, device=DEV)


def break_fp4_bytes(a):
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2)


def nvfp4_value_roundtrip(value):
    """[T,H,D] bf16 -> nvfp4 cache -> float32 reconstruction, scale 1.0."""
    from vllm.utils.torch_utils import nvfp4_split_data_scale
    num_blocks = (NUM_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE
    full_dim = DATA_DIM + SCALE_DIM
    key = torch.zeros_like(value)
    kc = torch.zeros(num_blocks, BLOCK_SIZE, KV_HEADS, full_dim,
                     dtype=torch.uint8, device=DEV)
    vc = torch.zeros_like(kc)
    slot_mapping = torch.arange(NUM_TOKENS, dtype=torch.long, device=DEV)
    scale = torch.tensor([1.0], dtype=torch.float32, device=DEV)
    reshape_and_cache_flash(key, value, kc, vc, slot_mapping, "nvfp4",
                            scale, scale)
    data_u8, sf_u8 = nvfp4_split_data_scale(vc.view(num_blocks, BLOCK_SIZE,
                                                    KV_HEADS, full_dim))
    sf = sf_u8.permute(0, 2, 1, 3).contiguous()
    sf_f32 = sf.view(torch.float8_e4m3fn).to(torch.float32)
    d = data_u8.permute(0, 2, 1, 3).contiguous()
    vals = break_fp4_bytes(d.reshape(-1, DATA_DIM)).reshape(
        num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    out = (vals.reshape(num_blocks, KV_HEADS, BLOCK_SIZE, SCALE_DIM, 16)
           * sf_f32.unsqueeze(-1)).reshape(num_blocks, KV_HEADS,
                                           BLOCK_SIZE, HEAD_SIZE)
    out = out.permute(0, 2, 1, 3).reshape(-1, KV_HEADS, HEAD_SIZE)
    return out[:NUM_TOKENS]


def _build_hadamard(d):
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(DEV)


def turboquant_value_roundtrip(value, preset):
    """[T,H,D] bf16 -> TQ cache -> reconstruction via 1-key decode."""
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=HEAD_SIZE)
    H = _build_hadamard(HEAD_SIZE)
    centroids, _ = solve_lloyd_max(HEAD_SIZE, cfg.centroid_bits)
    centroids = centroids.float().to(DEV)
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(DEV)

    # one token per block -> each "sequence" sees exactly its own token
    kv_cache = torch.zeros(NUM_TOKENS, BLOCK_SIZE, KV_HEADS,
                           cfg.slot_size_aligned, dtype=torch.uint8,
                           device=DEV)
    slot_mapping = (torch.arange(NUM_TOKENS, dtype=torch.int32, device=DEV)
                    * BLOCK_SIZE)
    v16 = value  # bf16 direkt — Kernel akzeptiert bf16; fp16-Cast wuerde
    # >65k-Magnituden schon VOR dem Kernel zerstoeren (Probe-Artefakt).
    # keys: unit-norm random (well-conditioned queries, error drops out)
    torch.manual_seed(99)
    key = torch.randn(NUM_TOKENS, KV_HEADS, HEAD_SIZE, device=DEV,
                      dtype=torch.bfloat16)
    key = key / key.norm(dim=-1, keepdim=True)
    triton_turboquant_store(key, v16, kv_cache, slot_mapping, H, midpoints,
                            mse_bits=cfg.key_mse_bits,
                            key_packed_size=cfg.key_packed_size,
                            value_quant_bits=cfg.effective_value_quant_bits,
                            key_fp8=cfg.key_fp8)
    query = key.contiguous()
    block_table = torch.arange(NUM_TOKENS, dtype=torch.int32,
                               device=DEV).unsqueeze(1)
    seq_lens = torch.ones(NUM_TOKENS, dtype=torch.int32, device=DEV)
    out = triton_turboquant_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, Pi=H, centroids=centroids,
        scale=1.0 / math.sqrt(HEAD_SIZE), mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=H)
    return out.to(torch.float32), cfg


def rel_l2(a, b):
    denom = torch.linalg.vector_norm(b)
    return float("nan") if denom.item() == 0 else \
        (torch.linalg.vector_norm(a - b) / denom).item()


def make_value(amax, outlier_tokens=4, outlier_channels=2):
    torch.manual_seed(7)
    v = torch.randn(NUM_TOKENS, KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16,
                    device=DEV)
    mask = torch.zeros_like(v, dtype=torch.bool)
    if amax:
        g = torch.Generator(device="cpu").manual_seed(11)
        toks = torch.randperm(NUM_TOKENS, generator=g)[:outlier_tokens]
        chans = torch.randperm(HEAD_SIZE, generator=g)[:outlier_channels]
        for t in toks:
            for h in range(KV_HEADS):
                for c in chans:
                    v[t, h, c] = amax * (1 if (t + h + c) % 2 else -1)
                    mask[t, h, c] = True
    return v, mask


def split_metrics(recon, ref, mask):
    ref32 = ref.to(torch.float32)
    bulk = rel_l2(recon[~mask], ref32[~mask])
    outl = rel_l2(recon[mask], ref32[mask]) if mask.any() else None
    return {"bulk_rel_l2": round(bulk, 5),
            "outlier_rel_l2": round(outl, 5) if outl is not None else None}


def main():
    scenarios = [("gaussian", None), ("sink_amax42k", 42000.0),
                 ("sink_amax125k", 125000.0)]
    presets = ["turboquant_k8v4", "turboquant_4bit_nc", "turboquant_3bit_nc"]
    results = {"head_size": HEAD_SIZE, "kv_heads": KV_HEADS,
               "tokens": NUM_TOKENS, "arms": {}}
    for name, amax in scenarios:
        v, mask = make_value(amax)
        arm = {}
        recon = nvfp4_value_roundtrip(v)
        arm["nvfp4_scale1"] = split_metrics(recon, v, mask)
        for p in presets:
            recon, cfg = turboquant_value_roundtrip(v, p)
            m = split_metrics(recon, v, mask)
            m["slot_bytes_aligned"] = cfg.slot_size_aligned
            arm[p] = m
        results["arms"][name] = arm
        print(name, json.dumps(arm))
    # bytes/token/head: nvfp4 = data 64 + scales 8 (per K and V each)
    results["bytes_per_token_head"] = {"nvfp4_v_only": DATA_DIM + SCALE_DIM}
    with open("/probe-out/RESULT_turboquant_vs_nvfp4_value.json", "w") as f:
        json.dump(results, f, indent=1)
    print("geschrieben: RESULT_turboquant_vs_nvfp4_value.json")


if __name__ == "__main__":
    main()
