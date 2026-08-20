# SPDX-License-Identifier: Apache-2.0
# NVFP4 KV online-Hadamard (>head_dim) probe (model-free, real kernel round-trip).
#
# Answers the maintainer question in llm-compressor#2936 ("larger hadamard
# size ...?"): is the Hadamard BLOCK SIZE the limiter for amax KV calibration,
# or do multi-channel sinks win at every practical size? Extends
# nvfp4_hadamard_probe.py (which uses a foldable per-head H of size 128) with
# a NON-foldable ONLINE rotation of size B in {1,128,256,512,1024} applied
# across the flattened kv-head axis (B>128 crosses head boundaries, so it
# cannot be folded into W_v/W_o and must run at write time; inverted at read
# time). Same real writer (reshape_and_cache_flash, nvfp4), same independent
# dequant, same sink provenance (42k/125k, 2 outlier channels), metrics in the
# ORIGINAL space. The amax observer measures amax in the space it sees
# (post-online-rotation), i.e. exactly what a real KV amax observer would bake.
# Pure measurement probe; the online path is not deployable as-is.
import json
import random

import torch

import vllm  # noqa: E402
from vllm.utils.torch_utils import nvfp4_split_data_scale  # noqa: E402

DEV = "cuda:0"
HEAD_SIZE = 128
KV_HEADS = 8              # 8*128 = 1024 channels -> B up to 1024
CHANNELS = KV_HEADS * HEAD_SIZE
BLOCK_SIZE = 16
NUM_BLOCKS = 64
NUM_TOKENS = 512
DATA_DIM = HEAD_SIZE // 2
SCALE_DIM = HEAD_SIZE // 16
FULL_DIM = DATA_DIM + SCALE_DIM

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def hadamard(n: int, device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h / (n ** 0.5)


def break_fp4_bytes(a: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2)


def dequant_linear(data_u8, sf_u8, global_scale):
    P, T, H, _ = data_u8.shape
    sf = sf_u8.permute(0, 2, 1, 3).contiguous()
    sf_f32 = sf.view(torch.float8_e4m3fn).to(torch.float32)
    d = data_u8.permute(0, 2, 1, 3).contiguous()
    vals = break_fp4_bytes(d.reshape(-1, DATA_DIM)).reshape(P, H, T, HEAD_SIZE)
    out = (
        vals.reshape(P, H, T, SCALE_DIM, 16)
        * (sf_f32 * global_scale).unsqueeze(-1)
    ).reshape(P, H, T, HEAD_SIZE)
    return out.permute(0, 2, 1, 3)


def rel_l2(a, b):
    denom = torch.linalg.vector_norm(b)
    if denom.item() == 0.0:
        return float("nan")
    return (torch.linalg.vector_norm(a - b) / denom).item()


def online_rotate(v_flat, Hb, block):
    """v_flat:[T, CHANNELS] -> rotate each contiguous block of size `block`."""
    if block == 1:
        return v_flat
    T = v_flat.shape[0]
    return (v_flat.reshape(T, CHANNELS // block, block) @ Hb).reshape(T, CHANNELS)


def run_case(amax, scale_mode, block, Hb,
             outlier_tokens=4, outlier_channels=2):
    torch.manual_seed(7 + KV_HEADS)
    random.seed(7 + KV_HEADS)
    num_slots = BLOCK_SIZE * NUM_BLOCKS
    slot_lst = random.sample(range(num_slots), NUM_TOKENS)
    slot_mapping = torch.tensor(slot_lst, dtype=torch.long, device=DEV)

    key = torch.randn(NUM_TOKENS, KV_HEADS, HEAD_SIZE,
                      dtype=torch.bfloat16, device=DEV)
    value = torch.randn(NUM_TOKENS, KV_HEADS, HEAD_SIZE,
                        dtype=torch.float32, device=DEV)

    outlier_mask = torch.zeros(NUM_TOKENS, KV_HEADS, HEAD_SIZE,
                               dtype=torch.bool, device=DEV)
    if amax is not None:
        for t in range(outlier_tokens):
            for c in range(outlier_channels):
                outlier_mask[t, 0, 8 * c] = True   # sinks in head 0
        value[outlier_mask] = float(amax)

    vref = value.clone()                            # original space reference
    v_flat = value.reshape(NUM_TOKENS, CHANNELS)
    v_rot = online_rotate(v_flat, Hb, block)
    v_in = v_rot.reshape(NUM_TOKENS, KV_HEADS, HEAD_SIZE).to(torch.bfloat16)

    kv = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, KV_HEADS, FULL_DIM,
                     dtype=torch.uint8, device=DEV)
    key_cache, value_cache = kv[:, 0], kv[:, 1]

    if scale_mode == "amax6":
        eff_amax = v_in.abs().amax().to(torch.float32)   # observer sees rotated
        v_scale = (eff_amax / 6.0).clone()
    else:
        v_scale = torch.ones((), dtype=torch.float32, device=DEV)
    k_scale = torch.ones((), dtype=torch.float32, device=DEV)

    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key, v_in, key_cache, value_cache, slot_mapping,
        "nvfp4", k_scale, v_scale,
    )
    torch.cuda.synchronize()

    v_data, v_sf = nvfp4_split_data_scale(value_cache)
    v_sf = v_sf.view(torch.uint8)
    blk = slot_mapping // BLOCK_SIZE
    off = slot_mapping % BLOCK_SIZE
    dq = dequant_linear(v_data.contiguous(), v_sf.contiguous(),
                        v_scale.item())[blk, off]                  # [T,H,128]
    dq_flat = dq.reshape(NUM_TOKENS, CHANNELS)
    if block > 1:
        dq_flat = online_rotate(dq_flat, Hb.T.contiguous(), block)  # invert
    dq = dq_flat.reshape(NUM_TOKENS, KV_HEADS, HEAD_SIZE)

    bulk = ~outlier_mask
    res = {
        "amax": amax, "scale_mode": scale_mode, "block": block,
        "foldable": block <= HEAD_SIZE,
        "v_scale_value": round(v_scale.item(), 4),
        "V_rel_l2_bulk": round(rel_l2(dq[bulk], vref[bulk]), 5),
        "V_bulk_zero_frac": round((dq[bulk] == 0).float().mean().item(), 5),
    }
    if amax is not None:
        res["V_rel_l2_outlier"] = round(
            rel_l2(dq[outlier_mask], vref[outlier_mask]), 5)
        res["V_outlier_recon_mean"] = round(
            dq[outlier_mask].mean().item(), 1)
    return res


def main():
    print(json.dumps({
        "probe": "nvfp4_online_hadamard_probe",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "tokens": NUM_TOKENS, "head_size": HEAD_SIZE, "kv_heads": KV_HEADS,
        "e4m3_subnormal_min": 2 ** -9,
    }), flush=True)
    Hbs = {b: hadamard(b, DEV) for b in (128, 256, 512, 1024)}
    rows = []
    for amax in (None, 42000.0, 125000.0):
        for block in (1, 128, 256, 512, 1024):
            Hb = Hbs[block] if block > 1 else None
            for scale_mode in ("unit", "amax6"):
                r = run_case(amax, scale_mode, block, Hb)
                rows.append(r)
                print(json.dumps(r), flush=True)
    json.dump(rows, open("RESULT_nvfp4_online_hadamard_probe.json", "w"),
              indent=1)
    print("written RESULT_nvfp4_online_hadamard_probe.json", flush=True)


if __name__ == "__main__":
    main()
