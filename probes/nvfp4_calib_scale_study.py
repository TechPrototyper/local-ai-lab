# SPDX-License-Identifier: Apache-2.0
# NVFP4 KV-cache calibration-scale study (model-free, real kernel round-trip).
# Extends the sm121 round-trip probe (vllm#50084 methodology): deterministic
# BF16 V inputs with sink-like outliers -> installed reshape_and_cache_flash
# (kv_cache_dtype="nvfp4") -> independent unpack+dequant (linear scale read,
# the patched sm12x consumer path) -> rel-L2 split into BULK vs OUTLIER parts.
# Compares per-tensor scale = 1.0 (uncalibrated default) vs amax/6 (what
# static amax KV calibration bakes). Outlier magnitudes match the measured
# provenance of a real calibrated checkpoint (v_scale 7000/20864 -> amax
# ~42k/~125k).
import json
import random

import torch

import vllm  # noqa: E402
from vllm.utils.torch_utils import nvfp4_split_data_scale  # noqa: E402

DEV = "cuda:0"
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 64
NUM_TOKENS = 512
DATA_DIM = HEAD_SIZE // 2
SCALE_DIM = HEAD_SIZE // 16
FULL_DIM = DATA_DIM + SCALE_DIM

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


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
    """NHD [P,T,H,*] -> float32 [P,T,H,HEAD_SIZE]; linear scale read."""
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


def run_case(amax: float | None, scale_mode: str, kv_heads: int = 4,
             outlier_tokens: int = 4, outlier_channels: int = 2):
    torch.manual_seed(7 + kv_heads)
    random.seed(7 + kv_heads)
    num_slots = BLOCK_SIZE * NUM_BLOCKS
    slot_lst = random.sample(range(num_slots), NUM_TOKENS)
    slot_mapping = torch.tensor(slot_lst, dtype=torch.long, device=DEV)

    key = torch.randn(NUM_TOKENS, kv_heads, HEAD_SIZE,
                      dtype=torch.bfloat16, device=DEV)
    value = torch.randn(NUM_TOKENS, kv_heads, HEAD_SIZE,
                        dtype=torch.bfloat16, device=DEV)

    outlier_mask = torch.zeros(NUM_TOKENS, kv_heads, HEAD_SIZE,
                               dtype=torch.bool, device=DEV)
    if amax is not None:
        # sink-like: first few tokens, one head, a couple of channels
        for t in range(outlier_tokens):
            for c in range(outlier_channels):
                outlier_mask[t, 0, 8 * c] = True
        value = value.clone()
        value[outlier_mask] = torch.tensor(amax, dtype=torch.bfloat16)

    kv = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, kv_heads, FULL_DIM,
                     dtype=torch.uint8, device=DEV)
    key_cache, value_cache = kv[:, 0], kv[:, 1]

    if scale_mode == "amax6":
        eff_amax = value.abs().amax().to(torch.float32)
        v_scale = (eff_amax / 6.0).clone()
    else:  # unit
        v_scale = torch.ones((), dtype=torch.float32, device=DEV)
    k_scale = torch.ones((), dtype=torch.float32, device=DEV)

    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping,
        "nvfp4", k_scale, v_scale,
    )
    torch.cuda.synchronize()

    k_data, k_sf = nvfp4_split_data_scale(key_cache)
    v_data, v_sf = nvfp4_split_data_scale(value_cache)
    v_sf = v_sf.view(torch.uint8)
    k_sf = k_sf.view(torch.uint8)

    blk = slot_mapping // BLOCK_SIZE
    off = slot_mapping % BLOCK_SIZE
    vref = value.to(torch.float32)
    kref = key.to(torch.float32)

    dq_v = dequant_linear(v_data.contiguous(), v_sf.contiguous(),
                          v_scale.item())[blk, off]
    dq_k = dequant_linear(k_data.contiguous(), k_sf.contiguous(),
                          k_scale.item())[blk, off]

    bulk = ~outlier_mask
    res = {
        "amax": amax, "scale_mode": scale_mode,
        "v_scale_value": round(v_scale.item(), 4),
        "V_rel_l2_all": round(rel_l2(dq_v, vref), 5),
        "V_rel_l2_bulk": round(rel_l2(dq_v[bulk], vref[bulk]), 5),
        "K_rel_l2_control": round(rel_l2(dq_k, kref), 5),
        "V_bulk_zero_frac": round(
            (dq_v[bulk] == 0).float().mean().item(), 5),
    }
    if amax is not None:
        res["V_rel_l2_outlier"] = round(
            rel_l2(dq_v[outlier_mask], vref[outlier_mask]), 5)
        res["V_outlier_recon_mean"] = round(
            dq_v[outlier_mask].mean().item(), 1)
    return res


def main():
    print(json.dumps({
        "probe": "nvfp4_calib_scale_study",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "tokens": NUM_TOKENS, "head_size": HEAD_SIZE,
    }), flush=True)
    rows = []
    # amax levels: none (control), 42k (mid sink, v_scale~7000),
    # 125k (max sink, v_scale~20864) — matching measured provenance.
    for amax in (None, 42000.0, 125000.0):
        for scale_mode in ("unit", "amax6"):
            if amax is None and scale_mode == "amax6":
                # amax6 of a pure-bulk tensor ~= abs.max/6 ~ 0.8 — include
                # as reference for "calibration on a sink-free layer"
                pass
            r = run_case(amax, scale_mode)
            rows.append(r)
            print(json.dumps(r), flush=True)
    json.dump(rows, open("RESULT_nvfp4_calib_scale_study.json", "w"), indent=1)
    print("written RESULT_nvfp4_calib_scale_study.json", flush=True)


if __name__ == "__main__":
    main()
