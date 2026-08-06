# SPDX-License-Identifier: Apache-2.0
# NVFP4 KV Hadamard-rotation probe (model-free, real kernel round-trip).
# Extends nvfp4_calib_scale_study.py (NVFP4_CAMPAIGN_2026-08-01/evidence):
# same synthetic V (Gaussian bulk + sink outliers at measured provenance
# magnitudes 42k/125k), same real writer (reshape_and_cache_flash, nvfp4),
# same independent linear-scale dequant — plus a SpinQuant-R2-style
# per-head Hadamard rotation along head_dim applied BEFORE quantization
# and inverted AFTER dequantization. Metrics are computed in the ORIGINAL
# space, so "rotate" measures exactly what folding H into Wv/Wo would do
# to KV-cache fidelity. Compares per-tensor scale 1.0 vs amax/6, each
# unrotated vs rotated.
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


def hadamard(n: int, device) -> torch.Tensor:
    """Sylvester Hadamard, orthonormal (H @ H.T = I)."""
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


def run_case(amax, scale_mode: str, rotate: bool, kv_heads: int = 4,
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
        for t in range(outlier_tokens):
            for c in range(outlier_channels):
                outlier_mask[t, 0, 8 * c] = True
        value = value.clone()
        value[outlier_mask] = torch.tensor(amax, dtype=torch.bfloat16)

    H = hadamard(HEAD_SIZE, DEV)
    vref = value.to(torch.float32)
    if rotate:
        v_in = (vref @ H).to(torch.bfloat16)  # what folded Wv·H would emit
    else:
        v_in = value

    kv = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, kv_heads, FULL_DIM,
                     dtype=torch.uint8, device=DEV)
    key_cache, value_cache = kv[:, 0], kv[:, 1]

    if scale_mode == "amax6":
        # calibrator measures amax in the space it sees (post-rotation)
        eff_amax = v_in.abs().amax().to(torch.float32)
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
                        v_scale.item())[blk, off]
    if rotate:
        dq = dq @ H.T  # what H^-1·Wo undoes at read time

    bulk = ~outlier_mask
    res = {
        "amax": amax, "scale_mode": scale_mode, "rotate": rotate,
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
        "probe": "nvfp4_hadamard_probe",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "tokens": NUM_TOKENS, "head_size": HEAD_SIZE,
    }), flush=True)
    rows = []
    for amax in (None, 42000.0, 125000.0):
        for rotate in (False, True):
            for scale_mode in ("unit", "amax6"):
                r = run_case(amax, scale_mode, rotate)
                rows.append(r)
                print(json.dumps(r), flush=True)
    json.dump(rows, open("RESULT_nvfp4_hadamard_probe.json", "w"), indent=1)
    print("written RESULT_nvfp4_hadamard_probe.json", flush=True)


if __name__ == "__main__":
    main()
