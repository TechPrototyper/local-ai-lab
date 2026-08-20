# SPDX-License-Identifier: Apache-2.0
# NVFP4 KV GPTQ-ablation probe (model-free, real kernel round-trip).
#
# Answers the maintainer question in llm-compressor#2936 ("...some other
# modifier like GPTQ might be needed?") on the KV-cache axis. Extends
# nvfp4_hadamard_probe.py: same real writer (reshape_and_cache_flash,
# kv_cache_dtype="nvfp4"), same independent linear-scale dequant, same
# sink provenance (42k/125k, 2 outlier channels), same metrics in the
# ORIGINAL space. New axis: the value bulk is now produced by a REAL
# value projection V = X @ W_v^T, and W_v is optionally quantized with a
# genuine GPTQ pass (Frantar et al. 2022: Hessian H=X^T X, Cholesky
# inverse, left-to-right column error propagation, symmetric int4 per
# output channel). Full 2x2x2 matrix:
#   {scale 1.0, amax/6} x {no rotation, R2 (head_dim Hadamard)} x {no GPTQ, GPTQ}.
# The point: GPTQ compensates WEIGHT-quant error; the amax bulk-erasure is
# a KV-cache scale effect downstream of the projection. This measures
# whether GPTQ (alone or with R2) rescues amax KV calibration. Sinks are
# massive input-driven activations and are planted post-projection in the
# reference exactly as in the prior probes, so GPTQ/RTN act on the bulk.
import json
import random

import torch

import vllm  # noqa: E402
from vllm.utils.torch_utils import nvfp4_split_data_scale  # noqa: E402

DEV = "cuda:0"
HEAD_SIZE = 128
HIDDEN = 256          # value-projection input dim (X columns)
BLOCK_SIZE = 16
NUM_BLOCKS = 64
NUM_TOKENS = 512
DATA_DIM = HEAD_SIZE // 2
SCALE_DIM = HEAD_SIZE // 16
FULL_DIM = DATA_DIM + SCALE_DIM
W_BITS = 4            # GPTQ target bit-width for the value projection

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


def quantize_int_sym(w_col, bits):
    """Symmetric per-(output-channel) int quantize of one weight column."""
    qmax = 2 ** (bits - 1) - 1            # 7 for 4-bit
    scale = (w_col.abs().amax(dim=0, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(w_col / scale), -qmax - 1, qmax)
    return q * scale


def gptq_quantize(W, X, bits=W_BITS, damp_frac=0.01):
    """Genuine GPTQ (Frantar et al. 2022). W:[out,in], X:[n,in]."""
    W = W.clone().float()
    d_out, d_in = W.shape
    H = (X.float().t() @ X.float()) * (2.0 / X.shape[0])
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    damp = damp_frac * torch.mean(torch.diag(H))
    H[range(d_in), range(d_in)] += damp
    # Hinv via Cholesky of H^-1 (upper-triangular factor), GPTQ convention.
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    for j in range(d_in):
        w = W[:, j]
        d = Hinv[j, j]
        q = quantize_int_sym(w.unsqueeze(1), bits).squeeze(1)
        Q[:, j] = q
        err = (w - q) / d
        W[:, j:] -= err.unsqueeze(1) * Hinv[j, j:].unsqueeze(0)
    return Q


def rtn_quantize(W, bits=W_BITS):
    return quantize_int_sym(W.float().t(), bits).t()  # per-out-channel


def run_case(amax, scale_mode, rotate, gptq, W, Wg, Wr, X,
             kv_heads=4, outlier_tokens=4, outlier_channels=2):
    torch.manual_seed(7 + kv_heads)
    random.seed(7 + kv_heads)
    num_slots = BLOCK_SIZE * NUM_BLOCKS
    slot_lst = random.sample(range(num_slots), NUM_TOKENS)
    slot_mapping = torch.tensor(slot_lst, dtype=torch.long, device=DEV)

    key = torch.randn(NUM_TOKENS, kv_heads, HEAD_SIZE,
                      dtype=torch.bfloat16, device=DEV)
    # Reference value bulk from the full-precision projection (head 0);
    # other heads are deterministic Gaussian filler.
    v0_ref = (X @ W.t())                                  # [tokens, 128], fp
    v0_used = (X @ (Wg if gptq else W).t())              # weight arm
    value_ref = torch.randn(NUM_TOKENS, kv_heads, HEAD_SIZE,
                            dtype=torch.float32, device=DEV)
    value_used = value_ref.clone()
    value_ref[:, 0, :] = v0_ref
    value_used[:, 0, :] = v0_used

    outlier_mask = torch.zeros(NUM_TOKENS, kv_heads, HEAD_SIZE,
                               dtype=torch.bool, device=DEV)
    if amax is not None:
        for t in range(outlier_tokens):
            for c in range(outlier_channels):
                outlier_mask[t, 0, 8 * c] = True
        # sinks are input-driven massive activations: identical in ref and
        # in every weight arm (planted post-projection, as prior probes do).
        value_ref[outlier_mask] = float(amax)
        value_used[outlier_mask] = float(amax)

    H = hadamard(HEAD_SIZE, DEV)
    vref = value_ref
    if rotate:
        v_in = (value_used @ H).to(torch.bfloat16)
    else:
        v_in = value_used.to(torch.bfloat16)

    kv = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, kv_heads, FULL_DIM,
                     dtype=torch.uint8, device=DEV)
    key_cache, value_cache = kv[:, 0], kv[:, 1]

    if scale_mode == "amax6":
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
        dq = dq @ H.T

    bulk = ~outlier_mask
    res = {
        "amax": amax, "scale_mode": scale_mode, "rotate": rotate,
        "gptq": gptq, "v_scale_value": round(v_scale.item(), 4),
        "V_rel_l2_bulk": round(rel_l2(dq[bulk], vref[bulk]), 5),
        "V_bulk_zero_frac": round((dq[bulk] == 0).float().mean().item(), 5),
    }
    if gptq:
        wq_ref = X @ W.t()
        res["W_gptq_rel_err"] = round(rel_l2(X @ Wg.t(), wq_ref), 5)
        res["W_rtn_rel_err"] = round(rel_l2(X @ Wr.t(), wq_ref), 5)
    if amax is not None:
        res["V_rel_l2_outlier"] = round(
            rel_l2(dq[outlier_mask], vref[outlier_mask]), 5)
        res["V_outlier_recon_mean"] = round(
            dq[outlier_mask].mean().item(), 1)
    return res


def main():
    torch.manual_seed(1234)
    # Structured calibration + weight for a KV value projection (head 0).
    X = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.float32, device=DEV)
    # a few input rows carry a massive-activation direction (sink source)
    X[0:4] += 6.0 * torch.randn(4, HIDDEN, device=DEV)
    W = (torch.randn(HEAD_SIZE, HIDDEN, dtype=torch.float32, device=DEV)
         / (HIDDEN ** 0.5))          # V bulk ~ unit variance, matches prior
    Wg = gptq_quantize(W, X)
    Wr = rtn_quantize(W)
    print(json.dumps({
        "probe": "nvfp4_gptq_ablation_probe",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "tokens": NUM_TOKENS, "head_size": HEAD_SIZE, "hidden": HIDDEN,
        "w_bits": W_BITS,
    }), flush=True)
    rows = []
    for amax in (None, 42000.0, 125000.0):
        for rotate in (False, True):
            for gptq in (False, True):
                for scale_mode in ("unit", "amax6"):
                    r = run_case(amax, scale_mode, rotate, gptq, W, Wg, Wr, X)
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    json.dump(rows, open("RESULT_nvfp4_gptq_ablation_probe.json", "w"),
              indent=1)
    print("written RESULT_nvfp4_gptq_ablation_probe.json", flush=True)


if __name__ == "__main__":
    main()
