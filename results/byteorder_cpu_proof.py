#!/usr/bin/env python3
"""Off-GPU byte-order proof for the NVFP4 dequant shim (Kanban #30/#32, Risiko #1).

vLLM's `dequant_nvfp4_kv_cache` (tests/kernels/quantization/nvfp4_utils.py) is
verified against the REAL GPU writer `reshape_and_cache_nvfp4` in
tests/kernels/attention/test_cache.py (torch.testing.assert_close). So if the
shim's dequant disagrees with `dequant_nvfp4_kv_cache` on identical bytes, it
disagrees with real vLLM pages — no GPU needed. Both are pure torch (CPU ok).

We separate the two byte-order questions:
  (1) fp4 NIBBLE order  -> run with a CONSTANT block scale (swizzle irrelevant).
  (2) block-SCALE layout -> run with per-block DISTINCT scales (exposes swizzle).
"""
import torch

torch.manual_seed(0)

# ---- vLLM reference (copied verbatim from nvfp4_utils.py) -------------------
kE2M1ToFloat = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0], dtype=torch.float32)

def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()  # even=low, odd=high
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    values = kE2M1ToFloat[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)

def dequant_nvfp4_kv_cache(fp4_data, block_scale, global_scale, head_size, block_size):
    data_dim = head_size // 2; scale_dim = head_size // 16
    sf_swizzled = block_scale.view(torch.uint8)
    batch_shape = sf_swizzled.shape[:-2]
    T, S = block_size, scale_dim; sg = S // 4
    sf_reshape = sf_swizzled.reshape(*batch_shape, T // 4, 4, sg, 4)
    ndim = sf_reshape.ndim
    perm = list(range(ndim - 4)) + [ndim - 4, ndim - 1, ndim - 3, ndim - 2]
    sf_linear = sf_reshape.permute(*perm).reshape(*batch_shape, T, S)
    sf_f32 = sf_linear.view(torch.float8_e4m3fn).to(torch.float32)
    shape = fp4_data.shape
    fp4_vals = break_fp4_bytes(fp4_data.reshape(-1, data_dim), torch.float32)
    fp4_vals = fp4_vals.reshape(*shape[:-1], head_size)
    return (fp4_vals.reshape(*shape[:-1], scale_dim, 16)
            * (sf_f32 * global_scale).unsqueeze(-1)).reshape(*shape[:-1], head_size)

# ---- shim CPU reference (copied verbatim from nvfp4_dequant_shim.py) --------
_E2M1_LUT = (0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0)
def e2m1_decode_reference(nibbles):
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32)
    return lut[nibbles.long()]
def dequant_nvfp4_side_reference(data_bytes, scale_bytes, global_scale, head_size):
    lo = data_bytes & 0xF; hi = (data_bytes >> 4) & 0xF
    vals = torch.empty((*data_bytes.shape[:-1], head_size), dtype=torch.float32)
    vals[..., 0::2] = e2m1_decode_reference(lo)
    vals[..., 1::2] = e2m1_decode_reference(hi)
    bs = scale_bytes.float().repeat_interleave(16, dim=-1)
    return vals * bs * global_scale

def shim_page_dequant(data, scale_u8, gs, head_size, block_size):
    # shim reads each (head, token) slice linearly, no unswizzle. data/scale:
    # [H, T, data_dim]/[H, T, scale_dim].
    scale_f8 = scale_u8.view(torch.float8_e4m3fn)
    return dequant_nvfp4_side_reference(data, scale_f8, gs, head_size)

def run(head_size, block_size, distinct_scales):
    H = 2; dd = head_size // 2; sd = head_size // 16; gs = 1.0
    data = torch.randint(0, 256, (H, block_size, dd), dtype=torch.uint8)
    # Safe E4M3 bytes: pick from {1.0,1.5,2.0,3.0} encodings, no NaN/inf.
    safe = torch.tensor([0x3C,0x3E,0x40,0x42], dtype=torch.uint8)  # 1,1.5,2,3
    if distinct_scales:
        idx = torch.arange(block_size * sd) % 4
        scale = safe[idx].reshape(1, block_size, sd).repeat(H, 1, 1)
    else:
        scale = torch.full((H, block_size, sd), int(safe[2]), dtype=torch.uint8)  # all 2.0
    ref = dequant_nvfp4_kv_cache(data, scale.view(torch.float8_e4m3fn), gs, head_size, block_size)
    shim = shim_page_dequant(data, scale, gs, head_size, block_size)
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), shim.flatten(), dim=0).item()
    maxerr = (ref - shim).abs().max().item()
    return cos, maxerr

print("=== NVFP4 byte-order: shim CPU ref vs vLLM dequant_nvfp4_kv_cache (real-page decoder) ===")
for hs in (128, 256):
    for T in (16,):
        c_const, e_const = run(hs, T, distinct_scales=False)
        c_dist, e_dist   = run(hs, T, distinct_scales=True)
        print(f"hs={hs} block={T}: CONSTANT-scale cos={c_const:.6f} maxerr={e_const:.4g} "
              f"| DISTINCT-scale cos={c_dist:.6f} maxerr={e_dist:.4g}")
print()
print("Interpretation:")
print("  CONSTANT-scale cos~1.0 => fp4 NIBBLE order + E2M1 LUT MATCH vLLM (even=low nibble).")
print("  DISTINCT-scale cos<1.0 => block-SCALE layout MISMATCH: vLLM stores scales")
print("  4x4-SWIZZLED across (block_size, scale_dim); the shim reads them linearly.")
print("  => shim needs the 4x4 SF unswizzle before it is real-page-correct.")

# ---- Proposed FIX: page-level shim dequant WITH the 4x4 SF unswizzle --------
def shim_page_dequant_FIXED(data, scale_u8, gs, head_size, block_size):
    """Corrected shim scale-read: unswizzle the 4x4 SF plane before the linear
    per-16 broadcast (mirrors vLLM dequant_nvfp4_kv_cache). data/scale_u8:
    [H, T, data_dim]/[H, T, scale_dim]."""
    sd = head_size // 16; T = block_size; S = sd; sg = S // 4
    batch = scale_u8.shape[:-2]
    sfr = scale_u8.reshape(*batch, T // 4, 4, sg, 4)
    nd = sfr.ndim
    perm = list(range(nd - 4)) + [nd - 4, nd - 1, nd - 3, nd - 2]
    scale_lin = sfr.permute(*perm).reshape(*batch, T, S)
    # now scale_lin[h,t,b] is the true block-b scale of token t -> linear read OK
    return dequant_nvfp4_side_reference(data, scale_lin.view(torch.float8_e4m3fn), gs, head_size)

def run_fixed(head_size, block_size):
    H = 2; dd = head_size // 2; sd = head_size // 16; gs = 1.0
    data = torch.randint(0, 256, (H, block_size, dd), dtype=torch.uint8)
    safe = torch.tensor([0x3C,0x3E,0x40,0x42], dtype=torch.uint8)
    idx = torch.arange(block_size * sd) % 4
    scale = safe[idx].reshape(1, block_size, sd).repeat(H, 1, 1)
    ref = dequant_nvfp4_kv_cache(data, scale.view(torch.float8_e4m3fn), gs, head_size, block_size)
    fixed = shim_page_dequant_FIXED(data, scale, gs, head_size, block_size)
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), fixed.flatten(), dim=0).item()
    return cos, (ref - fixed).abs().max().item()

print("=== proposed FIX (add 4x4 unswizzle) vs vLLM reference ===")
for hs in (128, 256):
    c, e = run_fixed(hs, 16)
    print(f"hs={hs} block=16: FIXED cos={c:.6f} maxerr={e:.4g}  ({'MATCH' if c>0.9999 else 'still off'})")
