#!/usr/bin/env python3
"""Repro-Ladder für jethacs Split-KV-NVFP4-Korruption (vllm#46329, 09-03).

Vergleicht BatchPrefillWithPagedKVCacheWrapper (FA2) auf identischem
NVFP4-Paged-KV: Gate aktiv (Split-KV off, Referenz) vs. Gate umgangen
(Split-KV on). Identische Eingabebytes; grobe Divergenz => Split-Pfad-Bug.
Greedy/fixed seeds: torch.manual_seed, deterministische Buffers.

Layout (aus vllm/utils/torch_utils.py nvfp4_kv_cache_full_dim):
  full_dim = head//2 (fp4-Daten, gepackt) + head//16 (fp8-e4m3-Scales)
  Scales konstant 1.0 (0x38), Daten-Nibbles random => gültig, seed-stabil.
"""
import argparse
import json

import torch
import flashinfer
import flashinfer.prefill as fip

E4M3_ONE = 0x38  # 1.0 in float8_e4m3fn


def build_cache(num_pages, page_size, num_kv_heads, head_dim, seed):
    """(data, sf): fp4-Daten uint8 (head//2) + fp8-e4m3-Scales (head//16),
    beide als Paged-Tensoren im NHD-Layout (pages, 2, page_size, H, dim)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    data = torch.randint(
        0, 256, (num_pages, 2, page_size, num_kv_heads, head_dim // 2),
        generator=g, dtype=torch.uint8, device="cuda",
    )
    # Positionsabhängige Scales (Zyklus über 8 e4m3-Werte 0.25..4.0):
    # falsche Chunk-Gewichtung beim Split-Merge kann sich so nicht in
    # Zufallsrauschen wegmitteln, sondern verschiebt Betraege sichtbar.
    sf_cycle = torch.tensor([0x30, 0x34, 0x38, 0x3C, 0x40, 0x3A, 0x36, 0x32],
                            dtype=torch.uint8, device="cuda")
    pos = torch.arange(num_pages * page_size, device="cuda") % 8
    sf = sf_cycle[pos].view(num_pages, page_size, 1, 1).expand(
        num_pages, page_size, num_kv_heads, head_dim // 16)
    sf = sf.unsqueeze(1).expand(num_pages, 2, page_size, num_kv_heads,
                                head_dim // 16).contiguous()
    return data, sf.view(torch.float8_e4m3fn)


def build_side_cache(num_pages, page_size, num_kv_heads, head_dim, seed):
    """Single-side (K-only or V-only) paged cache, same value pattern as
    build_cache but without the combined K/V axis — needed for the
    asymmetric head_dim_qk != head_dim_vo case (Gemma 4 VO-split), where K
    and the FULL-width V cache share head_dim_qk but only a head-dim slice
    of V (width head_dim_vo) is narrowed off for a given pass, mirroring
    FlashInferImpl._run_vo_split_prefill in vllm/v1/attention/backends/
    flashinfer.py. NHD layout (pages, page_size, H, dim)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    data = torch.randint(
        0, 256, (num_pages, page_size, num_kv_heads, head_dim // 2),
        generator=g, dtype=torch.uint8, device="cuda",
    )
    sf_cycle = torch.tensor([0x30, 0x34, 0x38, 0x3C, 0x40, 0x3A, 0x36, 0x32],
                            dtype=torch.uint8, device="cuda")
    pos = torch.arange(num_pages * page_size, device="cuda") % 8
    sf = sf_cycle[pos].view(num_pages, page_size, 1, 1).expand(
        num_pages, page_size, num_kv_heads, head_dim // 16).contiguous()
    return data, sf.view(torch.float8_e4m3fn)


def run_case(qo_len, kv_len, *, num_qo_heads, num_kv_heads, head_dim_qk,
             head_dim_vo, page_size, split_mode, fixed_split_size, seed,
             causal=True, batch=1, vo_chunk=0):
    """split_mode: 'gate' (Referenz) oder 'forced' (Gate umgangen).
    batch>1: gemischte kv-Längen [kv, kv//2, kv//4, kv+13, ...zyklisch] —
    prod-näher (Verify läuft gebatcht, Split-Scheduler verteilt über
    Requests).

    head_dim_qk == head_dim_vo (Standardfall): ein kombinierter [K|V]-Cache-
    Tensor, wie bisher. head_dim_qk != head_dim_vo (Gemma-4-VO-Split-Shape,
    z.B. 512/256): K UND die volle V-Seite werden bei head_dim_qk gebaut
    (Gemma 4 speichert V physisch genauso breit wie K/Q), dann wird V per
    narrow() auf die head_dim_vo-breite Chunk `vo_chunk` dieses Passes
    geschnitten — exakt der Aufruf-Stil von _run_vo_split_prefill (Tupel
    (k_cache, v_cache_i) / (k_sf, v_sf_i) statt kombiniertem Tensor)."""
    kv_lens = [[kv_len, max(page_size, kv_len // 2),
                max(page_size, kv_len // 4), kv_len + 13][i % 4]
               for i in range(batch)]
    pages_per = [(k + page_size - 1) // page_size for k in kv_lens]
    num_pages = sum(pages_per)
    gq = torch.Generator(device="cuda").manual_seed(seed + 1)
    q = torch.randn(qo_len * batch, num_qo_heads, head_dim_qk, generator=gq,
                    dtype=torch.bfloat16, device="cuda")
    qo_indptr = torch.arange(0, (batch + 1) * qo_len, qo_len,
                             dtype=torch.int32)
    kv_indptr = torch.tensor([0] + list(__import__("itertools").accumulate(
        pages_per)), dtype=torch.int32)
    last_lens = torch.tensor(
        [k - (p - 1) * page_size for k, p in zip(kv_lens, pages_per)],
        dtype=torch.int32)

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = fip.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")

    if head_dim_qk == head_dim_vo:
        data, sf = build_cache(num_pages, page_size, num_kv_heads,
                                head_dim_qk, seed)
        kv_cache, kv_sf = data, sf
    else:
        k_data, k_sf = build_side_cache(num_pages, page_size, num_kv_heads,
                                        head_dim_qk, seed)
        v_data_full, v_sf_full = build_side_cache(
            num_pages, page_size, num_kv_heads, head_dim_qk, seed + 2)
        data_step = head_dim_vo // 2  # packed e2m1, 2 elements per byte
        sf_step = head_dim_vo // 16  # one fp8 scale per 16 elements
        v_data = v_data_full.narrow(-1, vo_chunk * data_step, data_step)
        v_sf = v_sf_full.narrow(-1, vo_chunk * sf_step, sf_step)
        kv_cache, kv_sf = (k_data, v_data), (k_sf, v_sf)

    orig = fip._nvfp4_kv_requires_disabled_split_kv
    if split_mode == "forced":
        fip._nvfp4_kv_requires_disabled_split_kv = lambda dt: False
    try:
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=torch.arange(num_pages, dtype=torch.int32,
                                          device="cuda"),
            paged_kv_last_page_len=last_lens,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            causal=causal,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
            fixed_split_size=fixed_split_size,
        )
        out = wrapper.run(q, kv_cache, kv_cache_sf=kv_sf)
    finally:
        fip._nvfp4_kv_requires_disabled_split_kv = orig
    torch.cuda.synchronize()
    return out.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qo", default="1,2,3,5,9")
    ap.add_argument("--kv", default="2048,8192,32768,131072")
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128,
                    help="Fallback fuer --head-dim-qk/--head-dim-vo, wenn "
                    "die nicht gesetzt sind (symmetrischer Standardfall)")
    ap.add_argument("--head-dim-qk", type=int, default=None,
                    help="z.B. 512 fuer die Gemma-4-VO-Split-Shape")
    ap.add_argument("--head-dim-vo", type=int, default=None,
                    help="z.B. 256 fuer die Gemma-4-VO-Split-Shape; bei "
                    "!= head-dim-qk wird V als narrow()-Slice eines bei "
                    "head-dim-qk-breiten physischen Caches gebaut")
    ap.add_argument("--vo-chunk", type=int, default=0,
                    help="welcher head_dim_vo-breite Slice des vollen "
                    "V-Caches getestet wird (nur bei head-dim-qk != "
                    "head-dim-vo relevant; 0 = erster Chunk)")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--fixed-split", type=int, default=None,
                    help="fixed_split_size an plan() (None = Auto-Scheduler)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="/tmp/ladder.json")
    args = ap.parse_args()

    head_dim_qk = args.head_dim_qk if args.head_dim_qk is not None else args.head_dim
    head_dim_vo = args.head_dim_vo if args.head_dim_vo is not None else args.head_dim

    rows = []
    for qo in [int(x) for x in args.qo.split(",")]:
        for kv in [int(x) for x in args.kv.split(",")]:
            kw = dict(num_qo_heads=args.heads, num_kv_heads=args.kv_heads,
                      head_dim_qk=head_dim_qk, head_dim_vo=head_dim_vo,
                      page_size=args.page_size,
                      fixed_split_size=args.fixed_split, seed=args.seed,
                      causal=not args.non_causal, batch=args.batch,
                      vo_chunk=args.vo_chunk)
            ref = run_case(qo, kv, split_mode="gate", **kw)
            got = run_case(qo, kv, split_mode="forced", **kw)
            diff = (got - ref).abs()
            denom = ref.abs().max().clamp(min=1e-6)
            row = {
                "qo": qo, "kv": kv,
                "head_dim_qk": head_dim_qk, "head_dim_vo": head_dim_vo,
                "vo_chunk": args.vo_chunk if head_dim_qk != head_dim_vo else None,
                "max_abs": round(diff.max().item(), 6),
                "max_rel": round((diff.max() / denom).item(), 6),
                "frac_gt_1e2": round((diff > 1e-2).float().mean().item(), 6),
                "nan": bool(torch.isnan(got).any().item()),
                "corrupt": bool((diff.max() / denom).item() > 1e-2
                                or torch.isnan(got).any().item()),
            }
            rows.append(row)
            print(row, flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
