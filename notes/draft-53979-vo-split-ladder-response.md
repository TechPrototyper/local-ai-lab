# GEPOSTET 2026-09-05 (Tims Go) — https://github.com/vllm-project/vllm/pull/46329#issuecomment-5554547616

Ursprünglicher Entwurf unten, zur Nachvollziehbarkeit stehen gelassen.

---

Kontext: jethac hat am 2026-09-05 ~15:48 UTC in #46329 direkt gefragt, ob
unsere Ladder mit `head_dim_qk=512, head_dim_vo=256` gegen einen 512-breiten
Cache planen kann — die Shape, für die sein FlashInfer-Split-KV-Gate
(`_nvfp4_kv_requires_disabled_split_kv`) ursprünglich eingeführt wurde. Wir
haben `scripts/kvsweep/nvfp4_splitkv_ladder.py` entsprechend erweitert und
die volle Matrix für **beide** Hälften des 512-breiten V-Caches gefahren.

## Kommentar-Entwurf (Englisch, für GitHub)

> Ran your requested shape — both halves of the 512-wide V. Extended our
> ladder to support asymmetric `head_dim_qk`/`head_dim_vo`: K and the full
> 512-wide V side are built separately, V is `narrow()`-sliced to the
> `head_dim_vo` chunk for the pass — same call pattern as
> `_run_vo_split_prefill` in our #53979 impl (`(k_cache, v_cache_i)` /
> `(k_sf, v_sf_i)` tuples, not a combined tensor).
>
> Full sweep, qo 1..9 × kv {2k, 8k, 16k, 32k}, `head_dim_qk=512`,
> `head_dim_vo=256`, both `vo_chunk=0` and `vo_chunk=1`: **72/72 clean, no
> NaN**. `max_rel` ~0.0052 across the board for both chunks (well under our
> 1% corruption threshold), `max_abs` a flat 0.0625 on every row,
> `frac_gt_1e2` in the 11–18% band with no trend against qo or kv growth
> and no meaningful difference between the two chunks — reads like steady
> NVFP4 quantization noise, not an accumulating bug, and not something tied
> to which half of V gets sliced.
>
> Caveat: this ran on the gate as currently shipped (blanket sm120/121
> disable), not against your shape-scoped branch — we forced split-KV on
> directly via the same monkeypatch trick as the rest of the ladder, so
> it's testing the kernel behavior at this shape independent of which gate
> implementation sits in front of it.
>
> Script + both result sets, public:
> - https://github.com/TechPrototyper/local-ai-lab/blob/351c846/scripts/kvsweep/nvfp4_splitkv_ladder.py
> - https://github.com/TechPrototyper/local-ai-lab/blob/351c846/results/nvfp4-splitkv-ladder-512-256-vo0-2026-09-05.json
> - https://github.com/TechPrototyper/local-ai-lab/blob/351c846/results/nvfp4-splitkv-ladder-512-256-vo1-2026-09-05.json

## Offene Entscheidungen für Tim vor dem Posten
- Wo posten: auf #46329 (direkte Antwort auf jethacs Frage) oder auf #53979
  (unser eigener PR)? Empfehlung weiterhin: #46329, mit Verweis auf #53979.
- Commit-Hash `351c846` in den Links fixieren (aktueller `master`-Stand) oder
  lieber auf `master` verlinken (dann laeuft der Link mit, falls die Dateien
  spaeter noch angepasst werden)? Empfehlung: den Commit-Hash behalten —
  reproduzierbarer Verweis, der nicht unter uns wegdriftet.
