# Spark (sm121) GridBook+DFlash2 — evaluated, DISCARDED (speed)

**As of 2026-08-31. Decision: the Spark stays AQUA+DFlash2.** GridBook-13GB
was brought fully to a working state on the Spark (incl. Whisper headroom), but not adopted because of
~35 % decode loss. **On the Spark, speed is the top criterion right after
quality** (the box is bandwidth-limited and slow anyway) — a −35 % is unacceptable, even for +10 GB Whisper headroom.

## What worked (self-serve, without Rob re-export, without #52883)
The rdtand-3.8-gridbook is packaged multimodal but declared text (export bug).
Four programmatically validated interventions → loads + serves with DFlash2 on sm121:
1. **Prefix strip** `model.language_model.` → `model.` (header-only + config_groups/ignore/execution_contracts).
2. **NVFP4 activation contract re-attested** (`target_values_sha256` recomputed with gridbook's own function over SHORT names, gate green).
3. **BF16 embed swap** from AQUA (CB-quantized embedding not supported by the plugin path).
4. **BF16 lm_head swap + config_group group_8 removed** (DFlash2 requires an unquantized target LM head for candidate top-k — exactly the error that #52883 addresses upstream).

## Measurement — GridBook does NOT win on speed
| | GridBook+DFlash2 | AQUA+DFlash2 |
|---|---|---|
| single tok/s prose/count200/json/retr | 13.4 / 42.2 / 24.3 / 31.0 | **19.6 / 62.2 / 38.9 / 40.1** |
| batched-8 tok/s | 65.9 | **99.6** |
| GSM8K(30) | 0.867 | 0.90 |
| Tool fidelity / acceptance / KV pool | 1.0 / 0.27 / 775,766 | 1.0 / 0.27 / 775,766 |
| Footprint | **53 GB (+10 free → Whisper)** | 63 GB |

**~35 % slower** (CB dequant kernels vs. AQUA's NVFP4+FP8 on sm121). Quality +
tool calling on par, prefix cache works under DFlash2 too (#50897). The only
advantage (10 GB headroom → 2 Whisper) does not outweigh the speed loss.

## Consequence
- **Spark stays AQUA+DFlash2+NVFP4-KV** (`vllm-prod-nvfp4kv`). Rollback completed 2026-08-31.
- **Whisper transcriber:** for now NOT on the Spark (no headroom without GridBook) →
  Tim uses Apple Silicon / MacBook for it.
- **The cross-arch contrast stands:** GridBook wins on sm120/RTX (KV-scarce → headroom
  matters, MTP-Turbo, ~624k pool), loses on sm121/Spark (pool-rich → headroom irrelevant,
  CB dequant too slow). Platform-differentiated, as in the decision register.

## Artifacts (left on the Spark, for a possible follow-up)
`build_gridbook_fixed.py`, `recompute_contract.py`, `embed_swap_fixed.py`,
`lmhead_swap_fixed.py`, `cb-serve-dflash2-gridbook.sh`, `battery_gridbook.json`,
`battery_aqua.json`, as well as `gridbook-13gb-fixed`. **Clean path for the future:**
rdtand re-export text-native (`model.*` prefix, BF16 embed+lm_head, without MTP) → the
4-fix surgery would no longer be needed. But only relevant if the speed disadvantage
disappears on the kernel side — otherwise the Spark stays on AQUA anyway.
