# Spark (sm121) GridBook+DFlash2 — evaluiert, VERWORFEN (Speed)

**Stand 2026-08-31. Entscheidung: der Spark bleibt AQUA+DFlash2.** GridBook-13GB
wurde auf dem Spark voll zum Laufen gebracht (inkl. Whisper-Headroom), aber wegen
~35 % Decode-Verlust nicht übernommen. **Auf dem Spark gilt: Speed ist das
Spitzenkriterium direkt nach Qualität** (die Kiste ist bandbreiten-limitiert und
ohnehin langsam) — ein −35 % ist inakzeptabel, auch für +10 GB Whisper-Headroom.

## Was funktioniert hat (self-serve, ohne Rob-Re-Export, ohne #52883)
Die rdtand-3.8-gridbook ist multimodal-verpackt-aber-text-deklariert (Export-Bug).
Vier programmatisch validierte Eingriffe → lädt + serviert mit DFlash2 auf sm121:
1. **Prefix-Strip** `model.language_model.` → `model.` (header-only + config_groups/ignore/execution_contracts).
2. **NVFP4-Activation-Contract re-attestiert** (`target_values_sha256` mit gridbooks eigener Funktion über SHORT-Namen neu, Gate grün).
3. **BF16-embed-Swap** aus AQUA (CB-quantisierte Embedding vom Plugin-Pfad nicht unterstützt).
4. **BF16-lm_head-Swap + config_group group_8 entfernt** (DFlash2 verlangt unquantisierten Target-LM-Head für Candidate-TopK — genau der Fehler, den #52883 upstream adressiert).

## Messung — GridBook gewinnt NICHT auf Speed
| | GridBook+DFlash2 | AQUA+DFlash2 |
|---|---|---|
| single tok/s prosa/count200/json/retr | 13,4 / 42,2 / 24,3 / 31,0 | **19,6 / 62,2 / 38,9 / 40,1** |
| batched-8 tok/s | 65,9 | **99,6** |
| GSM8K(30) | 0,867 | 0,90 |
| Tool-Fidelity / Acceptance / KV-Pool | 1,0 / 0,27 / 775.766 | 1,0 / 0,27 / 775.766 |
| Footprint | **53 GB (+10 frei → Whisper)** | 63 GB |

**~35 % langsamer** (CB-Dequant-Kernels vs. AQUAs NVFP4+FP8 auf sm121). Qualität +
Tool-Calling gleichauf, Prefix-Cache greift auch unter DFlash2 (#50897). Der einzige
Vorteil (10 GB Headroom → 2 Whisper) wiegt den Speed-Verlust nicht auf.

## Konsequenz
- **Spark bleibt AQUA+DFlash2+NVFP4-KV** (`vllm-prod-nvfp4kv`). Rollback vollzogen 2026-08-31.
- **Whisper-Transcriber:** vorerst NICHT auf dem Spark (kein Headroom ohne GridBook) →
  Tim nutzt dafür Apple Silicon / MacBook.
- **Der Cross-Arch-Kontrast steht:** GridBook gewinnt auf sm120/RTX (KV-knapp → Headroom
  zählt, MTP-Turbo, ~624k Pool), verliert auf sm121/Spark (pool-reich → Headroom egal,
  CB-Dequant zu langsam). Plattform-differenziert, wie im Entscheidungs-Register.

## Artefakte (auf dem Spark belassen, für ein etwaiges Follow-up)
`build_gridbook_fixed.py`, `recompute_contract.py`, `embed_swap_fixed.py`,
`lmhead_swap_fixed.py`, `cb-serve-dflash2-gridbook.sh`, `battery_gridbook.json`,
`battery_aqua.json`, sowie `gridbook-13gb-fixed`. **Sauberer Weg für die Zukunft:**
rdtand-Re-Export text-nativ (`model.*`-Prefix, BF16 embed+lm_head, ohne MTP) → die
4-Fix-Surgery entfiele. Aber nur relevant, falls der Speed-Nachteil kernel-seitig
verschwindet — sonst bleibt der Spark ohnehin bei AQUA.
