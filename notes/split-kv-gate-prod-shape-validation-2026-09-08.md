# Split-KV-Gate: Validierung für unsere eigene Prod-Shape (Spark, sm121)

**Stand 2026-09-08. Alle Tests auf `vllm-prod-nvfp4kv` (Spark/GB10, 10.1.0.229),
ohne Downtime, ohne einen anderen Container anzufassen (Whisper/Reranker/
Embedder liefen die ganze Zeit durch).**

## DEPLOYED 2026-09-08 ~20:00 UTC (Tims Go)

Patch live gebracht, echter Container-Restart (kein synthetisches
Mikrobenchmark mehr): `/vllm-src` in `vllm-prod-nvfp4kv` läuft auf Branch
`exp/pc50897` (Commit `83a9ef7f4`, nicht `kernel/dflash2-nvfp4-sm120-v2`
wie zunächst angenommen — separates Spark-residentes Repo
`/home/timw/nvfp4-work/vllm-dflash2`, von unserem Mac-Checkout entkoppelt).
Original-Datei vorher gesichert (`/tmp/flashinfer.py.orig-backup-20260908`
im Container), Patch direkt an der realen Codestelle angebracht (passte
fast 1:1 auf unsere Referenzimplementierung), Container neu gestartet,
Änderung danach als Commit `6db27cfd1` im Container-Repo festgehalten.

**Ablauf:** Whisper + `vllm-prod-nvfp4kv` gestoppt (Tims Go, "Whisper bleibt
unten" für diese Aktion) → Datei getauscht → `vllm-prod-nvfp4kv` neu
gestartet → sauberer Boot, keine Fehler, `GPU KV cache size: 655,640
tokens`, `/health` → 200.

**Echte Serving-Validierung (nicht nur Kernel-Mikrobenchmark):**
- Kurzer Multi-Step-Test (zählen 1–20 + 17×23 Kopfrechnung über DFlash2):
  korrekt, kohärent.
- Needle-Test bei ~18.207 Prompt-Tokens (echter Request, nicht synthetisch):
  Codewort korrekt wiedergefunden und ausgegeben.
- Nur harmlose einmalige Triton-JIT-Warmup-Warnungen in den Logs
  (DFlash2-Kernel, unabhängig vom Split-KV-Patch), keine Fehler, kein OOM.
- Speicher/GPU nach den Requests unauffällig (91 GiB used, 4,2 GiB frei,
  Swap 10/5,1 GiB, GPU idle danach).

**Entscheidung: behalten, kein Rollback.** Whisper danach wieder
hochgefahren, volle Batterie (7 Container) durchgehend oben — nur
`vllm-prod-nvfp4kv` selbst war kurz unten.

## Nachtest 2026-09-08 ~20:20 UTC — die beiden verbliebenen Lücken geschlossen

Tim: Whisper nochmal runter für Headroom, dann die zwei offenen Punkte
(echte Nebenläufigkeit, echter Prefix-Cache-Hit/-Extend) live nachprüfen.

- **Langer Kontext an der Crash-Grenze, echtes Serving:** Needle bei
  **63.508 Prompt-Tokens** (fast exakt die 09-02-Crash-Region) — Codewort
  korrekt gefunden, `finish_reason: stop`, sauberer Durchlauf in ~63s,
  keine Fehler/Traceback/OOM.
- **Prefix-Cache-Hit (voller Treffer):** derselbe 63,5k-Prompt nochmal
  gesendet → **6,4s statt 63s** (~10× schneller durch Cache-Hit), Codewort
  weiterhin korrekt. Exerziert exakt "decode über eine lange gecachte KV",
  jethacs ursprünglichen Trigger-Fall.
- **Prefix-Cache-Extend:** derselbe 63,5k-Prefix + neuer ~12k-Tail mit
  zweitem Needle → 75.609 Tokens gesamt, 28,6s, zweites Codewort korrekt
  gefunden. Exerziert "prefix-cache extend" direkt.
- **Echte Nebenläufigkeit:** 3 Requests gleichzeitig gefeuert (kurz ~64
  Tokens, mittel ~18k, lang ~63,5k) — alle drei liefen parallel durch
  (4–9s je Request), alle drei Antworten korrekt (Paris /
  PURPLE-NARWHAL-7734 / CRIMSON-OCTOPUS-4471).
- **Speicher-Beobachtung (kein Patch-Problem, aber notiert):** nach dem
  ersten frischen 64k-Request stand der Swap bei 15/15 GiB (praktisch
  voll) — blieb danach stabil auf diesem Niveau (161 MiB frei), stieg
  nicht weiter. Bei voller Batterie + langem Einzelkontext ist die Box
  nah an der Kante, aber stabil.

**Damit sind beide zuvor offen benannten Lücken geschlossen.** Whisper
wieder hochgefahren, volle Batterie oben. Bewertung bleibt "vorsichtig
bestätigt, weiter beobachten" statt "für alle Zeit bewiesen" — aber jede
konkret benannte Lücke aus der ersten Runde ist jetzt mit einem echten
Request abgedeckt.

## Kontext

FlashInfers `_nvfp4_kv_requires_disabled_split_kv` (siehe Docstring im
laufenden Container) schaltet Split-KV für jede NVFP4-KV-Cache-Dtype auf
sm120/sm121 blanket ab — ursprünglich als empirischer Workaround gegen eine
nie root-caused Korruption bei `qo_len << kv_len`. jethac (vllm#46329) hat
diesen Bug inzwischen root-caused: er tritt nur bei Gemma-4s asymmetrischer
VO-Split-Shape (`head_dim_qk=512`/`head_dim_vo=256`) auf, nicht bei
symmetrischen Shapes. Unsere Produktion (Qwen3.5-Hybrid, `head_dim=256`,
`num_kv_heads=4`, symmetrisch) ist von dieser Shape nicht betroffen, zahlt
aber denselben Performance-Preis: DFlash2-Verify-Batches (`q_len` bis 8 bei
`num_speculative_tokens=7`) laufen über den FA2-`BatchPrefill`-Wrapper
(`_noncausal_prefill_wrapper`) und werden vom Gate gebremst.

Ziel dieser Runde: die Korrektheit UND den Performance-Gewinn für unsere
eigene Shape über alle relevanten Dimensionen absichern, bevor über einen
eigenen Carry-Patch (Bypass des Gates) nachgedacht wird.

## Getestete Shape

- Modell: Qwen3.5-Hybrid-Basis (unser `qwen3.8-27b-prismaaqua55`),
  `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`
  (symmetrisch, kein VO-Split nötig — `_vo_split_factor(256, True) == 1`).
- `page_size=32` (= Prod-`--block-size`).
- Non-causal (DFlash-Family-Verify: voller Präfix + block-interne
  Bidirektionalität).
- Skript: `scripts/kvsweep/nvfp4_splitkv_ladder.py`, erweitert um einen
  `--timing`-Modus (CUDA-Event-Latenzmessung plan()+run(), gate vs. forced),
  zusätzlich zum bestehenden Korrektheitsmodus.

## Dimensionen

1. **qo-Bereich:** 1, 2, 3, 5, 8 — deckt Teilakzeptanz/Warmup (kleine q_len)
   und den vollen `num_speculative_tokens=7`-Fall ab.
2. **Kontext:** 2.048 bis **262.144** (= Prods `max-model-len`), nicht nur
   bis zur Crash-Nachbarschaft (65.536).
3. **Gebatcht, gemischte KV-Längen** (`--batch 4`): vier Sequenzen
   unterschiedlicher Länge in einem Verify-Schritt — reflektiert echte
   Serving-Situationen, nicht nur eine isolierte Einzelsequenz.
4. **Wiederholungen:** 15 (+3 Warmup) pro Timing-Punkt für stabile Mediane.

## Ergebnisse

### Korrektheit (35 Punkte, qo×kv, Einzelsequenz)
Alle 35/35 sauber: `max_abs=0.0625`, `max_rel≈0.0052` konstant über den
gesamten Kontext- und qo-Bereich, `frac_gt_1e2` in der 14–17%-Band ohne
Trend, kein NaN, keine Korruption. Identisches Muster wie beim öffentlichen
Gemma-4-Sweep (reines NVFP4-Quantisierungsrauschen, keine Split-Merge-
Korruption).
→ `results/prod_correctness_full-2026-09-08.json`

### Korrektheit, gebatcht (batch=4, gemischte Längen)
2/2 sauber bei kv=32.768 und kv=131.072 (batch-intern gemischt mit
kv/2, kv/4, kv+13 je nach Slot).
→ `results/prod_correctness_batch4-2026-09-08.json`

### Latenz (Median aus 15 Wiederholungen, Einzelsequenz)

| qo | kv=2k | kv=8k | kv=16k | kv=32k | kv=65k | kv=131k | kv=262k |
|---|---|---|---|---|---|---|---|
| 1 | 1,33× | 2,10× | 2,69× | 3,10× | 3,26× | 3,28× | 3,34× |
| 2 | 1,31× | 2,11× | 2,70× | 3,04× | 3,11× | 3,18× | 3,14× |
| 3 | 1,80× | 3,48× | 4,16× | 5,03× | 5,40× | 6,55× | 6,93× |
| 5 | 1,93× | 3,71× | 4,80× | 5,86× | 4,63× | 5,65× | 7,11× |
| 8 | 1,91× | 3,65× | 4,74× | 5,86× | 6,56× | 6,84× | 7,01× |

(Speedup-Faktor "forced" ggü. "gate"; absolute ms in der Rohdatei.) Trend
bestätigt sich über den vollen Bereich: Gate skaliert linear mit dem
Kontext, Forced deutlich sublinear — der Vorteil wächst mit qo UND mit
Kontextlänge, bei `qo=8, kv=262144` (unserem realen Verify-Fall bei sehr
langem Kontext) **7,0×** schneller.
→ `results/prod_timing_full-2026-09-08.json`

### Latenz, gebatcht (batch=4)
`qo=8, kv=32768`: 2,65×. `qo=8, kv=131072`: 2,92×. Niedriger als die
Einzelsequenz-Werte bei gleichem nominalen kv, weil drei der vier
Batch-Slots kürzer sind (kv/2, kv/4) — verdünnt den Durchschnitt, bleibt
aber ein klarer, sauberer Gewinn.
→ `results/prod_timing_batch4-2026-09-08.json`

## Was diese Tests NICHT abdecken

- **Kein echter Serving-Pfad:** isoliertes FlashInfer-Kernel-Mikrobenchmark
  mit synthetischen Paged-KV-Buffern, kein Tokenizer/Scheduler, keine
  Modellgewichte. Realistisch für die Kernel-Latenz, aber kein End-to-End-
  Beweis für Antwortqualität unter echtem Traffic.
- **Kein echter Prefix-Cache-Hit:** jethacs Gate-Docstring nennt explizit
  "decode und prefix-cache extend" als Ursprungs-Trigger der Korruption,
  die er beheben wollte. Unser Benchmark bildet die kritische Shape nach
  (kurze Query, lange KV), aber nicht vLLMs echten Prefix-Cache-Mechanismus.
- Getestet auf Sparks aktueller FlashInfer/vLLM-Version zum Zeitpunkt dieses
  Laufs — kein Dauerlast-/Langzeit-Stabilitätstest.

## Fazit

Für unsere eigene Shape (Qwen3.5-Hybrid, `head_dim=256` symmetrisch) ist
über qo 1–8, Kontext 2k–262k und gebatchte gemischte Längen **keine
Korruption messbar**, bei gleichzeitig **1,3×–7,1× niedrigerer
Verify-Attention-Latenz** mit erzwungenem Split-KV. Damit steht die
Beweislage für einen eigenen, privaten Carry-Patch (Bypass von
`_nvfp4_kv_requires_disabled_split_kv` für unsere Deployment) sauber da —
nächster Schritt laut Tim: Patch vorbereiten, entscheiden ob/wie öffentlich
gemeldet, dann (sobald frei) dieselbe Prüfung auf sm120/RTX (MTP-on-
GridBook, `num_speculative_tokens=2`) wiederholen.
