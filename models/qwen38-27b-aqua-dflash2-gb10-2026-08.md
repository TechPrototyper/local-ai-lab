# Qwen3.8-27B PrismaAQUA · DFlash2 · DGX Spark (GB10) — Stand August 2026

Background-Seite zur Modellkarte. Jeder Konfigurations-Stand bekommt in
diesem Verzeichnis eine eigene Seite; die Karte verlinkt hierher, der
Weg zur Konfiguration steckt in den verlinkten Notes und im Recipe.

## Was ist es (Fakten)

| | |
|---|---|
| Modell | Qwen3.8-27B, dense, hybrid Linear-Attention (48/64 Layer GDN) |
| Quantisierung | PrismaQuant **AQUA** Mixed-Precision, ~5,5 bit (NVFP4-Gewichte + FP8-Attention), `compressed-tensors`, Apache-2.0 — [`rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm), ~24 GB |
| Spekulation | **DFlash2** Block-Diffusion-Drafter, Draft-Länge 7, Drafter selbst FP8-quantisiert (per-channel, 2,25 GB) |
| KV-Cache | FP8, unkalibriert (Scale 1.0) |
| Kontext | 262.144 Tokens pro Session |
| Zielgerät | NVIDIA DGX Spark (GB10, sm121, 128 GB unified, ~273 GB/s) |
| Serving | vLLM, sm12x-Custom-Line — Container (ARM64): [`containers/`](../containers/README.md); Source-Branch: [`dflash2-sm121`](https://github.com/TechPrototyper/vllm/tree/dflash2-sm121) |

## Gemessene Performance (22.08.2026, Prod-Konfiguration, greedy)

**Decode** (Reasoning-Traffic, 1–4 parallele Sessions):

| Sessions | aggregate | pro Session |
|---:|---:|---:|
| 1 | 42 tk/s | 42 |
| 2 | 73 | 40 |
| 3 | 87 | 35 |
| 4 | 111 | 34 |

Skaliert weiter: ~205 tk/s bei 8, Peak ~227 bei 16 Sessions
([Draft-Längen-Karte](../notes/dflash2-draft-length-map.md)). Acceptance
ist inhaltsabhängig: Reasoning ≈ 5,5, freie Prosa ≈ 2,3 (→ ~20 tk/s).

**Prefill** (frische Prompts, Zufallstext, cached=0):

| Kontexttiefe | Prefill |
|---:|---:|
| 22k | 1.479 tk/s |
| 66k | 1.175 |
| 176k | 788 |

**KV-Kapazität** — nach der Hausregel am Device-Maximum ausgewiesen
(112 GB KI-Budget = 128 − 16 GB System):

| Konfiguration | KV-Pool | Tokens |
|---|---:|---:|
| **Device-Maximum** (Solo, util 0.90, verifiziert) | 80 GiB | **1.771.995** ≈ 6,7 × 262k |
| Flotten-Betrieb (Embedder/Reranker/Whisper daneben) | 21,6 GiB | 478.334 ≈ 1,8 × 262k |

## Warum diese Quantisierung

1. **NVFP4-Gewichte (~5,5 bpp)** bringen die 27B auf ~17 GB — erst das
   schafft auf 128 GB Platz für Millionen KV-Tokens *und* eine Flotte.
   Qualität: volle Parität zum Referenz-Checkpoint (n=1319, McNemar
   p=0,88, [Note](../notes/qwen36-aura-head-to-head.md)).
2. **KV unkalibriert (Scale 1.0):** kalibrierte amax-Scales schaden bei
   4-bit-KV messbar — eigener Befund, upstream diskutiert
   ([Note](../notes/dflash2-full-split-verdict.md) · README-Finding 2026-08-01).
3. **FP8-KV statt NVFP4-KV** ist der Preis der DFlash2-Spekulation
   (non-causale Drafter-Attention liest kein NVFP4 auf sm12x) —
   verdict-belegt qualitätsneutral, ~4× Single-Stream-Speed als Gegenwert.
4. **FP8-Drafter:** Drafter-Qualität kann konstruktionsbedingt nur die
   Geschwindigkeit beeinflussen, nie den Output (verified-lossless) —
   gemessen acceptance-neutral und sogar schneller
   ([Note](../notes/dflash2-drafter-fp8-quant.md)).

## Wie starten

1. Container (ARM64) + Build-Stack: [`containers/`](../containers/README.md) — bis #52816/#52883 upstream gemerged sind, braucht es den [`dflash2-sm121`](https://github.com/TechPrototyper/vllm/tree/dflash2-sm121)-Branch (inkl. der #53122-Fixes für quantisierte Drafter).
2. Modell + Drafter von Hugging Face (nicht gated).
3. Serve-Kommando, Flags und Boot-Disziplin (Warmup-Einzelrequest!):
   [`recipes/dgx-spark-sm121.md`](../recipes/dgx-spark-sm121.md).

## Getestete Qualität

| Test | Ergebnis |
|---|---|
| GSM8K, voller Split (n=1319, paarweise vs. Baseline ohne Spekulation) | **95,83 %** vs. 95,15 % — McNemar p=0,122, qualitätsgleich ([Verdict](../notes/dflash2-full-split-verdict.md)) |
| Tool-Calling-Konformität | byte-identisch zum nicht-spekulativen Canon (4/5 Fälle, 5/5 konform) |
| Needle-Recall | exakt bis 240k Tokens (Kampagne 08/2026) |
| Workload-Abnahme (reale Agenten-Tasks, paarweise) | bestanden — inkl. Same-Config-Kontrolle ([Methodik-Note](../notes/kanban-workload-acceptance-and-flip-noise.md)) |

## Evolution

Der Weg zu diesem Stand, chronologisch: NVFP4-KV-Kalibrierungsbefund
(08-01) → GB10-Rezeptlinie (08-06) → DFlash2-First-Light + Gates
(08-20/21) → [Adoption](../notes/dflash2-full-split-verdict.md) (08-21)
→ [FP8-Drafter + #53122](../notes/dflash2-drafter-fp8-quant.md) (08-22).
Vollständige Findings-Historie: [README](../README.md#findings-so-far).
