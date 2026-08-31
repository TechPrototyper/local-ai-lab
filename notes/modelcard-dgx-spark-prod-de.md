# 262.000 Token Kontext, Spekulation und Prefix-Cache gleichzeitig — die Produktions-Konfiguration meines DGX Spark

*Stand 2026-08-29. Dies ist die tatsächlich laufende Produktions-Konfiguration meines kleinen Hobby-AI-Labs auf einem DGX Spark (GB10, 128 GB unified memory); alle Zahlen sind eigene Messungen, Rohdaten und englischsprachige Detail-Notes sind am Ende verlinkt.*

## Worum gehts?

ist die umgekehrte Maschine zur RTX 5090: Speicher ist im Überfluss da (128 GB, unified), aber die Speicherbandbreite ist knapp — also wird hier nicht um Gigabytes gekämpft, sondern um **Geschwindigkeit und Gleichzeitigkeit**. Die Produktions-Konfiguration, die dieses Lab dafür über Wochen erarbeitet hat, kombiniert drei Dinge, die lange als unverträglich galten:

**Erstens: Spekulatives Decoding in Produktion.** Ein kleiner Block-Diffusions-Drafter (DFlash2, 7 Tokens pro Schritt) schlägt vor, das große Modell verifiziert — beweisbar verlustfrei, die Ausgaben bleiben byte-identisch zum normalen Decoding. Auf Reasoning-Verkehr bringt das bis zu ~4× Single-Stream; strukturierte greedy Ausgaben erreichen ~51 tok/s, aggregiert wurden bis ~227 tok/s gemessen.

**Zweitens: ein 4-Bit-KV-Cache (NVFP4).** Seit dem heutigen Verdikt-Test (n=1319, gepaart) wissen wir: er kostet gegenüber dem 8-Bit-Cache **weder messbare Qualität noch Geschwindigkeit** (p=0,77; Perf-Parität) — halbiert aber die Cache-Bytes. Im selben Budget meldet die Engine damit **775.766 Tokens KV-Pool** statt ~478.000.

**Drittens — und das war bis vor zwei Tagen unmöglich: der Prefix-Cache *trifft* unter Spekulation.** Agenten senden pro Runde denselben großen Kontext erneut; bis vor Kurzem bekam der Cache unter Spekulation exakt 0 % Treffer (vllm#52244). Der Fix aus vllm#50897, von diesem Lab auf beide Architekturen portiert und validiert, läuft hier seit dem 28.08. in Produktion: **90 % Cache-Wiederverwendung** auf einem realen 12.800-Token-Replay, Antworten byte-identisch.

Qualität ist dabei mehrfach abgesichert: Die volle Batterie auf exakt dieser Konfiguration besteht GSM8K-250 mit 0,996, Tool-Calling 6/6, Needle 6/6, Determinismus 5/5; der große n=1319-Lauf der Nacht liegt mit 0,9704 innerhalb des gemessenen Rauschbodens zweier identischer Läufe. Ein Wiederholungslauf derselben Konfiguration diente zugleich als Rausch-Kontrolle — die Methodik, die alle Verdikte dieses Labs lesbar macht.

**Zum Selbst-Ausprobieren:** Weil ein Teil dieser Verbesserungen in der offiziellen vLLM-Entwicklung noch unterwegs ist, haben wir begonnen, **fertige Container** bereitzustellen — Serving-Images, in die wir ausgewählte, bereits validierte Verbesserungen aus der laufenden Entwicklung vorab integrieren, damit man diesen Stand ohne eigenes Bauen nachfahren kann. Das Spark-Image ist öffentlich auf [GHCR](https://github.com/users/TechPrototyper/packages/container/package/vllm-sm12x) (inklusive des Prefix-Cache-Fixes), Übersicht und Rezepte im [Container-Verzeichnis des Labs](https://github.com/TechPrototyper/local-ai-lab/tree/master/containers) — hier legen wir in den kommenden Wochen nach.

## Model-Card (Produktions-Konfiguration)

| Feld | Wert |
|---|---|
| **Konfigurationsname** | DGX-Spark-Produktion (sm121 / GB10) |
| **Basismodell** | Qwen3.8-27B (hybrid GDN + Full-Attention) |
| **Gewichte** | PrismaAQUA 5,5 bpp mixed-precision (NVFP4+FP8, KL-Fisher), 23,6 GB |
| **Drafter** | [`TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm`](https://huggingface.co/TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm) — DFlash2, fp8, `num_speculative_tokens: 7` |
| **Engine** | vLLM (sm12x-Linie mit vorab integrierten Verbesserungen: `f4c27c0da` + vllm#53122-Fix + vllm#50897) + FlashInfer 0.6.15 |
| **KV-Cache** | **NVFP4 (4 bit)** — seit 29.08., Verdikt-gedeckt (p=0,77 vs fp8, Perf-Parität) |
| **Kontext / Pool** | max-len 262.144 · **KV-Pool 775.766 Tokens** (21,6-GiB-Budget) |
| **Prefix-Caching** | an (vllm#50897) — **90 % Replay-Reuse** in Produktion gemessen |
| **Parser** | Reasoning (qwen3) + Tool-Calling (qwen3_coder) aktiv |

### Gemessene Werte

| Achse | Wert |
|---|---|
| **Qualität** (GSM8K n=250, volle Batterie, diese Config) | **0,996** · Tools 6/6 · Needle 6/6 · Determinismus 5/5 |
| Qualität (n=1319, Config-Familie, greedy) | 0,9704 — im Rauschboden identischer Wiederholungen (p=0,30-Kontrolle) |
| **Decode-Spektrum** (Single-Stream) | ~22 tok/s Prosa (temp 1,0) → **~51 tok/s** greedy strukturiert |
| Spekulations-Gewinn | bis ~4× Single-Stream auf Reasoning-Verkehr · Akzeptanz ~52–54 %, Ø ~4,8 |
| **Aggregat** | bis **~227 tok/s** (c=16, draft-length 7) |
| **Prefix-Replay** (12,8k-Token-Prompt, Produktion) | **90,0 % Cache-Reuse**, byte-identische Antworten |
| Batterie-Perf (32k-Testform) | 23,1–23,3 tok/s single · 51–54 tok/s batched-8 |

### Grenzen (ehrlich)

- Die ~4×/227-tok/s-Werte stammen aus den Verdikt-Messreihen der Vorwochen auf dieser Serving-Linie; das Decode-Spektrum ist stark workload-abhängig (Prosa vs. strukturiert — ein einzelner tok/s-Wert wäre irreführend).
- MTP bleibt bewusst aus (dokumentierte Tool-Calling-Regression).
- Der 4-Bit-KV-Cutover ist qualitäts- und speed-verdikt-gedeckt; der veröffentlichte Container trägt ihn noch nicht (der validierte Produktionspfad läuft derzeit über eingebundene Quell-Verzeichnisse — ein aktualisiertes Image ist ein offener Arbeitspunkt).
- Ein Wechsel der Gewichte auf die 20-GB-Scout-Variante (auf der RTX verdikt-gleich und ~10 % schneller) wird derzeit auf dieser Maschine vorbereitet; bei grünem Ergebnis wird diese Card aktualisiert.

## Links

[Nacht-Note 08-28 — Rollout + Replay-Zahlen (englisch)](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/night-2026-08-28-pc50897-scout-h2h.md) · [Cutover-Verdikt nvfp4/fp8](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/verdict-nvfp4kv-vs-fp8kv-cutover-n1319-2026-08-29.md) · [Nachlauf + Rauschboden](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/verdict-prod50897-nachlauf-noisefloor-n1319-2026-08-29.md) · [Prod-Batterie-Rohdaten](https://github.com/TechPrototyper/local-ai-lab/blob/master/results/RESULT_sm121-PROD-pc50897.json) · [Draft-Length-Map — 227 tok/s](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/dflash2-draft-length-map.md) · [Spark-Rezept](https://github.com/TechPrototyper/local-ai-lab/blob/master/recipes/dgx-spark-sm121.md) · [Das Lab-Notebook](https://github.com/TechPrototyper/local-ai-lab)
