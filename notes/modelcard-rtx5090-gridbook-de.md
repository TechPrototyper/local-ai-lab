# Qwen3.8-27B-Modell mit ~900.000 Token Kontext auf einer Consumer-Grafikkarte — die vermessen(st)e Konfiguration der RTX 5090

*Stand 2026-08-29. Alle Zahlen sind Einzelmessungen meines kleinen Hobby-AI-Labs, hier mit meiner RTX 5090; Rohdaten und englischsprachige Detail-Notes sind am Ende verlinkt.*

## Worum gehts?

Vor ein paar Wochen habe ich Euch von meiner Wette erzählt: einen Eigenbau-server mit einer NVIDIA RTX 5090 zur professionellen Inferenzmaschine zu machen. Warum ich das für möglich hielt, und was mich motiviert hat, findet sich im Artikel gut dokumentiert. Die RTX 5090 ist für eine Consumer GPU das ultimative Rechenmonster, und der Speicherdurchsatz ist ebenfalls jenseits allem, was sich sonst in der Kategorie am Markt bewegt (zur Vergleich Apple's diese Woche angekündigter und ab Ende September erhältliche M5 Ultra liegt bei gut 2/3 der Performance zum Durchsatz, beim Rechnen trennen die beiden immer noch Faktoren). Nun hat die RTX 5090 einen ziemlich restriktiven Engpass, nämlich den VRAM, der zwar schnell ist, aber mit 32 GB VRAM vergleichsweise klein. Ein 27B-Modell muss schon stark optimiert werden, und das haben wir auch bereits gemacht: in unserer bisherigen Produktionsquantisierung (~5,5 bpp, mixed-precision) kamen wir ohne nennenswerten Qualitätsverlust ggü. BF16 durch eine geniale gemischte Quantisierung auf ca. 23,6 GB Footprint für die Gewichte— der Rest wird dann zum Kontext (KV-Cache), also zum Arbeitsgedächtnis. Die Frage meines Sommers (Juli Urlaub, August Elternzeit) war: **Wie klein dürfen die Gewichte werden, bevor die Qualität messbar nachgibt?** Und was kann man sonst noch tun, um den Engpass VRAM besser zu bewirtschaften? Jedes gesparte Gigabyte zählt, d.h. jede Optimierung muss die effektive Nutzung von VRAM ins Zentrum stellen, immer unter der Maßgabe, dass es bei der Optimierung nicht zu einer Qualitätseinbuße kommt, oder zu unverhältnismäßigem Verzicht auf Performance.

Die Antwort, Stand heute, lautet: **wir sind nun für Qwen3.8.27b auf 13 GB — ohne messbaren Qualitätsverlust auf Verdikt-Niveau.** Das 13-GB-Artefakt (GridBook, eine Produkt-Codebook-Quantisierung von Rob Tand) landet im gepaarten GSM8K-Test über 1.319 Aufgaben auf **exakt derselben Trefferzahl** wie die 23,6-GB- und auch eine neuere, ebenfalls nochmals optimierte 20-GB-Variante — 1287 von 1319, McNemar p=1,0 gegen beide, bei identischer Konfiguration und greedy Decoding. Die wenigen abweichenden Aufgaben verteilen sich symmetrisch (9:9) und liegen innerhalb des Rauschbodens, den zwei byte-identische Wiederholungsläufe desselben Setups erzeugen. Langkontext-Nadeltests (needle) und Determinismus-Wiederholungen bestehen vollständig.

Was die kleineren Gewichte kaufen, haben wir anschließend vermessen: Bei Produktions-Form (262k Max-Kontext, 97 % Speichernutzung, 4-Bit-KV-Cache in NVFP4) meldet die Engine einen **KV-Pool von 898.037 Tokens** — das 2,6-fache der bisherigen Produktionsform. Dazu ~60 tok/s Single-Stream, 4.500–6.500 tok/s Prefill, und — dank des heute quer durch den Stack validierten Prefix-Cache-Fixes (vllm#50897) — **13- bis 42-fache Beschleunigung**, wenn ein großer Kontext erneut gesendet wird: ein 88.000-Token-Prompt fällt von ~16–18 Sekunden auf 0,4–1,4 Sekunden. Für Agenten-Workloads, die pro Runde denselben Riesenkontext neu schicken, dürfte das der praktisch wertvollste Einzelwert sein.

Spekulatives Decoding (DFlash2) haben wir ehrlich mitvermessen: Auf strukturierten Ausgaben (Zählen, JSON) liefert es auch auf dem GridBook-Target **149–156 tok/s** (Akzeptanz 72 %) — auf offener Prosa bricht die Draft-Akzeptanz gegenüber dem früheren Target jedoch auf ~24 % ein, womit der Prosa-Gewinn derzeit entfällt. Die Ursache ist plausibel verteilungsbedingt; ein auf die aktuelle Quantisierung hin nachgezogener Drafter ist der naheliegende Fix und wird durch Agenten in den kommenden Tagen abgewickelt! Update folgt an dieser Stelle.

Das ganze Setup steht auf der Schulter von großen Köpfen aus der ganzen Welt. Allen voran ist Rob Tand (New York) (https://www.linkedin.com/in/rob-tand/) und sein PrismaQuant zu nennen; der vLLM-basierte Serving-Pfad ist das große Werk von Jetha Chan (Tokyo) (https://www.linkedin.com/in/jethac/) und so geht es weiter mit wertvollen Findings zum Prefill-Cache, zu Speculative Decoding, sowie zu eigenen Messungen und kleineren Beiträgen. AI-Community Work, vereint im Bestreben, Consumer Grade GPUs ans Limit zu führen, ihnen zu entlocken, wozu sie wirklich im Stande sind, und die Voraussetzungen dafür zu schaffen, dass starke Modelle mit voller Qualität, schnell und zuverlässig auf kleineren GPUs laufen.

**Zum Selbst-Ausprobieren:** Weil ein Teil dieser Verbesserungen in der offiziellen vLLM-Entwicklung noch unterwegs ist, haben wir begonnen, **fertige Container** bereitzustellen — Serving-Images, in die wir ausgewählte, bereits validierte Verbesserungen aus der laufenden Entwicklung vorab integrieren, damit man diesen Stand ohne eigenes Bauen nachfahren kann. Die ersten Images (für RTX 5090 und DGX Spark) sind öffentlich auf [GHCR](https://github.com/users/TechPrototyper/packages/container/package/vllm-sm12x), Übersicht und Rezepte im [Container-Verzeichnis des Labs](https://github.com/TechPrototyper/local-ai-lab/tree/master/containers) — hier legen wir in den kommenden Wochen nach, allerdings: Mileage may vary, Elternzeit ist vorbei, keine Nachtschichten mehr, denn ab Dienstag darf ich wieder arbeiten!

## Model-Card (Serving-Konfiguration)

| Feld | Wert |
|---|---|
| **Konfigurationsname** | GridBook-13GB auf RTX 5090 (sm120) |
| **Basismodell** | Qwen3.8-27B (hybrid GDN + Full-Attention) |
| **Gewichte** | [`rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm) — GridBook FP8-CB-Produkt-Codebook, **13 GB** |
| **Engine** | vLLM (sm12x-Linie mit vorab integrierten Verbesserungen: v4@`2cf8b8a` ∪ vllm#50897 ∪ #53979-SWA-Guard) + `pip install gridbook==0.8.8` + FlashInfer |
| **KV-Cache** | **NVFP4 (4 bit)**, `--kv-cache-dtype nvfp4` |
| **Kontext / Pool** | max-len 262.144 · **KV-Pool 898.037 Tokens** (util 0,97) |
| **Prefix-Caching** | an (vllm#50897); Replay-Hits 65–79 % gemessen |

### Gemessene Werte (eine Karte, einzelne Läufe)

| Achse | Wert |
|---|---|
| **Qualität** (GSM8K n=1319, greedy, gepaart) | **0,9757** — identisch mit 23,6-GB- und 20-GB-Artefakt (je 9:9, p=1,0) |
| Needle (12k/24k Wörter × 3 Tiefen) | 6/6 |
| Determinismus (5 Prompts × 3 Läufe) | 5/5, identische Hashes |
| **Decode, Prosa** (256 Tok, greedy, warm) | ~60 tok/s |
| **Decode, strukturiert** (mit DFlash2-Spec, fp8-KV) | **149–156 tok/s** (Akzeptanz 72,3 % / Ø 6,06) |
| **Prefill** (19.7k–88k-Token-Prompts, kalt) | ~4.500–6.500 tok/s |
| **Prefix-Replay** (88k-Token-Prompt) | 17,7 s → **1,36 s (13×)** · im Spec-Setup 15,97 s → **0,38 s (42×)** |
| **Sessions** (256-Tok-Läufe parallel) | c=2: 98 · c=4: 154 · c=8: 191 tok/s aggregiert (~24 tok/s je Session ab c≥5) |
| Sessions mit Spec (fp8-KV, bis c=7) | c=7: 258,8 tok/s aggregiert (~37 je Session) |

### Grenzen

- **NVFP4-KV × Spekulation** ist derzeit bewusst blockiert: der Drafter ist durchgehend Sliding-Window (2048), und für non-causal+SWA auf dem FA2-NVFP4-Pfad fehlt der Kernel-Paritätstest (Jetha Chans #53979-Finding; unser Guard). Auf dem DGX Spark läuft die Kombination produktiv mit byte-identischen Gates — der formale Test ist der nächste Arbeitspunkt.
- **Prosa-Spekulation** lohnt auf diesem Target derzeit nicht (Akzeptanz ~24 %); ein GridBook-getunter Drafter ist der geplante Fix.
- Tool-Calling wurde in dieser Konfiguration nicht mitgetestet (Parser-Flags nicht gesetzt); GSM8K ist eine Aufgabe, kein Benchmark-Universum; Langkontext-*Qualität* jenseits des Needle-Rasters (mehrere hundert k) ist noch unvermessen.
- Alle Zahlen: eine Karte, ein Tag, einzelne Läufe — indikativ, kein Benchmark-Suite-Anspruch.

## Links

[Detail-Note (englisch)](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/endgegner-gridbook-rtx-2026-08-29.md) · [Verdikt-Rohdaten](https://github.com/TechPrototyper/local-ai-lab/blob/master/results/RESULT_sm120-gridbook13-n1319.json) · [Sweep Max-Pool](https://github.com/TechPrototyper/local-ai-lab/blob/master/results/SWEEP_MAXPOOL.json) · [Sweep Turbo](https://github.com/TechPrototyper/local-ai-lab/blob/master/results/SWEEP_TURBO.json) · [Qualitäts-Triage 08-23](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/gridbook-13gb-quality-holds.md) · [#50897-Validierung auf der RTX](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/pc50897-sm120-cache-under-spec-2026-08-29.md) · [Das Lab-Notebook](https://github.com/TechPrototyper/local-ai-lab)
