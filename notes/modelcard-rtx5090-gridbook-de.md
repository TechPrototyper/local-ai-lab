# Ein 27B-Modell mit ~900.000 Token Kontext auf einer Consumer-Grafikkarte — die vermessene Endgegner-Konfiguration der RTX 5090

*Kurzartikel + Model-Card (deutsch), Stand 2026-08-29. Alle Zahlen sind
Einzelmessungen dieses Labors auf einer RTX 5090; Rohdaten und
englischsprachige Detail-Notes sind am Ende verlinkt.*

## Der Kurzartikel

Eine RTX 5090 hat 32 GB Speicher. Ein 27B-Modell belegt davon in unserer
bisherigen Produktionsquantisierung (~5,5 bpp, mixed-precision) rund
23,6 GB — der Rest wird zum KV-Cache, also zum Arbeitsgedächtnis für den
Kontext. Die Frage des Sommers war: **Wie klein dürfen die Gewichte
werden, bevor die Qualität messbar nachgibt?** Jedes gesparte Gigabyte
wird direkt zu mehr Kontext.

Die Antwort, Stand heute, lautet: **mindestens bis 13 GB — ohne messbaren
Qualitätsverlust auf Verdikt-Niveau.** Das 13-GB-Artefakt (GridBook, eine
Produkt-Codebook-Quantisierung von Robert Tand) landet im gepaarten
GSM8K-Test über 1.319 Aufgaben auf **exakt derselben Trefferzahl** wie
die 23,6-GB- und die 20-GB-Variante — 1287 von 1319, McNemar p=1,0 gegen
beide, bei identischer Konfiguration und greedy Decoding. Die wenigen
abweichenden Aufgaben verteilen sich symmetrisch (9:9) und liegen
innerhalb des Rauschbodens, den zwei byte-identische Wiederholungsläufe
desselben Setups erzeugen. Langkontext-Naldeltests (needle) und
Determinismus-Wiederholungen bestehen vollständig.

Was die kleineren Gewichte kaufen, haben wir anschließend vermessen: Bei
Produktions-Form (262k Max-Kontext, 97 % Speichernutzung, 4-Bit-KV-Cache
in NVFP4) meldet die Engine einen **KV-Pool von 898.037 Tokens** — das
2,6-fache der bisherigen Produktionsform. Dazu ~60 tok/s Single-Stream,
4.500–6.500 tok/s Prefill, und — dank des heute quer durch den Stack
validierten Prefix-Cache-Fixes (vllm#50897) — **13- bis 42-fache
Beschleunigung**, wenn ein großer Kontext erneut gesendet wird: ein
88.000-Token-Prompt fällt von ~16–18 Sekunden auf 0,4–1,4 Sekunden.
Für Agenten-Workloads, die pro Runde denselben Riesenkontext neu
schicken, dürfte das der praktisch wertvollste Einzelwert sein.

Spekulatives Decoding (DFlash2) haben wir ehrlich mitvermessen: Auf
strukturierten Ausgaben (Zählen, JSON) liefert es auch auf dem
GridBook-Target **149–156 tok/s** (Akzeptanz 72 %) — auf offener Prosa
bricht die Draft-Akzeptanz gegenüber dem gewohnten Target jedoch auf
~24 % ein, womit der Prosa-Gewinn derzeit entfällt. Die Ursache ist
plausibel verteilungsbedingt (der Drafter wurde gegen die
AQUA/BF16-Verteilung trainiert); ein auf GridBook nachgezogener Drafter
wäre der naheliegende Fix und ist als Arbeitspunkt notiert.

Nichts hiervon ist ein Alleingang: Die Gewichte stammen von Robert Tand
(PrismaQuant), der Serving-Pfad steht auf jethacs
Consumer-Blackwell-Linie (vllm#46329), der Prefix-Cache-Fix auf
ZJY0516s vllm#50897, und unser eigener Beitrag — der
Non-Causal-NVFP4-Seam (#53977/#53978/#53979) — trägt seit heute auch
den im Review erbetenen Sliding-Window-Guard. So sieht es aus, wenn
eine Community an derselben Karte zieht.

## Model-Card (Serving-Konfiguration)

| Feld | Wert |
|---|---|
| **Konfigurationsname** | „Endgegner" — GridBook-13GB auf RTX 5090 (sm120) |
| **Basismodell** | Qwen3.8-27B (hybrid GDN + Full-Attention) |
| **Gewichte** | [`rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm) — GridBook FP8-CB-Produkt-Codebook, **13 GB** |
| **Engine** | vLLM (sm12x-Carry-Line: v4@`2cf8b8a` ∪ vllm#50897 ∪ #53979-SWA-Guard) + `pip install gridbook==0.8.8` + FlashInfer |
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

### Grenzen (ehrlich)

- **NVFP4-KV × Spekulation** ist derzeit bewusst blockiert: der Drafter
  ist durchgehend Sliding-Window (2048), und für non-causal+SWA auf dem
  FA2-NVFP4-Pfad fehlt der Kernel-Paritätstest (jethacs #53979-Finding;
  unser Guard). Auf sm121 läuft die Kombination produktiv mit
  byte-identischen Gates — der formale Test ist der nächste Arbeitspunkt.
- **Prosa-Spekulation** lohnt auf diesem Target derzeit nicht
  (Akzeptanz ~24 %); ein GridBook-getunter Drafter ist der geplante Fix.
- Tool-Calling wurde in dieser Konfiguration nicht mitgetestet
  (Parser-Flags nicht gesetzt); GSM8K ist eine Aufgabe, kein Benchmark-
  Universum; Langkontext-*Qualität* jenseits des Needle-Rasters (mehrere
  hundert k) ist noch unvermessen.
- Alle Zahlen: eine Karte, ein Tag, einzelne Läufe — indikativ, kein
  Benchmark-Suite-Anspruch.

## Provenienz

[Endgegner-Note (en, Details)](endgegner-gridbook-rtx-2026-08-29.md) ·
[Verdikt-Rohdaten](../results/RESULT_sm120-gridbook13-n1319.json) ·
[Sweeps](../results/SWEEP_MAXPOOL.json) ([Turbo](../results/SWEEP_TURBO.json)) ·
[Qualitäts-Triage 08-23](gridbook-13gb-quality-holds.md) ·
[#50897-Validierung](pc50897-sm120-cache-under-spec-2026-08-29.md)
