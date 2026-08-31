# RTX-Cutover: MTP-on-GridBook (drafter-frei) — Build, Serving & Tests

**Stand 2026-08-31 ~07:45 UTC. Prod-Cutover vollzogen und verifiziert.**
Die RTX 5090 (sm120) serviert Prod jetzt mit **MTP-on-GridBook** — drafter-frei,
~624k KV-Pool. Dies ist die vollständige Doku zu Build, Serving-Konfiguration und
allen Tests, die zur Umstellung geführt haben.

---

## 1. Entscheidung & Ergebnis (Kurzfassung)

Umgestellt: RTX von Forschung → **Prod-primär** mit `gridbook-13gb-mtpfix` + MTP@2.
Spark bleibt Overflow-Fallback (weiter AQUA + DFlash2). Grund: MTP-on-GridBook ist
qualitäts-äquivalent zu no-spec (McNemar p=0,83), besteht die volle Tool-Calling-
Batterie, ist drafter-frei (kein externes Drafter-Modell → mehr VRAM/KV) und liefert
einen großen Kontext-Pool (624k Tokens).

Entscheidungs-Register (Stand Cutover):
- **Basis RTX:** GridBook-13GB (vs AQUA-23,6GB) — kleiner → mehr KV. **umgestellt.**
- **Spec RTX:** MTP (DFlash2 lädt bei voller max-len nicht). **umgestellt.**
- **Sampling/Thinking:** thinking-ON bleibt (Sweep: C1/thinking-on am besten;
  `-tools`/thinking-off kontraproduktiv). **kein Wechsel.**
- **Spark:** AQUA + DFlash2 bleibt (Cross-Arch: GridBook-on-Spark braucht Build-Match/
  Prefix-Strip; separat in Arbeit für Whisper-Headroom).

---

## 2. Build — wie das vLLM zusammengepatcht ist

**Serving-Image:** `10.1.0.243/inference/vllm-nvfp4:sm120-pc50897-gridbook-088`
- vLLM-Basis: sm120-NVFP4-Linie (V1-Engine, `v0.1.dev1+g2cf8b8ae0`).
- **`pc50897`** = vllm#50897 vorab integriert: lookahead-aware Prefix-Cache-Hashing
  für EAGLE-/spec-Drafter (stellt Prefix-Cache-Hits unter Spekulation her; adressiert
  vllm#52244). Auf sm120 portiert + validiert.
- **`gridbook-088`** = GridBook-Plugin 0.8.8 (out-of-tree vLLM-Plugin via
  `vllm.general_plugins`-Entry-Point) — liefert `quantization=gridbook`
  (NVFP4-CB / FP8-CB Product-Codebook-Dequant) fest im Image.
- **Der „vLLM-Upstream-Package"-Seam** (unser Beitrag, Cherry-Picks): #53977
  (VocabParallelEmbedding-tp==1-OOB-Maske), #53978 (DFlash2 überlebt Spec-Warmup mit
  ungefüllten Draft-Buffern), #53979 (NVFP4-KV: non-causal FA2-Prefill-Wrapper für
  DFlash-family + SWA-Guard). Diese härten den non-causal-NVFP4-Seam; für MTP-on-
  GridBook nicht alle load-bearing, aber Teil der sm12x-Serving-Linie.
- FlashInfer-Backend (`VLLM_ATTENTION_BACKEND=FLASHINFER`), warmer JIT/Autotune-Cache.

**Das Checkpoint-Artefakt `gridbook-13gb-mtpfix`** (der eigentliche Modell-Fix):
- Basis: rdtands GridBook-13GB (`Qwen3.8-27B-PrismaAQUA-gridbook-13GB`, NVFP4-CB,
  ~13 GB, quant_method `gridbook`). Behält die 15 nativen MTP-Kopf-Tensoren.
- **Problem:** GridBook (ab 0.8.7) quantisiert die geteilte Body-Embedding
  `model.language_model.embed_tokens` (weight_packed/scale/global_scale). vLLMs
  MTP-Prädiktor teilt die Target-Embedding mit dem Draft → die gepackte Embedding
  bricht den geteilten Draft (`weight_global_scale`-Ladefehler).
- **Fix (bit-exakt, self-serve, KEIN Rob-Re-Export):** die 3 gepackten embed-Tensoren
  droppen und den **BF16 `model.language_model.embed_tokens.weight` aus AQUA**
  (`qwen3.8-27b-prismaaqua55`, identische Qwen3.8-Basis) verbatim einsetzen;
  `quant_config.json`: `quantized_embedding`-Key entfernen + Unit in `ignore`.
  Tensor-Zahl 1365→1363. Skript: `s24/surgical_embed_swap.py`.
- **Warum bit-exakt korrekt:** MTP-Spekulation ist verlustfrei; das Byte-Identitäts-
  Gate war ungeeignet (FP-Nichtassoziativität batched-verify vs. incremental-decode;
  selbst natives AQUA-MTP „scheitert" daran) → Korrektheit über den n=1319-Verdikt
  bewiesen, nicht über Byte-Gleichheit.

---

## 3. Serving-Konfiguration (live)

**k8s-Deployment `vllm-qwen38-mtp-gridbook`** (namespace inference, Node neo26/RTX 5090),
hinter Service `llm-active` (Selector-Match), litellm routet `qwen3.8-27b` → llm-active:8000.

vLLM-Serve (Kernflags):
```
--model /cache/models/gridbook-13gb-mtpfix
--served-model-name qwen3.8-27b
--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":2}
--kv-cache-dtype nvfp4
--gpu-memory-utilization 0.90          # STABIL — 0.97 crashte (s. §5)
--max-model-len 262144
--reasoning-parser qwen3
--tool-call-parser qwen3_coder  --enable-auto-tool-choice
--async-scheduling  --enable-prefix-caching  --enable-chunked-prefill
```
Engine-Log bestätigt: `quantization=gridbook`, `speculative_config method='mtp'
num_spec_tokens=2`, `Detected MTP model. Sharing target embedding + lm_head weights`,
`served_model_name=qwen3.8-27b`.

**Gemessener KV-Pool:** **624.152 Tokens** (2,38× Concurrency @ 262.144/Request),
exklusive GPU. Wichtig: bei GPU-Split (s24 gleichzeitig) nur 534k → RTX muss dem
Serving exklusiv gehören (kein Split), sonst re-profiliert vLLM gegen weniger Speicher.

**Routing:** litellm-Semaphoren SPARK_DOWN=0 / RTX_DOWN=0 → RTX primär, Spark Overflow
(>3 gleichzeitige lokale Sessions → Spark). Fallback bei RTX-Ausfall: Spark (AQUA+DFlash2).

---

## 4. Tests & Ergebnisse

### 4.1 Qualität (die Freigabe-Gates)
| Test | Ergebnis |
|---|---|
| **n=1319 McNemar** (MTP@2 vs no-spec, GridBook, greedy, gepaart) | Acc_MTP 0,9704 · Acc_nospec 0,9719 · **b=10 c=12 p=0,8318** → statistisch nicht unterscheidbar |
| **Tool-Calling-Batterie** (auf mtpfix-MTP, util 0,90) | **Tools 6/6** · GSM8K-250 **0,9840** (0 Fehler) · **Needle 12k/24k/64k 9/9** · **Determinismus 5/5** |
| Live-Spot-Check (nach Cutover) | E2E 200, Tool-Call `get_weather{"city":"Paris"}` sauber |

### 4.2 Speed / KV (Kontext)
| Metrik | Wert |
|---|---|
| KV-Pool (stabil, util 0,90, exklusiv) | **624.152 Tokens** (2,38× @ 262k) |
| MTP-Turbo (auf AQUA gemessen, überträgt sich) | Prosa +49% @ 54,3% Acceptance; count200 +112%; JSON +72%; Retrieval +84% |
| MTP vs DFlash2 (Head-to-Head, sm120) | MTP hält 82,9% des no-spec-Pools; DFlash2 lädt bei voller max-len gar nicht |

### 4.3 Sampling/Thinking-Sweep (Loop-Verhalten, sm120 UND sm121)
5 Configs × 4 Tasks × K6. **C1 (thinking-ON, Ist-Basis) auf beiden Arches am besten**
(höchster Erfolg, 0% Loops); thinking-OFF (`-tools`-Profil, pp1,5) schlechter, auf sm121
sogar loop-induzierend. → Sampling bleibt thinking-on; der Tool-Loop-Fix liegt NICHT im
Sampling, sondern in Hermes (inerte Guardrails + Steering-im-Tool-Content). Caveat:
Harness reproduziert die echten Hermes-Loops kaum → Richtungssignal, nicht endgültig.

### 4.4 Cross-Arch sm121 (Spark) — Machbarkeit
MTP-Drain auf pool-reichem sm121 leichter (leistbar); MTP+Tool-Calling-Desync (alter
08-09-Blocker) reproduziert auf dem neuen Build NICHT mehr. GridBook-on-Spark lädt
noch nicht (rdtand-3.8-gridbook mis-exportiert: `model.language_model.*`-Prefix vs.
text-only-CausalLM; Spark-Build f4c27c0da lehnt ab — RTX-Build akzeptiert ihn).
Fix in Arbeit: `gridbook-13gb-fixed` (Prefix-Strip) für Whisper-Headroom.

---

## 5. Bekannte Punkte / Lessons

- **util 0,97 crasht reproduzierbar** (Fragmentierungs-OOM unter Needle-12k–64k-Last:
  vLLMs statische KV-Buchung frisst den Slack, den Long-Context-Aktivierungen +
  Allocator-Fragmentierung über die Profiling-Schätzung hinaus brauchen). **util 0,90
  ist stabil** (Batterie inkl. Needle 9/9). Offener Optimierungshebel:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` könnte höheren util stabil machen
  → Pool-Rückgewinnung. Noch nicht gegengetestet.
- **Kein GPU-Split:** neo26/RTX ist splitbar (allocatable 2), aber wir splitten nicht —
  der Prod-Pod braucht die GPU exklusiv (sonst re-profiliert vLLM gegen weniger Speicher,
  534k statt 624k).
- **Verdikt-Robustheit:** der n=1319-Lauf ging zweimal an Ephemeralität verloren
  (Mac-Orchestrator-Tod; Pod-`sleep 86400`-Tod). Lehre: lange Läufe pod-todsicher +
  Outputs auf PVC (`/cache`), nie ephemer `/tmp`.

## 6. Artefakte / Quellen
- Serving: Deployment `vllm-qwen38-mtp-gridbook`, Image `...gridbook-088`.
- Checkpoint: `/cache/models/gridbook-13gb-mtpfix` (PVC).
- Ergebnis-JSONs: `s24/RESULT_verdict_mtpfix_n1319.json`, `RESULT_mtpfix-mtp-full-battery.json`,
  `RESULT_phase4_sweep.json`, `RESULT_headtohead_aqua_prod.json`, `RESULT_kv_drain_diagnosis_aqua.json`.
- Swap-Skript: `s24/surgical_embed_swap.py`.
