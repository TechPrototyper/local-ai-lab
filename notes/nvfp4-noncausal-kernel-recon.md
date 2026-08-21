# Recon: NVFP4-KV + non-causal DFlash2 drafter attention on sm120/sm121

**Status:** DRAFT — Recon-Spike (Kanban #29). Kein GPU-Lauf, reine Code-/Architektur-Archäologie.
**Datum:** 2026-08-20 (Nacht). **Autor:** Session-B Agent.
**Quellen:** lokaler Fork-Checkout `~/Projects/vllm-upstream-work` (Branch
`jethac/spark/hijinks-020-aeon-qwen-dflash-sm121a`, HEAD `e2a8197a9` — der im Auftrag
genannte `e24d26638` ist lokal nicht vorhanden, Branch-Inhalt ist inhaltlich äquivalent),
flashinfer PR **#3684** (MERGED, via `gh`, read-only), vLLM PRs **#52816/#52883** (beide OPEN),
InferenceDream (`custom_kv_kernel.cu`, `kdev.py`, `RESUME_NIGHT_KERNEL_OPT.md`).

Zeilennummern beziehen sich auf den obigen Branch-Stand.

---

## 1. The gap (Kurzfassung)

DFlash/DFlash2-Verify braucht Attention, bei der ein kleiner **Query-Block** von
`1 + num_speculative_tokens` Tokens (bei `num_speculative_tokens=15` → **16 Query-Tokens/Req**)
**den gesamten committeten Prefix voll sieht UND innerhalb des Draft-Blocks bidirektional** ist
(Block-Diffusion). Der Prefix-K/V liegt als NVFP4-Page-Pool vor.

Diese Kombination — **non-causal Query-Block über NVFP4-KV-Pages** — existiert an **keiner**
Stelle des Stacks:

- **flashinfer:** Der NVFP4-Paged-Prefill-Pfad (FA2, sm120/121) aus #3684 ist im vLLM-Aufruf
  effektiv **causal-only** verdrahtet; ob der Kernel selbst `mask_mode ≠ causal` / `custom_mask`
  über NVFP4-Pages korrekt trägt, ist **ungetestet** (siehe §4-Option-A).
- **vLLM:** Zwei harte Guards (Backend-Selektion + hartkodiertes `causal=True` im Build)
  verhindern, dass der non-causale Drafter überhaupt auf den FlashInfer-NVFP4-Pfad kommt. Der
  Drafter wird auf ein non-causal-fähiges Backend (FLASH_ATTN) gezwungen, dessen KV-Pages **nicht
  NVFP4** sind (bf16/fp16; fp8 nur via `kv_cache_dtype_skip_layers`). Ergebnis: getrennter/gepolsterter
  Page-Pool für die Drafter-Layer → Page-Padding-Blowup (~41 GiB @32k). Der Skip-Layer-Ausweg ist
  genau dieser Blowup und damit tot.

**Payoff bei Auflösung:** Spekulation **und** 4-bit-KV gleichzeitig auf jeder speichergebundenen
Karte — der zentrale speed↔kv-max-Konflikt wird aufgelöst statt gemanagt.

---

## 2. Where exactly it lives (Dateien + Zeilen)

### 2a. Guard #1 — Backend-Selektion schließt FlashInfer für non-causal aus
- `vllm/v1/attention/backend.py:229-237` — `AttentionBackend.supports_non_causal()` liefert per
  Default **`False`**. **FlashInferBackend überschreibt dies NICHT** (kein Override in
  `vllm/v1/attention/backends/flashinfer.py`).
- `vllm/v1/attention/backend.py:323-324` — `validate_configuration(...)`: bei
  `use_non_causal and not cls.supports_non_causal()` → invalid-reason
  `"non-causal attention not supported"`. Damit fällt FlashInfer für den Drafter raus.
- `vllm/v1/attention/backends/flash_attn.py:113-114` — nur **FLASH_ATTN** (plus rocm_attn,
  rocm_aiter_fa, flex_attention) liefert `supports_non_causal() == True`.
- `vllm/v1/attention/backends/flash_attn.py:70-74` — FLASH_ATTN
  `supported_kv_cache_dtypes = ["auto","float16","bfloat16"]` → **kein fp8, kein nvfp4**.
- `vllm/config/speculative.py:114` — der Kommentar sagt es explizit:
  *"DFlash needs a non-causal-capable backend like FLASH_ATTN"*.
- Wahl-/Init-Pfad: `vllm/v1/spec_decode/dflash.py:73-82` (`_create_draft_vllm_config` setzt
  `use_non_causal=not self.dflash_causal`), Assertion in `dflash.py:286-293`
  (jedes Layer muss `attn_metadata.causal is False` erfüllen), Drafter-Backend-Init in
  `vllm/v1/worker/gpu_model_runner.py:6816-6819`.

**Netto:** non-causal ⇒ nur FLASH_ATTN qualifiziert ⇒ Drafter-KV kann nicht NVFP4 sein.

### 2b. Guard #2 — FlashInfer-Build verdrahtet `causal=True` hart
Selbst wenn Guard #1 umgangen würde:
- `vllm/v1/attention/backends/flashinfer.py:1445-1463` — `prefill_wrapper.plan(..., causal=True, ...)`
  **hartkodiert**; `common_attn_metadata.causal` wird ignoriert, **kein `custom_mask`/`packed_custom_mask`
  plumbing**.
- Ebenso `flashinfer.py:1350` (Cascade) und die trtllm-Prefill-Pfade — überall `causal=True`.
- Der einzige non-causal-Prefill im Backend ist der **DCP-Context-Run**
  (`flashinfer.py:450 causal=False` / `:466 causal=True` = Two-Pass Context+NewTokens mit
  LSE-Merge) — aber der ist für den FA2-NVFP4-Pfad **explizit gesperrt**
  (`flashinfer.py:877-880`: `use_fa2_nvfp4_kv and use_dcp` → `NotImplementedError`).

### 2c. Der NVFP4-FA2-Pfad selbst (sm120/121)
- `vllm/v1/attention/backends/flashinfer.py:815-835` — `is_kvcache_nvfp4` +
  `use_fa2_nvfp4_kv = is_device_capability_family(120)`; sm100 → trtllm-gen, sonst Fehler.
- `flashinfer.py:995-1005` — Prefill-Wrapper `BatchPrefillWithPagedKVCacheWrapper(..., backend="fa2")`
  für den NVFP4-KV-Fall; `flashinfer.py:1024-1042` analog Decode.
- `flashinfer.py:1145-1146` — für FA2-NVFP4 wird `prefill_use_trtllm=False` erzwungen (also der
  native FA2-Prefill-Pfad, der prinzipiell Masken kennt).
- V-Scale-Deswizzle-Flag: `flashinfer.py:82, 273-278, 881-886`
  (`-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`).
- NVFP4-Page-Geometrie: `vllm/utils/torch_utils.py:415 nvfp4_kv_cache_full_dim`,
  `:472 nvfp4_kv_cache_split_views`.

### 2d. Skip-Layer-Ausweg (der tote Pfad)
- `vllm/config/cache.py:115 kv_cache_dtype_skip_layers`,
  `vllm/model_executor/layers/attention/attention.py:253-265` (Layer → native dtype),
  `vllm/platforms/interface.py:596`. Diese Layer bekommen eine **andere Page-Geometrie** →
  Padding auf die größere Page-Breite → der genannte ~41 GiB-Blowup @32k.

### 2e. DFlash2-Delta (Upstream, OPEN, im Fork gecherrypickt)
- vLLM **#52816** (SubSir): `DFlash2DraftModel` = DFlash + grouped dynamic depthwise conv +
  Candidate-Selector; V2-Runner: `model_executor/models/qwen3_dflash2.py`,
  `v1/worker/gpu/spec_decode/dflash2/speculator.py`, `.../__init__.py`, `registry.py`.
- vLLM **#52883** (oceanplexian, stacked auf #52816): LM-Head-Guard-Fix (unquantized head bei
  quantisiertem Body).
- **Wichtig:** Conv + Selector sind Zusatz-Sublayer; die **Attention-Mask-Semantik** erbt DFlash2
  unverändert vom DFlash-Proposer (§3). Für dieses Kernel-Problem ist DFlash == DFlash2.

---

## 3. Required mask semantics (präzise)

Aus `vllm/v1/spec_decode/dflash.py`:
- **Geometrie:** `num_query_per_req = 1 + num_speculative_tokens` (`dflash.py:104`). Bonus-Token +
  `num_speculative_tokens` Mask/Proposal-Tokens sind die **einzigen Queries**; der gesamte
  Kontext (akzeptierter Prefix) ist **K/V** und wird per `precompute_and_store_context_kv` vorab in
  den Cache geschrieben (`dflash.py:238, 265-269`). Bei `num_speculative_tokens=15` → **16 Queries/Req**.
- **Mask:** `new_cad.causal = self.dflash_causal` (Default **False**, `dflash.py:71, 189`) und
  Draft-Config `use_non_causal = not dflash_causal` (`dflash.py:80`). Der Proposer **assertet**
  `causal is False` für alle Draft-Layer (`dflash.py:286-293`).
- **Effektiv:** Voll-non-causal über `[Prefix | Block]`. Da der Block **nach** dem gesamten
  Prefix angehängt wird, ist "voll-non-causal" **identisch** zur Auftrags-Formulierung
  *"kausal zum Prefix + bidirektional im Draft-Block"*:
  jede der 16 Query-Positionen sieht (a) **alle** Prefix-Spalten (rechteckige Voll-Sicht) und
  (b) **jede** andere Block-Position (nicht-trianguläre `16×16`-Submatrix). Kein Sliding-Window,
  kein Soft-Cap-Zwang.
- **Konsequenz für die Kernel-API:** Der reine Block-Diffusion-Fall braucht **kein** bespoke
  `custom_mask` — eine FA2-Paged-Append-Prefill-`plan(causal=False)` liefert genau diese
  Semantik (Query = die letzten `qo_len` Positionen der KV-Sequenz, non-causal ⇒ jede Query sieht
  jede KV-Position). Ein **packed custom_mask** wäre nur nötig, falls DFlash2 später eine
  strengere Intra-Block-Struktur (z.B. Tree-/Kausal-Sub-Maske) einführt.

---

## 4. Lösungsarchitekturen A / B / C

### Option A — Mask-Mode-Erweiterung der bestehenden NVFP4-FA2-Kernelfamilie in flashinfer
**Idee:** Den FA2-NVFP4-Paged-Prefill (#3684, `include/flashinfer/attention/prefill.cuh`) für
`mask_mode = kNone` (non-causal) bzw. `kCustom` über NVFP4-Pages nutzbar machen; vLLM so verdrahten,
dass der Drafter-Prefill mit `causal=False` (statt hartem `causal=True`) plant.

**Evidenz pro:**
- #3684 fasst **genau diesen Kernel** an (prefill.cuh +99/-41, scheduler.cuh, page.cuh,
  `flashinfer/prefill.py` +132/-19) und enthält bereits einen **Custom-Mask-Fix**:
  *"plan(): move `mask_indptr` to the custom mask's device before `segment_packbits`"* — d.h.
  die **Masken-Plumbing des Paged/Ragged-Prefill-Wrappers ist vorhanden und wurde sogar repariert**.
- Der FA2-Prefill ist der generische FA2-Pfad, der `custom_mask`/`packed_custom_mask` prinzipiell
  kennt (im fp16/fp8-Fall Standard).

**Evidenz contra / offene Punkte:**
- #3684 validiert numerisch nur **causal** ("multi-tile seq 512/1024 + causal all sane") und
  symmetrische Shapes; **non-causal / custom_mask über den NVFP4-V-Load-Pfad ist nicht
  nachgewiesen**. Risiko: der FP4-V-Load + `[all-data|all-SF]`-SF-Layout + asymmetrische Offsets
  interagieren mit Mask-Iteration/Tile-Skipping.
- **Split-KV ist für NVFP4 hart abgeschaltet** (#3684-Zusatzfix: mid-block-Split verletzt die
  16-Element-FP8-Block-Alignment → Retrieval-Cliff). Drafter-Verify ist **exakt** die
  Extend-Geometrie (`qo_len=16 ≪ kv_len`), die den Scheduler sonst in Split-KV treibt → der Pfad
  läuft **single-split** (korrekt, aber ohne Flash-Decoding-Parallelismus). Perf-Fußnote, kein Blocker.
- Wo Kausalität hart sitzt: `CTA_TILE_Q`/Schedule-Dispatch (`scheduler.cuh`) + die Tile-Iteration
  in `prefill.cuh` (mask-abhängiges KV-Tile-Skipping). Bei non-causal entfällt das Skipping, d.h.
  potentiell **mehr** KV-Tiles pro Query-Tile — bei 16 Queries vernachlässigbar.
- vLLM-Seite: Guard #1 (`supports_non_causal` Override für FlashInfer, gated auf `use_fa2_nvfp4_kv`)
  **und** Guard #2 (`causal` aus `common_attn_metadata` statt hart `True`, + optional custom_mask-
  Plumbing durch `plan()`).

**Scope:** **mittel–groß.** flashinfer-Kernel-Verifikation/-Anpassung (unbekannte Tiefe) + zwei
vLLM-Guards + Metadaten-Plumbing. Perf-Erwartung: **hoch** (nativer FA2, tensor-core-nah), sobald
non-causal korrekt.

### Option B — On-the-fly-Dequant-Shim (NVFP4-Pages → bf16/fp8 nur für Drafter-Attention)
**Idee:** Nur für die (wenigen) Drafter-Attention-Aufrufe die relevanten NVFP4-Prefix-Pages in
einen **Scratch-Buffer** nach bf16 (oder fp8) dequantisieren und den **bestehenden non-causal
FLASH_ATTN-Kernel** darauf laufen lassen. Der persistente KV-Pool bleibt durchgehend NVFP4 (kein
Page-Padding, kein zweiter Pool).

**Evidenz/Infra:**
- Dequant-Bausteine existieren bereits: `nvfp4_kv_cache_split_views` (`torch_utils.py:472`) und der
  Triton-Dequant `trtllm_prefill_attn_kvfp8_dequant` (`flashinfer.py:292-394`) als Vorlage für einen
  NVFP4→bf16-Scratch-Dequant.
- FLASH_ATTN kann non-causal (`flash_attn.py:113`) und bf16-KV (`:70-74`) — passt direkt.

**Memory-/Latenz-Abschätzung (Drafter, pro Layer, pro Req):**
- Zu dequantisierender Kontext ≈ `seq_len × num_kv_heads × head_size` bf16. Beispiel Gemma-Voll-
  Attn (HS=256, NKV klein): @32k, HS=256, NKV=2, bf16 = `32768·2·256·2 B ≈ 32 MiB/Layer/Req`.
  Als **einmaliger** Scratch (wiederverwendet über Layer/Req, nicht persistent) ist das eine kleine
  feste Reserve, **nicht** das ~41 GiB-Padding — der springende Vorteil gegen den Skip-Layer-Ausweg.
- Latenz: zusätzlicher Dequant-Read/Write des Prefix pro Drafter-Forward. Da der Drafter nur
  16 Queries hat, aber den **vollen** Prefix als K/V liest, ist der Dequant memory-bound ~ die
  Attention selbst — grob **~2× Bandbreite** des Drafter-Attention-Anteils. Drafter-Attention ist
  ein kleiner Bruchteil des Gesamt-Steps → moderater, planbarer Overhead; **keine** neuen MMA-Kernel.

**Scope:** **klein–mittel.** Ein Dequant-Shim + Scratch-Allocator + Drafter-Attention-Routing.
Perf-Erwartung: **moderat** (Bandbreiten-Overhead), **Korrektheit trivial verifizierbar**
(Dequant→SDPA ist die Referenz selbst). **Bester Correctness-first-Startpunkt.**

### Option C — Eigener non-causal NVFP4-KV-Kernel from scratch
**Nur falls A strukturell scheitert** (Kernel trägt non-causal nicht) **und B perf-technisch nicht
reicht.** **Wichtig:** ein solcher Kernel existiert bereits substanziell in InferenceDream —
`custom_kv_kernel.cu` (870 Zeilen): NVFP4-KV-Attention, Modi 0 (K+V NVFP4) / 1 (K fp8 / V NVFP4),
HS=128/256-Templates, Stream-Fix, **MTP-kompatibel**, **bit-exakte Parität** vs. Stock verifiziert.
Kausalität sitzt an **einer** Stelle: `custom_kv_kernel.cu:216`
`int num_keys = seq_lens[s_idx] - q_len + q_offset + 1;` (obere Key-Grenze pro Query = eigene Position).
Für Block-Diffusion: `num_keys` so, dass jede Query den **vollen** Prefix + **alle** Block-Positionen
sieht (Prefix-Länge + volle Block-Sicht statt `+ q_offset + 1`). Split-K-Variante (`:395ff`) analog.

**Scope:** **mittel** (nicht groß, weil Kernel + Split-K-Skelett + Parität schon da sind), aber
**Perf-Ceiling niedrig** (scalar software-dequant, **keine** Tensor-Cores — belegt in
`RESUME_NIGHT_KERNEL_OPT.md`). Für 16 Drafter-Queries über langen Prefix jedoch akzeptabel, da
memory-bound. Perf-Erwartung: **niedrig–moderat**; Vorteil = volle Kontrolle + vorhandene Parität.

---

## 5. Recommended path

**Zweistufig: B zuerst (Correctness-Gate + sofortiger Nutzen), A als Perf-Ziel parallel sondieren.**

1. **B als Correctness-first-Prototyp** — löst den 41-GiB-Blowup sofort (durchgehend NVFP4-Pool),
   Korrektheit ist gegen Dequant-SDPA trivial verifizierbar, keine neuen MMA-Kernel, kleiner Scope.
   Liefert das erste "Spekulation + 4-bit-KV auf einer Karte"-Ergebnis.
2. **A als Perf- Upgrade** — parallel eine **Mikro-Probe** (kein vLLM nötig): den flashinfer-FA2-
   NVFP4-Paged-Prefill direkt mit `mask_mode=kNone`/`custom_mask` gegen eine bf16-Referenz testen.
   Trägt der Kernel non-causal korrekt, ist A der Endzustand (nativer Speed) und B fällt als
   Fallback zurück. Trägt er es nicht → A = Kernel-Patch in prefill.cuh, oder C.
3. **C nur** wenn A-Probe rot **und** B-Perf unzureichend — mit dem InferenceDream-Kernel als Basis.

---

## 6. Phased plan (Correctness-Gate vor Perf-Phase)

**Phase 0 — Referenz & Fixtures (kein GPU nötig).**
- `kdev.py`-Oracle (InferenceDream) um einen **non-causal / Block-Diffusion-Referenzpfad** erweitern:
  `ref_attention(..., causal=False)` bzw. eine `[Prefix voll | Block voll]`-Maske; Gate **cos > 0.99**
  gegen Dequant-KV-SDPA, Batch-Fall mit variablen `seq_lens`/`q_lens=16` (`kdev.py:139-166`).
- DFlash2-Mask-Fixture: `qo_len=16`, langer Prefix, NVFP4-Pages, gegen bf16-Ground-Truth.

**Phase 1 — Correctness-Prototyp (Option B).**
- Dequant-Shim NVFP4→bf16-Scratch (Vorlage `flashinfer.py:292-394`), Drafter-Attention auf
  FLASH_ATTN non-causal. **Gate:** bit-nahe Parität (cos > 0.99) drafter-output vs. voll-bf16-Drafter,
  Acceptance-Rate unverändert vs. fp8/bf16-Baseline. **Erst nach grünem Gate** → Perf.

**Phase 1' (parallel) — A-Machbarkeitsprobe.**
- Standalone flashinfer-Test: FA2-NVFP4-Paged-Prefill mit non-causal/custom_mask, `max_abs_err`
  gegen bf16 (analog #3684-Validierung `~0.0047`). Rot/Grün entscheidet A vs. C.

**Phase 2 — Perf & Integration.**
- Bei grünem A: vLLM Guard #1 (`supports_non_causal` Override gated auf `use_fa2_nvfp4_kv`) +
  Guard #2 (`causal` aus Metadata, optional custom_mask-Plumbing durch `plan()`); E2E gegen bf16-
  Facts (Retrieval-Needle wie #3684, radix/prefix-cache **on**).
- Bei B als Endzustand: Scratch-Allocator-Tuning, Overhead-Messung, ggf. Dequant-Fusion.

**Risiken (größte zuerst):**
- **A:** FA2-NVFP4-Kernel trägt non-causal/custom_mask evtl. nicht korrekt (FP4-V-Load × Mask-Tile-
  Iteration ungetestet) → Fallback B/C. **Split-KV-Gate** macht Drafter-Verify single-split (Perf-Fußnote).
- **B:** Dequant-Overhead pro Drafter-Forward (memory-bound, ~2×); Scratch-Lebenszeit/CUDA-Graph-
  Stabilität (Buffer-Adressen stabil halten — vgl. DFlash-Context-Buffer-Muster `dflash.py:42-62`).
- **Fork-Drift:** DFlash2 (#52816/#52883) ist **OPEN** upstream; Mask-Semantik kann sich mit
  Selector/Conv noch ändern (aktuell erbt sie unverändert von DFlash).
- **Timing-Wette:** Upstream-Issue erst nach Prototyp-Entscheid posten (Auftrag).
