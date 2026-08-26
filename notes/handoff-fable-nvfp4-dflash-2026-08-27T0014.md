# Handoff → Fable · NVFP4-KV × DFlash non-causal · der PR-Sprint

**Erstellt:** 2026-08-27T00:14 (+0200) · **Von:** Session-Agent (Opus 4.8) für **Tim** (@TechPrototyper)
**Für:** **Fable** — übernimmt federführend, hat **heute Nacht (2026-08-27) freien Zugriff auf BEIDE GPUs** (RTX 5090 / sm120 und GB10 / sm121).
**Modus:** ⚡ **OFFENSIVE / Sprint.** Devise: *Let's rock'n'roll, wir sprinten bis zum Ende durch. Diesen PR fahren wir ein.*

> **Lesehaltung — das Wichtigste zuerst.** Dieses Dokument ist ein **Anker und Analysestart, kein Analyseende.**
> Tim: *"Er soll auf keinen Fall blind auf Einschätzungen vertrauen, sondern sie als Anker und aktuellen Ausgangspunkt betrachten."*
> Deshalb ist jede Aussage getaggt:
> **✅ VERIFIED** = im Repo/`gh`/Result-JSON belegt (mit Fundstelle) · **🧭 EINSCHÄTZUNG** = meine Interpretation, mit „so widerlegst du es" · **❓ OFFEN** = ungeklärt.
> Wenn etwas riecht: **verifiziere neu** (Cheatsheet §13). Vertraue nichts blind — auch nicht mir.

---

## §0 · Lage in einem Absatz

Unsere GitHub-Beteiligungen bei `vllm-project/vllm` sind in einer Welle explodiert (jschmied, Tejas, hcl, **jethac**). Das ist **eine** Welle über **zwei orthogonale Linien**: (A) **NVFP4-KV-Cache** auf Consumer/SoC-Blackwell und (B) **DFlash/DFlash2 Block-Diffusion-Speculative-Decoding**. jethac ist **unser Verbündeter** und treibt die NVFP4-Linie. Tim hat ihm vor ~5 Tagen privat eine Mail mit drei technischen Punkten geschickt; jethac hat **nicht mit Worten geantwortet, sondern mit zwei Commits** (ce2fece, 7a5cf14) — und uns die Interpretation überlassen. Genau an der **Naht zwischen (A) und (B)** — *non-causal Drafter-Attention über NVFP4-KV-Pages auf sm12x* — liegt unser Vorsprung: wir haben das auf dem 5090 für **DFlash-v1 schon e2e zum Laufen gebracht** (>100 tok/s), während es für **DFlash2 (Produktionsmodell) noch am Base-FlashInfer-Wrapper blockt.** Das ist der Hebel für den eigenen PR.

---

## §1 · Cast (wer ist wer)

| Handle | Rolle | Relevanz für uns |
|---|---|---|
| **jethac** | Unser Verbündeter. Designer & Treiber der NVFP4-KV-Linie. | flashinfer#3684 (VO-split NVFP4 paged-prefill, **MERGED** `8f9ad200`) + **vLLM#46329** (Serving-Seite, **OPEN**). Hat Tim bei #46329 gedankt (`d671b2530`) und auf die Mail mit Code geantwortet (§4). |
| **jschmied** | DFlash/DFlash2-Spec-Decode-Linie. | Orthogonal zu jethac (Beweis §5). Alternativer Support-Kandidat, falls NVFP4-Weg stockt. |
| **Tejas / hcl** | Weitere Beteiligte der Welle. | Peripher; kein Konflikt mit unserer Linie identifiziert. Bei Bedarf nachverifizieren. |
| **Rob Tand** (GitHub `RobTand`, HF `rdtand`) | PrismaQuant-Ökosystem. **Name ist „Rob Tand", nicht „Rob Tandler".** | AURA / AQUA / **GridBook** (§8). Liefert unsere Produktionsquants. |
| **Tim** (@TechPrototyper) | Lab-Owner. | Zwei GPUs, Produktions-Fleet, alle Messungen. |

---

## §2 · Die zwei Linien und die Naht

**Linie A — NVFP4-KV auf sm12x (jethac):**
- flashinfer#3684 = asymmetrischer **VO-split NVFP4 paged-prefill**-Kernel (FA2), **MERGED 2026-08-13** als `8f9ad200`. Damit ist die **Kernel-Seite upstream**.
- vLLM#46329 = die **Serving-Seite** dieser Linie, **noch OPEN**, wartet(e) auf Rebase (Konflikte seit 08-07). ✅ Tim ist dort bereits **credited contributor** (Produktionsdatenpunkt + 5090/GB10-Validierung, siehe `notes/upstream-contributions.md`).

**Linie B — DFlash/DFlash2 Speculative Decoding (jschmied et al.):**
- DFlash **v1** support ist bereits upstream (inkl. Blackwell-Fixes vllm#48167/#50065).
- DFlash**2** (Drafter für aktuelle Modelle) lebt in offenen PRs **vLLM#52816** (conv + candidate selector) + **#52883** (LM-Head-Bugfix). Bei uns cherry-gepickt (10 exklusive Commits, nicht gemerged).

**Die Naht (unser Gebiet):** Der DFlash-Verify braucht **non-causal** Attention (ein `1+num_speculative_tokens`-Query-Block sieht den vollen Prefix **und** ist blockintern bidirektional). Der NVFP4-FA2-Pfad aus #3684 ist in vLLM effektiv **causal-only** verdrahtet. **Non-causal-Query-Block über NVFP4-KV-Pages existiert nirgends im Stack** → entweder fp8-KV (halbiert Kontext, falsch auf memory-bound 5090) oder der `--kv-cache-dtype-skip-layers`-Ausweg (Page-Padding-Blowup ~41 GiB @32k, tot). **Auflösung = Spekulation UND 4-bit-KV gleichzeitig auf jeder speichergebundenen Karte.** Das ist der Payoff.

Vollständige Code-Archäologie (Dateien+Zeilen, Guards, Optionen A/B/C): **`notes/nvfp4-noncausal-kernel-recon.md`** — Pflichtlektüre, das technische Herz.

---

## §3 · Tims Mail an jethac (vor ~5 Tagen) — die drei Punkte

Kontext, den Tim mitgeschickt hat: DFlash-v1 (Qwen3.6-27B) und DFlash2 (Qwen3.8-27B) laufen bei uns, Draft-Length 7, mit belastbaren Proof-Points. Die **drei technischen Punkte**:

1. **sm12x-Decode soll auf dem FA2-Pfad bleiben, nicht auf die XQA/trtllm-gen-API wechseln** — XQA ist ~0.7–1.2 % langsamer und degradiert die MTP-Acceptance.
2. **VocabParallelEmbedding tp==1 OOB-Bug** — Spec-Decode-Padding-Token-ID `-1` wird ungemaskt in `F.embedding` gereicht (out-of-bounds).
3. **NVFP4-Dequant-Referenz vs. Store-Kernel-Scale-Layout** — die Dequant-Hilfe muss dem Scale-Layout des Store-Kernels entsprechen (K linear geschrieben; V nur auf CC<12 swizzled, auf sm12x linear).

> ❓ **OFFEN für Fable:** Der exakte Mail-Wortlaut liegt im Sitzungstranskript, nicht im Repo. Falls du ihn 1:1 brauchst: `/Users/timw/.claude/projects/-Users-timw-Projects-local-ai-lab/4191a373-12a2-45cd-b71a-486a340d1603.jsonl`. Die drei Punkte oben sind die verifizierte Substanz.

---

## §4 · jethacs Reaktion — zwei Commits, keine Worte

Tim bekam heute (08-26) diese zwei Commits in CC:

| Commit | Titel | Mappt auf Mail-Punkt |
|---|---|---|
| **`ce2fece`** | `[Attention] NVFP4 KV: keep sm12x decode on the FA2 path, not the XQA API` | → **Punkt 1** (direkt, wörtlich) |
| **`7a5cf14`** | `[Test][Kernel] NVFP4 KV: match dequant helper to the store kernel's scale layout` | → **Punkt 3** (direkt, wörtlich) |

🧭 **EINSCHÄTZUNG:** jethac hat **Punkt 1 und Punkt 3 direkt aufgegriffen** und in Code gegossen — das ist die Antwort. **Punkt 2 (Embedding-OOB) blieb unberührt** — steht als eigenständiges, kleines, sauberes Gap da.
*Frühere Geste:* jethac hat Tim bei #46329 bereits gedankt (`d671b2530`). Zusammengenommen: **jethac hat uns effektiv eingeladen.** Die Offensive baut darauf auf.
*So widerlegst du es:* `gh` die beiden Commits ziehen (§13), Diffs gegen die Mail-Punkte prüfen — decken sie wirklich 1 und 3? Ändert 7a5cf14 genau das V-linear/K-linear-Layout?

---

## §5 · Konflikt-Check jschmied ↔ jethac — Entwarnung

✅ **VERIFIED (Stand der Analyse):** **Null Datei-Overlap, null Thread-Overlap** zwischen jschmieds DFlash-Linie und jethacs NVFP4-Linie. Es gibt **keine Arbeit „gegen jethac"**. Beide Linien sind orthogonal; unsere Naht-Arbeit **verbindet** sie, statt Partei zu ergreifen.
*So widerlegst du es:* Datei-Listen beider PR-Sets schneiden (`gh pr diff --name-only`), Thread-Teilnehmer vergleichen.

---

## §6 · Das technische Herz — non-causal NVFP4, und der **korrigierte** Ist-Stand

> ⚠️ **Wichtigste Korrektur gegenüber älteren Notizen/Zusammenfassungen:** Die bequeme These *„Option A ist trivial grün, der Kernel trägt non-causal ohne Änderung"* ist **so nicht belegt.** Die Result-JSONs zeichnen ein **differenzierteres und ehrlicheres** Bild. Genau das ist Gold für den Sprint — lies §6 + §7 zusammen.

**Die drei Lösungsarchitekturen (aus `nvfp4-noncausal-kernel-recon.md`):**
- **Option A** — Mask-Mode-Erweiterung der bestehenden NVFP4-FA2-Kernelfamilie (`plan(causal=False)`/`custom_mask`). Perf-Ceiling **hoch** (nativer FA2), Machbarkeit über NVFP4-V-Load **ungetestet**.
- **Option B** — On-the-fly-Dequant-Shim (NVFP4-Pages → bf16-Scratch nur für Drafter-Attention) auf FLASH_ATTN. Scope **klein–mittel**, Korrektheit **trivial verifizierbar**, Perf **moderat** (bandbreiten-limitiert). „Bester Correctness-first-Startpunkt."
- **Option C** — Eigener non-causal NVFP4-Kernel (InferenceDream `custom_kv_kernel.cu`, 870 Z., bit-exakte Parität). Nur falls A scheitert **und** B perf-technisch nicht reicht. Perf-Ceiling **niedrig** (scalar dequant, keine Tensor-Cores).
- **Empfehlung der Recon:** **B zuerst (Correctness-Gate), A parallel als Perf-Ziel sondieren.**

**Was tatsächlich passiert ist (chronologisch, ✅ aus Result-JSONs):**

1. **08-20/21 · A-Microprobe (#31):** **RED — aber wegen Versions-Skew**, nicht wegen Kernel-Verdikt. Image `f4c27c0da` ist **älter** als der Shim-Branch; flashinfer 0.6.15 akzeptiert den String `'nvfp4'` als `kv_data_type` nicht (`AttributeError: torch has no attribute 'nvfp4'`). Der Shim-Triton-Gate war ebenfalls **BLOCKED** (Triton-CompilationError gegen 3.7.1, wieder Skew, **kein Mathe-Fehler**). → **Kein sauberes A-GO, aber auch kein A-NO-GO.** Der Kernel wurde nie fair getestet. Quelle: `RESULT_nvfp4_noncausal_microprobe.json`, `RESULT_nvfp4_shim_triton_gate.json`.

2. **08-21 · DFlash-**v1** + NVFP4-KV e2e (Qwen3.6-27B), Image `sm120-shim-6b86a309f`, env `VLLM_DFLASH_ALLOW_NONCAUSAL_NVFP4=1`:** ✅ **GRÜN.** Nach zwei Fixes (GDN `index_select`-OOB in der Spec-Plumbing **und** dem `dflash.py:335` causal-Attribut-Seam) läuft **DFlash-v1 + NVFP4-KV speculative decoding im Graph-Mode auf sm120** — **bricht die 100-tok/s-Marke:** count-to-200 **373.8 tok/s**, realistische Prosa **101.3 tok/s** (vs. no-spec-Baseline **67.4**). Quelle: `RESULT_e2e_dflash_nvfp4_spec.json` (headline) + `RESULT_e2e_dflash_nvfp4_verdict.json`.

3. **08-22 · DFlash**2** + NVFP4-KV auf sm120 (Qwen3.8-27B, **Produktionsmodell**), untracked/**neuestes** File:** ⚠️ **SPEC weiterhin BLOCKIERT.** Node-Safety + Warmup-OOM gefixt; KV-Headline erfasst (**mit** Spec 8.642 tok / 2.11×, **no-spec** 50.790 tok / 12.40× — der Spec-Pfad kostet ~5.9× KV: Eagle3-Aux-Layer + 7 Draft-Slots + hybrid GDN/mamba-State). Aber Spec-Engine-Init stirbt: **`NotImplementedError: FlashInfer non-causal attention is not supported with NVFP4 KV cache` (`flashinfer.py:_get_prefill_wrapper`).** Das File hält ausdrücklich fest: **„korrigiert den Gate-1-Record: ‚non-causal subsumed by base' ist FALSCH zur Laufzeit; Base-FlashInfer-NVFP4-KV hat keinen non-causal prefill wrapper."** Der no-spec-Fallback servierte das NVFP4-KV-Target **live** (Kohärenz PASS, Greedy-Determinismus PASS, ~55.6 tok/s). Quelle: `RESULT_e2e_dflash2_nvfp4_sm120.json`.

🧭 **Die entscheidende Lesart für den Sprint:** Der **Shim+Guard-Lift-Pfad ist auf DFlash-v1 bewiesen** (>100 tok/s, e2e, sm120) — aber (a) er steckt im v1-Image `sm120-shim-6b86a309f`, **nicht** im DFlash2-Image, weshalb DFlash2 gegen den **nackten** Base-Wrapper läuft und die `NotImplementedError` wirft; und (b) **native Option-A-Parität (fa2-NVFP4 non-causal vs bf16 auf echten vLLM-Pages) ist weiterhin unbewiesen** (Microprobe nur skew-rot). **Der konkrete Sprint-Zielpunkt:** den bewiesenen Guard-Lift/Shim in die DFlash2-Linie ziehen **und/oder** die Option-A-Microprobe skew-frei wiederholen. Beides mündet **direkt** in jethacs `ce2fece` (FA2-Pfad halten) + `7a5cf14` (Dequant-Scale-Layout).
*So widerlegst du es:* die drei JSONs selbst lesen (§13), Image-Tags/Branch-SHAs prüfen — läuft das DFlash2-Image wirklich ohne den `6b86a309f`-Shim? Ist `VLLM_DFLASH_ALLOW_NONCAUSAL_NVFP4` dort gesetzt?

---

## §7 · Harte Evidenz — Result-Files (ehrlich getaggt: Screening vs. Verdict)

| Datei | Was | Ergebnis | Tier |
|---|---|---|---|
| `RESULT_nvfp4_noncausal_microprobe.json` | Option-A-Microprobe (kernel trägt non-causal?) | **RED (Versions-Skew, kein Verdikt)** | Recon-Spike |
| `RESULT_nvfp4_shim_triton_gate.json` | Shim-Triton-Kernel-Gate | **BLOCKED (Skew, kein Mathe-Fehler)** | Recon-Spike |
| `RESULT_e2e_dflash_nvfp4_spec.json` | DFlash-v1 + NVFP4 e2e Spec | **GRÜN, >100 tok/s (373.8 / 101.3)** | e2e-Praxis |
| `RESULT_e2e_dflash_nvfp4_verdict.json` | DFlash-v1 A-Praxis-Verdikt + Fit | GDN-index-OOB gefixt; **Option-A-Parität offen** | e2e-Praxis |
| `RESULT_e2e_dflash2_nvfp4_sm120.json` ⭐ **untracked/neuestes** | DFlash2 + NVFP4 sm120 | **Spec BLOCKED (non-causal NVFP4); no-spec live PASS** | e2e-Praxis |
| `RESULT_dflash2_n250_base_nvfp4.json` | Paired GSM n=250 Baseline (GB10) | Baseline-Arm, Determinismus PASS | Screening (±2.7 pp) |

**Method-Disziplin des Labs (einhalten!):** paired McNemar exakt; **Screening n=250 (±2.7 pp) ≠ Verdikt n=1319**; „measurably excellent". Spekulation ist unter Greedy **qualitätsneutral** (Drafter beeinflusst nur Acceptance/Speed, nie die Output-Distribution) — Quality-Gates messen also v.a. Regressionen im Target, nicht im Drafter.

---

## §8 · Rob Tands GridBook & das PrismaQuant-Ökosystem

**PrismaQuant (Rob Tand):** AURA (KL-Fisher mixed-precision), **AQUA/PrismaAQUA** (NVFP4-Weights + fp8-Attention, ~5.5 bpp, ~24 GB), **GridBook** (FP8-CB Produkt-Codebook, **13 GB**).

**GridBook-Befund (✅ `notes/gridbook-13gb-quality-holds.md`, GB10/sm121, 2026-08-23):** GridBook 13 GB **hält Qualität gegen** unsere 24-GB-Produktion (AQUA) — task-äquivalent auf **drei** Achsen: GSM8K n=250 **96.0 % vs 96.4 %** (McNemar exakt **p=1.0**), Tool-Calling PASS=PASS, Needle @~32k gleich. Der intrinsische +4.56 % PPL-Verlust (den Rob selbst publiziert hat: KL 0.0917, WikiText-2 9.792 vs 9.365) **schlägt auf Task-Ebene nicht durch.**
**Warum es zählt:** 13 statt 24 GB ≈ **verdoppelter KV-Headroom auf einer 32-GB-Karte** — realer Schritt für den 5090.
**Ehrliche Grenzen:** n=250 ist Triage, nicht Verdikt; gemessen auf **GB10**. Der **RTX-Payoff — läuft GridBook auf sm120, und komponiert es mit NVFP4-KV** für ~1M-Token-Kontext auf 32 GB — ist **separat, nicht abgedeckt.** 🧭 Das ist ein **zweiter potenzieller Sprint-Hebel**, falls die non-causal-Naht länger braucht.
Serviert via out-of-tree-Plugin + One-File-Cherry-pick (dev693 `qwen3_5.py`, strippt den VL-Text-Tower-Prefix), auf dediziertem Tree — berührt Produktion nie.

---

## §9 · NVFP4-Experimente auf RTX (sm120) & DGX (sm121) — was läuft, was publiziert ist

**Produktion RTX 5090 (sm120, 32 GB), ✅ `nodes/rtx-5090.md`:** derselbe 27B-NVFP4-Quant wie GB10. **NVFP4-KV** macht 27B mit Langkontext auf 32 GB überhaupt möglich (~469k Token, ~4× vs fp8). ~72 tok/s single-stream. Spec-Decode **aus** (weil DFlash2 fp8-KV erzwingt = falscher Trade auf memory-bound; skip-layers = 41-GiB-Blowup). **Was für diese Karte bleibt: genau die upstream NVFP4-non-causal-Kernel-Arbeit dieses Sprints.**

**Produktion GB10 (sm121, 128 GB unified), ✅ `nodes/dgx-spark-gb10.md`:** Qwen3.8-27B AQUA, **NVFP4-Weights + fp8-KV + DFlash2-Spec (draft length 7)** seit 08-21 in Produktion — ~4× single-stream, bis ~227 tok/s aggregat, qualitäts-verdict-gleich. Seit 08-22 ist der **Drafter selbst fp8-quantisiert** → −1.6 GB zurück in den Pool: **21.6 GiB = 478k Token**, ~45 tok/s single-stream. Bandbreite (~273 GB/s) ist der definierende Constraint → Box ist auf **Langkontext + viele parallele Sessions** ausgelegt.

**Weitere NVFP4-Experimente (✅ Repo):**
- `notes/laguna-nvfp4-vs-int4.md` — NVFP4 vs INT4-Marlin auf 117B-MoE: **kein Accuracy-Penalty** auf Screening-Level (HumanEval 96.7 % vs 95.0 %, n=120), Richtung eher pro NVFP4.
- `notes/dflash2-sm121-first-light.md` — DFlash2 first light auf GB10 (2.2×, Tool-Calling bleibt sauber; „speculation breaks tool-calling" war nie über Spekulation — es waren ngram-Parser-Korruption + still-gedroppte Sampling-Params).
- Kalibrierungs-Studie, Hadamard-Probe, TurboQuant-KV, byteorder/blockdiff-Oracles — siehe `notes/upstream-contributions.md`.

**Publiziert vs. nicht:**
- ✅ **Publiziert:** flashinfer#3684-Validierung; vLLM#46329-Produktionsdatenpunkt; #50288 GDN; llm-compressor#2936 Kalibrierung; #52816 fused-scale-Gap; #53334 TurboQuant; PrismaQuant PR#80. HF: `rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm`, Tims **fp8-Drafter**, portabler GB10-DFlash2-Container, vllm-sm12x-Container-Repo. **Alles in `notes/upstream-contributions.md` als neutraler Record.**
- ❌ **Nicht (noch) publiziert:** die non-causal-e2e-Durchbrüche auf sm120 (DFlash-v1 >100 tok/s), der Shim/Guard-Lift, die DFlash2-sm120-Blocker-Analyse. **Das ist der ungehobene PR-Stoff.**

---

## §10 · Einschätzungen (Challenge-Bar — jede mit „so widerlegst du sie")

- **[E1] jethac hat eingeladen.** Zwei Commits = Antwort auf Mail-Punkt 1+3; frühere #46329-Danksagung. → *Widerlegen:* Diffs vs. Mail-Punkte prüfen; sind es wirklich unsere Themen oder Zufall?
- **[E2] Der eigene PR liegt an der Naht, nicht im Kernel.** Kernel (#3684) ist merged; das offene Stück ist die **vLLM-Serving-Verdrahtung** (Guards + causal-Plumbing) + der non-causal-Pfad für DFlash2. → *Widerlegen:* zeigt die skew-freie Option-A-Microprobe, dass der Kernel non-causal **doch nicht** trägt? Dann verschiebt sich der PR Richtung Kernel-Patch/Option C.
- **[E3] „Weg B" (reshape_and_cache_flash-NaN als eigenes Gap) ist tot.** jethacs `7a5cf14` war ein **Dequant-Referenz**-Fix, kein Kernel-Bug — er hat es adressiert. → *Widerlegen:* `7a5cf14`-Diff lesen; bleibt ein NaN-Pfad im Store-Kernel offen?
- **[E4] Punkt 2 (Embedding-OOB) ist ein sauberes, kleines, unberührtes Gap** — Kandidat für einen eigenständigen Mini-PR, falls die Naht zäh wird. → *Widerlegen:* prüfen, ob jemand (Tejas/hcl?) das inzwischen angefasst hat.

---

## §11 · Offene Fragen für Fable

1. Läuft das **DFlash2-Image** wirklich **ohne** den `6b86a309f`-Shim/Guard-Lift? (Wenn ja: den Shim reinziehen ist der kürzeste Weg zu DFlash2+NVFP4 auf sm120.)
2. Trägt der **fa2-NVFP4-Kernel non-causal** wirklich (skew-freie Microprobe, `max_abs_err` vs bf16 ~ #3684-Niveau ≈ 0.0047)? Grün → Option A = Endzustand; Rot → B/C.
3. Deckt sich unser Shim-Ansatz mit jethacs `7a5cf14`-Scale-Layout, oder müssen wir uns daran angleichen?
4. Ist Punkt-2-OOB upstream noch offen?
5. Welcher Hebel bringt den **klarsten, kleinsten, review-baren PR** — DFlash2×NVFP4-Serving-Lift, oder Embedding-OOB, oder GridBook×NVFP4-KV auf sm120?

---

## §12 · Nachtsprint-Runbook (beide GPUs frei)

> Sicherheits-/Prozess-Leitplanken aus den Result-Files ernst nehmen: **prod `vllm-qwen38`/`vllm-aura38` zuerst auf 0/0** (nie GPU-Co-Tenant); Node-MemAvailable watchen (Abort-Gate <15 GiB); RAM-Limit-Anhebung auf control-plane-Node **nur mit Tim-OK** (INFOBOARD-Regel); PVC-AOT-Cache wiederverwenden; nichts committen ohne Gate.

**Phase 0 — Rebase & Skew beseitigen.** vLLM#46329 auf aktuellen main rebasen (war Konflikt-blockiert). Ein **skew-freies Image** bauen, in dem flashinfer-Version, Shim-Branch (`6b86a309f`) und DFlash2-Cherry-picks (`#52816/#52883`) **konsistent** sind — der dominierende Fehler der 08-20-Microprobe war reiner Versions-Skew.

**Phase 1 — RTX 5090 / sm120 · Regressions-Anker (bekannt-grün reproduzieren).** DFlash-**v1** + NVFP4-KV e2e mit `VLLM_DFLASH_ALLOW_NONCAUSAL_NVFP4=1` **erneut grün fahren** (Ziel: count-200 ~373 / Prosa >100 tok/s reproduzieren). Das ist der stabile Boden.

**Phase 2 — RTX 5090 / sm120 · der Zielpunkt ⭐.** DFlash**2** + NVFP4-KV: den bewiesenen Guard-Lift/Shim in die DFlash2-Linie ziehen, den `NotImplementedError`-Wrapper-Block auflösen. **Parallel:** skew-freie **Option-A-Microprobe** (fa2-NVFP4 non-causal vs bf16-Parität). Rot/Grün entscheidet A vs. B als Endzustand. **Gate:** Kohärenz + Greedy-Determinismus + Acceptance unverändert.

**Phase 3 — GB10 / sm121 · frischer Datenpunkt & Cross-Arch-Bestätigung.** Denselben Fix auf sm121 bestätigen (Produktions-Arch für DFlash2). Optional: **GridBook × NVFP4-KV auf sm120** anprobieren (der explizit offene RTX-Payoff aus §8).

**Phase 4 — Landing.** Kleinste review-bare Einheit schnüren: Serving-Guard-Lift (Guard #1 `supports_non_causal`-Override gated auf `use_fa2_nvfp4_kv` + Guard #2 `causal` aus Metadata statt hart `True`) mit paired Quality-Gate + tok/s-Fakten. An jethacs `ce2fece`/`7a5cf14` andocken (FA2-Pfad, Scale-Layout). Ehrliche Method-Tags (Screening vs Verdict). **Diesen PR fahren wir ein.**

---

## §13 · Re-Verifikations-Cheatsheet (nichts blind glauben)

```bash
# jethacs zwei Commits + Diffs gegen Mail-Punkte 1/3
gh api repos/vllm-project/vllm/commits/ce2fece --jq '.commit.message'
gh api repos/vllm-project/vllm/commits/7a5cf14 --jq '.commit.message'
gh pr view 46329 --repo vllm-project/vllm --json state,title,mergeable,reviews
gh pr view 46329 --repo vllm-project/vllm --json commits --jq '.commits[].oid' | grep d671b25   # frühere Danksagung

# Konflikt-Check jschmied ↔ jethac (Datei-Overlap)
gh pr diff 52816 --repo vllm-project/vllm --name-only > /tmp/dflash.txt
gh pr diff 46329 --repo vllm-project/vllm --name-only > /tmp/nvfp4.txt
comm -12 <(sort /tmp/dflash.txt) <(sort /tmp/nvfp4.txt)   # leer = kein Overlap

# Der korrigierende Ist-Stand — die drei JSONs SELBST lesen
python3 -c "import json;print(json.load(open('results/RESULT_e2e_dflash2_nvfp4_sm120.json'))['status'][:600])"
python3 -c "import json;print(json.load(open('results/RESULT_e2e_dflash_nvfp4_spec.json'))['headline'])"
python3 -c "import json;print(json.load(open('results/RESULT_nvfp4_noncausal_microprobe.json'))['attempt_1_checkout_script_as_written'])"

# Die Guards im Code (Recon §2 nachprüfen)
grep -n "supports_non_causal" vllm/v1/attention/backends/flashinfer.py vllm/v1/attention/backend.py
grep -n "causal=True" vllm/v1/attention/backends/flashinfer.py
```

---

## §14 · Datei- & Artefakt-Karte

**Pflichtlektüre (Repo):**
- `notes/nvfp4-noncausal-kernel-recon.md` — technisches Herz (Guards, Optionen A/B/C, Phased Plan, Dateien+Zeilen). jethac-Branch `jethac/spark/hijinks-020-aeon-qwen-dflash-sm121a` HEAD `e2a8197a9`.
- `notes/dflash2-sm121-first-light.md` — DFlash2-Grundlagen, Tool-Calling-Mythos entlarvt.
- `notes/upstream-contributions.md` — neutraler Upstream-Record (was publiziert ist).
- `notes/gridbook-13gb-quality-holds.md` — Robs GridBook-Befund.
- `nodes/rtx-5090.md`, `nodes/dgx-spark-gb10.md` — die zwei GPUs, Configs, Constraints.
- `results/RESULT_e2e_dflash2_nvfp4_sm120.json` (⭐ untracked, neuestes), `..._e2e_dflash_nvfp4_spec.json`, `..._verdict.json`, `..._nvfp4_noncausal_microprobe.json`, `..._nvfp4_shim_triton_gate.json`.

**Pins & Tags (aus den Result-Files, verifizieren vor Nutzung):**
- flashinfer **0.6.15**, Branch `pr3684`, relevanter Fix-Commit `2ed09bd3`; Kernel MERGED `8f9ad200`.
- Shim-Image `sm120-shim-6b86a309f` (DFlash-v1 grün); Base-Image `sm120-nvfp4-f4c27c0da-t213` (skew, älter). GB10-Image `vllm-sm121:f4c27c0da`.
- Guard-Lift-Env: `VLLM_DFLASH_ALLOW_NONCAUSAL_NVFP4=1`.
- Modelle: Prod `qwen3.8-27b` (PrismaAQUA-5.5bit, DFlash2); v1-Linie `qwen3.6-27b` (PrismaAURA-5.5, DFlash-v1, hybrid GDN).

**Provenienz dieses Dokuments:** aufgebaut aus dem vollständigen Repo-Read (upstream-contributions, recon, gridbook, node-pages, laguna, dflash2-first-light) + den fünf Result-JSONs. Vorgänger im Scratchpad (`HANDOFF-fable-nvfp4-dflash.md`) ist damit **abgelöst**; dies ist die maßgebliche De-Luxe-Version im Repo.

— Ende. Rock'n'roll, Fable. 🚀

---

## Addendum (Fable, 2026-08-27 ~01:45) — Korrekturen nach Re-Verifikation

Dieses Handoff war an mehreren Stellen hinter dem realen Stand (Quelle: Branch-Doku `kernel/dflash2-nvfp4-sm120-v2` §17/§18, MyCluster-Log, `gh`):

1. **§6/§7/§12-Phase-2 obsolet:** DFlash2+NVFP4-KV-Spec auf sm120 ist seit dem §18-Run **nicht mehr blockiert, sondern serviert e2e** (82.5 tok/s prose +48%, Acceptance 0.539, Determinismus byte-identisch). Fixes: `952ed2fe2` (non-causal FA2-NVFP4-Wrapper) + `96d4568e7` (Selector-Codebook-Clamp, ein **dritter** `-1`-Sentinel-Fundort). Siehe [`dflash2-nvfp4-sm120-spec-serves.md`](dflash2-nvfp4-sm120-spec-serves.md) und das aktualisierte RESULT-JSON.
2. **#52816 ist seit 2026-08-24 MERGED** (hier als offen geführt).
3. **GridBook×NVFP4-KV auf sm120 lief bereits** (877K-Kontext-Config, LiteLLM-Modell `gb`, zeitweise Primärroute; MyCluster 08-24). Rollback-Grund war **#52244** (Prefix-Cache × Spec × hybrid-GDN = 0% Hits), nicht Qualität oder Speicher.
4. **[E1] relativiert:** `ce2fece` credited @ssubbotin (Thread-Report 08-25), `7a5cf14` credited @hclsys/@seanyourhighness — die Commits antworten primär auf **öffentliche Thread-Reports**, nicht (nur) auf Tims Mail. Die Community arbeitet parallel an derselben Naht; das Zeitfenster ist real.
5. **[E3]-Detail:** `7a5cf14` pinnt das Layout **umgekehrt** zur Mail-Behauptung: auf sm12x K **und** V linear (V-Swizzle nur CC<12). Unser Unswizzle-Befund war Option-B-Shim-spezifisch (§17: „deswizzle_port_needed: False").
6. Jethacs Branch wurde force-gepusht (`e2446da20` → `7a5cf1431`); `_v3_on_jethac` ist stale und trägt den non-causal-Wrapper nicht — Rebase v4 nötig. #46329 ist gerebased und MERGEABLE.
