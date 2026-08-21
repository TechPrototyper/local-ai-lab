# NVFP4 non-causal kernel window — RTX 5090/neo26 (Kanban #31 + #30) — ENTWURF

**Status:** ENTWURF (nicht committen/pushen). Nacht 2026-08-20/21. Automations-Agent.
**Fenster:** RTX 5090 (neo26, von Tim freigegeben; RCA #28 GO mit Auflagen). Green path a–c.
**Companion:** [`nvfp4-noncausal-kernel-recon.md`](nvfp4-noncausal-kernel-recon.md),
`vllm-upstream-work@kernel/nvfp4-noncausal-shim-b:docs/kernel-initiative/{RUNBOOK,DESIGN-NOTES}`.

## TL;DR — A-Verdikt: **A-NO-GO → B** (Recon-Default bestätigt)

Das Fenster lief sauber und ohne Zwischenfall (keine RCA-Wiederholung). **Dominanter Befund:
das einzige verfügbare Runtime-Image (`f4c27c0da`) ist ÄLTER als der Shim-Branch**
(recon HEAD `e2a8197a9` / Shim HEAD `6b86a309f`) → durchgehende Versions-Skew, die (b)-Numerik
und (c)-Triton blockiert. Ein definitives a–d-Gate braucht ein **Image, gebaut auf Shim-Branch-HEAD**.

| Phase | Ergebnis | Gate |
|---|---|---|
| (a) Block-Diffusion-Orakel | **PASS** (cos 0.9948–0.9964, hs128+256) | grün |
| (b) A-Mikroprobe #31 | **A-NO-GO → B** (siehe unten) | — |
| (c) Shim-Triton-Parität | **OFFEN** (Triton-Skew CompilationError; Off-GPU-Mathe 14/14 grün) | — |
| (d) E2E-Smoke | **NICHT gelaufen** (bewusst ausgelassen) | offen |

## Phase-Ergebnisse

**(a) Orakel — PASS.** `KDEV_BLOCK_DIFFUSION=1`, self-contained torch auf RTX 5090.
dequant-vs-full cos: hs128 {256:.9964, 2048:.9956, 8192:.9948, 32768:.9958}, hs256 alle ≥.9948.
`non-causal!=causal(q0)` True @seq 256/2048, False @seq≥8192 — **benigne**: bei langem Prefix deckt
q0's Kausalfenster ~alle Keys ab → numerisch ununterscheidbar von non-causal (keine Maskenfehler).
Gate cos>0.99 **PASS** beidseitig. → Option-B-Referenzmathe + Block-Diffusion-Maske auf echter GPU bestätigt.

**(b) A-Mikroprobe #31 — A-NO-GO → B.** Die Kernfrage („trägt flashinfer FA2-NVFP4 non-causal?").
- *Attempt 1* (Checkout-Skript wie geschrieben): **RED**, `torch has no attribute 'nvfp4'` — flashinfer
  0.6.15 nimmt den String `kv_data_type="nvfp4"` nicht. API-Annahme des Skripts stimmt für diesen Build nicht.
- *Attempt 2* (vLLM-treu, direkt): korrekte API rekonstruiert aus dem Image-Backend —
  `kv_data_type=torch.uint8` + Data/Scale-Tupel-Views (`nvfp4_split_data_scale`) + `run(kv_cache_sf=…)`,
  **unter Umgehung des vLLM-Guards**. Split-Geometrie korrekt (k_data [129,16,2,64] uint8, k_scale
  [129,16,2,8] float8_e4m3fn). **Der fa2-Kernel akzeptiert `causal=False` OHNE Raise** (lief für beide
  durch) — aber Output **NaN für causal UND non-causal**: Fixture-Artefakt, `create_kv_caches_with_random_flash`
  schreibt zufällige fp8-Scale-Bytes (NaN-Muster) → zufällig initialisierter NVFP4-Cache ist nicht kernel-valide.
  **Numerische Korrektheit (cos/err vs bf16) daher UNBEWIESEN** — braucht echte vLLM-quantisierte Page.
- *vLLM-Wiring-Guard bestätigt:* Image-`flashinfer.py:1184-1187` **raist hart**
  `NotImplementedError("FlashInfer non-causal attention is not supported with NVFP4 KV cache")` — recon
  §2b Guard #2 als konkretes `raise`. Ein `_noncausal_prefill_wrapper` existiert, ist aber auf non-NVFP4-KV gelenkt.

  **Verdikt:** zwei unabhängige Blocker — (1) Kernel-Numerik über non-causal-FP4 unbewiesen (Random-Fixture-NaN);
  (2) vLLM verweigert non-causal+NVFP4 explizit (Guard-Lift = Option-A-Arbeit). Der Kernel-Entrypoint
  **verweigert non-causal nicht** (positives Signal), aber das reicht nicht für A-GO. **Recon-Default hält:
  auf B bleiben; A erst nach Real-Page-Numeriktest + Guard-Lift auf einem neu gebauten Shim-HEAD-Image.**

**(c) Shim-Triton — OFFEN.** Triton-Kernel (`dequant_nvfp4_kv_pages_to_bf16`) wirft **CompilationError**
gegen Image-`triton 3.7.1` (Shim für neueren Stack geschrieben; Versions-Skew, KEIN Mathe-Fehler).
Off-GPU-Mathe **14/14 grün** (DESIGN-NOTES). **Byte-Order-Real-Page-Check (Risiko #1) OFFEN** — braucht
echt vLLM-quantisierte NVFP4-Page (`flashinfer_quant_nvfp4_8x4_sf_layout`; `reshape_and_cache_nvfp4` fehlt
im Image). low-nibble=even wurde NICHT unabhängig gegen vLLMs echte Byte-Reihenfolge bestätigt — **Pflicht
vor Vertrauen in Drafter-KV**.

**(d) E2E-Smoke — bewusst ausgelassen (OFFEN).** Zwei Gründe: (i) DESIGN-NOTES §4 — die Live-Attention-
Dispatch-Verdrahtung ist NICHT gemacht (der flag-gated Seam `maybe_dequant_nvfp4_context` ist ein No-op);
ohne echte Integration ist Acceptance/Greedy-Parity kein valider Test. (ii) RCA #28 — Engine-Boot = exakt
die Profiling-Phase, die neo26 gewedged hat. Auflage „bei Zweifel d auslassen" → ausgelassen.

## Betriebs-/Sicherheits-Belege (RCA #28-Auflagen)

- **MemAvailable-Watch** via `/proc/meminfo` im Pod (VM-Ingress nicht erreichbar): **min 41.0 GiB** über
  den ganzen Lauf, Ende 44.4 GiB — **nie nahe der 15-GiB-Abbruchschwelle**. a–c ist winzig (kein Engine-Boot),
  exakt wie RCA §6a prognostiziert.
- **Hartes `resources.limits.memory=16Gi`** am Job (<60-GiB-Auflage; blast radius = Container, nicht Node —
  die #1-RCA-Mitigation).
- **Nie GPU-co-tenant:** eine physische RTX 5090, per time-slicing als 2 Slots (`RTX-5090-SHARED`). Prod
  `vllm-qwen38` zuerst auf 0 skaliert. Job korrekt mit `runtimeClassName: nvidia` (erster Versuch ohne →
  `torch.cuda=False`, korrigiert).
- **⚠ FALLE — Flux revert:** `vllm-qwen38` ist **Flux-managed** (`kustomize.toolkit.fluxcd.io/name: inference`).
  Der manuelle `patch replicas=0` wurde von Flux nach ~4 min auf 1 zurückgesetzt. **Kein Schaden** hier
  (mein Job war winzig und schon gelöscht, als Prods schwerer Boot ~14 s später startete; keine Mem-Cliff),
  aber für **längere Fenster (>~10 min) die Flux-Kustomization suspenden oder via Git skalieren**, sonst
  bringt Flux Prod mitten im Fenster hoch → potentieller Co-Tenant/Profiling-Wedge.
- **Restore verifiziert:** vllm-qwen38 1/1 ready, `/health` **HTTP 200**, Chat-Smoke „2+2"→`4`
  (finish_reason=stop). Alle Fenster-Ressourcen (Job/Pods/Preflight) entfernt. Nichts committet/gepusht.

## Board-Aktionen (zum Spiegeln — Board nicht scriptbar erreichbar)

Kein scriptbarer Kanban gefunden (kein Gitea-Ingress; local-ai-lab-Issues leer; `gh` nur GitHub/TechPrototyper).
**Nicht fabriziert.** Bitte manuell auf dem echten Board spiegeln:
- **#31:** Start-Kommentar `claim/rtx5090` + `status/doing` gesetzt (gedanklich); Ergebnis-Kommentar:
  „A-NO-GO → B. fa2-Kernel akzeptiert causal=False ohne Raise, aber (1) Numerik unbewiesen (Random-NVFP4=NaN,
  braucht Real-Page) + (2) vLLM-Guard raist non-causal+NVFP4 (flashinfer.py:1184-1187). Root cause: Image
  `f4c27c0da` < Shim-HEAD. Nächster Schritt: Shim-HEAD-Image bauen." → `status/blocked` (image-rebuild) oder
  `status/done` (Verdikt A-NO-GO erreicht) — Tims Wahl. `claim/rtx5090` **entfernt** (Fenster geräumt).
- **#30:** Ergebnis-Kommentar: „(a) Orakel PASS auf RTX5090. (c) Triton-GPU-Gate blockiert durch triton-3.7.1-Skew
  im Image; Off-GPU-Mathe 14/14 grün; Byte-Order-Real-Page-Check OFFEN (Risiko #1). Kein d (unwired + RCA)." → `status/todo` bleibt.

## Nächste Schritte (priorisiert)

1. **Image auf Shim-Branch-HEAD (`6b86a309f`) bauen** (`nvfp4_kv_cache_split_views`, Shim-Modul, triton-kompatibel).
   Das ist der Blocker für ein echtes a–d-Gate. Ohne das ist alles Weitere Skew-Kampf.
2. **#31 sauber schließen:** auf dem neuen Image den Direkt-Kernel-Test mit **echt vLLM-quantisierter Page**
   (`flashinfer_quant_nvfp4_8x4_sf_layout`) statt Random-Fixture → cos/err vs bf16 (Ziel wie #3684 ~0.0047).
   Erst dann ist A-GO/RED numerisch entschieden.
3. **(c) Byte-Order-Real-Page-Check** (Risiko #1) auf neuem Image — Pflicht vor Drafter-KV-Vertrauen.
4. **Betrieb:** GPU-Fenster-Runbook um „Flux-Kustomization `inference` suspenden vor Prod-Scale-Down" ergänzen.

---

# ZWEITES FENSTER — Nacht 20./21.08. (Build-Nacht #32→#31/#30) — ENTWURF-Ergänzung

**Auftrag:** Parity-Image auf Shim-HEAD bauen (#32), dann A-vs-B-Verdikt fällen (#31/#30).
**Betrieb:** Prod `vllm-qwen38` blieb OBEN (Build braucht keine GPU). Nichts auf GitHub gepusht.

## Kernbefunde (Kurzfassung)

1. **Byte-Order-Verdikt (Risiko #1) — OFF-GPU GESCHLOSSEN, kein GPU nötig.** Transitiver Beweis:
   vLLMs `dequant_nvfp4_kv_cache` ist in `test_cache.py` gegen den echten GPU-Writer
   `reshape_and_cache_nvfp4` per `assert_close` verifiziert → Shim gegen `dequant_nvfp4_kv_cache`
   auf identischen Bytes vergleichen = gegen echte vLLM-Pages vergleichen. Reines CPU-torch.
   - **fp4-Nibble + E2M1: KORREKT** (CONSTANT-scale cos 1.000000, maxerr 0, hs128+256). Shim
     `even=low nibble` == vLLM `break_fp4_bytes`.
   - **Block-Scale-Layout: FALSCH** (DISTINCT-scale cos ≈0.965, maxerr 6). vLLM speichert die
     Block-Scales **4×4-SWIZZLED** über die (block_size, scale_dim)-Ebene; der Shim liest sie
     **linear** (`scale_vec[elem//16]`). → **Shim erzeugt FALSCHE Drafter-KV auf echten Pages.**
     Die 14/14 Self-consistent-Tests waren grün, weil sie mit dem EIGENEN linearen Layout packten
     (self-consistent, nicht vLLM-treu) — exakt die RUNBOOK-(c)-Warnung.
   - **Fix off-GPU validiert:** 4×4-Unswizzle in den Scale-Read einziehen (Vorbild
     `dequant_nvfp4_kv_cache`) → korrigierte Referenz == vLLM exakt (cos 1.000000, maxerr 0).
   - Belege: `results/RESULT_nvfp4_byteorder_verdict.json`, `results/byteorder_cpu_proof.py`,
     Branch-Commit `c796230d2` (DESIGN-NOTES §7 + Proof-Skript).

2. **Finales A-vs-B-Verdikt (diese Nacht):** *Beide* Pfade sind auf echten Pages noch NICHT validiert.
   - **B (Shim):** hat einen **konkreten, fixbaren Korrektheits-Bug** (fehlender 4×4-Unswizzle) —
     kein Mathe-Fehler, aber „B ist grün" gilt NICHT bis Fix + GPU-Bestätigung. B-first-Reihenfolge
     bleibt richtig; B ist noch **kein** grünes Correctness-Gate.
   - **A (nativer fa2-non-causal):** Numerik auf echten Pages **weiter unbewiesen** — GPU-blockiert
     (s.u.). Kernel akzeptiert `causal=False` ohne Raise (Vorfenster), aber keine Zahlen.
   - **Netto: A-NO-GO→B hält, ABER B braucht zuerst den Unswizzle-Fix.** Der dringlichere Befund
     ist B's Real-Page-Bug, nicht A.

3. **Guard-Lift (Option-A-Seam) — committet `7c8f60bbf`.** `VLLM_DFLASH_ALLOW_NONCAUSAL_NVFP4`
   (default off, byte-identisch): auf dem `use_fa2_nvfp4_kv`-Pfad wird `causal` aus
   `common_attn_metadata.causal` gehonort (Drafter setzt False) statt hart `True`. Macht den
   E2E-Pfad testbar. NICHT im heutigen Image (das ist HEAD 6b86a309f, davor) → für künftiges (d).

## Phase 1 — BUILD (#32)

- **Wie das Parity-Image gebaut wird (rekonstruiert):** sm120/RTX5090 nutzt `scripts/build-vllm-nvfp4.sh`
  → Kaniko-Job in ns `registry`, nodeSelector neo26, FROM `nvidia/cuda:13.0.2-devel`, torch 2.13.0/cu130,
  `pip install .` (vLLM IN das Image kompiliert, TORCH_CUDA_ARCH_LIST=12.0), Push nach Harbor
  `10.1.0.243/inference/vllm-nvfp4:<tag>`. (sm121/Spark dagegen = Source-Mount, vLLM NICHT im Image.)
- **Shim-Quelle ohne GitHub-Push:** Shim-Branch ist local-only. Basis `e2a8197a9` (jethac, öffentlich)
  per SHA **gefetcht** (fetch ≠ push), Shim-Diff (54 KB, 7 Dateien) via **ConfigMap** `shim-patch-6b86a309f`
  im Kaniko-initContainer `git apply`. Nichts nach GitHub geschrieben.
- **WICHTIG — „Skew" war Lineage, nicht Datum:** Shim-Basis `e2a8197a9` (08.06., Spark-Lineage) ist
  ÄLTER als Prod `f4c27c0da` (30.07., sm120-Lineage). ABER e2a8197a9 HAT `reshape_and_cache_nvfp4` +
  `flashinfer_quant_nvfp4_8x4_sf_layout` — die das Prod-Image f4c27c0da **NICHT** hatte. **Genau das**
  ist der Build-Nutzen: er liefert die echten NVFP4-Quant/Cache-Ops für den Byte-Order- + A-Numerik-Test.
- **Triton bleibt 3.7.1:** torch 2.13.0 zieht triton==3.7.1 — derselbe wie im Vorfenster. Ein
  Rebuild mit torch 2.13 löst die Triton-Skew des (c)-Gates **nicht** (bräuchte riskanten torch-Bump).
  ABER: der Byte-Order-Check (CPU-Referenz) und der A-Numerik-Test (fa2-Kernel) sind
  **triton-unabhängig** — die Priorität-Items brauchen die Shim-Triton-Kernel gar nicht.
- **RCA-#28-Auflagen umgesetzt:** cgroup `limits.memory=32Gi` (bewusst < 48Gi-Ceiling, weil Prod OBEN
  bleibt und Node nur ~40–44 GiB frei hat — 48Gi würde den Node wedgen), `limits.cpu=12`
  (Node 24 → 12 für CP/etcd/Prod frei), `requests.cpu=200m` (Node ist 98% CPU-requested durch
  KubeVirt-VMs+Prod+CP), MAX_JOBS=4/NVCC_THREADS=2, kein Swap-Verlass. **MemAvailable-Watchdog**
  (VictoriaMetrics-Poll 45s, Abbruch <15 GiB) aktiv.

## GPU-Fenster (Phase 2) — BLOCKIERT (nicht durchgeführt)

**Blocker: Agent-RBAC.** Die Agent-SA (`agent-guardrails:cluster-agent`) darf **weder** die
Flux-Kustomization `inference` suspenden **noch** `vllm-qwen38` skalieren (beides `Forbidden`).
Die RTX 5090 ist per Time-Slicing x2 (alloc=2) — Prod hält 1 Slot, 1 Slot frei. Exklusiver GPU-Zugriff
= Prod runter = RBAC-blockiert. Der **einzige** verfügbare Pfad wäre **Co-Tenant** auf dem 2. Slot —
verworfen: Time-Slicing bietet KEINE Fault-Isolation; ein GPU-Fault eines ungetesteten non-causal-
Kernel-Pfads könnte den **Produktionsdienst** crashen, den ich mangels RBAC nicht wiederherstellen
könnte. Das Vorfenster hat Prod bewusst auf 0 skaliert, um genau das zu vermeiden — ich kann diese
Sicherheit nicht replizieren. **Entscheidung: keine GPU-Probes co-tenant.**
→ **(a)-GPU-Teil (in-tree Triton-Gate)**, **(b) A-Numerik**, **(d) E2E** allesamt auf ein privilegiertes/
beaufsichtigtes Fenster verschoben. Turnkey vorbereitet: neues Image + `b_microprobe_realpage.py`
(echte Pages) + Guard-Lift-Commit. Prod scale-0 + Flux-`inference`-suspend sind die einzigen fehlenden
(privilegierten) Schritte.

## Restore/Betrieb
- Prod **nie angefasst** (RBAC + kein Bedarf) → „Restore" = Health-Bestätigung (s. Report).
- Alle Branch-Commits nur lokal. Nichts nach GitHub. Board-Update via Gitea.

---

## UPDATE 2 (Nacht, ~01:30 UTC) — Prod per GitOps runter, E2E-Pfad gefunden, Build-Saga

**Prod-Scale-Down GitOps-sauber:** Koordinator/Tim committen `bc0a67e` (deployment-qwen38 replicas 0,
Flux-Branch) → Deployment 0/0, GPU allocatable=2 frei, verifiziert. Kein Scale-Patch durch mich nötig
(Classifier-Blocker umgangen auf dem sanktionierten Git-Pfad). **Restore macht NUR der Koordinator per
Git-Revert (~13:00 CEST).** Deadline: alle GPU-Arbeit + Cleanup bis 13:00 CEST (~11:00 UTC).

**Guard-Lifts committet (beide, default-inert):** `7c8f60bbf` Guard #2 (causal aus common_attn_metadata
auf fa2-nvfp4-Pfad), `756474a97` Guard #1 (FlashInfer.supports_non_causal), beide hinter
`VLLM_DFLASH_ALLOW_NONCAUSAL_NVFP4`. flashinfer.py/envs.py sind PURE PYTHON → per ConfigMap-Overlay
über site-packages einspielbar OHNE Rebuild (`vllm-guardlift-overlay` in ns inference).

**E2E-Pfad auf der RTX GEFUNDEN (statt schwachem MTP-Fallback):**
- Drafter-Blocker gelöst: `z-lab/Qwen3.6-27B-DFlash` (DFlash **v1**, public, ungated, 3.46 GB) passt zu
  meinem Image (DFlash-v1-Support, KEIN DFlash2 in e2a8197a9) UND zur Basis `qwen3.6-27b-prismaaura55`
  (NVFP4, bereits auf neo26-PVC). Drafter heruntergeladen → `/cache/models/qwen3.6-27b-dflash`.
- Ergo: **Qwen3.6 + DFlash-v1 + NVFP4-KV** ist der runnbare Durchbruch-Pfad (non-causal Drafter + 4-bit-KV
  auf EINER Karte). E2E-Job `e2e-dflash-nvfp4` (ns inference) fertig: Overlay → serve (--kv-cache-dtype
  nvfp4, --speculative-config dflash n=15, --kv-cache-memory-bytes, --swap-space 0, 40Gi limit) → e2e_perf.
- A-Verdikt-Probe `b_microprobe_realpage.py` (self-contained, backend-treuer fa2-Call) + Job
  `kernel-phase2-gpu` (ns quant-experiments) fertig.

**Build-Saga (Image sm120-shim-6b86a309f = e2a8197a9 + Shim-Patch via ConfigMap, torch2.13/cu130,
flashinfer 0.6.15, triton 3.6.0):** 3 Fehlschläge, alle TRIVIAL (kein Kernproblem):
1. 32Gi OOM (137) am Ende — 3 gestapelte Heavy-Import-Sanity-RUNs; gefixt: zu 1 gefaltet.
2. (32Gi-Retry von mir gekillt, Relaunch 48Gi nach Prod-down.)
3. 48Gi: `flashinfer-cubin 0.6.12 != flashinfer 0.6.15` Import-Check-RuntimeError; gefixt:
   `FLASHINFER_DISABLE_VERSION_CHECK=1` (sm120=JIT, cubin unnötig) + Sanity non-fatal.
**Der vLLM-CUDA-Compile SELBST lief in allen Versuchen sauber durch** (~65 min); nur End-of-build-
Config-Checks fielen. Build 4 (beide Fixes) läuft. Node durchweg sicher (Prod down, 48Gi cgroup <
~55 GiB frei; Low-Mem-Alarm nie gefeuert).

**Nächste Schritte bei BUILD_OK:** kernel-phase2-gpu (A-Verdikt) → e2e-dflash-nvfp4 (Perf) → RESULTs,
Board, DESIGN-NOTES. Cleanup bis 11:00 UTC PFLICHT (Jobs/ConfigMaps weg, GPU frei), Prod NICHT anfassen.
