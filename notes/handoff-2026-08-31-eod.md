# Handoff — End of Day 2026-08-31

## 2026-09-04 — INCIDENT ENDE: LiteLLM stabil ohne DB (Master-Key-only, Tim-Entscheid)
ROOT-CAUSE final: Die Virtual-Key-DB-Integration fror den einzigen litellm-
Worker periodisch KOMPLETT ein (auch `/` hing; DB-TCP ok => synchroner
Loop-Block bei DB-Tasks). Tim: „Ein Master-Key reicht, idealerweise ohne DB."
LÖSUNG (dauerhaft, Git/Flux, Commit 02f133a): DATABASE_URL + database_url
raus, store_model_in_db false, master_key via Env bleibt, custom_auth
Keyless für lokale Modelle bleibt. = 40-Tage-stabiler Zustand + Modell-
Reduktion (14 lokale) + Gate-Lift-RTX + gehärtete Probes.
VERIFIZIERT: Health 0,00s (kein Hang), R0 stabil, Hermes keyless 200
(RTX 0,56s / Spark-overflow 0,77s), Master-Key 200.
Virtual Keys (n8n Budget/RPM) BEWUSST ZURÜCKGESTELLT — bräuchte DB +
Stabilisierung (mehr Worker / Loop-Block-Root-Cause); nicht den Prod-Aufwand
wert solange Master-Key + Keyless reichen. n8n kann Keyless auf lokale
Modelle ODER den Master-Key nutzen.
PROZESS-LEHRE (Tim): bedingte/richtungsweisende Prod-Entscheidungen erst
fragen, nicht eigenmächtig ([[prod-eingriffe-erst-fragen]]); nach jedem Fix
stabilisieren+verifizieren statt Rollouts stapeln.


Sauberer Übergabepunkt für einen frischen Kontext. Alle Standard-Stränge, Zustände, und wie
Watcher & co wieder anlaufen.

## 0) Watcher wieder starten (im neuen Kontext)
Die **vLLM-Watch** lief als `/loop 2h`. Zum Neustart im neuen Kontext exakt diesen Prompt geben:

> `/loop 2h Führe die vLLM-Watch aus notes/handoff-vllm-robtand-2026-08-27.md §7/§8 aus (read-only, keine Kommentare/Posts, keine GPU-Aktionen): PR-Status #53977/#53978/#53979 (Reviews, Labels — v.a. ob `ready` gesetzt), letzte Kommentare #46329 (jethac Fold-in-Antwort?), Status #52244 + #50897 + #52883 (Rebase/Bewegung), PR #53122 (Reaktionen auf unseren Prod-Datenpunkt, Maintainer-Review?), Issue #53334 (Antwort/Branch von TDHSC?), jethac-Branch nvfp4-fa2-consumer-blackwell auf Force-Push prüfen, RobTand/prismaquant Commits/Releases. Nur bei relevanten Änderungen: kurzen Eintrag ins §8-Watch-Log der Handoff-Datei schreiben und Tim knapp berichten; sonst noop.`

**Watch-Baseline (Stand letzter Tick 2026-08-31 ~19:35 UTC, alles noop):** #53977/78/79 kein
`ready`, REVIEW_REQUIRED; #46329 letzter Kommentar jethac 08-29 00:52; #53122 letzter Kommentar
unser (08-27); #53334 letzter AbdulrahmanHashem 08-31 05:22 (keine TDHSC-Antwort); #52244 nur
Metadaten-Bumps (jschmied 08-24 letzter echter Kommentar, „kein Invest"); #50897/#52883 unbewegt
(`needs-rebase`); jethac-Branch @ `7a5cf143` (kein Force-Push); prismaquant Commit `55d43e33`
(08-30, Test-Infra), Release v0.16.2.

## 1) Hermes-Fork & Deployment — HEUTE der große Schritt ✅
- **Deploy-Linie = Branch `consolidation`** auf Fork [TechPrototyper/hermes-agent](https://github.com/TechPrototyper/hermes-agent)
  (jetzt **Default-Branch** → README zeigt „What this version does that upstream doesn't").
- **Produktion:** Gateway läuft via launchd `ai.hermes.gateway` aus **`~/.hermes/hermes-agent`**
  (Checkout auf `consolidation`, **hermes 0.21.0**). WICHTIG: die launchd-Plist ist die Quelle der
  Wahrheit — zeigte früher auf `~/Projects/hermes-agent-test1` (jetzt umgezeigt, Backup in
  `~/.hermes/update-backups`). **`hermes-agent-test1` ist obsolet/löschbar.**
- **6 Carry-Patches (CP-1…CP-6):** adaptive routing, tool-call-Guardrails (keep-both), patch-schema
  + Loop-Cleanup, docker-kaniko, **CP-5 Base-Skill-Reconciliation** (`hermes skills reconcile`),
  **CP-6 Graph-API-E-Mail (Graph-only, kein IMAP/SMTP/POP)**. Live verifiziert.
- **Update-Modell:** `updates.parked_branch_strategy: update_in_place` in `~/.hermes/config.yaml`
  → `hermes update` merged origin/main IN `consolidation` statt wegzuschalten. Alternativ sauberer
  Rebase im Arbeits-Repo + Fork-Push + Deploy-Pull.
- **E-Mail:** `EMAIL_AUTH_MODE=graph` in `~/.hermes/.env`; Konto `hermes1_mac@sjanasek.de`, Token
  auto-refreshed. Details/Registry: `notes/carry-patches-registry.md`, Design: `notes/cp5-…`,
  Cutover: `notes/hupdate-v7-and-checkout-convergence.md`. Memory: `hermes-deploy-linie-consolidation`.
- **Safety-Netze:** Tag `pre-consolidation-cutover-20260831`, Plist/Config/.env-Backups in update-backups.
- **Offen/optional:** sauberen linearen Rebase nachholen (heute an einem `eol`-Filter im Arbeits-Repo
  gescheitert, daher Merge-Commit); test1 löschen; LiteLLM-Loop-Breaker deployen (siehe §3).

## 2) AI-Lab Serving (RTX / Spark)
- **RTX (sm120, neo26):** drafter-freies **MTP-on-GridBook** primär (624k KV-Pool, McNemar p=0.83,
  Graph… äh Tool-Batterie grün). Image `sm120-pc50897-gridbook-088`, Checkpoint `gridbook-13gb-mtpfix`.
- **Spark (sm121):** **AQUA+DFlash2+NVFP4-KV** als Overflow, volle Batterie, **kein Whisper**
  (GridBook-auf-Spark verworfen: ~35% langsamer — `notes/spark-gridbook-eval-2026-08-31.md`).
- **LiteLLM:** `SPARK_DOWN=0`, `RTX_DOWN=0`.

## 3) LiteLLM Tool-Loop-Carry-Patches (offen, nicht deployed)
- **CP-1 greedy-coerce** + **CP-2 loop_breaker** liegen in `MyCluster/…/litellm/` **gestaged**,
  offline getestet, **nicht deployed**. Registrierung in `config.yaml` `callbacks` + Live-Test an
  echtem Loop-Mitschnitt ausstehend (Tims Sequenz). Die Hermes-Seite (CP-3-Signal-Cleanup) ist
  bereits in `consolidation`. Wurzeln: Memory `qwen38-tool-loop-rootcauses`.

## 4) GitHub-Stränge (Überblick)
- **Fork:** TechPrototyper/hermes-agent @ `consolidation` (default). Sync: Fork == Deploy == `fa02bd44`.
- **Upstream vLLM PRs:** unsere Seam #53977/#53978/#53979 + #46329 (jethac-Review) offen, kein `ready`;
  #50897/#52883/#53122/#53334 im 2h-Watch. Voller Kontext: `notes/handoff-vllm-robtand-2026-08-27.md`.

## Nächster Kontext — Startreihenfolge
1. vLLM-Watch `/loop` (Prompt oben) neu starten.
2. Optional: LiteLLM CP-1/CP-2 deployen + Live-Loop-Test (Tims Go).
3. Optional: test1 löschen, linearen Rebase nachholen.

## Morgen — PRIORISIERT: bojiang3-Track (#53334 / TurboQuant)
Kontext: bojiang3 hat zwei PRs, die unsere beiden #53334-Bugs fixen —
**#54085** (fp16-Clamp vor Quantisierung → Beobachtung 2, inf-Overflow) und
**#54086** (klare Meldung statt `'auto'` → Beobachtung 1, Message-Fix + reicht
unsere Hybrid-Scope-Frage an Maintainer weiter). AbdulrahmanHashem gab auf #53334
eine off-target Speed-Abfuhr („<40 t/s"), die technisch nicht zu diesen zwei PRs
passt (Clamp/Meldung ≠ Decode-Hot-Path). Freundlicher Aufschlag + Vernetzungs-
angebot ist raus: [#53334 comment 5486190018](https://github.com/vllm-project/vllm/issues/53334#issuecomment-5486190018).
- **[P1] #54085 hinter ihn stellen:** fp16-Clamp auf echten sm120/sm121-Checkpoints
  gegen die ~125k-Sink-Verteilung **validieren**, gepaarte Zahlen in den PR posten
  → Merge erleichtern. Kein eigener Konkurrenzfix.
- **[P1 — TIMS ENTSCHEIDUNG OFFEN] #54086 Hybrid-Scope:** wollen wir Hybrid-
  TurboQuant-**Support** (dann Lücke separat tracken) oder reicht „out-of-scope +
  klare Meldung"? Bis Entscheidung: Position im Kommentar bewusst offen gehalten.
- **[P2] Allianz kultivieren:** bojiang3 arbeitet unser Terrain (GB10/sm121, Quant-KV,
  Kernel-Guards: #53748/#54013/#54306). Als Reviewer/Mit-Zeuge für unsere
  **Parity-Probe (#53979-Linie)** vormerken; seine angrenzenden PRs reziprok stützen.
- **Abgrenzung:** unsere NVFP4-Seam-Linie (#53977/78/79 + #50897-Rebase + SWA-Parity)
  bleibt eigenständig — anderer Quant-Pfad, kein Überlapp, kein „Einreihen" dort.

## Nacht-Stand 2026-09-01 (Far-Window + Parity)
**Prod UNANGETASTET** (`vllm-qwen38-mtp-gridbook` 1/1 auf neo26). Ich habe Prod
bewusst NICHT gekippt: Cluster ist single-node neo26, DGX/Spark läuft extern, und
der saubere Failover war nachts nicht sicher verifizierbar (LiteLLM-Endpunkte
zeigten auf 10.1.0.229/10.1.0.190 — kein eindeutiger separater DGX-Endpunkt
erkennbar). Blinder Scale-to-0 = Risiko Total-Ausfall über Nacht → gestoppt.
Lektion als Memory [[rtx-erst-runter-dann-arbeiten]] hinterlegt.
- **Startklar für supervised Morgen-Lauf:** `scripts/farwindow/RUNBOOK.md`
  (Reihenfolge: DGX-Failover verifizieren → RTX runter → Safety-Net → Serve hoch
  → dry-run gegen Research-Serve → First-Pass → Teardown+Restore).
  Serve-Manifest gestaged + client-validiert: `scripts/farwindow/serve-farwindow.json`
  (`max-model-len 524288`, `max-num-seqs 1`, eigene Labels → LiteLLM routet nie drauf).
  Harness `scripts/farwindow/needle_determinism_probe.py` (Chat-Endpoint,
  max_tokens=800 — gegen Prod dry-getestet, scored real).
- **Scope-Wahrheit:** das ist **First-Pass Needle+Determinismus**, NICHT das
  Verdikt-tier Qualitäts-Gate (kein Long-Doc-Set/Referenz-Harness im Repo — wird
  nicht über Nacht fabriziert).
- **Parity-Scaffold fertig (lokal, kein Verdikt):** `scripts/parity_fa2_nvfp4_swa/PLAN.md`
  (+ CPU-Referenz aus `results/byteorder_cpu_proof.py`; Kernel-Arm = supervised TODO).
- **Watch** läuft weiter read-only alle 2h (Cron `ce0f32e9`).

## §8 Watch-Log
- **2026-09-01 ~02:04 UTC — #53334:** AbdulrahmanHashem **zieht den Speed-Einwand
  zurück**: *„my bad i thought this was a different PR fix for TQ giving garbage
  output with GDN on sm120"* ([comment 5487666309](https://github.com/vllm-project/vllm/issues/53334#issuecomment-5487666309)).
  → bojiang3s #54085/#54086 sind nicht mehr angefochten (bestätigt unsere Lesart:
  Fehlattribution). **Neuer Lead:** er verweist auf einen *anderen* PR = „TQ fix
  for garbage output with **GDN on sm120**" mit <40 t/s — potenziell relevant für
  unsere GDN-Hybrid-Linie (#52244/#50897-Umfeld) bzw. TurboQuant-Pfad auf der RTX.
  Welchen PR genau er meint, ist offen. Kein Post (Watch read-only).
- **2026-09-01 ~05:07 UTC — #46329:** neuer, unabhängiger RTX-5090-Nutzer **maxpla3**
  hat **jethacs Feature-Branch gebaut & gebootet** (`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`),
  Log zeigt **421k Tokens KV-Cache**, will benchmarken
  ([comment 5489187665](https://github.com/vllm-project/vllm/pull/46329#issuecomment-5489187665)).
  → dritter 5090 auf genau unserer NVFP4-FA2-Stack-Basis; positives Momentum für
  #46329/#53977 (dead-weight-Buffer/KV-Pool-Thema). Branch weiter `7a5cf143` (kein
  Force-Push). Kein Post (Watch read-only).
- **2026-09-01 ~08:09 UTC — #46329:** **sblantipodi** an maxpla3:
  *„gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 is less intelligent than the
  standard checkpoint, I would use the unsloth NVFP4 instead"*
  ([comment 5490927822](https://github.com/vllm-project/vllm/pull/46329#issuecomment-5490927822)).
  → NVFP4-Export-Qualitätsdebatte (gittensor vs. unsloth) auf dem Thread — unser
  Kerngebiet (wir haben n=1319-Verdikt-Daten für AQUA/GridBook-Exporte). Unbelegte
  Behauptung, kein Bezug zu unseren PRs. Kein Post; kein Handlungsbedarf (wir
  drängen uns nicht auf).
- **2026-09-08/09 — #55976 (jethacs isolierter Writer-Fix, funktional identisch
  zu unserem eigenen verworfenen "Fix 1"):** zwei unabhängige Hardware-Red/Green-
  Bestätigungen eingetroffen: **hclsys** (08-08 23:37 UTC) *„sm_121 (GB10) red/green
  for this patch, on the record here"* (DGX Spark GB10, CUDA 13.0/torch 2.12.1) und
  **TensorRaya** (09-09 04:13 UTC) *„sm_120 red/green for this patch, on the record
  here as well"* (RTX PRO 4500 Blackwell, CUDA 13.0/torch 2.13.0). Beide bestätigen
  den Fix unabhängig von unserer eigenen Spark-Validierung — konvergente externe
  Bestätigung genau der Kernel-Stelle, die wir selbst schon vor der PR-Öffnung als
  Kandidaten identifiziert hatten. Kein neuer Maintainer-Review, PR weiterhin offen/
  `MERGEABLE`, nur `bug`-Label. Kein Post (Watch read-only).
- **2026-09-09 ~06:00 UTC — #55976, Nachschlag mit quantifizierten Zahlen (WICHTIG,
  aber Entwarnung für uns):** TensorRaya (05:32 UTC) liefert die Messung, die den
  Bug vom NaN-Level auf echten Attention-Output hebt: `main`s geswizzelter
  V-Scale-Writer erzeugt **rel. L2-Fehler ≈0,34 / Cosine ≈0,946** gg. bf16-
  Referenz (linearer Writer aus dieser PR: **≈0,12 / ≈0,993**) — Faktor **~2,7×**
  mehr Fehler, ohne NaN, über KV-Längen 256/1024/2048, 5 Seeds. Kernbefund:
  **Needle/Recall-Tests erkennen diese Degradation nicht** (27B findet den
  String trotzdem, Output ist aber messbar verschlechtert). hclsys (05:52 UTC)
  reproduziert exakt dasselbe auf sm_121 (GB10) — bestätigt `major < 12` als
  korrekten Split für beide sm12x-Teile.
  → **Direkt geprüft, ob uns das betrifft (read-only `git log`/`grep` im
  laufenden `vllm-prod-nvfp4kv`-Container auf dem Spark, keine GPU-Aktion):**
  **Wir sind NICHT betroffen.** Unser Writer-Kernel
  (`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`) hat bereits
  `swizzle_v_sf = get_device_prop()->major < 12` fest verdrahtet — auf sm12x
  wird V linear geschrieben, exakt der in #55976 vorgeschlagene Fix. Commit
  `7a5cf1431` (Store-Contract-Pin) ist in unserer Container-Historie enthalten.
  Grund: wir bauen auf jethacs #46329-Linie, nicht auf vLLMs `main` — `main`
  hat den Fix noch nicht (daher TensorRayas "V swizzled"-Baseline), unsere
  Basis hatte ihn schon vor Wochen. Kein Post (Watch read-only), keine Aktion
  nötig — reine Bestätigung, dass unsere Architektur-Entscheidung (Fork statt
  main) uns hier bereits abgesichert hat.
- **2026-09-09 ~08:00-09:00 UTC — jethac-Branch `nvfp4-fa2-consumer-blackwell`:
  Force-Push bestätigt** (explizit gecheckt, wie angefragt). Head-SHA
  `ae449f10f6` → `534310f788`, gleiches Datum (2026-09-07 15:53 UTC), **exakt
  gleiche Commit-Message** ("VO split: record the measured reason bf16/FP8
  stay unsplit"), aber anderer Parent/Tree-Hash → **Rebase auf neueren main-
  Stand**, kein inhaltlicher Änderungsinhalt an jethacs eigenem Commit. Da
  #46329 exakt diesen Branch als Head führt, erklärt das auch dessen
  `updatedAt`-Bump auf 08:09 UTC (Rebase-Commit selbst 23:00 UTC 09-08, kurz
  nach seinem "Option 1 — I'll open it"-Kommentar zu #55976) — vermutlich um
  die PR nach dem Split-Vorschlag frisch/mergeable zu halten. Keine neuen
  Kommentare auf #46329 seit dem letzten Tick. Kein Post (Watch read-only).
- **2026-09-09 ~20:44 UTC — RobTand/prismaquant #478** (Merge): "Fix streamed
  Qwen rotary positions for Transformers 5.17" — Transformers 5.17 hat
  Qwens Text-Position-Expansion in den Model-Forward verschoben; prismaquants
  gestreamte Probes/Costing umgehen den Forward und crashten dadurch bei der
  direkten Rotary-Call. Fix betrifft `prismaquant/model_profiles/qwen3_5.py`
  (unsere Modellfamilie) — reine Kompatibilitäts-Reparatur in RobTands
  Export-Tooling für Transformers 5.17, keine Aussage über unsere aktuell
  deployten Checkpoints. Nur relevant, falls wir künftig mit prismaquant auf
  Transformers-5.17-Basis neu exportieren. Kein Post (Watch read-only).

## §9 Arbeits-Session 2026-09-01 — Hermes-Audit A–Z (Loops fixen)
Tim: „sind wir sicher / Einzelrepo fertig / alle Carry-Patches drin / warum laufen
die Loops noch?" → vollständiges A–Z-Audit, dann Vorbereitung + Trockenlauf, Live-
Deploy bewusst gegatet. Details: Memory [[qwen38-tool-loop-rootcauses]] (§2026-09-01)
+ `carry-patches-registry.md` (CP-2).
- **(a) Sicher:** ja — Tags `pre-consolidation-cutover-20260831` + `pre-update-20260831-230707`,
  Voll-Backups in `~/.hermes/update-backups/`. Gateway stabil.
- **(b/c) Einzelrepo/Carry-Patches:** `consolidation` = Default am Fork, Deploy trackt
  `fork/consolidation @ fa02bd44`. Alle 6 Hermes-CPs sind saubere Commits (adaptive routing,
  Guardrail-Bremse, CP-3 Loop-Cleanup, kaniko, CP-5 3-Wege-Skill-Reconcile, CP-6 Graph-Mail).
- **(d) Loop-Wurzel — EINE echte Lücke:** LiteLLM `loop_breaker` (CP-2, Force-Progress) war nie
  in `callbacks` verdrahtet → sendet nie den Header, den Hermes CP-3 (im laufenden Gateway) liest
  → Kette tot. Deployte ConfigMap zudem 8d alt (kein greedy-coerce). **KORREKTUR:** kein 3.6→3.8-
  Retarget nötig — Primär-Loop ist schon `qwen3.8-27b` thinking-ON (= Sweep-Optimum), Ladder ist
  extern (Antigravity/Codex/Claude), smart_routing aus.
- **Erledigt:** loop_breaker in `kustomization.yaml` + `config.yaml:670` gestaged; lokaler
  kustomize-Render grün (neuer Hash `gfdh2g5552`); Detektor+Prune offline grün, FP-frei;
  Runtime-Artefakt `contributors/emails/…` via `git skip-worktree` entschärft.
- **DEPLOYED 2026-09-02** (Tim fuhr `kubectl apply -k`; Strategie `maxSurge:0/maxUnavailable:1`
  belassen — nie 2 Pods): ConfigMap `gfdh2g5552`, Pod `litellm-6cdc695b76-m6zwd` 1/1 restarts=0,
  startup-complete ohne Fehler → **loop_breaker + CP-1 greedy-coerce LIVE**, kein Hermes-Restart.
  Kill-Switch `LOOP_BREAKER=0`. **OFFEN: End-to-End-Realprobe** an echtem Loop (Detektor-Fenster
  greift an Tims realer Form? CP-3 konsumiert Header?) — Canary-Import im Container war geblockt.
- **Bewusst gehalten:** test1-Löschung — `apply-tool-patches.sh:14 HERMES_REPO` zeigt noch drauf
  (hupdate-v7-Konvergenz); nicht blind gelöscht.
- **2026-09-03 CAUSE-FIX (Realprobe deckte auf):** loop_breaker feuerte in Prod 15+× identisch
  (`reps=3`) ohne zu heilen. Wurzel: `adaptive_model_routing.py:110` schwenkt Tool-Turns auf
  `qwen3.8-27b-tools` = thinking-OFF/pp1.5 = Sweep-loop-INDUZIEREND (meine 09-01-Audit „‑tools ist
  dead" war falsch — 2 Router: smart_model_routing AUS, adaptive_model_routing AKTIV). CP-3-Header→
  Collapse klebt nicht (Streaming-Header-Weg fragil). **Fixes LIVE:** LiteLLM `-tools`+`-tools-overflow`
  → C1 (thinking-ON, ConfigMap `2945m45k6m`); Hermes CP-4 `max_identical_calls` 50→6. Ursache weg +
  2 Netze. Test = nächster DMS-Task. Details: Memory [[qwen38-tool-loop-rootcauses]] §09-03.
- **2026-09-02 ~00:37 UTC — #53977 (unser Seam-PR):** Label **`needs-rebase`** neu
  gesetzt (vorher nur `bug`) — Merge-Konflikt mit Upstream-main. Aktionspunkt: Seam
  auf aktuellen main rebasen (stacked auf #46329). Weiter `REVIEW_REQUIRED`, kein
  `ready`. #53978/#53979 (noch) ohne `needs-rebase`. Kein Post (Watch read-only).
- **2026-09-02 ~05:14 UTC — #46329 (Stack-Basis):** mergify flaggt **Merge-Konflikte**,
  Ping an jethac zum Rebase. Zusammen mit #53977-`needs-rebase` (Tick 28): die ganze
  Seam-Stack-Basis (jethacs #46329 + unsere #53977) driftet gegen bewegten main.
  Reihenfolge bei Rebase beachten (erst #46329/jethac, dann unsere Seam nachziehen).
  Kein Post (Watch read-only).
- **2026-09-02 ~07:27 UTC — #46329 REBASE + direkter Ping an uns:** jethac hat die
  Stack-Basis **auf aktuellen main (`aae36577f`) rebast, neuer HEAD `9590d2464`**
  (force-push, vorher `7a5cf1431`; 10 Commits, keine funktionalen Änderungen; nur 2
  Patches berührt — Gemma-mm-prefix-Konflikt in `flashinfer.py` „keep both" pinned-mem
  + mm_groups; NVFP4-KV decode-`plan()` nur Kontext-Shift). Checks: ruff clean,
  gemma4-routing 6/6 (CPU); **keine CUDA-Re-Validierung auf dem Head** → jethac lädt
  ausdrücklich zu „fresh sm12x boot from anyone with a card" ein.
  **AKTION (Ping an @TechPrototyper):** *„#53979 will want the same treatment —
  `git rebase --onto 9590d2464 7a5cf1431`, at most the one mm_groups/prefill_groups
  conflict at the pinned-memory block."* Gilt analog für unsere Seam (#53977 hat
  `needs-rebase`). #46443 ist bereits re-stacked (`864f785a5`).
  → Zwei Aktionspunkte fürs wache Fenster (kein Auto-Push/GPU im Watch):
  (1) Seam #53977/#53979 onto `9590d2464` rebasen; (2) optional sm120/sm121-Boot-
  Datenpunkt auf dem rebasten Head liefern (unsere Spur + Allianz-Zug).
- **2026-09-02 ~12:32 UTC — #53979 `needs-rebase`:** wie von jethac vorhergesagt jetzt
  auch #53979 als `needs-rebase` markiert. → **beide** Seam-PRs (#53977 + #53979)
  brauchen Rebase onto `9590d2464` (jethacs Kommando: `git rebase --onto 9590d2464
  7a5cf1431`, erwarteter Einzelkonflikt am pinned-memory-Block). #53978 (noch) ohne.
  Kein Post (Watch read-only).

- **2026-09-02 ~20:16 UTC — #46329: maxpla3 tritt zum Benchmark an (wie von
  Tim vermutet).** 8-Run-Kurven auf RTX 5090 (Treiber 595.84/CUDA 13.2, alter
  PR-Stand `7a5cf1431`, HF-Modell sakamakismile/Qwen3.8-27B-MTP-NVFP4):
  fp8/nvfp4 × ±MTP@3. Sein Befund + offene Frage an den Thread: **„harsh
  drop in decode speed with growing context size for nvfp4+mtp — expected?"**
  → Das ist EXAKT unsere MTP-KV-Drain-Achse; wir haben Prod-Daten und die
  Antwort (KV-Pool-Ökonomie, n=1319-Verdikt, deshalb MTP@2 drafter-frei +
  775k-Pool statt Speed-Maximierung). Steilvorlage für einen Daten-Antwort-
  Kommentar — NUR MIT TIMS GO (Watch read-only). Nebenbefund: er misst auf
  dem Vor-Rebase-Stand; jethacs 9590d246 + unsere v5-Linie wären der
  aktuelle Vergleichspunkt.

- **2026-09-03 ~01:5x UTC — UNSER Post auf #46329 (Tims Go, Spock-Ton):**
  Antwort auf maxpla3 mit vollem KV-Sweep (RTX 5090, v5-Linie, GridBook-13GB,
  MTP@2, NVFP4-KV, 775k-Pool; 8 Runs/Stufe, 1k→262k):
  Decode 81,7→5,8 tok/s (14×), **Akzeptanz stabil 0,58–0,67** → Einbruch ist
  KEIN Akzeptanz-Kollaps, sondern KV-/Attention-Kostenwachstum × Draft-Breite
  (konsistent mit seinem härteren Knick bei mtp@3). Baseline-Hinweis auf
  9590d2464. Kommentar: issuecomment-5517213586. Rohdaten:
  `results/kvsweep-rtx-v5-{near,far}.json`. Notebook-Vertiefung offen (Task 8).

- **2026-09-03 ~22:37 UTC (18 min nach unserem Post) — ptorsten (Patrik
  Torstensson, META) auf #46329: Schwergewichts-Analyse.** 2×GB10 TP=2,
  NVFP4-KV @512k, Pool 2,06× vs bf16. Drei Befunde + Follow-up:
  (1) **Non-causal Drafter werden abgelehnt** — Builder liest runner-weites
  `cache_config.cache_dtype` statt per-KV-Group → EXAKT die Problemklasse
  unseres #53979; er hat es selbst per-KV-Group gescoped und DFlash bootet.
  (2) Scheduler-Blockgröße 1424 unter nvfp4 → Doku-Falle.
  (3) **GB10-Unified-Memory: >64k bei util 0.85 unsicher** — NVRM
  NV_ERR_NO_MEMORY, 132k-Cold-Prefill riss den SoC weg (65 W, nicht
  thermisch); Prefill-Working-Memory skaliert mit Kontext und wird beim
  Profiling nicht erfasst (#46307). ↔ möglicher Bezug zu UNSEREM
  Spark-CUBLAS-Crash vom 09-02 (64k-Request)!
  **→ 2026-09-03 ~02:5x UTC: Kollaboration ANGENOMMEN (Tims Go), unsere
  Antwort gepostet (issuecomment-5517611768):** #53979-Querverweis (zwei
  unabhängige Implementierungen), Sweep-Wiederholung mit gepatchtem
  FlashInfer auf sm120+sm121 binnen eines Tages angeboten (Harness teilbar),
  GB10-CUBLAS-Datenpunkt für #46307 beigesteuert, deterministische
  Repro-Ladder angeboten („count us in").
  **Follow-up erklärt UNSERE Sweep-Kurve:** FA2-nvfp4-Route hat
  single-token-only CUDAGraph (Multi-Query-Verify läuft eager, −42% kurz)
  und FlashInfers `_nvfp4_kv_requires_disabled_split_kv` erzwingt
  Split-KV-off → Verify streamt den ganzen KV aus einer CTA-Row (−78% bei
  Tiefe). „Root-causing that corruption is what makes nvfp4 KV viable with
  speculative decoding; **happy to help reproduce**." → Kollaborationsangebot
  eines Meta-Engineers; Antwort NUR MIT TIMS GO (Watch read-only).

- **2026-09-03 ~02:37 UTC — jethac antwortet (auf uns + ptorsten): voller
  Mechanismus + konkreter Arbeitsauftrag an UNS.** (a) Bestätigt: flache
  Akzeptanz entlastet den Drafter; Rebase nicht behavioral. (b) Mechanismus:
  sm12x + NVFP4 → trtllm-gen-Decode-Kernel gecleart → `supports_spec_as_
  decode=false` → MTP-Verify (q_len 2–9) läuft als Prefill durch FA2-
  BatchPrefill → SEIN Gate `_nvfp4_kv_requires_disabled_split_kv`
  (flashinfer 6dc2c589, nie root-gecaust) schaltet Split-KV ab → linearer
  Zeitanstieg mit Kontext; Kurz-Kontext-Offset = single-token-only
  CUDAGraph. (c) **Plan: Korruption root-causen, Gate droppen, Verify auf
  spec-as-decode.** (d) **Ask an uns: Repro-Ladder liefern** (greedy, fixed
  seeds, kleinste korrupte qo/kv-Shape) — er arbeitet dagegen auf einer
  5090 und liefert den FlashInfer-Patch für unsere Sweeps (Task #10).
  (e) **Co-Autorenschaft:** zieht das per-KV-Group-Scoping mit ptorsten UND
  uns als Co-Authors in den Branch (#53979-Anerkennung) + 1424-Doku-Note.
  Repro-Ladder-Bau = GPU-Arbeit → supervised, nicht nachts (RTX-Regel).

- **2026-09-03 Vormittag — Repro-Ladder gebaut+gefahren+gepostet
  (issuecomment-5520861396): STRUKTURIERTES NEGATIV.** Direktes A/B auf
  `BatchPrefillWithPagedKVCacheWrapper` (Gate an vs. umgangen), identische
  NVFP4-Buffer (positions-zyklische e4m3-Scales gegen Wegmitteln), Matrix:
  qo{1,2,3,5,9} × kv{2k…262k} × split{auto,256,512,1024} × causal{±} ×
  ragged{±} × batch{1,8 gemischt} — **keine Korruption in keiner Zelle**
  (max_rel ≤0,0056 = bf16-Reassoziation, 0 NaN; No-Split-Zellen bitidentisch).
  Konjunktiv-Deutung im Post: Trigger läge außerhalb dieser Achsen
  (CUDAGraph, TP>1, Modell-echte Daten, Drift seit 6dc2c589 — Gate evtl.
  liftbar nach E2E-Validierung). Harness im Post eingebettet
  (`scripts/kvsweep/nvfp4_splitkv_ladder.py`), Rohdaten
  `results/nvfp4-splitkv-ladder-2026-09-03.tgz`. Angebot erneuert:
  Patch-Validierung sm120+sm121 binnen eines Tages. neo26-Lärm-Check davor:
  alles grün (Load 1,9 / 28 °C / GPU 5 W / Lüfter <1000 RPM) — Verdacht
  Mechanik/Spindel, remote nicht weiter diagnostizierbar.

- **2026-09-03 ~05:51 UTC — AbdulrahmanHashem jetzt auch auf #46329**
  (freundlich, „awesome work"): meldet **Regression auf 2× 5060 Ti (sm120)**
  — ältere Builds desselben PR ok, aktueller Stand + #50288 je <15 tok/s
  tg bei **GPTQ/INT4**-Modellen (RedHatAI + AutoRound Qwen3.8-27B-INT4),
  Prefill unauffällig. NICHT unsere NVFP4-KV-Achse (kein nvfp4-Cache) —
  plausibel eine Upstream-Drift, die der 9590d246-Rebase mitgenommen hat
  (GPTQ-Decode-Pfad/CUDAGraph auf sm120?). Beobachten; falls Tim
  reagieren will: gemäß [[keine-direktangriffe-fragen-oder-ignorieren]]
  auf die freundliche Sachfrage eingehen wäre unproblematisch. Kein Post
  (Watch read-only).

- **2026-09-03 ~20:22 UTC — prismaquant: 3 Merges von Rob** (erste Bewegung
  seit 08-30): #160 „derive the confident cut, stamp it, refuse the empty
  lane", #159 Fisher-Row-Weight-Clip konsolidiert (eine Env-Regel), #146
  Pre-Guard-Trust-Admission programmatisch lesbar. Kein neues Release
  (weiter v0.16.2). Relevanz: Quantizer-Weiterentwicklung unter unseren
  GridBook/AQUA-Checkpoints — bei nächstem Release prüfen, ob Re-Export
  der 13GB-Checkpoints ansteht.

- **2026-09-03 ~22:37 UTC — maxpla3 liefert den Gate-Lift-Patch + E2E-Beleg:**
  Branch `maxpla3/vllm nvfp4-sm120-kv-mtp-decode-fix` mit Env-Schalter
  `VLLM_NVFP4_KV_SPLIT_KV=1` (Monkeypatch des Gates — dieselbe Technik wie
  unsere Ladder). Validierung auf 5090 (Qwen3.8-27B-MTP-NVFP4, head 256):
  Needle+Prefix-Cache-Batterie 60k–200k, ~6,5 Mio Tokens aus dem Cache,
  **63/64 Retrieval, 0 degenerierte Outputs MIT Split-KV an**; der eine Miss
  reproduziert auch MIT Gate (Retrieval-Schwierigkeit, keine Korruption).
  Benchmark zeigt Speed-Recovery. → **Drittes unabhängiges Negativ**
  (nach unserer Kernel-Ladder + seiner Batterie): die 6dc2c589-Korruption
  reproduziert auf aktuellem sm120-Stack nicht. Gate-Lift rückt näher;
  jethacs Root-Cause-Verdikt steht aus. **Unser nächster Zug (Task #10,
  supervised mit Tims Go):** `VLLM_NVFP4_KV_SPLIT_KV`-Carry-Patch auf
  unsere v5-Linie (2 Zeilen) + kvsweep-Rerun sm120/sm121 + unsere
  Greedy-E2E-Batterie = vierte unabhängige Validierung + unsere eigene
  Decode-at-Depth-Heilung (5,8→? tok/s @262k).

- **2026-09-04 ~03:0x UTC — UNSER Gate-Lift-A/B gepostet
  (issuecomment-5534004391), Task-#10-Rerun sm120 ERLEDIGT.** Carry-Patch
  `c8278b18` (env-gated, off-by-default) auf v5-Linie; Messung im
  Research-Pod (Prod-Spec-Klon, Hot-Patch): **Gate-lifted Decode 100→62,5
  tok/s über 1k→262k — 9,9× bei 262k** (aktiv: 6,3). Methodenfund:
  Stream-Timing überzeichnet auf dem gelifteten Pfad (394 tok/s
  physikalisch unmöglich) → **Differenzmessung** (mt16 vs mt512,
  Rate=496/Δt), OFF-Kontrolle mit derselben Methode = Baseline bestätigt.
  Akzeptanz beide Arme 0,56–0,81. **Needle 9/9 exakt** (32k/131k/262k ×
  3 Tiefen, greedy, doppelt = byteidentisch); Achtung Harness-Falle:
  erst 3/9 wegen max_tokens=200 (<think>-Budget!), mit 900 perfekt.
  Prod restauriert (RTX-Fingerprint verifiziert), Flag in Prod AUS bis
  Root-Cause-Verdikt + Verdikt-tier-Qualitätsgate (Endgegner-Regel).
  Rohdaten `results/splitkv-*`. OFFEN: sm121-Arm (zugesagt), Prod-Adoption
  = Image-Rebuild ab c8278b18 + Gate — TIMS ENTSCHEIDUNG.

- **2026-09-04 ~01:12 UTC — paulocarinhena (neu im Thread, 2×5060 Ti TP=2):
  orthogonaler Fund** — `--max-num-seqs` UNTER der größten
  CUDAGraph-Capture-Size lässt Decode-Batches ohne passenden Graph still
  eager laufen: bei 6 Sessions **+106 %/Session** durch max-num-seqs 6→8
  (Aggregat 330→505 tok/s), Kosten nur ~1 % KV-Pool. Möglicher Teilfaktor
  von AbdulrahmanHashems Regression. **Für uns:** RTX (max-num-seqs 2,
  capture [1,2]) sauber; **Spark-Prod (max-num-seqs 32, capture [1,2,4])
  läuft bei Batch >4 eager** → Tuning-Kandidat capture [1,2,4,8,16,32]
  für den Overflow-Concurrency-Fall — eigener A/B nötig (Task-Kandidat,
  nicht dringend). Kein Post (Watch read-only).

- **2026-09-04 ~08:xx UTC — sm121-A/B gepostet (issuecomment-5536685306),
  Task #10 KOMPLETT.** Tims Zuschnitt umgesetzt: nur LLM-Container getauscht,
  Embedder/Reranker-Batterie lief durch (wxO ungestört), RTX trug Prod.
  v5-Image + Hot-Patch, AQUA + natives MTP@2. **A/B (Differenzmethode,
  Flag je Arm via /proc/<pid>/environ verifiziert):** 8k 15,5→17,1 (1,1×),
  64k 10,3→16,2 (1,6×), 262k 4,5→13,7 (**3,0×**); Lifted-Kurve nahezu
  flach 19,1→13,7. Needle 9/9 byteidentisch. QA-Zwischenfall transparent
  im Post: erste OFF-Messung traf nach Port-Kollision still den alten
  ON-Serve — bis auf 3. Nachkommastelle identische Zahlen verrieten es;
  Lehre: Flag-Zustand IMMER am Prozess verifizieren, nie nur am Log.
  Deutung (Konjunktiv): kleinerer Hebel als sm120 (3,0× vs 9,9×) passte
  zur geringeren SM-Zahl des GB10. Beide Arches jetzt mit gepaartem A/B +
  sauberem Retrieval; Batterie-Angebot für Root-Cause-Patch erneuert.
  Prod-Endzustand verifiziert: RTX primär (fe3247505), Spark-alt Overflow
  (f4c27c0da), SPARK_DOWN=0. Rohdaten `results/sm121-*`.

- **2026-09-04 ~07:09 UTC — maxpla3 präzisiert die Beweislage (wichtig):**
  Das Gate entstand für einen E2E-Fehler auf **Gemma 4 E2B/E4B** auf dem
  **asymmetrischen VO-Split-Pfad** (qk=512/vo=256); alle vier Negative
  (unsere Ladder, seine Batterie, unsere sm120+sm121-A/Bs) testen den
  SYMMETRISCHEN Pfad mit großen Qwen-Modellen single-stream — „evidence
  about our regime, not evidence the underlying bug is fixed". Lücke:
  kleiner Gemma-4-NVFP4-Checkpoint, asymmetrischer Pfad, Prefix-Cache-Last.
  **Option für uns (Tims Entscheidung):** Gemma-4-E2B-NVFP4-Checkpoint
  selbst quantisieren (llmcompressor/PrismaQuant) + Asym-Ladder auf der
  5090 — würde die letzte Lücke schließen ODER den Bug endlich echt
  reproduzieren (beides wertvoll für jethacs Root-Cause). Kein Post.

- **2026-09-04 ~12:2x UTC — GATE-LIFT IN PRODUKTION (Tims Go).** Kette:
  Build `sm120-v5-c8278b18` (Patch gebacken, Flag off-by-default) →
  GridBook-088-Overlay → Deployment `sm120-v5-c8278b18-gridbook-088` mit
  `VLLM_NVFP4_KV_SPLIT_KV=1` (Pod-Env), **`--max-num-seqs` 2→4** +
  LiteLLM **`MAX_LOCAL=4`** (Kapazität und Overflow-Schwelle jetzt
  konsistent — die alte Diskrepanz 2 vs 3 ließ Tims dritten Request lokal
  anstehen statt zum Spark zu gehen). Qualitätsbatterie grün: greedy
  korrekt+deterministisch, Tool-Call sauber, **Decode @64k 85,1 tok/s**
  (vorher gated ~20; Needle-9/9-Beleg von heute Nacht gilt — identischer
  Quellstand). Beide LiteLLM-Routen mit neuem Fingerprint `gc8278b1850`
  verifiziert. Auslöser: Tims Parallel-Last-Messung (7–18 tok/s) = live
  erlebte Gate-Kurve + Schwellen-Mismatch. GitOps-Drifts (Image, args,
  env, MAX_LOCAL) für Reconcile notiert. Anlass-Fenster: Spark trug
  allein, gefühlter Hänger = Prefill-Dominanz des Alt-Stacks (74k prompt
  vs 1,4k generiert) — mit RTX-Rückkehr erledigt.

- **2026-09-04 — prismaquant fc6abe3b** ("Resolve the architecture
  provenance conflict literally left in main") — Nachzügler zum 09-03-Merge-
  Schwung, kein Release (weiter v0.16.2). Beobachten, kein Handlungsbedarf.

- **2026-09-04 ~16:xx UTC — Watch-Tick, zwei Nachtraege:**
  (1) **#53977 + #53979 `needs-rebase` ist WEG** (beide `updated_at` fast
  identisch: 09-02 14:39:37Z / 14:39:49Z — 12s auseinander, also im selben
  Zug rebast). Schliesst die am 09-02 ~07:27 UTC geloggte Aktion (jethacs
  `git rebase --onto 9590d2464 7a5cf1431`) ab; war bisher nicht im Log
  bestaetigt. Beide weiter `REVIEW_REQUIRED`, kein `ready`, `mergeable_state:
  blocked` (Review, kein Merge-Konflikt mehr). #53978 unveraendert (nie
  needs-rebase, `updated_at` 08-27, keine Bewegung).
  (2) **#52244 hat jetzt `needs-rebase`** (neu gegenueber Baseline/bisherigen
  Ticks — vermutlich mergify-Auto-Flag durch denselben main-Drift). Keine
  neuen Kommentare (jschmied 08-24 weiter letzter echter Kommentar).
  Sonst reines Noop: #46329 letzter Kommentar unveraendert (maxpla3
  07:08:55Z 09-04, bereits geloggt); #53122/#50897/#52883/#53334 keine
  Bewegung seit letztem Tick; jethac-Branch weiterhin `9590d246` (kein
  neuer Force-Push); prismaquant kein neuer Commit seit fc6abe3b.
  Kein Post (Watch read-only).

- **2026-09-05 ~08:xx UTC — Watch-Tick: #46329 (Stack-Basis) wieder
  `needs-rebase` + `CONFLICTING`.** mergify hat um 00:31:24Z erneut Konflikt
  gegen main gemeldet und jethac zum Rebase gepingt (zweiter Drift-Zyklus
  seit dem 09-02-Rebase auf `9590d2464`). jethac-Branch
  `nvfp4-fa2-consumer-blackwell` bisher **unveraendert** (`9590d246`, kein
  neuer Force-Push) — Rebase steht noch aus. Unsere eigene Seam bleibt
  vorerst unberuehrt: #53977/#53978/#53979 alle weiter `MERGEABLE`/`BLOCKED`
  (kein `needs-rebase`, kein `ready`), #53979 zusaetzlich frisch mit unserem
  CodeRabbit-Fix-Commit (`41758a0aa4`, heute gepusht) aktuell. **Sobald
  jethac force-pusht, muss #53979 vermutlich erneut rebast werden**
  (gleiches Muster wie 09-02: erst #46329, dann unsere Seam nachziehen).
  Sonst reines Noop: #52244 unveraendert `needs-rebase`/`CONFLICTING` (kein
  neuer Kommentar seit jschmied 08-24); #50897/#52883/#53122/#53334 keine
  Bewegung; prismaquant kein neuer Release (weiter v0.16.2), aber
  durchgehende Commit-Aktivitaet an Tessera/W4A4 (`8a77b0ad` Merge #196
  „pq-tessera-audit" + 4 Fix-Commits, 05:39-06:45 UTC) — weiter rein
  experimentell, nichts fuer RTX/Spark actionable (Standing-Regel „What's
  in it for us" bleibt „nichts"). Kein Post (Watch read-only).

- **2026-09-05 ~16:2x UTC — WICHTIG: jethac-Branch geforce-pusht + inhaltliche
  Antwort mit direkter Anfrage an uns.** Branch `nvfp4-fa2-consumer-blackwell`
  neuer HEAD `b423d395d` (vorher `9590d2464`, seit 00:27:35Z). #46329 selbst
  **`needs-rebase` wieder weg**, `MERGEABLE`/`BLOCKED` (rebased onto main,
  neuer PR-Head `b423d395d`); #46443 analog auf `c8d9ae7a5`.
  **jethacs Kommentar (15:48:55Z) — Kern:** Er behaelt den FlashInfer-Gate
  (`_nvfp4_kv_requires_disabled_split_kv`, sein eigener) bei, weil bisher NUR
  symmetrische Head-Shapes (128/256) getestet sind — unsere Ladder, maxpla3s
  Needle-Runs, das 5090/GB10-A/B, paulocarinhenas Woche — alle sauber, aber
  keiner deckt die eigentliche 512/256-Asymmetrie (Gemma-4-VO-Split) ab, fuer
  die der Gate ursprünglich eingefuehrt wurde. Er hat den Gate stattdessen auf
  der Shape-Achse verschaerft (`head_dim_qk`/`head_dim_vo` statt pauschal
  sm120/121): symmetrische Heads bis 256 bekommen split-KV zurueck, die
  512/256-VO-Split-Konfiguration bleibt gegatet. Neuer Branch in seinem
  FlashInfer-Fork: `jethac/flashinfer@nvfp4-splitkv-gate-shape-scope`
  (`b8256000`, 15:46:06Z). Validierungsplan: gate-on/off A/B heute auf 5090 +
  GB10.
  **DIREKTE ANFRAGE AN @TechPrototyper:** *„if your ladder can plan with
  head_dim_qk=512, head_dim_vo=256 against a 512-wide cache (qo 1..9, kv to
  32k) that would be the first direct look at the shape the gate exists
  for."* — exakt unsere VO-Split-Arbeit aus #53979. Waere die erste direkte
  Messung der Shape, fuer die der Gate ueberhaupt existiert.
  **Konsequenz fuer #53979:** unser Branch (`41758a0aa4`) sitzt noch auf dem
  ALTEN jethac-Head; Rebase onto `b423d395d` steht noch aus (Konflikt
  vermutlich wieder nur am pinned-memory-Block, analog 09-02).
  Sonst Noop: #52244/#50897/#52883/#53122/#53334 unveraendert seit letztem
  Tick; prismaquant neuer Commit `468e7534` (Merge #200 „pq-197-registry-
  leak", 16:15 UTC) — Bugfix, nicht Tessera, kein neuer Release, nicht
  actionable. **Kein Post (Watch read-only) — Rebase + ggf. Ladder-Lauf fuer
  512/256-Shape sind Aktionspunkte fuers wache Fenster, brauchen Tims Go.**

- **2026-09-05 ~22:04 UTC — AKTION ausgefuehrt (Tims Go): Rebase #53979 +
  Ladder-Antwort auf jethacs 512/256-Anfrage.**
  (1) **Rebase:** `feat/nvfp4-noncausal-dflash` von altem jethac-Head (lokale
  Kopie von `9590d2464`, Hash `5441f78839`) per `git rebase --onto b423d395d
  5441f78839` auf jethacs neuen Head gehievt — **kein Konflikt** diesmal
  (glatter als 09-02). Neuer Head `0cd789b397`, `ast.parse`+`ruff check`
  gruen, gepusht (`--force-with-lease`). PR wieder `MERGEABLE`/`BLOCKED`,
  CodeRabbit-Fixes erhalten.
  (2) **Ladder erweitert:** `scripts/kvsweep/nvfp4_splitkv_ladder.py` unter-
  stuetzt jetzt asymmetrisches `head_dim_qk`/`head_dim_vo` (K + volle
  512-breite V-Seite separat gebaut, V per `narrow()` auf die
  head_dim_vo-Chunk geschnitten — exakt der Aufruf-Stil von
  `_run_vo_split_prefill` in unserer eigenen `flashinfer.py`, Tupel
  `(k_cache, v_cache_i)`/`(k_sf, v_sf_i)` statt kombiniertem Tensor).
  (3) **Sweep gefahren** (Tim, im laufenden `vllm-prod-nvfp4kv`-Container auf
  dem Spark, **ohne Battery-/LLM-Downtime** — Skript ist ein separater,
  kurzlebiger Prozess, ~400 MB transient neben der laufenden Produktion):
  volle Matrix qo 1..9 × kv {2048, 8192, 16384, 32768}, `head_dim_qk=512`,
  `head_dim_vo=256`, `vo_chunk=0` — **36/36 Kombinationen `corrupt: false`,
  kein NaN**, `max_rel` durchgehend ~0.0052 (weit unter der 1%-Schwelle),
  `max_abs` konstant 0.0625 ueber alle Zeilen, `frac_gt_1e2` schwankt
  11–17 % ohne erkennbaren Trend über qo/kv (spricht fuer gleichbleibendes
  NVFP4-Quantisierungsrauschen statt eines sich aufschaukelnden Bugs).
  Ergebnis gesichert: `results/nvfp4-splitkv-ladder-512-256-vo0-2026-09-05.json`.
  **Das ist die erste direkte Messung der 512/256-Shape, fuer die jethacs
  Gate ueberhaupt existiert** — sauber negativ (keine Korruption gefunden).
  (4) **`vo_chunk=1` (zweite Haelfte des 512-breiten V) nachgezogen** (Tim,
  22:19 UTC, gleicher Weg): ebenfalls **36/36 `corrupt: false`, kein NaN**,
  `max_rel` 0.005208–0.005263, `frac_gt_1e2` 11.7–17.7 % — praktisch
  deckungsgleich mit chunk 0. Damit ist die **gesamte** 512-breite V-Seite
  abgedeckt, nicht nur eine Haelfte.
  Beide Ergebnisse + das erweiterte Skript **oeffentlich gepusht** (Tims
  Entscheid: „Script/Rohdaten oeffentlich verlinken ist immer seriös"):
  `results/nvfp4-splitkv-ladder-512-256-vo0-2026-09-05.json` (`939ad72`),
  `results/nvfp4-splitkv-ladder-512-256-vo1-2026-09-05.json` (`351c846`),
  Skript `scripts/kvsweep/nvfp4_splitkv_ladder.py` — nur diese Dateien
  gescoped committet, Handoff-Notes/Drafts/uebriger Uncommitted-Rest bewusst
  unangetastet gelassen.
  **GEPOSTET 2026-09-05 (Tims Go):** Antwort-Kommentar auf #46329 raus —
  https://github.com/vllm-project/vllm/pull/46329#issuecomment-5554547616
  (72/72 sauber, beide vo_chunks, oeffentliche Links auf Skript + beide
  Ergebnisdateien). Damit ist jethacs direkte Anfrage beantwortet; naechster
  Schritt liegt bei ihm (sein eigenes A/B auf 5090+GB10, dann ggf. PR gegen
  FlashInfer fuer den shape-scoped Gate).

- **2026-09-06 ~09:20 UTC — WICHTIG: jethac force-pusht mit GEMESSENEM
  Korrektheits-Bug — bestaetigt (und verschaerft) genau unseren eigenen
  Fix 5.** Branch `nvfp4-fa2-consumer-blackwell` neuer Head `8bc0d3b109`
  (vorher `b423d395d`, seit 09:11:39Z). Kein neuer Kommentar auf #46329 dazu
  (unser Post vom 09-05 bleibt der letzte) — der Push kam OHNE Ankuendigung
  im Thread, nur ueber Branch/PR-`updatedAt`-Bump sichtbar.
  **Neuer Commit:** „[Attention] NVFP4 KV: refuse cudagraph decode capture
  for VO-split groups on sm12x" — inhaltlich **exakt CodeRabbit-Finding 5**,
  das wir in #53979 bereits (praeventiv, ohne GPU-Beleg) gefixt hatten
  (`get_cudagraph_support()` → `NEVER` statt `UNIFORM_SINGLE_TOKEN_DECODE`
  bei VO-Split). jethac hat es **unabhaengig selbst gefunden UND auf echter
  Hardware gemessen**: Gemma-4-E2B-it/-12B-it mit kalibrierten NVFP4-KV-
  Checkpoints auf GB10 — **chat needle retrieval 0/8** unter
  `FULL_DECODE_ONLY`/Default-`FULL_AND_PIECEWISE` (captured Full-Decode-Graph
  um den per-Step-geplanten Prefill-Wrapper spielt stale Plan-Daten ab →
  Garbage-Decode), **8/8 sauber** unter `PIECEWISE`/eager/nach dem Fix.
  Prefill war schon korrekt; nur Decode unter captured Graph war kaputt.
  Head-256-Modelle (Gemma 3, Qwen) nutzen den echten Decode-Wrapper und
  waren nicht betroffen. Co-Authored mit „Claude Fable 5.1" (nicht Sonnet) —
  jethac nutzt offenbar ein anderes Modell/Setup fuer sowas.
  **Bedeutung fuer uns:** validiert unseren eigenen Fix 5 nachtraeglich als
  real und schwerwiegend (kein theoretisches Capture-Safety-Detail, sondern
  stiller Korrektheits-Bug mit garantiertem Garbage-Output). Sein Check ist
  simpler (`spec.head_size > 256` direkt) als unserer (`_vo_split_factor
  (max_head_size, True) > 1`) — beide fuer `is_fa2_nvfp4=True` funktional
  aequivalent, aber unserer ruft `_vo_split_factor` auf, das bei fehlendem
  `VLLM_NVFP4_KV_VOSPLIT` eine `ValueError` werfen kann — ein Seiteneffekt,
  den seine Version nicht hat.
  **KONSEQUENZ (Aktionspunkt fuers wache Fenster, kein Post/GPU jetzt):**
  Rebase von #53979 auf `8bc0d3b109` wird an dieser Stelle einen Duplikat-
  Konflikt geben (beide fuegen die gleiche `NEVER`-Fruehschluss-Zeile vor
  demselben `return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` ein) —
  Aufloesung vermutlich: seine (gemessene, einfachere) Version behalten,
  unsere redundante `_vo_split_factor`-Variante fallen lassen, statt „keep
  both". Braucht Tims Go vor dem naechsten Rebase/Push.
  Sonst Noop: #53977/#53978/#53979 unveraendert (`MERGEABLE`/`BLOCKED`,
  kein `ready`); #52244/#50897/#52883/#53122/#53334 keine Bewegung;
  prismaquant kein neuer Commit seit `fa229bec`.

- **2026-09-06 ~16:52 UTC — GROSSER Fund: jethac liefert die vollstaendige
  Root-Cause-Analyse fuer „Gemma-4+NVFP4-KV = Garbage", zweiter Bug
  gefunden, DIREKTE ANFRAGE an uns.** Neuer Kommentar auf #46329 (kein
  neuer Force-Push auf `nvfp4-fa2-consumer-blackwell`, weiter `8bc0d3b109`
  — der zweite Fix liegt in einem separaten PR).
  **Bug 1 (bereits gestern geloggt, jetzt mit Zahlen):** FULL-Cudagraph-
  Decode-Capture um den VO-Split-Prefill-Wrapper — Fix `NEVER` fuer sm12x-
  NVFP4-Gruppen mit `head_size > 256` (unser eigener Fix 5, unabhaengig
  bestaetigt). Tabelle: Gemma-4-E2B/-12B nvfp4 **0/8 vorher, 8/8 danach**
  auf 5090 UND GB10, durchgehend; Gemma-3-1B (Kontrolle) unveraendert
  7-8/8.
  **Bug 2 (NEU, eigener PR #55559 „Propagate quantized-KV scales to
  KV-sharing layers"):** KV-Sharing-Layer (E2B's letzte 20 von 35 Layern
  lesen Layer 13/14s Cache) dequantisieren mit Scale 1.0 statt der
  kalibrierten Werte (k~0.2, v~2.7), weil der Checkpoint fuer die
  Sharing-Layer selbst keine Scales hat (keine eigenen K/V-Projektionen).
  FlashInfer faltet `k_scale` in den Softmax-Scale ein → 20 von 35 Layern
  liefen ~5x zu scharf. Degradiert nur Qualitaet statt zu kollabieren
  (Per-Layer-Fehler vs. bf16 springt bei Layer 15 von ~0.2 auf 0.4-0.9),
  deshalb hat der simple „capital of France"-Smoke-Test das nie gefangen.
  **Wichtiger Nebenbefund:** die oeffentlichen „NVFP4"-Gemma-4-Checkpoints
  auf dem Hub haben GAR KEINE KV-Scales — mit `--kv-cache-dtype nvfp4`
  produzieren die IMMER Garbage, unabhaengig von diesen Fixes (Log zeigt
  `Using KV cache scaling factor 1.0`). Braucht einen mit 4-bit
  `kv_cache_scheme` kalibrierten Checkpoint.
  Repro-Repo mit fertigen Wheels (sm120 x86_64, sm120+sm121 aarch64, PR-
  Head + #55559 cherry-gepickt): https://github.com/jethac/vllm-gemma4-nvfp4-kv-repro
  **DIREKTE ANFRAGE AN @TechPrototyper (+ @hclsys, @maxpla3):** „if you run
  the probe against a calibrated E2B on your hardware I'd like to see the
  result either way." Split-KV-Gate-Verschaerfung von gestern steht separat
  weiter — er will dafuer noch einen FlashInfer-PR aufmachen (nicht das,
  was Gemma-4 kaputt gemacht hat).
  **Konsequenz fuer uns:** zwei offene Fragen fuers wache Fenster (kein
  GPU/Post jetzt, Watch read-only) — (1) Rebase-Konflikt-Aufloesung fuer
  Bug 1 wie gestern notiert, jetzt zusaetzlich (2) ob/wie wir auf die
  direkte Probe-Anfrage reagieren (Repro-Skript laufen lassen, mit
  kalibriertem E2B-Checkpoint — haben wir einen? AQUA-Checkpoints sind
  i.d.R. Qwen3.8, nicht Gemma-4). Beides braucht Tims Go.
  Sonst Noop: #53977/#53978/#53979/#52244/#50897/#52883/#53122/#53334
  keine weitere Bewegung; prismaquant nur laufende Commits, kein neuer
  Release.

- **2026-09-07 ~00:07 UTC — jethac force-pusht erneut, aber nur ein Test
  (keine funktionale Aenderung), kein neuer Kommentar.** Branch
  `nvfp4-fa2-consumer-blackwell` neuer Head `1a6c2ce875` (vorher
  `8bc0d3b109`). Einziger neuer Commit: „[Test] sm12x e2e: Gemma 4 NVFP4
  KV needle retrieval with cudagraphs on" — genau der e2e-Regressionstest,
  den er in seinem Root-Cause-Post angekuendigt hatte, reine Testdatei
  (`tests/v1/e2e/general/test_gemma4_nvfp4_kv_sm12x.py`, 100 Zeilen, keine
  Aenderung an `flashinfer.py`). Serviert standardmaessig seinen eigenen
  oeffentlichen kalibrierten Checkpoint `jethachan/gemma-4-E2B-it-NVFP4KV-
  calib` (ueberschreibbar via `VLLM_TEST_GEMMA4_NVFP4KV_MODEL`), prueft
  Needle-Retrieval bei Filler-Laengen 0/50/200/800 Woertern.
  **Beantwortet unsere gestrige offene Frage 2 teilweise:** wir muessen
  keinen eigenen Gemma-4-Checkpoint kalibrieren — jethacs
  `gemma-4-E2B-it-NVFP4KV-calib` ist oeffentlich und genau fuer diesen Test
  gedacht; koennte auch fuer seine Probe-Anfrage direkt genutzt werden.
  Sonst Noop: #53977/#53978/#53979 unveraendert `MERGEABLE`/`BLOCKED`;
  #52244/#50897/#52883/#53122/#53334 keine Bewegung; prismaquant kein
  neuer Release.

- **2026-09-07 ~02:47 UTC — jethac schliesst den Gemma-4-NVFP4-Strang ab:
  komplette Modellfamilie gruen, FlashInfer-PR fuer den Split-KV-Gate ist
  jetzt oben.** Neuer Kommentar auf #46329 (kein weiterer Force-Push,
  Branch weiter `1a6c2ce875`).
  **Family-Re-Run:** RTX PRO 6000 Blackwell (sm120, 96 GB), Wheel von
  `a421d4a13` (PR-Head + #55559), Checkpoints vor Ort mit llm-compressor
  kalibriert, gleiche Needle-Probe (0/50/200/800 Filler-Woerter):
  Gemma-4-E4B-it **8/8**, Gemma-4-31B-it **8/8**, Gemma-4-26B-A4B-it (MoE)
  **8/8** — jeweils gleichauf mit der bf16-KV-Referenz. Zusammen mit den
  E2B/12B-Zahlen von 5090+GB10 aus dem vorigen Kommentar ist damit **die
  gesamte Gemma-4-Linie gruen unter Default-Config, auf drei verschiedenen
  sm120/sm121-Karten**; 26B-A4B (MoE) brauchte keine MoE-seitigen Aenderungen.
  Serve-Logs zeigen durchgehend `cudagraph_mode=PIECEWISE` — die neue Regel
  greift wie gedacht. Rohdaten im Repro-Repo unter
  `results/rtxpro6000-sm120-a421d4a13/`. Eigene Einschraenkung: Needle-Checks
  gegen bf16-Referenz, keine volle Eval-Suite — fangen genau die zwei
  Fehlermodi (Garbage-Decode, verlorene Needle), keine breitere
  Qualitaetsaussage.
  **Zwei lose Enden geschlossen:** (1) e2e-Regressionstest `1a6c2ce87`
  (von uns gestern geloggt) verifiziert gegen Pre-Fix-Head (alle 4 Needles
  verfehlt) vs. Fix-Head (besteht). (2) FlashInfer-PR fuer den
  Split-KV-Gate ist jetzt oben: **flashinfer-ai/flashinfer#4993**
  („scope the NVFP4 split-KV gate on sm120/121 to the VO-split plan
  shape") — **kein Dependency von #46329**: bringt Decode-Durchsatz fuer
  symmetrische Head-Modelle mit NVFP4-KV, Gemma 4s 512-breite Heads
  bleiben so oder so gegated.
  **Fuer uns:** die direkte Probe-Anfrage von gestern ist damit im
  Kern beantwortet (er hat selbst schon die volle Familie durchgemessen,
  inkl. groesserer Karte als unsere) — unsere eigene Reaktion darauf ist
  jetzt optional/nice-to-have statt noetig fuer seine Entscheidung.
  #4993 (FlashInfer) ist neu auf der Watch-relevanten Liste, falls
  interessant fuers naechste Fenster (nicht in der urspruenglichen
  Watch-Liste, nur beilaeufig erwaehnt).
  Sonst Noop: #53977/#53978/#53979/#52244/#50897/#52883/#53122/#53334
  keine weitere Bewegung; prismaquant kein neuer Release.

- **2026-09-07 ~06:40 UTC — SEHR WICHTIG: jethac hat unsere eigenen
  CodeRabbit-Fixes fuer #53979 (weitgehend) selbst nachgebaut, direkt auf
  seiner Basis, PLUS zusaetzliche Fixes, die wir nie gemacht hatten.**
  Branch `nvfp4-fa2-consumer-blackwell` erneut force-gepusht, neuer Head
  `d3c77077a1` (vorher `1a6c2ce875`). Kein neuer Kommentar auf #46329 dazu
  (nur ueber den Push sichtbar). Drei neue Commits (History umgeschrieben,
  auch der e2e-Test von gestern hat jetzt einen neuen Hash `24068b11c5`):
  1. **„Address review: VO-split cudagraph scope, mm-prefix rejections,
     NVFP4 scale validation"** — fuenf CodeRabbit-Findings, explizit
     „Raised in review by @coderabbitai" (CodeRabbit hat offenbar AUCH
     seine PR/seinen Branch reviewt). Darunter: **unser Fix 3** (DCP/Sink-
     Rejections liefen im `mm_prefix_enabled`-Block NACH dem Vision-Reset —
     Flag zuerst neu pruefen, wortgleiche Diagnose wie unsere), **unser
     Fix 1** (DeviceGuard vor `get_device_prop()`, fast identische
     Formulierung), sowie **DREI Findings, die WIR NIE gemacht haben:**
     `_mm_prefix_prefill_spans` liest das deprecated `seq_lens_cpu` vor dem
     eigenen Guard → erzwingt einen host sync, der unter
     `VLLM_GPU_SYNC_CHECK=error` crasht; compressed-tensors akzeptierte
     `ATTN_HEAD`-Scale-Strategie fuer NVFP4-KV (`num_bits=4`), obwohl der
     Store-Kernel nur EINEN Scalar pro Seite liest → jeder Head haette
     Head-0s Scale bekommen; `swizzle_scale_offset()` brauchte
     `scale_dim % 4 == 0`, was nur indirekt ueber `block_size` geprueft war
     — `head_size=80` haette in einen fremden Head/Page geschrieben,
     `head_size<64` durch Null geteilt (SM12x/lineares V-Scale-Layout
     ausgenommen).
  2. **„[Test] sm12x e2e"** — derselbe Test wie gestern, nur umgehasht.
  3. **„Confine the VO split to the NVFP4 path"** — **exakt unser Fix 6**,
     inhaltlich fast wortgleich zu unserer eigenen Begruendung: der
     `HEAD_DIM_VO<=256`-Guard in FlashInfers FA2-Kernel ist eine
     FP4-spezifische Smem-Spezialisierung, 16-bit/FP8 erreichen den
     CTA_TILE_Q=32-Pfad und brauchen HEAD_DIM_VO>=512 gar nicht gesplittet
     — also `_vo_split_factor()` auf `is_fa2_nvfp4` gaten statt nur auf
     `head_size`. Damit zieht sich auch der Cudagraph-`NEVER`-Check (den er
     in Commit 1 kurz auf ALLE Dtypes verbreitert hatte) automatisch wieder
     auf NVFP4 zusammen. Co-Authored mit „Claude Opus 5 (1M context)".
  **Konsequenz fuer #53979 (Aktionspunkt, kein GPU/Rebase jetzt):** wenn
  wir jetzt rebasen, wird ein Grossteil unseres eigenen Fix-Commits
  vermutlich zum No-Op/Konflikt, weil jethacs Basis unsere Fixes 1/3/6
  bereits (in praeziserer Form) enthaelt. Nur unser Fix 4 (Sliding-Window-
  Guard) und Fix 5 (urspruengliche Cudagraph-NEVER-Version, jetzt eh durch
  seine bessere Version ersetzt) und Fix 2 (Test-Import-Guard) bleiben
  vermutlich als echter eigener Beitrag uebrig. Sorgfaeltiger Soll/Ist-
  Vergleich vor dem naechsten Rebase noetig — Tims Go einholen.
  Sonst Noop: #53977/#53978/#53979 unveraendert `MERGEABLE`/`BLOCKED`;
  #52244/#50897/#52883/#53122/#53334 keine Bewegung; prismaquant kein
  neuer Release.

- **2026-09-07 ~xx:xx UTC — AKTION ausgefuehrt (Tims Go): #53979 sauber auf
  jethacs Head `d3c77077a1` rebast, Soll/Ist-Abgleich per echtem Probe-
  Rebase statt nur Analyse.** Klarstellung vorab (fuer Tim, nicht nur
  Watch): **#53977 und #53978 haengen NICHT an jethacs Branch** — reine
  unabhaengige Bugfix-PRs (VocabParallelEmbedding-OOB bzw. DFlash2-Warmup-
  OOB), die nur zufaellig mal zeitgleich `needs-rebase` durch normale
  main-Drift hatten. Nur **#53979** sitzt auf #46329.
  **Probe-Rebase-Ergebnis (4 Konflikte, alle real geloest, nicht nur
  behauptet):** Fix 1 (DeviceGuard), Fix 3 (mm_prefix-Reihenfolge), Fix 5
  (Cudagraph-NEVER), Fix 6 (VO-Split-Gating) **fliegen raus** — jethacs
  Basis hat sie bereits, teils praeziser (seine Cudagraph-Version ohne
  `_vo_split_factor`-Seiteneffekt). Fix 2 (Test-Import-Guard) und Fix 4
  (Sliding-Window-Guard, gehoert zu Code, den nur wir eingefuehrt haben)
  **bleiben** als echter eigener Beitrag. Diff auf #53979 geschrumpft von
  3 Dateien/57 Zeilen auf 2 Dateien/34 Zeilen. `ast.parse`+`ruff check`
  gruen, gepusht (`--force-with-lease`), neuer Head `7ad10effcf`,
  `MERGEABLE`/`BLOCKED`.
  **Deploy-Frage geklaert (Fork-Recherche in unseren eigenen Notizen):**
  unsere Produktion (Qwen3.8-27B/DFlash2/NVFP4-KV, `vllm-prod-nvfp4kv`
  auf Spark, Image `vllm-sm121:f4c27c0da`) haengt an KEINEM Teil dieses
  Strangs — sie hat den non-causal-NVFP4-Wrapper schon seit 08-22 als
  **eigenen privaten Carry-Patch** ("the seam", Commit `952ed2fe2`),
  unabhaengig vom oeffentlichen PR-Merge-Status. Der komplette Gemma-4-
  VO-Split-Bug-Komplex (Cudagraph-Capture, KV-Sharing-Scale) ist fuer uns
  strukturell irrelevant — wir servieren kein Gemma-4, nur Qwen3.8 (kein
  head_size>256). **Entscheidung: kein Redeploy/Rebuild noetig, Prod
  bleibt unangetastet auf dem stabilen `f4c27c0da`-Stand;** Umstellung auf
  die offizielle Upstream-Version erst spaeter, rein als Wartungs-
  Vereinfachung, wenn #46329/#53979 gemerged und jethacs Branch sich
  beruhigt hat.
  Sonst Noop: #52244/#50897/#52883/#53122/#53334 keine Bewegung;
  prismaquant kein neuer Release.

- **2026-09-07 ~20:xx UTC — Watch-Tick, kein Force-Push, ein neuer
  CodeRabbit-Review bereits triagiert:** jethacs Branch hat einen
  weiteren Commit (`ae449f10f6`, "record the measured reason bf16/FP8
  stay unsplit") — reiner Kommentar/Doku-Commit, baut linear auf
  `d3c77077a1` auf, **kein Force-Push**, unsere Rebase-Basis fuer
  #53979 (`7ad10effcf`) bleibt gueltig. Auf #53979 kam heute 07:39 UTC
  ein neuer CodeRabbit-Review (1 actionable comment, `get_cudagraph_support`-
  Gating praeziser auf `kv_quant_mode.is_nvfp4 && head_size>256` statt
  implizit) — das ist der Kommentar, den Tim bereits erfragt hat; bereits
  mit ihm besprochen: Minor/Performance, betrifft jethacs geerbten Code,
  kein Handlungsbedarf.
  Sonst Noop: #53334 weiterhin keine TDHSC-Antwort (letzter Kommentar
  2026-09-01); #53122 weiterhin `MERGEABLE`, 4 Reactions, kein Maintainer-
  Review; #52244/#50897/#52883 weiterhin `OPEN`/`CONFLICTING` ohne
  inhaltliche Bewegung; prismaquant normale Dev-Aktivitaet (PR #344
  gemergt), kein neuer Release seit v0.16.2.

- **2026-09-08 ~12:16-13:16 UTC — #46329: offener Lead aus Issue #53334
  aufgeklaert (kein TDHSC, sondern hclsys/AbdulrahmanHashem).** hclsys
  erklaert Abdulrahmans "<40 t/s"-Einwand oeffentlich als denselben
  Split-KV-Gate-Bug, den jethac bereits am 2026-09-03 root-caused hatte
  (Kommentar 5519478703, bisher nicht vollstaendig erfasst): auf sm12x ist
  der trtllm-gen-Decode-Kernel unter NVFP4 gesperrt -> MTP-Verify-Batches
  (q_len 2-9) fallen auf den FA2-`BatchPrefill`-Pfad -> dort greift
  `_nvfp4_kv_requires_disabled_split_kv` und schaltet Split-KV zwangsweise
  ab -> Verify-Pass streamt die komplette KV aus einer CTA-Zeile, Decode
  bricht mit wachsendem Kontext linear ein. Abdulrahmans eigenes A/B ohne
  MTP (NVFP4 41 t/s vs. FP8 38 t/s) bestaetigt sauber: KV-Pfad selbst
  intakt, ganze Luecke sitzt im MTP-Verify-Weg. **Unser eigener A/B-Sweep
  wird von hclsys explizit als Beleg fuer die Gate-Aufhebung zitiert**
  ("TechPrototyper's A/B sweep ... found no corruption anywhere, which is
  reasonable evidence the gate can go, but that's jethac's call") — reine
  Fremdwirkung, kein Zutun unsererseits. Noch kein gemergter Fix; Stand
  laut hclsys: "today it's one or the other: nvfp4 KV without MTP, or fp8
  with MTP". #53334 damit aus Watch-Sicht erledigt (Lead identifiziert,
  keine offene TDHSC-Spur mehr zu verfolgen). Kein Post (Watch read-only).

- **2026-09-08 ~18:19-19:32 UTC — WICHTIG: main ist unter #46329/#53979
  weggedriftet, betrifft direkt unsere eigene "Fix 1"-Historie
  (DeviceGuard/Writer), Split-Vorschlag von TensorRaya laeuft.**
  PR #55908 (mgoin) ist heute in main gelandet und haengt `test_cache.py`
  in einen bestehenden B200-CI-Job — dadurch wird ein bisher nur in
  Threads bekannter Bug (#50084) jetzt in vLLMs eigenem Test sichtbar:
  `test_reshape_and_cache_flash[nvfp4]` ist auf sm120 gegen aktuellen main
  rot (V-Assert, ~11% Elemente falsch, NaN als groesste Abweichung —
  Swizzled-Write/Linear-Read-Signatur). TensorRaya hat das auf einer
  RTX PRO 4500 (sm120) verifiziert: Stock-Writer 24 failed/0 passed, mit
  jethacs V-Scale-Gate (`get_device_prop()->major < 12`, DeviceGuard vor
  die Query gezogen — **exakt unsere eigene, im Rebase gedroppte "Fix 1"**,
  jethac hat sie unabhaengig genauso implementiert) 24 passed. Mergify hat
  #46329 wegen main-Drift `needs-rebase` gesetzt, das schlaegt auf unser
  gestapeltes #53979 durch (jetzt ebenfalls `needs-rebase`/`CONFLICTING`).
  **hclsys reproduziert das exakt auf sm121/GB10 (unsere Architektur!):**
  gegen main `1a522b6949` (inkl. #55908), GB10/aarch64/CUDA 13.0,
  `TORCH_CUDA_ARCH_LIST=12.1` — main pur: 24 failed/0 passed; nur
  `nvfp4_kv_cache_kernels.cu` aus #46329 rein: 24 passed. Hat zusaetzlich
  geprueft, dass die Datei isoliert kompiliert (kein Dispatch-Signatur-
  Wechsel, main-Caller unberuehrt) und dass #55908s
  `v_scales_swizzled = not is_device_capability_family(120)` fuer sm121
  schon korrekt ist (kein Extra-Fall fuer GB10 noetig).
  **TensorRaya schlaegt vor, den Writer-Fix als eigene, kleine Bugfix-PR
  ("Fixes #50084", ~30 Zeilen, ein File, Test schon im Baum) aus #46329
  herauszuloesen** — Grund: #46329 hatte seit drei Monaten keinen
  Maintainer-Kommentar, vermutlich wegen der Groesse, nicht des Inhalts.
  Zwei Optionen an jethac gestellt (er macht's selbst, oder TensorRaya
  macht's mit ihm als Co-Author) — **Deadline "morgen Abend CEST"**, sonst
  macht TensorRaya es eigenmaechtig. Fuer uns: das ist im Kern unsere
  eigene "Fix 1", die wir beim 09-07-Rebase zugunsten jethacs Version
  gedroppt hatten — falls die Herausloesung kommt, koennte dieser Teil
  deutlich schneller mergen als das restliche #46329-Paket. Naechster
  Watch-Tick: pruefen, ob die neue Standalone-PR aufgetaucht ist. Kein
  Post (Watch read-only, betrifft nicht direkt unseren eigenen PR-Inhalt).
  Sonst noop: #52244/#50897/#52883 weiterhin ohne inhaltliche Bewegung
  (nur main-Drift-Status-Rauschen); #53122 unveraendert; jethac-Branch
  kein neuer Commit seit 09-07 (kein Force-Push); prismaquant normale
  Dev-Aktivitaet, kein neuer Release seit v0.16.2.

- **2026-09-08 ~22:52 UTC — Vorhersage aus dem letzten Tick eingetroffen:
  jethac hat die Standalone-Writer-Fix-PR selbst eroeffnet.** Antwort auf
  TensorRayas Vorschlag: "Option 1 — I'll open it." → **PR #55976
  "[Bugfix][Kernel] NVFP4 KV: write V block scales linearly on
  SM120/SM121"**, nur `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`,
  Label nur `bug` (nicht in die #46329-Themengruppe verstrickt) — genau
  unsere eigene, beim 09-07-Rebase gedroppte "Fix 1", jetzt als fokussierte
  Einzel-PR mit sm120- UND sm121/GB10-Daten im Rueckenwind. Sollte deutlich
  leichter/schneller zu reviewen sein als das #46329-Gesamtpaket. Kein
  Force-Push auf jethacs Feature-Branch (weiterhin `ae449f10f6`) — die neue
  PR ist ein separater Branch/PR, nicht Teil von #46329 selbst.
  Sonst noop: #53977/#53978/#53979/#52244/#50897/#52883/#53122 keine
  inhaltliche Bewegung ueber main-Drift-Rauschen hinaus; #53334 weiterhin
  ohne neue TDHSC-Antwort (letzter Kommentar unveraendert 09-01); prismaquant
  normale Dev-Aktivitaet (Push + PR + Issue-Kommentar), kein neuer Release.

## 2026-09-04 — INCIDENT: LiteLLM-Kill-Loop nach Virtual-Key-DB-Anbindung (Fable, selbstverschuldet verlängert)
Auslöser: Tims Multi-Session-Report + Wunsch nach mehr Speed. Kette von
Fehlern MEINERSEITS, ehrlich protokolliert:
- **Root-Cause des „Totalversagens":** Der Virtual-Key-Commit (dad3e41,
  DB + store_model_in_db) verlangsamt LiteLLMs Startup so, dass die alte
  Liveness-Probe (6×20s=120s) den Pod mitten im DB-Init killt → Kill-Loop →
  sporadisch 503/502/connection-refused für ALLE Routen. NICHT die GPUs.
  Fix: Probes gehärtet (liveness 30×30s, readiness 10×15s, startup 40×10s,
  timeout 5→10) — Commit 93931f1. **litellm jetzt 1/1, R=0, stabil.**
- **Timeout-Kaskade (real, gefixt):** lange non-streaming Requests → 502
  (Traefik) → allowed_fails=1 warf Gruppe in Cooldown → 503 für alle.
  Fix: allowed_fails 1→20, cooldown 30→5, request_timeout→1200, Traefik
  ServersTratsport litellm-longtimeout 1200s (d059290).
- **MEIN OOM-FEHLER:** util 0.90→0.97 gesetzt (Tims KV-Pool-Wunsch) →
  CUDA-OOM (GridBook13GB+MTP-Draft, nur 244MiB frei) → RTX CrashLoop.
  Zurück auf 0.92 (9dc5eb8). LEHRE: util 0.97 ist mit diesem Modell-Stack
  NICHT möglich; ~0.92 = Obergrenze (~570k KV-Pool). Der „Endgegner-775k"
  galt für ein anderes Setup (ohne MTP-Draft-Modell im VRAM).
- **STREAMING über LiteLLM noch KAPUTT (offen):** non-streaming 200/0,5s,
  streaming Timeout. iterator-hook-Fix (async+sync) half NICHT → Ursache
  tiefer (LiteLLM-Core/custom_auth/Traefik-SSE-Buffering). Hermes streamt →
  betroffen. NÄCHSTER SCHRITT: callbacks isolieren, dann Traefik-SSE prüfen.
- **Prozess-Lehre (Tim-Feedback sinngemäß):** nach JEDEM Fix stabilisieren +
  verifizieren, DANN erst nächste Änderung. Nicht mehrere Rollouts stapeln.
- **GPUs die ganze Zeit gesund:** RTX 81 tok/s / Spark 60 tok/s.

## 2026-09-02 — RUNDUMSCHLAG: Cherry-Picked-vLLM auf neue Basis (Tims Auftrag, Fable)
**Alles auf jethacs rebasten Head `9590d2464` (= main `aae36577f` + 10 NVFP4-Commits)
hochgezogen.** Arbeits-Repo: `~/Projects/vllm-upstream-work`.
- **Neue Integrationslinie `integrate/v5-9590d246`** (ersetzt das v4-Merge-Gebilde):
  jethac-Head + unsere 5 Patches **linear** (VPE-mask → embed-outside → clamp →
  non-causal-Wrapper [wandte sich sauber an!] → SWA-Guard) + **pr-50897-Merge**
  (57 Konflikt-Hunks über 20 Dateien gegen 3 Wochen Upstream-Drift aufgelöst;
  alle Entscheidungen + 3 Review-Punkte im Merge-Commit `2280d0237` dokumentiert;
  Kernpunkte: `num_prefill_lookahead`∪`use_eagle_prefix_cache_hashing`,
  Block-Drop-Gate komponiert, `prefer_cross_layer_blocks` überall gestrichen
  [upstream #51718 entfernte die API], Mooncake-Job-Ledger upstream-seitig,
  Eagle-Finish-Capture nach `capture_finished_eagle_save`/`request_finished_all_groups`
  umgehomet, 2 Mooncake-Eagle-Tests transparent @skip). ruff check+format clean (54 Dateien).
- **Seam-PRs rebast & gepusht:** #53977 → `d726b9a15` (VPE-Fix in den neuen
  tp==1-Early-Return verlegt; range-diff zeigt exakt die Relokation),
  #53979 → `71a1133e8` (range-diff `=`, byte-identisch). **Beide wieder MERGEABLE.**
  #53978 bewusst NICHT angefasst (kein needs-rebase, Review-Zustand erhalten).
- **Nicht validiert:** GPU-Batterie auf v5 steht aus (Image-Build kaniko + Batterie,
  supervised; RTX-Regel beachten). CPU-Tests auf dem Mac nicht machbar
  (vLLM-Dep-Kette); auf Cluster-CPU nachziehen (jethacs "CPU box"-Flow).
- **Kommentar-Entwürfe** (#53977/#53979-Rebase-Notes + optionales ZJY0516-Fold-in):
  `notes/draft-pr-rebase-comments-2026-09-02.md` — **posten nur mit Tims Go**.
- **Nächste Schritte:** (1) sm120-Image v5 bauen (kaniko), Batterie; (2) Spark-
  Container analog; (3) nach grün: Kommentare posten + ggf. ZJY0516-Angebot.
- **2026-09-02 ~14:39 UTC — #53977/#53979:** mergify hat die `needs-rebase`-Labels
  **entfernt** — die 09-02-Rebases (d726b9a15 / 71a1133e8) sind vollständig
  anerkannt; #53979 MERGEABLE, #53977 in Neuberechnung. Rebase-Bogen geschlossen;
  weiter `REVIEW_REQUIRED`, kein `ready`. Kein Post (Watch read-only).

### RTX-Build-Stand (Task: v5-Container)
- **Build 3 (`vllm-nvfp4-build-1788365728`) FAILED nach ~61 min** — Ursache jetzt
  eindeutig: **GitHub rate-limitet anonyme git-Klone von der Cluster-NAT-IP**
  (`could not read Username` = 401-Prompt). Diesmal traf es CMakes
  FetchContent-Clone von `nvidia/cutlass` (3 Schnell-Retries ohne Pause, alle
  gescheitert); unser vLLM-Fetch mit 5×30s-Retry kam dagegen DURCH. Build 2
  (52 min) war dasselbe Muster am vLLM-Fetch — kein Flake, systematisch.
- **Fix in `build-vllm-nvfp4-v5.sh` (credential-frei):** cutlass +
  vllm-flash-attn werden vorab mit Retry-Schleifen (6×60s) geklont, Tags
  dynamisch aus dem Quellbaum (`CUTLASS_REVISION` / `GIT_TAG`), dann via
  offizieller vLLM-Hooks `VLLM_CUTLASS_SRC_DIR`/`VLLM_FLASH_ATTN_SRC_DIR`
  eingespeist (Gate: nur wenn `.git` existiert, sonst FetchContent-Fallback).
  Token-Variante bewusst verworfen (Tims gh-Token in Pod-Spec/Zwischenlayern).
- **Builds 4+5 (Tims Go via `!`): ebenfalls Rate-Limit-Tote.** Build 4: cutlass
  trotz 6×60s-Retries zu (Drossel klebt am Repo); Build 5: jetzt auch
  jethac/vllm gesperrt — GitHub hat nach den Retry-Wellen **git-über-HTTPS von
  der Cluster-IP komplett dicht gemacht**. codeload-HTTPS dagegen offen
  (von der Cluster-IP verifiziert: HTTP 200 fürs cutlass-Archiv).
- **DOPPELFUND dabei: Builds 3–5 bauten die FALSCHE QUELLE** — Script-Defaults
  zeigten auf jethac@9590d246 statt unserer v5-Linie; zudem war `2280d0237`
  ein Zwischenstand, der finale Merge-Head ist **`fe3247505`** (auf Fork
  verifiziert). Korrektes Tag: `sm120-v5-fe32475`.
- **Build 6 läuft (`vllm-nvfp4-build-1788373082`): vollständig git-freier
  Build.** Alle Quellen als gepinnte codeload-Tarballs (vLLM fe3247505 mit
  `VLLM_VERSION_OVERRIDE=0.26.1rc1.dev0+gfe3247505` für setuptools_scm ohne
  .git; cutlass v4.4.2 via `VLLM_CUTLASS_SRC_DIR`; vllm-flash-attn 06bdd47c
  + csrc/cutlass-Pin 62750a2b via `VLLM_FLASH_ATTN_SRC_DIR`, Pin-Drift-Check
  bricht hart ab; flashinfer 00054844 + 3rdparty-Pins b46b16d0/87686768/
  c3aed4b6/05e4243f). Alle SHAs authentifiziert vom Mac aufgelöst.
  RTX bleibt geflaggt (RTX_DOWN=1).
- **✅ ERFOLG 09-02 ~19:50 UTC — RTX LÄUFT AUF v5.** Build 8 komplett
  (`sm120-v5-fe32475`; Builds 6/7 scheiterten an übersehenen FetchContent-
  Externals → alle 7 [triton_kernels/flashkda/qutlass/tml-fa4/MSA/FlashMLA/
  DeepGEMM] + Submodule als Tarballs vorgeladen; Lektion:
  `TRITON_KERNELS_SRC_DIR` muss aufs Python-Paketverzeichnis zeigen, nicht
  auf die Repo-Wurzel). GridBook-088-Overlay gebaut, Deployment auf
  `sm120-v5-fe32475-gridbook-088`, Tests grün: MTP aktiv (Akzeptanz 2,69),
  GridBook geladen, greedy deterministisch, 95 tok/s, Version
  `0.26.1rc1.dev0+gfe3247505`. **RTX_DOWN=0**, LiteLLM-Route verifiziert
  (Fingerprint = v5-Engine). Imperative Drifts für GitOps-Reconcile:
  Image-Tag, CPU-Request 4→3 (firecrawl nahm heute 750m), Power-Cap-Gate
  570→550 (YAML gefixt, Live-Objekt folgt beim nächsten Rollout).
  Spark-Rebuild (Task 5) offen — sm121-Weg braucht dieselbe
  Tarball-Behandlung (gleiche NAT-IP, gleiche git-Sperre).

### Spark-v5-Rollout (Nacht 09-02/03) — Befund DFlash2-Draft-Loader
- Build auf GB10 erfolgreich (Image `vllm-sm121:v5-fe3247505`, editable
  /vllm-src+/fi ersetzt, torch 2.13 aarch64, cubin-Fix per docker commit;
  Sanity: vllm 0.26.1rc1.dev0+gfe3247505 | flashinfer 0.6.18).
- **BEFUND: DFlash2-Draft-Load bricht auf der v5-Basis** — Target lädt
  sauber (5/5 Shards), aber beim Draft (`qwen3.8-27b-dflash2-fp8`,
  compressed-tensors): `AttributeError: 'MergedColumnParallelLinear' object
  has no attribute 'data'` in `linear.py:753 weight_loader` via
  `dflash/utils.py:42 load_dflash_model` → Upstream-Drift im Loader-Pfad
  vs. Robs Draft-Format (3 Wochen Drift; NICHT unsere Patches — generischer
  Lade-Pfad). Container crash-loopte (5×). **Tagesaufgabe:** Loader-Drift
  bisecten; relevant auch für jethac/#53978-Kontext.
- ~~Nacht-Pragmatik: v5 OHNE Spec-Decode~~ → **VON TIM GEKIPPT (09-03
  ~00:30): ROLLBACK auf den alten f4c27c0da-Stack.** 10 tok/s spec-los vs.
  30–40 alt = „absolut nicht tragbar, klares no go"; Lektion als Memory
  `performance-degradation-ist-rollback-kriterium` verankert (Degradation
  dokumentieren ≠ akzeptieren). Ablauf: Sweep pausiert (Teildaten 1k–64k
  gesichert), RTX_DOWN=0 (RTX trägt, verifiziert), Drain-Watcher wartet auf
  Spark-Leerlauf (Agent lief noch) → dann docker-Swap zurück auf
  `vllm-prod-nvfp4kv` (alt). **v5-Image `vllm-sm121:v5-fe3247505` bleibt
  liegen**; Re-Rollout nur mit DFlash-Fix + Durchsatz ≥ alt (Task #9).

### Spark-Incident 09-02 ~18:00 UTC (Prod-Ausfallfenster ~15 min)
- `vllm-prod-nvfp4kv` auf gx10 (10.1.0.229) crashte mit
  **`CUBLAS_STATUS_INTERNAL_ERROR`** in einem bf16-GEMM (`cublasGemmEx`,
  CUDA_R_16BF) — mitten in einem **64k-Token-Long-Context-Request mit
  DFlash-Spec-Decode (num_spec_tokens=7)**, kv_cache_usage nur 14%.
  Docker-Restart (unless-stopped) + Modell-Reload ≈ 15 min = Tims
  Fehlerfenster (RTX_DOWN=1, kein zweiter Failover).
- Erster Crash in ~2 Tagen Laufzeit; keine Xid-Einträge im Kernel-Log,
  GPU 39°C, Requests laufen wieder; **Failover-Kette end-to-end verifiziert**
  (LiteLLM `qwen3.8-27b` → overflow → Spark antwortet).
- **Merkposten:** sporadisch, aber Long-Context+DFlash ist genau unsere
  MTP/DFlash-Beobachtungsachse — bei Wiederholung Muster ernst nehmen
  (Async-CUDA-Fehler können vom eigentlichen Kernel wegzeigen).

### Loop-Breaker-Dauerfeuer-Bug (CP-2) — Befund 09-02 abends, Fix gestaged
- **Symptom (Tim):** Tool-Loops weg, aber Agent macht gar nichts mehr — nur
  noch identische Status-Reports („Verstanden. Hier der Stand… EIN nächster
  Schritt…"), auch unter /goal; sagt wörtlich „kein Tool mehr".
- **Root Cause (verifiziert im Code):** Nach einer Intervention antwortet das
  Modell mit Text; Hermes sendet aber weiter die volle Historie inkl. alter
  Loop-Region. `_detect_loop` arbeitet auf der GEFILTERTEN tool_call-Turn-
  Liste — deren Tail bleibt für immer der alte identische Run → Breaker feuert
  bei JEDEM Folgerequest, `tool_choice="none"` dauerhaft, `_DIRECTIVE`
  („Fasse zusammen… nenne den EINEN Schritt") wird jedes Mal injiziert —
  das Modell spiegelt sie wortwörtlich. Agent ist dauerhaft tool-los.
- **Fix:** Wächter `_loop_is_active()` — Breaker feuert nur, wenn der JÜNGSTE
  Assistant-Turn ein tool_call-Turn ist (Loop läuft JETZT). Offline-Replay
  4/4 grün (aktiv-identisch ✓ feuert, stale+Text ✗, legitime Arbeit ✗,
  aktiver Zyklus ✓). **DEPLOYED 09-02 ~19:51 UTC mit Tims Go** (CM
  `litellm-config-v5fb2`, Rollout sauber) + **E2E-verifiziert in Prod:**
  Positiv-Canary (aktiver 3×-Loop) → INTERVENTION geloggt, Header
  `x-litellm-loop-intervention` emittiert, Text-Synthese statt Tool;
  Negativ-Canary (Tims Szenario: alter Loop + Text + neuer User-Turn) →
  KEINE Intervention, Modell ruft wieder Tools auf. Damit auch die am 09-01
  offene „End-to-End-Realprobe" abgehakt. GitOps-Konvergenz der
  loop_breaker.py-Quelle in `MyCluster/.../litellm/` steht noch aus.
- Offen bleibt (hupdate v7 / CP-Reihe): Hermes-seitiger Konsument des
  `x-litellm-loop-intervention`-Headers (Historie dauerhaft aufräumen) —
  der eigentliche Langfrist-Fix. Und Tims Wunsch: Tool-Fehlschläge
  systematisch analysieren/notieren → Skill-Deklarationen optimieren.
- Ungeklärt: 408-Byte-Truncation von test_bidirectional.py (26,5s-Write) —
  Umfeld der Intervention, Mechanismus noch nicht belegt; beobachten.
