# Handoff — End of Day 2026-08-31

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
