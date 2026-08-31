# Carry-Patch-Registry

Laufender Index aller **Carry-Patches** — unsere lokalen Modifikationen, die wir über
Updates/Upgrades hinweg forward-porten, weil sie (noch) nicht upstream sind oder
bewusst lab-spezifisch bleiben. Jede Änderung an Inferenz-Gateway (LiteLLM),
Agent (Hermes) oder Engine (vLLM) gehört hier eingetragen. Prinzip (Tim): keine Angst
vor Carry-Patches — Wartbarkeit ist Präferenz, nicht Veto gegen die optimale Lösung.

**Konvention:** Signale/Meta laufen **out-of-band** (Header/Metadatenfeld), NIE über
`msg.content` (Runaway-Lehre vom Overflow-Banner).

---

## Aktiv / gestaged

### CP-1 · LiteLLM greedy-coerce  · Layer: LiteLLM · Status: **staged**
- **Was:** kaltes Sampling auf qwen3.8 (client-gesetzt, temp≤0.1) wird aufs Anti-Loop-
  Profil (temp0.7/top_p0.8/pp1.5) gezwungen statt nur geloggt.
- **Warum:** greedy ist bei qwen3.8 das dokumentierte Endlos-Wiederholungs-Anti-Pattern.
- **Wo:** `MyCluster/…/litellm/overflow_notice.py` (`_greedy_check` → coerce im pre_call_hook).
- **Abschaltbar:** `GREEDY_COERCE=0`. **Re-Apply:** erweitert Tims 17.08.-`_greedy_check`.

### CP-2 · LiteLLM loop-breaker (Force-Progress + Debloat) · Layer: LiteLLM · Status: **staged, offline-getestet**
- **Was:** erkennt Tool-Call-Loops im Request-Tail (identisch K≥3 **oder** Zyklus
  Periode 2–5, ≥2×) → (1) `tool_choice:"none"` + harte Direktive (Zwang zum Fortschritt),
  (2) Prune der wiederholten (assistant-tool_call + tool-result)-Paare, (3) Out-of-Band-
  Signal `x_loop_intervention` für Hermes.
- **Warum:** Loops nicht mit Fehler abbrechen, sondern zum Fortschritt zwingen + den
  aufgeblähten Kontext retten. Auf Inferenz-/Ansteuerungs-Ebene, nicht im Verhalten.
- **Wo:** `MyCluster/…/litellm/loop_breaker.py` (neu). Registrieren in `config.yaml`
  `litellm_settings.callbacks`. **Abschaltbar:** `LOOP_BREAKER=0`.
- **Test:** offline grün (identisch/Zyklus/keine-False-Positives). **Offen:** Signal-
  Zustellung (Header vs. Feld) verifizieren; live an einem echten Loop-Mitschnitt testen
  vor Deploy.

### CP-3 · Hermes loop-signal-cleanup · Layer: Hermes · Status: **IMPLEMENTIERT** (Commit `eae7ea3d74`, Branch `consolidation`)
- **Was:** Hermes liest den Header `x-litellm-loop-intervention` → kollabiert die Loop-
  Turns in seiner EIGENEN Historie dauerhaft (Bloat an der Wurzel, statt jeden Turn
  neu senden + LiteLLM neu detektieren lassen).
- **Wo (aktuelle Anker, Repo war mitten in der Consolidation):** 5 Edits —
  `agent/agent_init.py` (`_loop_intervention_state=None`-Init) · `run_agent.py`
  (`_capture_loop_intervention` + im Anthropic-Header-Pfad) · `agent/chat_completion_helpers.py`
  (Streaming `_stream_created`-Wiring) · `agent/message_sanitization.py`
  (`collapse_loop_turns` — nur komplette Gruppen, erstes Vorkommen + Marker + Post-Loop-Tail,
  nie tool_call/result trennen; in `__all__`) · `agent/conversation_loop.py`
  (Consume-once, fail-open, im Post-Tool-Prune-Block). Volle Spec: [[cp3-hermes-loop-cleanup-spec]].
- **Test:** offline 17/17 grün (identisch/Zyklus/Multi-Call-Paarung/no-op-Safety).
  **Offen:** Live-Validierung gegen echten Loop-Mitschnitt mit deployter CP-2 (Header-Empfang).
- **Kontrollfluss-Caveat:** LiteLLMs `tool_choice=none` erzeugt eine Text-Antwort ohne
  Tool-Call → der Post-Tool-Block greift in genau DEM Turn evtl. nicht; Collapse ist
  fail-open/verzögerungssicher (spätestens beim nächsten Tool-Turn/Resume, CP-2 prunt ohnehin
  erneut). Kein Fehler, höchstens Verzögerung.
- **Architektur-Hinweis:** Upstream-Hermes hat seit der Spec EIGENE Identical-Call-Behandlung
  (`observe_call` Notice + Debloat-Stub, jetzt in `tool_guardrails.py` neben unserer
  Zyklen-Erkennung → keep-both gemerged). CP-2/CP-3 bleiben als **Gateway-Netz** (greifen
  modell-/clientseitig, nicht nur für Hermes) + Root-Cause-Persistenz-Hygiene. Kein Konflikt:
  Upstream *notice-t*, unser Netz *erzwingt Fortschritt* (tool_choice=none) + *kollabiert Historie*.

### CP-5 · Basis-Skill-Reconciliation (3-Wege statt Upstream-Verlust) · Layer: Hermes · Status: **IMPLEMENTIERT** (Commit `9ce660653f`, Branch `consolidation`)
- **Problem:** `sync_skills` schützte lokal editierte Basis-Skills durch *skip* — aber wenn
  Upstream **denselben** Skill verbesserte (Sync-Matrix **Fall 4**: ours≠origin **und**
  theirs≠origin **und** ours≠theirs), ging die Upstream-Innovation **still verloren**. Genau
  die Falle, die wir ausdrücklich vermeiden müssen.
- **Lösung:** Fall 4 → *ours* nie überschreiben, **gemeinsamen Vorfahr snapshotten**
  (`~/.hermes/skill-baseline/`), *theirs* stagen (`~/.hermes/skill-reconcile/`), 3-Wege-Merge
  in die Queue. `hermes skills reconcile [name] [--apply]` (+ `/skills reconcile`).
- **Merge (billigst-zuerst):** trivial (nur eine Seite geändert) → `git merge-file`/diff3 →
  **LLM** bei Konflikt/Prosa/fehlender Basis. **Review-gated Apply** (Backup, Re-Baseline,
  Manifest-Advance). **Fail-open:** ohne Backend diff3-Konfliktmarker, Apply blockiert.
- **Backend node-agnostisch** (Tims Muster: beim RTX-Update den **Spark** treiben, beim
  Spark-Update die **RTX**, am Mac Codex/Antigravity/Claude): `HERMES_SKILL_RECONCILE_BACKEND`
  = `http` (OpenAI-kompatibel, `..._URL/_MODEL/_API_KEY`) | `cmd` (Prompt via stdin) | `none`.
- **Sicherheit:** Stores liegen **außerhalb** des Skills-Baums (Loader sieht nie halb-gemergte
  SKILL.md); eigener `_rmtree_reconcile`-Guard (spiegelt den #48200-Wipe-Schutz). Self-healing:
  Queue wird pro Sync auf die aktuelle Fall-4-Menge reduziert.
- **Wichtig (Bestandsaufnahme-Korrektur):** Upstream hatte nur die **Block**-Hälfte
  (`.org-provenance`/`user_modified`-skip). Die „adopt+nachpatchen"-Hälfte existierte **nie**
  als unser Code — die Consolidation hat nichts fallengelassen. CP-5 baut sie erstmals.
  Voller Design-Kontext: [[cp5-skill-reconciliation-design]].
- **Test:** offline grün (voller Fall-4-Zyklus, LLM-Konflikt-Pfad, No-Backend-Block, Trivial,
  Self-Heal); bestehende Sync-Suite unverändert grün. **Offen:** Live-Merge gegen echten
  Upstream-Skill-Bump mit konfiguriertem Peer-Backend.

### CP-6 · Microsoft-Graph-API-E-Mail-Adapter · Layer: Hermes (Gateway-Plugin) · Status: **IMPLEMENTIERT / carried** (Commit `82ef37a4d4`, Branch `consolidation`)
- **Was:** `plugins/platforms/email/graph_adapter.py` (`GraphEmailAdapter`, OAuth2/Graph API:
  Mail.ReadWrite/Mail.Send, Polling, sendMail, Attachments, Whitelist) + `__init__.py`
  Auth-Mode-Selektor (`EMAIL_AUTH_MODE=graph|imap`). Für Tenants, wo IMAP durch MFA/Tenant-
  Policy blockiert ist (M365 Business).
- **Warum Carry-Patch:** Tims eigenes Feature, **nicht in Upstream** (origin/main hat nur den
  IMAP/SMTP-`adapter.py`). **Brauchen wir immer** (Tim) → reist als Commit auf `consolidation`
  über Rebases mit. Ursprünglich aus `hermes-agent-test1@feature/graph-api-email-adapter`
  cherry-gepickt.
- **Verifiziert:** importiert gegen consolidation, subklasst `BasePlatformAdapter`, alle
  Abstract-Methods implementiert, Interface trotz alter Basis nicht gedriftet.
- **Rein Graph, kein IMAP/SMTP/POP** (Tim 2026-08-31, „jetzt und für immer"): der ursprüngliche
  SMTP-Send-Fallback wurde **entfernt** (Commit `39b54e0432`) — Graph-sendMail-Fehler wirft laut,
  statt still auf SMTP zu degradieren. `EMAIL_AUTH_MODE=graph` aktiviert (in `~/.hermes/.env`);
  der Auth-Selektor registriert dann nur `GraphEmailAdapter`, die IMAP-`adapter.py` bleibt inaktiv.
- **LIVE seit 2026-08-31**: Deploy-Checkout `~/.hermes/hermes-agent` steht auf `consolidation`,
  Konto `hermes1_mac@sjanasek.de` (Token in `~/.hermes/mail_oauth_token.json`). Ein Re-Auth kann
  nötig werden — Tim: „wenn wir das nochmal testen müssen, so be it."
- **Achtung Lifecycle:** falls Upstream je einen eigenen Graph-E-Mail-Adapter bringt → keep-both/
  Reconcile wie bei den anderen Carry-Patches.

---

## Bestehend (Bestandsaufnahme)

### CP-0a · `-tools`/`-no-think`-Modellnamen-Routing · Layer: LiteLLM(+Hermes) · Status: **teils inert**
- **Was:** base=Chat, `-tools`=Tool-Calling, `-no-think`=Oneshot — je eigenes Sampling-Profil.
- **Befund:** der `-tools`-Auto-Swap erreicht den Hermes-Agent-Loop NICHT (er läuft immer
  base). Und `-tools` = thinking-off/pp1.5 ist laut Sweep **kontraproduktiv** → bei
  Neuverdrahtung auf **thinking-on** umkalibrieren. Siehe [[qwen38-tool-loop-rootcauses]].
- **Wo:** `litellm/config.yaml` (Profile), Swap-Logik gehört in LiteLLM (neu zu bauen).

### CP-4 · GridBook+DFlash2 auf dem Spark · Layer: vLLM/Checkpoint · Status: **EVALUIERT → VERWORFEN (Speed)**
- **Ergebnis:** funktioniert (self-serve 4-Fix-Kette statt #52883-Cherry-Pick: prefix-strip
  + contract-re-attest + BF16-embed-swap + BF16-lm_head-swap), Qualität ≈ AQUA, Whisper
  passt — **aber ~35 % langsamer** (CB-Dequant vs. AQUAs NVFP4+FP8 auf sm121). Auf dem
  bandbreiten-limitierten Spark ist Speed Spitzenkriterium nach Qualität → **nicht übernommen.**
- **Konsequenz:** Spark bleibt AQUA+DFlash2; Whisper → MacBook. Details:
  [[spark-gridbook-eval-2026-08-31]] (`notes/spark-gridbook-eval-2026-08-31.md`).
  Fix-Skripte liegen auf dem Spark, #52883 bleibt der saubere Upstream-Weg *falls* der
  Kernel-Speed-Nachteil je verschwindet.

## In der Serving-Linie (Image, kein Laufzeit-Patch)
Der „vLLM-Upstream-Package"-Seam #53977/#53978/#53979 + Prefix-Cache #50897 (pc50897)
+ GridBook-Plugin 0.8.8 sind im Image `sm120-pc50897-gridbook-088` gebacken — kein
Carry-Patch zur Laufzeit, aber Teil des reproduzierbaren Stands (siehe Rezept + Cutover-Doku).
