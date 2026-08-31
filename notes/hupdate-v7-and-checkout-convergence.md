# hupdate v7 + Checkout-Konvergenz (Stand 2026-08-31)

Kontext: `~/.hermes/bin/hupdate.sh` (v6) ist Tims Update-Wrapper. Beim Durchgehen kam die
Frage auf, ob wir ihn alteriert haben und wie der Update künftig laufen soll — im Licht der
Consolidation (Carry-Patches als git-Commits) und CP-5 (Basis-Skill-Reconciliation).

## Befunde (read-only)
- **hupdate.sh NICHT alteriert** (mtime 2. Juli). Er hatte die LLM-Skill-Merge-Logik schon —
  in **Phase 5** (nicht im Repo, daher zunächst übersehen). Aber: 2-Wege (ours-Backup vs.
  upstream-new, keine echte Basis), Modell hartcodiert `qwen3.6-27b-no-think`, klassifiziert
  nur MERGE/RESTORE, **RESTORE auto, MERGE nur als „review manuell" geflaggt** — nicht gemergt.
- **Phase 6 (Vollkopie-Tool-Patches)** ist laut `~/.hermes/tool-patches/README.md` (19.08.)
  bereits stillgelegt: „Vollkopien überschrieben neuere Upstream-Features; Änderungen gehören
  als Commit auf `current`." tool-patches-Dir enthält keine `.py`.
- **Zwei Checkouts:**
  - `~/Projects/hermes-agent` — remote nur `origin`=NousResearch. Trägt `current` **und**
    `consolidation` (origin/main + 4 Carry-Patches + CP-3 + CP-5). Frisch, clean.
  - `~/Projects/hermes-agent-test1` — remotes `fork`=TechPrototyper **+** `origin`=NousResearch.
    Auf `feature/graph-api-email-adapter`. **~5991 Commits hinter origin/main.** hupdates
    `HERMES_REPO`. Uncommitted: `tools/file_tools.py` = **906-Zeilen-Vollkopie mit den
    V4A-Beispielen** (= unser CP-3 patch-schema-Intent, längst als Commit in consolidation) →
    genau das Vollkopie-Anti-Pattern, überschreibt 666 Zeilen neueren Upstream.

## Entscheidung (Tim): (2) vor (1)
Erst die Checkout-Situation klären, DANN wird `consolidation` die Deploy-Linie.

## AUSGEFÜHRT (2026-08-31, Tims Go: Fork-Push OK / hermes-agent = Deploy-Repo / Email droppen NUR wenn Graph sauber)
- **Graph-API-E-Mail-Gate:** consolidation hatte NUR IMAP/SMTP (`plugins/platforms/email/adapter.py`),
  KEIN Graph. Also test1-Branch NICHT gedroppt, sondern der Email-Adapter-Commit `3afb6c5455`
  auf consolidation **cherry-gepickt** (`graph_adapter.py` 878 Z. + `__init__.py` Auth-Mode-Selector,
  `EMAIL_AUTH_MODE=graph`). Neuer Tip `82ef37a4d4`. Cherry-pick clean.
- **„läuft sauber" (statisch verifiziert):** importiert gegen consolidation, `GraphEmailAdapter`
  subklasst `BasePlatformAdapter`, ALLE Abstract-Methods implementiert, Interface trotz 5991-Commit-
  Lücke nicht gedriftet. Laufzeit gegen echtes M365-Postfach = Tims Bestätigung.
- **SMTP-Send-Fallback (Tim-Entscheidung 2026-08-31: BLEIBT):** verifiziert, dass er auf
  **denselben Account** geht — `msg["From"]=self._address`, `server.login(self._address,
  self._user_password)`. Nur Senden fällt zurück; Empfang ist reines Graph. Konform zu „kein
  fremder SMTP". Als **CP-6** in die Carry-Patch-Registry aufgenommen (Commit `82ef37a4d4`).
- **Fork-Push:** `fork`=TechPrototyper/hermes-agent zu `hermes-agent` hinzugefügt; `consolidation`
  → `fork/consolidation` gepusht (Tracking gesetzt). 7 Commits über origin/main.
- **Deploy-Repo:** v7-Vorschlag `HERMES_REPO` → `/Users/timw/Projects/hermes-agent` gesetzt (TODO raus).
  v7 bleibt Vorschlag; Aktivierung (`mv hupdate.v7.proposed.sh hupdate.sh`) + test1-Ablösung pendent.

## CUTOVER VOLLZOGEN (2026-08-31, Tims Go „Stell um, vorwärts")
- **v7 scharf**: `~/.hermes/bin/hupdate.sh` = v7 (v6 gesichert als `hupdate.v6.bak`). `HERMES_REPO`
  → `/Users/timw/Projects/hermes-agent`. `bash -n` grün, Banner „live seit 2026-08-31".
- **Graph-E-Mail produktiv-bewährt** (Tim): Konto `hermes1_mac@sjanasek.de`, sendet täglich,
  empfängt. consolidations `graph_adapter.py` ist **byte-identisch** zur laufenden test1-Version
  → kein Regressionsrisiko. Kein separater Smoke nötig.
- **„Nichts verlieren" verifiziert:** alle Graph-Live-Daten liegen in `~/.hermes`
  (`graph_creds.json`, `mail_oauth_token.json`, `email_whitelist.txt` — alle vorhanden), Konto ist
  secret-/config-getrieben (`EMAIL_ADDRESS`/`graph_creds.json`), nicht im Code. Secrets in
  `~/.hermes/.env`. Die **11 test1-exklusiven Nicht-Secret-Flags** (BROWSER_*/TERMINAL_*/*_DEBUG)
  wurden additiv nach `~/.hermes/.env` migriert. Backups beider `.env` in `~/.hermes/update-backups`.

## KORREKTUR (2026-08-31): der echte Deploy-Ort war ein DRITTER Checkout
Beim ersten `hupdate`-Lauf zeigte sich: der laufende Gateway läuft aus **`~/.hermes/hermes-agent`**
(Branch `current` @ 741f82f96, 39 ahead / 318 behind origin/main), NICHT aus `~/Projects/hermes-agent`.
`hermes update` managed diesen Ort und **weigerte sich, den geparkten Branch fortzuschreiben**
(by design, Commit `8ce8ffd42`). Kein Notfall — der Gateway lief; nur der Code-Update-Schritt
wurde übersprungen.

**Nichts-verlieren definitiv (Patch-ID):** die einzigen Commits im Deploy-`current`, die nicht in
origin/main sind, sind exakt unsere **3 Carry-Patches**; alle in consolidation. Der Rest
(bot-mode/desktop/delegation) ist Upstream (nur umbenannt). → consolidation ist echter Superset.

**Korrektur meiner E-Mail-Aussage:** `EMAIL_AUTH_MODE` war ungesetzt → Live-E-Mail lief über
**IMAP/SMTP**, nicht Graph. Der Graph-Adapter war gebaut, aber schlafend.

## FINALER CUTOVER (2026-08-31, Tim: „auf Graph umstellen, jetzt und für immer, kein IMAP/SMTP/POP")
- **`updates.parked_branch_strategy: update_in_place`** in `~/.hermes/config.yaml` gesetzt (in die
  bestehende `updates`-Section gemerged, Backup in update-backups) → `hermes update` merged künftig
  origin/main IN unseren Branch, statt wegzuschalten.
- **SMTP-Fallback entfernt** (`39b54e0432`) → `graph_adapter.py` ist rein Graph. `EMAIL_AUTH_MODE=graph`
  in `~/.hermes/.env`.
- **Deploy-Checkout `~/.hermes/hermes-agent` auf `consolidation` umgestellt** (fork/consolidation
  @ `e5c9709052`). Safety-Tag `pre-consolidation-cutover-20260831` auf altem current;
  triviale Contributor-Email verworfen; `stash@{1}` (Tims alter WIP) unangetastet.
- **README-Feier** auf consolidation (Version, Rückstand ~227, unsere Verbesserungen).
- v7 hupdate live (HERMES_REPO zeigt auf ~/Projects/hermes-agent — nur für hupdates Backup/Diff/
  Reconcile-Phasen; der eigentliche Code-Deploy ist ~/.hermes/hermes-agent, jetzt auf consolidation).

Offen (Tims Operations): **Gateway-Restart** → Graph-E-Mail live. Ggf. Graph-Re-Auth. `test1`
obsolet/löschbar (Daten alle gesichert/in ~/.hermes). Rollback: `git checkout pre-consolidation-cutover-20260831`. Optional: SMTP-Fallback — bleibt (gleicher Account, Tim-Entscheidung).

## Konvergenzplan (Vorschlag, NICHT ausgeführt)
1. **`consolidation` auf den Fork pushen**: `hermes-agent` hat nur `origin`=NousResearch. Fork-
   Remote (`TechPrototyper/hermes-agent`) hinzufügen, `consolidation` dorthin pushen.
2. **Email-Adapter retten**: `feature/graph-api-email-adapter` (test1, 1 Commit `3afb6c5455`)
   auf `consolidation` cherry-picken, falls noch gewollt (steht auf uralter Basis).
3. **test1 neu aufsetzen ODER stilllegen**: test1 ist 5991 Commits alt. Sauberer: `consolidation`
   als Deploy-Checkout nutzen (entweder `hermes-agent` selbst, oder test1 auf
   `fork/consolidation` hart zurücksetzen nach Backup). Die uncommittete file_tools.py-Vollkopie
   **verwerfen** (Intent ist in consolidation als Commit).
4. **hupdate `HERMES_REPO`** auf den consolidation-tragenden Checkout zeigen lassen.

Offene Punkte für Tim: Fork-Push-Ziel bestätigen; Email-Adapter behalten ja/nein; test1
zurücksetzen vs. hermes-agent als Deploy nehmen.

## hupdate v7 (Vorschlag, gebaut, nicht scharf)
Datei: `~/.hermes/bin/hupdate.v7.proposed.sh` (Kopie von v6, `bash -n` grün). Chirurgisch:
- **Phase 5** → treibt **CP-5** `hermes skills reconcile` (echter 3-Wege). Backend node-agnostisch
  via `HERMES_SKILL_RECONCILE_*` (RTX-Update → Spark, umgekehrt). `--apply-reconcile` wendet
  konfliktfreie Merges an; sonst stagen + Review. Ersetzt die 2-Wege-Klassifizierung.
- **Phase 6** → Vollkopie-`*.py` **RETIRED** (nur noch Warnung bei Legacy-Funden + Zeiger aufs
  Commit-Modell); **hupdate-Selbst-Update (.sh) bleibt**.
- **Phasen 0–4, 7–9 unverändert** (Backup, `hermes update`, Diff-Report, Verify, Config-Migration,
  Inference-Invarianten).
- `HERMES_REPO`-TODO-Kommentar (auf consolidation-Checkout umstellen, sobald Konvergenz steht).

Aktivierung erst nach Review + Checkout-Konvergenz: `mv hupdate.v7.proposed.sh hupdate.sh`.
