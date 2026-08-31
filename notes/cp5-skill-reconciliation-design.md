# CP-5 · Basis-Skill-Reconciliation — Design

Zweck: lokal angepasste **Basis-Skills** über Hermes-Updates hinweg schützen **ohne**
Upstream-Verbesserungen an denselben Skills zu verlieren. Beides zusammen — nicht nur
das eine (Block) oder das andere (blind übernehmen). Implementiert als Carry-Patch CP-5
(Hermes, Branch `consolidation`, Commit `9ce660653f`).

## Die Sync-Matrix (pro Basis-Skill)
Drei-Wege-Zustand: `origin` (letzte von uns ausgelieferte Upstream-Version, Baseline),
`ours` (aktuell on-disk), `theirs` (neue Upstream-Version).

| Fall | ours vs origin | theirs vs origin | ours vs theirs | Aktion |
|---|---|---|---|---|
| 1 | = | = | — | skip (nichts) |
| 2 | = | ≠ | — | auto-update (Upstream übernehmen) |
| 3 | ≠ | = | — | keep ours (User-Edit schützen) |
| **4** | **≠** | **≠** | **≠** | **RECONCILE** (3-Wege-Merge) |
| 4' | ≠ | ≠ | **=** | keep ours + re-baseline (konvergiert, nichts zu tun) |

Vor CP-5 fiel Fall 4 unter „keep ours, skip" → Upstream-Innovation still verloren.
Fall 4' (ours==theirs trotz stale Baseline, z.B. User hat auf aktuelle Bundled-Version
zurückgesetzt) bleibt bewusst beim alten Verhalten (kein Reconcile).

## Fluss
1. **Detektion** (`tools/skills_sync.py::sync_skills`): Fall 4 → *ours* nie überschreiben,
   *theirs* nach `~/.hermes/skill-reconcile/<name>/theirs/` stagen, Eintrag in
   `~/.hermes/skill-reconcile/pending.json`, Meldung `reconcile_needed`.
2. **Baseline-Snapshot** (`~/.hermes/skill-baseline/<install_rel>/`): bei **jedem sauberen
   Schreiben** (Fresh-Install, Auto-Update, identische Kopie) mitgeschrieben — liefert den
   gemeinsamen Vorfahr, den das Manifest (nur Hash) nicht hat.
3. **Reconcile** (`tools/skills_reconcile.py`, deferred, `hermes skills reconcile <name>`):
   per Datei billigst-zuerst — trivial (nur eine Seite ≠ base) → `git merge-file`/diff3 →
   **LLM** bei Konflikt / SKILL.md-Prosa / fehlender Basis. Ergebnis nach `…/merged/`.
4. **Apply** (`--apply`, review-gated): Backup (`<skill>.reconcile.bak`), *merged* → live,
   Baseline := theirs, Manifest-origin := bundled_hash, Pending-Eintrag weg. Danach ist der
   Skill Fall 3 (keep ours) — self-healing, kein Dauer-Flag.

## Backend (node-agnostisch)
Tims Betriebsmuster: der Reconcile-LLM läuft auf einem **Peer** — beim RTX-Update der
**Spark**, beim Spark-Update die **RTX**; am Mac alternativ Codex/Antigravity/Claude.

    HERMES_SKILL_RECONCILE_BACKEND = http | cmd | none   (default none → nur diff3)
    http:  ..._URL (OpenAI-kompatibel /v1/chat/completions), ..._MODEL, ..._API_KEY
    cmd:   ..._CMD (Shell-Kommando; Prompt via stdin, gemergte Datei auf stdout)

Fail-open: kein/unerreichbares Backend → diff3-Konfliktmarker in `merged/`, Apply blockiert,
bis der Konflikt manuell aufgelöst ist. Nie ein stiller Falsch-Merge.

## Sicherheit / Invarianten
- Baseline- und Reconcile-Stores liegen **außerhalb** von `~/.hermes/skills/` → der Skill-
  Loader entdeckt nie eine gestagete/halb-gemergte SKILL.md.
- Eigener `_rmtree_reconcile`-Guard (spiegelt den #48200-Katastrophen-Wipe-Schutz von
  `_rmtree_writable`): löscht nur strikt unterhalb der beiden Store-Roots.
- Apply nie destruktiv (Backup vor Ersetzen, Restore bei Fehler).
- Nur komplette 3-Wege-Merges pro Datei; Binärdateien mit beidseitiger Änderung → keep ours + Flag.

## Offen
- Live-Merge gegen einen echten Upstream-Skill-Bump mit konfiguriertem Peer-Backend testen
  (Prompt-Qualität, Frontmatter-Erhalt, große SKILL.md).
- Optional: Reconcile-Schritt in den `hermes update`-Report einhängen (heute eigener Befehl).
- Optional: mehrere Kandidaten/Voting über die zwei Peers (RTX+Spark) für kritische Skills.
