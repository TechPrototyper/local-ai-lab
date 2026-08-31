# Handoff · vLLM-Upstream & Rob-Tand-Stränge · 2026-08-27 (nach dem PR-Sprint)

**Erstellt:** 2026-08-27 (Fable 5, Session aff5b9cd) für **Tim** (@TechPrototyper)
**Für:** die nächste Session auf diesem Strang (Upstream-Interaktion, PrismaQuant/GridBook-Verfolgung).
**Status:** UNTRACKED, bewusst nicht committet — Tim entscheidet über Verbleib.

> **Lesehaltung (Lektion aus dem Vorgänger-Handoff):** Anker, kein Evangelium.
> Tags: ✅ VERIFIED (mit Fundstelle) · 🧭 EINSCHÄTZUNG (mit Widerlegungsweg) · ❓ OFFEN.
> Das Vorgänger-Handoff (00:14 desselben Tages!) war in vier Kernpunkten überholt —
> Re-Verifikation vor Vertrauen, §7-Cheatsheet nutzen.

---

## §1 · Lage in einem Absatz

Der PR-Sprint ist gefahren und **publiziert**: Drei PRs gegen `vllm-project/vllm`
(#53977 VPE-tp1-OOB+Test, #53978 DFlash2-Warmup-OOBs, #53979 non-causal-FA2-NVFP4-Seam
**gestackt auf jethacs #46329** mit explizitem Fold-in-Angebot) plus Datenpunkt-Kommentar
in #46329 — alles nach grüner, gepaarter Cross-Arch-Revalidierung (sm120 auf frischem
Integrations-Image, sm121 auf der Prod-Lineage; byte-identischer Greedy-Determinismus
überall). Das öffentliche Notizbuch ist vollständig re-baselined (Framing, Node-Pages,
Recipes, Pipeline-Story, Trias-Note). Was jetzt zählt: **Review-Verlauf begleiten,
jethacs Fold-in-Antwort, #52244 als RTX-Adoptions-Gate, und die GridBook-Restgates.**

## §2 · Was publiziert ist (✅ alles selbst gefiled, 27.08. ~04:00 CEST)

| Artefakt | Link/Ref | Inhalt |
|---|---|---|
| PR1 | vllm-project/vllm **#53977** | VocabParallelEmbedding tp==1 OOB-Maske + Regressionstest (`2 passed` auf 5090). Unabhängig von KV-Dtype/Drafter. Branch `fix/vpe-tp1-oob-mask` |
| PR2 | vllm **#53978** | DFlash2-Warmup: embed-outside-compile + Selector-Codebook-Clamp (gegen gemergten #52816-Code). Branch `fix/dflash2-warmup-oob` |
| PR3 | vllm **#53979** | Der Seam: non-causal Prefill-Wrapper für FA2-NVFP4 (nur fa2; trtllm-gen bleibt causal-only). **Gestackt auf #46329** (Basis = jethacs Head `7a5cf14`, letzter Commit = Delta). Fold-in-Angebot im Body. Branch `feat/nvfp4-noncausal-dflash` |
| Kommentar | #46329, issuecomment-5433626787 | Verknüpft alle drei, Messdaten, "missing link"-Framing, Validierungsangebot |
| Notebook | local-ai-lab bis `a8b3782` | Seam-Note, Trias-Note, Pipeline-Story, Re-Baseline, Mixed-Precision-Präzisierung |
| Fork-Branches | TechPrototyper/vllm | obige drei + `integrate/dflash2-nvfp4-v4` @ `2cf8b8ae0` (= main ∪ #46329@7a5cf14 ∪ PR1+2+3) |

**Messstand für Diskussionen** (✅ `results/RESULT_nvfp4_spec_crossarch_revalidation.json`):
sm120 (Image `sm120-v4-2cf8b8a-t2`): count-200 **190.9 vs 55.9 (3.4×)**, Step-Reasoning
**135 tok/s** (Accept 0.653–0.791), Essay-Prosa Parität (Accept ≈0.20 — Spec regressiert
nicht bei Draft-Miss); KV-Pool 43,996 tok @3GiB/32k. sm121 (erste DFlash2+NVFP4-Serve
überhaupt dort): Prosa **+72%**, count-200 **5.9×**, c=2 **+70%**; Pool 47,304.
⚠️ Acceptance ist stark content-abhängig — **immer das Spektrum nennen, nie eine
Einzelzahl**; die alte 82.5/0.539 war ein günstigerer Prosa/Thinking-Mix, keine
Stack-Differenz. sm120-Arm lief bf16-Drafter, sm121-Arm fp8-Drafter (acceptance-neutral
per Verdikt 08-21).

## §3 · Watch-Liste (die eigentliche Arbeit der nächsten Session)

1. ❓ **Reaktionen auf #53977/#53978/#53979**: CI-Läufe, DCO-Bot, Reviewer-Kommentare.
   Bei Redaktionswünschen: Ton halten (Konjunktiv, messungsbasiert, freundlich).
2. ❓ **jethacs Antwort auf das Fold-in-Angebot** (#53979/#46329). Beide Wege sind für
   uns gut: Fold-in = er trägt es, wir sind co-credited; standalone = unser PR auf
   seiner Basis. Sein Rebase-Rhythmus: er force-pusht — **vor jeder Arbeit
   `git fetch jethac` und Basis prüfen** (Lektion: `_v3_on_jethac` war stale).
3. ❓ **#52244** (Prefix-Cache × Spec × hybrid-GDN, Y-aang; 26.08. CONFLICTING):
   DAS Gate für RTX-Spec-Adoption (Board-Issue #38). Validierungsangebot steht im
   #46329-Kommentar. Wenn es sich bewegt: sm120-Validierung anbieten/fahren.
4. ❓ **#52883** (DFlash2 LM-Head, jschmied) — noch offen; unsere PR2 liegt im selben
   Strang. jschmied ggf. als Reviewer für #53978 gewinnen.
5. 🧭 **Konkurrenzlage**: ssubbotin (RTX PRO 6000, MTP), hclsys (GB10), Sean arbeiten
   aktiv im #46329-Umfeld — die Commits `ce2fece`/`7a5cf14` credittten SIE, nicht uns
   (wichtigste Korrektur am Vorgänger-Handoff). Unsere Nische (DFlash2×NVFP4 non-causal)
   ist besetzt & publiziert, aber der Thread ist heiß. *Widerlegen:* Thread-Verlauf lesen.

## §4 · Rob-Tand-Strang (PrismaQuant/GridBook)

- ✅ **Stand**: AQUA 5.5bpp = Prod beide Boxen. GridBook 13GB hält Qualität (n=250,
  drei Achsen, p=1.0; `notes/gridbook-13gb-quality-holds.md`). Die volle Trias lief
  am 24.08. auf der RTX (877k-Config) — Rollback wegen #52244, nicht wegen der Trias
  (`notes/gridbook-nvfp4-dflash2-rtx-triad.md`).
- ✅ **GELÖST (27.08., HF-Sichtung):** Das Qwen3.8-GridBook-13GB ist
  `rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm` (HF, 18.08.; „built for
  a 16 GB card", 12.98 GB inkl. Codebooks). Board #10-Rest (4.5bpp) bleibt offen —
  liegt NICHT auf Robs HF. Neu dort außerdem (23./24.08.): `Qwen3.8-27B-
  PrismaScout-AQUA-20GB` (+`-Vision-20GB`) und ein Refresh des 5.5bit-Artefakts;
  DSv4-87GB-spark refreshed 21.08. prismaquant 0.16.0/0.16.2 (21./22.08.):
  CB-Lane-Sharding, exaktes Byte-Accounting (`read_gb_per_token`), MoE-A-Side-
  Bit-Accounting-Fix (dense Qwen3.8 unberührt; DSv4/Ornith/Hy3-Vergleiche betroffen).
  Vor eigenen Quant-Läufen (#11/#22): auf 0.16.2 gehen. Board-Kommentare 381 (#10),
  382 (#18).
- **Offene GridBook-Gates** (Board #37-Nachbarschaft, README-Next-up): (a) Acceptance
  unter GridBook-Target (Speed-Frage, Drafter auf BF16/AQUA trainiert), (b) Far-Window-
  Pass auf der 877k-Config, (c) n=1319-Verdikt. (d) Board #10: 4.5bpp-Artefakt bei Rob
  anfragen; #11: eigener CB-Re-Quant; #22 (ready): eigenen AURA-Quant auf HF publizieren.
- **PrismaQuant beobachten**: Rob schiebt nach (dev693-Loader war schon mal 13 Tage
  voraus). `RobTand/prismaquant` Releases/Commits checken; PR#80-Beziehung ist gut.
- ⚠️ **Claim-Disziplin** (Tim, 27.08., Board **#37**): KEINE lauten "BF16-Qualität"-
  oder "besser als NVFP4"-Claims — implizit über den Mixed-Precision-Aufwand
  kommunizieren. Zurückgestellte Messung dazu: uniform-NVFP4 (`dsikka-nvfp4`, liegt
  auf dem Spark) vs. AQUA, gepaart — bewusst NICHT terminiert.

## §5 · Ressourcen & Verhaltensregeln (Kurzfassung; Cluster-Details im Schwester-Handoff)

- **GPU-Fenster nur mit Tims expliziter Freigabe, `date` vor jeder Box-Aktion
  verifizieren.** (27.08. 01:46: Shutdown 14 min zu früh gestartet — dokumentierter
  Fehler, Wiederherstellung nötig. Nie wieder.)
- Prod-Restore-Stand: beide Boxen servieren (RTX `e2446da2-t1`, Spark fp8-KV-Config).
  **v4-Image ist NICHT prod-promoted** — braucht Verdikt-Gate (n=250→1319).
- Spark-Experiment-Tree `~/nvfp4-work/vllm-dflash2-v2exp` (Branch `exp/nvfp4-spec-v2`)
  liegt bereit; Prod-Tree nie anfassen.
- Method-Disziplin: gepaart, McNemar exakt, Screening n=250 ≠ Verdikt n=1319,
  Spec unter Greedy qualitätsneutral (Determinismus-Gate als Beweis).

## §6 · Naheliegende nächste Züge (Vorschlag, kein Auftrag)

1. Morgens/täglich: §3-Watch (15 min, `gh`-Cheatsheet unten).
2. Bei jethac-Fold-in: Commit übergeben, #53979 entsprechend schließen/umbauen.
3. NVFP4-KV-Cutover-Gate auf dem Spark vorbereiten (n=250-Triage im nächsten Fenster)
  — der ≈2×-Pool wartet nur auf Qualitätsbestätigung.
4. GridBook-Artefakt-Frage (§4) klären, dann Acceptance-Messung planen.

## §7 · Re-Verifikations-Cheatsheet

```bash
# PR-Status + CI + Reviews
for n in 53977 53978 53979; do gh pr view $n --repo vllm-project/vllm \
  --json state,mergeable,reviews,statusCheckRollup --jq '{n:'$n', s:.state, m:.mergeable, r:(.reviews|length)}'; done
gh pr view 46329 --repo vllm-project/vllm --json comments --jq '.comments[-5:][] | {a:.author.login, d:.createdAt, b:(.body[0:150])}'
gh pr view 52244 --repo vllm-project/vllm --json state,mergeable,updatedAt
gh pr view 52883 --repo vllm-project/vllm --json state,updatedAt
# jethacs Branch-Bewegung (force-pusht!)
cd ~/Projects/vllm-upstream-work && git fetch jethac nvfp4-fa2-consumer-blackwell --dry-run
# Rob Tand — BEIDE Repos beobachten (gridbook = eigenes Repo, unabhängig von prismaquant!)
for r in RobTand/gridbook RobTand/prismaquant; do echo "== $r =="; \
  gh api repos/$r/commits --jq '.[0:3][] | {d:.commit.author.date[:10], m:(.commit.message|split("\n")[0][:60])}'; \
  gh release list --repo $r -L 2; done
# Baselines (Stand 2026-08-30): gridbook Latest v0.9.0 (08-24, "TP/EP as plain vLLM plugin"), Commit d8cef3fb;
#   prismaquant Latest v0.16.2 (08-22), Commits 08-30 nur docs. Achtung gridbook 0.8.7 = quantisierte embed_tokens (unser MTP-Ladeproblem).
# Messstand
python3 -c "import json;d=json.load(open('results/RESULT_nvfp4_spec_crossarch_revalidation.json'));print(d['verdict'])"
```

**Schlüsselpfade:** Patch-Stack/Fork-Checkout `~/Projects/vllm-upstream-work`
(Remotes: origin=vllm-project, fork=TechPrototyper, jethac). Notebook local-ai-lab
@ `a8b3782`. PR-Text-Entwürfe/Runbooks der Sprint-Nacht: Session-Scratchpad
aff5b9cd (flüchtig).

— Ende Handoff A.

---

## §8 · Watch-Log

### 2026-08-27 ~10:30 UTC (Session nach dem Sprint)

- ✅ **#53977/#53978/#53979**: alle OPEN, DCO pass, keine menschlichen Reviews.
  Einziger CI-„Fail" ist `pre-run-check` = **Gating, kein Code-Problem**: volle CI
  läuft erst mit Label `ready`/`verified` oder ab 4 gemergten PRs des Autors (wir: 0).
  Der Workflow verbietet KI-Agenten explizit, das Label anzufragen → wenn, dann
  **Tim selbst** via vLLM-Slack `#pr-reviews`. claude[bot]-„Reviews" sind leere
  Platzhalter (Fork-PRs, Auto-Review disabled; Maintainer kann `@claude review` triggern).
- ❓ **jethac Fold-in**: noch keine Antwort auf unser Angebot (Kommentar 02:33 UTC).
  Branch `nvfp4-fa2-consumer-blackwell` unverändert @ `7a5cf14` → Stack-Basis aktuell.
- ✅ **#46329-Thread seit unserem Kommentar**: hclsys (04:59 UTC) bestätigt jethacs
  Root-Cause zum sm12x-NaN-Scale-Befund — Kernel ok, Test-Reader war falsch
  (2/256 fp8e4m3-Patterns sind NaN; linear statt swizzled gelesen). Für sm121 relevant,
  entlastet den Kernel-Pfad.
- ❓ **#52244**: weiter CONFLICTING (`needs-rebase`), keine neuen Commits/Kommentare;
  `updatedAt` 08:19 heute ohne sichtbares Timeline-Event (Metadaten). jschmied hat am
  24.08. seine Messung korrigiert und den Defekt **bestätigt** (Harness-Fehler zuvor:
  `cp -a`-venv mit absoluten Shebangs → alle Arme liefen ungepatcht).
- 🆕 **#50897 auf die Watch-Liste**: ZJY0516 nennt es in #52244 den „more proper fix"
  (lookahead-aware Prefix-Cache-Hashing für EAGLE-style Drafter, ZJY0516). Ebenfalls
  OPEN + CONFLICTING, zuletzt 24.08. → RTX-Adoptions-Gate hat jetzt **zwei** Kandidaten.
- **#52883**: unverändert seit 20.08.
- **prismaquant**: v0.16.2 (22.08., Docs/Math-Re-Underwrite-Korrekturen), seither ruhig.

**Fazit:** Nichts erfordert sofortiges Handeln. Blocker für Review-Fortschritt ist
das `ready`-Label (menschlicher Kanal). Nächster sinnvoller eigener Zug bleibt §6.3
(NVFP4-KV-Cutover-Triage) bzw. §4 (GridBook-Artefakt lokalisieren) — beides GPU-
fenster- bzw. klärungsabhängig.

### 2026-08-29 ~00:xx UTC (Nacht-Watch-Tick)

- 🆕 **#53977/#53978/#53979 getriaged**: haben jetzt Themen-Labels
  (#53978 `speculative-decoding/qwen/mrv2/dflash`, #53979
  `nvidia/quantization/dflash`, #53977 `bug`) — vorher unlabeled. Also von
  einem Maintainer/Bot kategorisiert, aber **weiterhin kein `ready`-Label**,
  Reviews weiter nur claude[bot]-Platzhalter, mergeable UNKNOWN. Full-CI-Gate
  unverändert (menschlicher Kanal, Tims Zug).
- 🟢 **#53122** (MERGEABLE): Unser Prod-Datenpunkt (sm121, fp8-Drafter) +
  Addendum (öffentlicher Checkpoint) stehen seit 08-27 17:44/45. **Cross-Vendor-
  Korroboration**: `dtandersen` (08-27 17:11) bestätigt den Patch auf **Intel
  Arc B70** (Qwen 3.8-27B stabil) — d.h. die Fix-Linie hält nicht nur auf
  Blackwell. Noch kein Maintainer-Review. Positiv, kein Handlungsbedarf.
- ❄️ **#52244 / #50897 / #52883**: `updatedAt`-Bumps (08-28), aber **keine neuen
  menschlichen Kommentare** — #50897s jüngster Eintrag ist unser eigener
  Validierungs-Kommentar (08-28 05:07), noch **keine Antwort von ZJY0516/
  Maintainer**. #52244 weiter CONFLICTING. Metadaten, nicht substanziell.
- ⚪ **#46329**: seit 08-27 04:59 (hclsys) **keine neuen Kommentare**; jethac-
  Branch `nvfp4-fa2-consumer-blackwell` unverändert @ `7a5cf14` (kein Force-Push,
  Stack-Basis aktuell). **#53334**: TDHSC noch keine Antwort/Branch seit unserem
  Mentoring-Kommentar (08-27 13:30). **prismaquant**: still v0.16.2 (08-22).

**Fazit:** Nichts Top-Prio, nichts das Umpriorisierung/Antwort erfordert. Zwei
milde Positiva (PRs getriaged; #53122 cross-vendor bestätigt). Review-Gate bleibt
das `ready`-Label. Weiter „durchziehen wie besprochen".

### 2026-08-29 ~02:1x UTC (Nacht-Watch-Tick — RELEVANT: jethac-Review)

- 🟢🟢 **#46329/#53979 — jethac hat geantwortet (00:52 UTC), volle Peer-Review.**
  Das ist der High-Prio-Kanal; Kernpunkte:
  1. **Sequencing/Fold-in geklärt:** #53979 bleibt **standalone unter unserem
     Namen** — „standalone isn't just acceptable, it's correct". Fold-in nur für
     Fixes an *seinem* Code (Co-authored-by), neue Capability stapelt mit
     **Autor-Attestierung intakt**. Reputationsgewinn wie gewünscht.
  2. **Approach bestätigt:** „The approach is right and the plumbing claim checks
     out." `causal=False` auf sm12x end-to-end getraced; 17-Zeilen-Delta
     ausreichend; non-nvfp4-non-causal byte-identisch; sm100f raist korrekt.
     Two-arch paired-Validation = „exactly the standard this line needs", 5.9×
     auf sm121 „a strong result".
  3. **Komposition mit #46443 (DiffusionGemma) sauber** — 3-Wege-Merge
     konfliktfrei, Seams komplementär.
  4. **EINE substanzielle Finding vor Merge — non-causal + sliding window:**
     shared plan reicht `window_left` unbedingt durch, FlashInfer wendet SW
     in-kernel unabhängig vom Mask-Mode an → `plan(causal=False,
     window_left>=0)` = ungetestete left-only-Staircase. **Nicht von unseren
     validierten Armen erreichbar** (Qwen3.8, full-attn Drafter, `window_left=-1`),
     aber vom `use_swa=True`/`layer_types=None` DFlash-Family (MiMo-V2.5-Pro-FP4-
     DFlash). **Zwei akzeptierte Lösungen:** (1) `raise` für `use_fa2_nvfp4_kv and
     not causal and window_left >= 0` bis Kernel-Parity-Test existiert, oder (2)
     symmetrisches Window via custom mask (wie grouped path). Beide lassen unsere
     Configs unberührt.
  5. Tooling-Note: `dequant_nvfp4_kv_cache(..., swizzle_sf=...)` @ `7a5cf143` ist
     jetzt kanonische Referenz (K linear überall, V linear auf sm12x).
- **Aktion:** Read-only-Watch → **nicht autonom gepostet** (Attestierungs-Regel:
  Code auf *unserer* PR unter Tims Namen ist Tims Zug). Guard-Entwurf (Lösung 1)
  vorbereitet für Tim: `notes/draft-53979-swa-guard-response.md`. Empfehlung:
  Option 1 (Guard) sofort → entblockt Merge, minimal, hält Attestierung; Option 2
  als Folge-PR wenn Kernel-Parity-Test steht.
- Rest unverändert (kein `ready`; #50897/#53122/#53334 weiter unser Kommentar
  zuletzt; jethac-Branch @ `7a5cf14`; prismaquant v0.16.2).

### 2026-08-29 ~05:xx UTC (Tick — prismaquant-Bewegung)

- 🆕 **prismaquant**: Rob Tand hat **PR #86** von externem Contributor `smb209`
  gemergt (05:05 UTC) — wrapped-MoE-FP8-Source-Fix (`moe_imatrix` FP8
  serialized-scale-Contract in packed-expert replay + streaming-export
  namespace-bridge). **Kein neues Release** (bleibt v0.16.2). Für unseren Track
  (PrismaAQUA/AURA, nicht wrapped-MoE) nicht direkt relevant; zeigt Community-
  Health (Rob nimmt Fremd-PRs auf). Keine Aktion.
- Sonst unverändert: #53977/78/79 kein `ready`; #46329/#53979 letzter Kommentar
  weiter jethac 00:52; jethac-Branch @ `7a5cf143`.

### 2026-08-30 ~06:2x UTC (erster Fremd-Kommentar auf #53977)

- 🆕 **#53977** (unser VocabParallelEmbedding-tp==1-OOB-Fix): erster externer
  Kommentar — **maxpla3** (06:22): *„Related PR #51508"*. Verifiziert: **#51508**
  = *anderer, aber verwandter* Bug (GDN/KDA silent recurrent-state corruption für
  stale zero-accept spec rows; `num_accepted_tokens=0` indiziert State-Slots OOB →
  Korruption/CUDA-Crash). **Macht #53977 NICHT redundant** (anderer Codepfad:
  Embedding-OOB-Maskierung vs. num_accepted_tokens-Indizierung); beide härten
  Spec-OOB auf Hybrid-GDN. maxpla3 = Autor von #51508 (open, needs-rebase) →
  freundlicher Cross-Ref, potenzieller „unter Gleichen"-Kontakt wie jethac.
  Tim informiert; kein Post ohne seine Freigabe. Optionaler Zug: Cross-Ref zurück.
- Rest unverändert: kein `ready`; #46329 jethac 00:52; #50897/#52883/#53122/#53334
  jüngster Kommentar weiter unser; jethac-Branch @ `7a5cf143`; prismaquant sehr
  aktiv über Nacht (#94/#98 gemergt), aber reine Eigenentwicklung.

### 2026-08-30 ~16:3x UTC (gridbook v0.9.1 Release + prismaquant cache-fix)

- 🆕 **gridbook v0.9.1** (Release 15:10 UTC, von v0.9.0): SM120-relevant —
  „ground SM120 dual-family CB validation" + SM120-compile-receipt; expandierte
  NVFP4/FP8-Research-Kernels (K1-K48), warp-resident v2-Trellis-Decoder (ABI 3),
  fix persistent-B-Metadata @ NVFP4 K25. **Nichts MTP/embedding-spezifisch**
  (unser `gridbook-13gb-mtpfix`-Swap unberührt), aber SM120-CB-Validierung +
  Kernel-Erweiterung sind fürs RTX-GridBook-Serving relevant → beim nächsten
  GridBook-Build im Blick behalten.
- 🆕 **prismaquant** `fix(cache): fail closed on activation sidecar corruption`
  (#110, 08-30) — Cache-Robustheit (fail-closed statt stiller Korruption). Kein
  neues Release (v0.16.2). Konzeptuell nahe unserem Sidecar-Thema, aber betrifft
  prismaquants Activation-Cache, nicht den MTP-embed-Sidecar. Keine Aktion.
- Sonst unverändert: PRs kein `ready`; #52244 nur Metadaten-Bump (upd 16:25, kein
  neuer menschlicher Kommentar); jethac-Branch @ `7a5cf143` (kein Force-Push).

### 2026-08-31 ~07:5x UTC (#53334 — neuer Community-Kommentar)

- 🆕 **#53334** (Tims sm121-TurboQuant-KV-Issue): neuer Kommentar von **Abdulrahman-
  Hashem** (05:22 UTC) — NICHT TDHSC: *„the PRs that fix this issue are both either
  wrong or need rebase because they reduce generation speed to less than 40 t/s."*
  D.h. ein Dritter kritisiert die (Community-)Fix-PRs für #53334 als speed-schädlich
  (<40 t/s). Betrifft NICHT unsere Seam-PRs (#53977-79) direkt, sondern die #53334-
  Fixes. Datapoint: Interesse am Issue wächst + Perf-Bedenken. Kein Handlungsbedarf
  (read-only Watch), aber im Blick behalten falls jemand unsere Arbeit hereinzieht.
- Rest unverändert: PRs kein `ready`; #52244 Metadaten-Bump (05:58); jethac @ `7a5cf143`;
  gridbook v0.9.1 / prismaquant v0.16.2 (nur Test-Fix), keine neue Bewegung.

---

## §9 · Beschlossen (Tim, 27.08.): Zwei Vorhaben, sobald GPUs frei

**Auf dem Board angelegt** (Gitea `gitadmin/AIResearchAndDevelopment`, Zugriff jetzt
scriptbar via scoped Token `~/.config/gitea/token`, write:issue+read:repository):
**#40** (50897-Validierung) und **#41** (jschmied-Allianz), beide `status/ready`;
Lage-Update als Kommentar 378 auf #38. Inhalt:

1. **#50897-Validierung (ZJY0516, „proper fix" fürs RTX-Spec-Gate)** — Jethac-Playbook:
   gepaarte Validierung DFlash2×NVFP4 auf sm120+sm121 gegen #50897 (Port auf lauffähige
   Basis nötig, PR ist CONFLICTING; wie Leoyzen porten). Danach messungsbasiertes
   Validierungs-/Befund-Angebot im PR-Thread. Nutzen: Reputation beim Schwergewicht
   (80 merged PRs) + entsperrt Board-#38 (RTX-Spec-Adoption). Empirie pro #50897 liegt
   vor (jschmieds GB10-Messung: 52244 bewegt nichts, 50897 löst).
2. **jschmied-Allianz (#52883)** — Peer-Aufbau „unter Gleichen": sein #52883 auf
   unserer Hardware validieren + freundliches Review; Koordinations-Kommentar wegen
   Datei-Überschneidung `qwen3_dflash2.py` (#52883 ↔ unser #53978). Gegenzug-Erwartung
   (nicht einfordern, ergibt sich): unabhängige Validierung unserer PRs → Pfad zum
   `ready`-Label. Sein GB10 = direkter sm121-Vergleichspartner; räumliche Nähe
   (Neuenhagen) als Bonus für echten Mitstreiter-Aufbau.

3. **TDHSC-Mentoring (#53334, unser TurboQuant-Issue)** — 27.08., mit Tims Freigabe
   gepostet: [issuecomment-5439861335](https://github.com/vllm-project/vllm/issues/53334#issuecomment-5439861335).
   Erstcontributor „Teddy" (0 PRs, New-Grad, USA) hatte das Issue am 22.08. geclaimt,
   seither still. Kommentar: Wegweiser (beide Befunde separabel, Code-Pointer
   `turboquant/config.py` bzw. `triton_turboquant_store.py`, Repro braucht keine
   Spezial-Hardware) + Validierungsangebot sm120/sm121 wenn er einen Branch aufsetzt.
   Dritter Netz-Faden: Mentoring-Rolle neben ZJY0516 (Schwergewicht) und jschmied (Peer).
   → #53334 in den 2h-Watch aufgenommen.

4. **Scout-20GB H2H (Board #42, Tim 27.08. nachmittags):** heute Nacht, Priorität
   **direkt hinter #40/#41**. `rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB` (19,97 GB/
   18,6 GiB, verifiziert) vs. Prod-AQUA-5.5 (23,63 GB/22,0 GiB) → 3,4 GiB KV-Headroom
   auf der RTX. Stufenplan: Boot-Smoke (Abschmieren → Notiz+abhaken) → paired n=250
   Screening, beide Arme im v4-Image (alle eigenen Codeänderungen, beide Arme gleich)
   → bei Erfolg n=1319-Verdikt als Folge-Gate. Alles reporten, auch Fehlschläge.

5. **#53122-Datenpunkt (27.08. abends, Tims Go):** Prod-Validierung aus dem Bestand
   gepostet ([Kommentar](https://github.com/vllm-project/vllm/pull/53122#issuecomment-5442968584)
   + [Addendum](https://github.com/vllm-project/vllm/pull/53122#issuecomment-5442978186)
   mit unserem öffentlichen Drafter `TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm` als
   Test-Artefakt). Kern: #53122 läuft seit 22.08. bei uns in Prod (fp8-Drafter voll
   quantisiert, acceptance-neutral, −1,6 GB → KV-Pool 478k). Kein GPU-Einsatz nötig
   gewesen — Quelle war `notes/dflash2-drafter-fp8-quant.md`. Kontext: #53122 ist das
   Sammelbecken der Quantisierte-Drafter-Fixes; jschmied ist dort de-facto Co-Autor
   (sein Branch gemerged, GB10-Verifikationsmatrix) → stärkt #41-These. #53122 im
   2h-Watch (Job 99918a91). **Prüfkandidat notiert:** `YourHighnessLA/Qwen3.8-27B-
   DFlash2-NVFP4` (publizierter NVFP4-Drafter — potenziell weitere Drafter-Schrumpfung
   auf der RTX; W4A16-Kernel-Frage auf sm12x offen, siehe Drafter-Note Schlussabsatz).

**Nachtfenster 28.08. ABGESCHLOSSEN ~02:00, Prod restored+verifiziert (RTX healthy,
Spark-Batterie komplett, Flux resumed):** Alle drei Vorhaben gelandet — #40 validiert
(0→90,9 % Replay-Hits, tokengenau auf beiden Archs, byte-identisch; Board 414),
#41 gepostet, #42 Screening PASS (p=1.0, ~10 % schneller, **+39,4 % KV-Pool** im
Prod-Zuschnitt: 342.604→477.569; Board 413/416). Publiziert @ deaf912 (Note +
5 Result-Files + README-Next-up). OFFEN: Tims Go für den #50897-Kommentar
(`scratchpad/draft_50897_comment.md`); n=1319-Verdikte für Scout-Adoption und
NVFP4-KV-Cutover als Folge-Gates.

**Morgen-Nachtrag 28.08. (~07:40):** #50897-Kommentar GEPOSTET (Tims Go;
50897#issuecomment-5448682111). **Risiko-Rollout auf Spark-Prod VOLLZOGEN**
(Tims Beschluss „wir riskieren mal was"): `vllm-aura38` auf Tree
`vllm-dflash2-pc50897` (= Prod-Lineage 58f998f84 + 50897-Kern @dd02ed4da1);
Rollback-Container `vllm-aura38-pre50897` + Skript liegen bereit. Live: 90,0 %
Replay-Hits auf Prod-Config; Battery deployed-config: GSM-250 0,9960, Tools 6/6,
Needle 6/6, Det 5/5, 21,9/157,0 tok/s. Publiziert @ dccb493 + eae1f2d; Board:
#40/#41 geschlossen (done), #38-Kommentare 424/425, #42-Kommentar 416.
**OFFEN: n=1319-Verdikte** (Spark-Prod-50897 nachlaufend, Scout-Adoption,
NVFP4-KV-Cutover) + jschmied/ZJY0516/TDHSC-Reaktionen im Watch.

**Ursprünglicher Zwischenstand:** #41 ERLEDIGT um ~00:50 — Befund statt Validierung:
jschmieds Fix auf main obsolet (#52816-Refactor), dieselbe Falle lebt in
`_apply_head` weiter; Probe publiziert (a747347), Retarget-Kommentar mit Tims Go
gepostet (52883#issuecomment-5446093771), Board #41 Kommentar 412. #40: beide
Ports fertig (sm120 Fork-Branch `...-pc50897`, sm121 v2exp `exp/pc50897` @83a9ef7f4);
sm120-Baseline reproduziert den Bug exakt (Replay 12812 Tokens → **0 Hits**,
Extended 8736/12821; byte-identisch). Gepatchte Arme booten.

**Gates vor Start:** (a) GPU-Fenster mit Tims expliziter Freigabe (§5-Regeln!) —
für heute Nacht erteilt (Reihenfolge #40 → #41 → #42), (b) Aktualitätscheck
unmittelbar davor (PR-Status, jethac-Branch, ob #50897/#52883 sich bewegt haben
oder gemergt/überholt sind). **Y-aang/#52244: kein Invest.**
Bis zum Fenster: Füße still, nur 2h-Watch (läuft, siehe unten).
