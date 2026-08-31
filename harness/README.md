# Harness engineering

The inference lab makes tokens fast and correct; this track makes the
**agent harness** that consumes them work well. The harness in use here is
[Hermes](https://github.com/NousResearch/hermes-agent). The work falls into
three buckets: upstream contributions, a maintained carry-patch line on a
public fork, and the operational model that keeps that fork deployable
without falling behind upstream.

The fork lives at
[**TechPrototyper/hermes-agent**](https://github.com/TechPrototyper/hermes-agent),
default branch `consolidation`, currently based on **Hermes v0.21.0**
(2026-08-31). Its README leads with what the fork does that upstream doesn't;
this report is the engineering story behind it.

## Upstream: landed

- **Azure Foundry provider** with OpenAI/Anthropic API-mode selection —
  submitted as [PR #9029](https://github.com/NousResearch/hermes-agent/pull/9029),
  landed in main as
  [`3a7653dd`](https://github.com/NousResearch/hermes-agent/commit/3a7653dd1f0c7499646d3867822f6a588e49b68c)
  (April 2026). Made Hermes usable against Foundry-hosted frontier models
  alongside local endpoints.

## The carry-patch line

Not everything belongs upstream, and not everything offered gets taken —
both are fine. What stays valuable is maintained as **carry patches**: clean,
individually-scoped commits on `consolidation`, rebased onto every upstream
release. The current set:

| # | Carry patch | What it does |
|---|---|---|
| CP-1 | **Adaptive model routing** ([PR #77098](https://github.com/NousResearch/hermes-agent/pull/77098), declined upstream) | Content-based profile selection: different tasks get different model/sampling configs through one LiteLLM-backed endpoint. Our Qwen 3.6 and 3.8 deployments want different routing, and the harness — not the user — should pick per task. |
| CP-2 | **Differentiated tool-call brake** | Identical-call caps, cycle detection (A,B,C,A,B,C…), and escalation-to-steering instead of a hard abort. A plausibility layer over the tool chain: agents that loop get redirected, not killed. Merged "keep-both" with upstream's own newer identical-call notice/de-bloat, so both mechanisms coexist. |
| CP-3 | **Patch-tool schema examples + Graph loop-breaker cleanup** | Concrete call examples in the patch schema (measurably fewer malformed calls from smaller models), folded into the dynamic V4A layer so the lean base schema is untouched. Plus the Hermes half of a force-progress loop-breaker: it consumes an out-of-band signal from the inference gateway and **collapses the repeated tool-call turns out of its own durable history** — bloat removed at the root. |
| CP-4 | **kaniko-compatible Docker chmod** | Build-portability fix for the serving images. |
| CP-5 | **Base-skill reconciliation** | See below — the skills-update problem, solved. |
| CP-6 | **Microsoft Graph API email adapter** | See below — a whole new gateway channel. |

`consolidation = upstream release + carry patches`, verified before it
becomes the deployed line. Custom settings (config, persona, escalation
ladder) never live in the branch — they are profile-provisioned
([platform](../platform/README.md)) — and that separation is what keeps the
rebase mechanical.

## Microsoft Graph email in the gateway (CP-6)

Hermes's bundled email channel speaks IMAP/SMTP. On a Microsoft 365 tenant
with MFA / conditional-access, IMAP is often disabled outright — so the agent
simply can't do email. CP-6 adds a **Graph API adapter** (`GraphEmailAdapter`,
OAuth2 with `Mail.ReadWrite`/`Mail.Send`): polling, `sendMail`, attachments,
whitelist filtering, and automatic token refresh. An `EMAIL_AUTH_MODE`
selector chooses `graph` or `imap` at registration.

The deployment runs **Graph only — no IMAP, SMTP, or POP** (a deliberate
call: a `sendMail` failure should surface loudly, not silently degrade to
SMTP). In production it connects to the tenant mailbox, refreshes its OAuth
token on its own, and filters inbound against the whitelist — a first-class
M365 channel where upstream had none.

## The skills-update problem — solved (CP-5)

Several bundled Hermes skills define tools with **conditionally required
parameters** — `cronjob` needs `schedule`+`prompt` only for `action=create`,
`patch` needs `path`/`old_string`/`new_string` only for `mode=replace`, and
so on. JSON Schema cannot enforce that, so weaker models omit the parameters
and spiral into error loops. The fix that works is prompt-level: explicit
call examples per action, imperative wording, negative examples.

But editing a bundled skill creates a maintenance trap. On the next upstream
update you face a fork in the road:

- **Block** — keep your edit, skip the upstream version. Safe, but you
  *silently lose* whatever upstream improved in that same skill.
- **Overwrite** — take upstream, lose your edit.

Upstream Hermes (and the old whole-file patch model) only ever did the first.
CP-5 adds the missing third option. `sync_skills` now snapshots the last
upstream version it shipped as a **common ancestor**; when a skill is edited
locally **and** changed upstream (the "case 4" collision), it preserves ours,
stages theirs, and queues a reconcile. `hermes skills reconcile` then runs a
real **3-way merge** (base / ours / theirs) — cheapest-first: trivial →
`git merge-file` (diff3) → an **LLM merge** on genuine conflict. The LLM
backend is node-agnostic: on an RTX update it can be driven from the Spark and
vice-versa, or from a local model. Review-gated apply, always backed up,
self-healing across syncs. Upstream skill improvements are adopted *and* our
customizations re-applied — the trap is gone.

The general pattern — **baseline snapshot → diff → reconcile → verify** — is
the whole trick for anyone who tunes bundled skills, and it's now built in.

## The update & deployment model

Rebasing carry patches by hand every release does not scale, and a stock
`hermes update` actively fights a maintained fork: it wants `main` and skips
(loudly) when it finds the checkout parked on a feature branch. Two things
make the fork deployable:

1. **`parked_branch_strategy: update_in_place`** — configured so `hermes
   update` *merges* the upstream release into `consolidation` instead of
   switching away from it. Local commits survive; the update is atomic and
   leaves a safety tag. The carry line advances with upstream, in place.
2. **Fork-as-deploy-line** — the running gateway checks out `consolidation`
   from the fork. Updating is either a clean rebase-in-work-repo → push →
   deploy pull, or the in-place merge above; both keep the deployment on our
   line with every carry patch intact.

**Lesson learned the hard way:** know exactly which checkout your gateway
actually launches from. A production cutover here surfaced *three* Hermes
checkouts on one machine — a work repo, an installed tree, and the one the
launchd service actually ran. The service's `ProgramArguments` is the single
source of truth; verify it before declaring a deploy done. Post-cutover the
gateway runs `consolidation` (v0.21.0), all six carry patches verified present
by patch-id and by content, Graph email live.

| Layer | Content | Lives in |
|---|---|---|
| 0 | Upstream release tag | `NousResearch/hermes-agent` |
| 1 | Carry patches (code) | commits on `consolidation`, public fork |
| 2 | Own / edited skills | reconciled in place (CP-5), never overwritten |
| 3 | Profiles: config, persona, escalation ladder | profile provisioning ([platform](../platform/README.md)) |
