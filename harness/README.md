# Harness engineering

The inference lab makes tokens fast and correct; this track makes the
**agent harness** that consumes them work well. The harness in use here is
[Hermes](https://github.com/NousResearch/hermes-agent); the work falls into
upstream contributions, a maintained carry-patch line, and a running fight
with skill schemas.

## Upstream: landed

- **Azure Foundry provider** with OpenAI/Anthropic API-mode selection —
  submitted as [PR #9029](https://github.com/NousResearch/hermes-agent/pull/9029),
  landed in main as
  [`3a7653dd`](https://github.com/NousResearch/hermes-agent/commit/3a7653dd1f0c7499646d3867822f6a588e49b68c)
  (April 2026). Made Hermes usable against Foundry-hosted frontier models
  alongside local endpoints.

## The carry-patch line

Not everything belongs upstream, and not everything offered gets taken —
both are fine. What stays valuable is maintained as **carry patches** on a
public fork ([TechPrototyper/hermes-agent](https://github.com/TechPrototyper/hermes-agent))
and re-applied on every upstream release:

- **Adaptive model routing** — content-based profile selection
  ([PR #77098](https://github.com/NousResearch/hermes-agent/pull/77098),
  declined upstream): different tasks get different model configs through
  one LiteLLM-backed endpoint. Concretely: our Qwen 3.6 and 3.8 deployments
  want different sampling and routing values, and the harness — not the
  user — should pick them per task.
- **Differentiated tool-call brake** — identical-call caps, cycle
  detection, escalation to steering instead of hard abort. A plausibility
  layer over the tool chain: agents that loop get redirected, not killed.
- **Patch-tool schema examples** — concrete call examples in the tool
  schema, which measurably reduces malformed calls from smaller models.

### The release pipeline

Rebasing carry patches by hand every release does not scale, so a small
pipeline rebuilds the working branch on every upstream tag:

| Layer | Content | Lives in |
|---|---|---|
| 0 | Upstream release tag | `NousResearch/hermes-agent` |
| 1 | Carry patches (code) | commits on `current`, public fork |
| 2 | Own skills (unknown upstream) | skill directories — never on `current` |
| 3 | Profiles: config, persona, escalation ladder | profile provisioning ([platform](../platform/README.md)) |

`current = upstream tag + carry patches`, verified by a semantic gate
before it becomes the base image anywhere. Custom settings never live in
the branch — that separation is what keeps the rebase mechanical.

## The skills update problem

Several bundled Hermes skills define tools with **conditionally required
parameters** — `cronjob` needs `schedule`+`prompt` only for
`action=create`, `patch` needs `path`/`old_string`/`new_string` only for
`mode=replace`, and so on. JSON Schema cannot enforce that, so models
omit the parameters and spiral into error loops.

The fix that works is prompt-level, not schema-level: explicit call
examples per action, imperative wording, and negative examples ("this
call will fail"). Those improvements live as a **patch set with its own
diff log**, re-applied and re-verified after every Hermes update — the
same update problem anyone who tunes bundled skills has. If that's you:
the pattern (baseline snapshot → diff → re-apply → verify) is the whole
trick, and it's worth automating on day one.
