# Overflow-banner runaway (fixed 2026-08-11)

**Symptom:** under `SPARK_FORCE=1`, every LiteLLM response had an
overflow-mode banner prepended to `msg.content`. The client (Hermes)
replays conversation history back to the model, so the context saturated
with identical banner strings across turns. A model conditioned on
highly-repetitive context continues the repetition — it started generating
the banner endlessly, running to the `max_tokens` ceiling (65536) on every
affected call: multi-hour hangs, non-streaming tool-calls that never
returned.

**Fix:** the banner was removed without replacement, cluster-/LiteLLM-side
(`litellm/overflow_notice.py` in the MyCluster ops repo, commit
`b38b71e`). It was a pure display feature with no functional value — not
worth this failure mode.

**Deliberately not fixed with a `max_tokens` cap.** User's call: if the
model wants to output a long response, let it. Quality/runaway risk is
checked instead via introspection — `~/.hermes/logs/agent.log`, the `out=`
token count and `finish_reason` per call — rather than by truncating
legitimate long output.

If a visible overflow indicator is wanted again: out-of-band (a response
header or metadata field), **never** via `msg.content` — anything in the
model-visible content gets replayed into the model's own future context.
