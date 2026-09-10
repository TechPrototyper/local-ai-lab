# The inference gateway: two boxes, one endpoint

*September 2026.* The RTX 5090 and the DGX Spark have opposite bottleneck
orders ([the framing](../../README.md#the-framing)): one is memory-bound and
fast, the other memory-rich and slower. Clients should not have to know. A
LiteLLM proxy in front of both makes them one OpenAI-compatible endpoint, and
the interesting part is not the proxy — it is the policy that decides which
box answers, the semaphores that take a box out of the pool honestly, and the
measurement that set the routing threshold.

This is the counterpart to the [MCP gateway](../../platform/mcp-gateway.md)
on the platform track. That one is the door agents come *in* through; this
one is the door their tokens go *out* through.

## Topology

```
clients (harnesses, IDEs, pipelines)
        │
        ▼
LiteLLM  :4000  ── four callbacks on every request ──┐
        │                                            │
        ├─► llm-active ──► vLLM on the RTX 5090      latency tier, primary
        │                  (../neo26/)               ~60 tok/s single, 262k, NVFP4-KV
        │
        ├─► *-overflow ──► vLLM on the DGX Spark     capacity tier, overflow
        │                  (nodes/dgx-spark-gb10.md) 128 GB unified, 262k
        │
        ├─► Jetson: nomic-embed, bge-m3              edge tier, always on
        └─► Spark: qwen3-embed-8b, rerankers
```

Every local chat alias — `qwen3.8-27b`, `-no-think`, `-tools` — exists twice
in [`config.yaml`](config.yaml): plain, and with an `-overflow` twin pointing
at the Spark. Clients only ask for the plain name. Two mechanisms decide when
the twin answers instead:

| Case | Mechanism | Where |
|---|---|---|
| RTX **busy** | the concurrency gate rewrites the model name before dispatch | [`overflow_notice.py`](overflow_notice.py) |
| RTX **unreachable** | LiteLLM's native `fallbacks` | `config.yaml` |

They are different failures and they need different tools. `fallbacks` only
fires on an error; a full box does not error, it queues. Deployment-level
`max_parallel_requests` would have been the built-in answer, but in this
LiteLLM line it does not trigger a fallback without Redis. Hence the callback.

## The gate, and the measurement that set it

The gate keeps a count of local requests in flight in a process dictionary
and sends the next request to the twin once the count reaches `MAX_LOCAL`.
The dictionary is correct only because the proxy runs `--num_workers=1`;
moving to multiple workers means moving that state to Redis first, and the
manifest says so.

`MAX_LOCAL` was 4, with the reasoning "local parallel capacity equals
`--max-num-seqs 4`". That reasoning is wrong, and the way it was found out is
the method this notebook runs on.

The operator observed that inference *blocked* when the RTX was full, and
asked whether the overflow was firing too late. `--max-num-seqs` is a
ceiling, not a capacity. What the engine actually runs is in
`vllm:iteration_tokens_total`, sampled over 4.7 days and 7,745 requests:

| Tokens per scheduler iteration | Share | Cumulative |
|---:|---:|---:|
| ≤ 1 | 19.05% | 19.05% |
| ≤ 8 | **79.38%** | **98.42%** |
| ≤ 16 | 1.25% | 99.68% |
| > 256 (prefill) | 0.30% | 100% |

With MTP at two speculative tokens, every running sequence needs three
tokens per iteration. Eight tokens or fewer is one or two sequences. In
98.4% of all iterations the RTX runs one or two sequences, not four. The
batch budget (`--max-num-batched-tokens 4096`) is never the constraint —
99.9% of iterations come nowhere near it.

The live cross-check, eight samples six seconds apart, both boxes at once:

```
RTX run/wait = 1/0   3/0   2/1   1/1   1/1   2/1   3/0   3/0
Spark run/wait = 5/0 4/0   4/0   3/0   4/0   4/0   5/0   3/0
```

The RTX shows waiting requests at `running` = 1–3, i.e. *before* a
`MAX_LOCAL` of 4 could ever fire. `vllm:num_requests_waiting_by_reason`
names the reason: `capacity`. Preemptions: 0. KV cache: 4–13%. It is a
slot-and-scheduling limit, not memory.

What it is **not**: not a KV problem (4–13%, no preemptions), not a
reachability problem (the Spark answered, the overflow targets were
correct), not a mean-latency problem (93.4% of requests wait ~0 s, p99 ≈ 5 s).
It strikes in the load peaks — which is exactly where it is noticed.

**Change:** `MAX_LOCAL` 4 → **2**. Rollback is one line. Before the change,
the Spark was already carrying six concurrent requests at 62% KV utilisation
from other traffic; where that base load comes from and how much headroom is
left is the open question the gate metrics below exist to answer.

## Two more findings from the same afternoon

**The TTL had drifted.** In-flight entries expire after `ACTIVE_TTL`, which
the code claimed equals LiteLLM's `request_timeout`. `request_timeout` had
been raised to 3600 s for long builder turns; `ACTIVE_TTL` stayed at 600.
Since then a request running longer than ten minutes fell out of the counter
*while still holding a vLLM slot*, and the gate under-counted. Only 2 of
7,745 requests crossed even 480 s — but they are precisely the long turns the
timeout was raised for. The bug was waiting for its use case. Both values are
env on the Deployment now, next to each other, with the invariant in the
comment.

**The gate was blind.** There was no metric for the in-flight count and none
for the routing decision. "Is the overflow firing at the right moment?" could
only be reconstructed from two vLLM queues. Now exported on `:9105`:

| Metric | Meaning |
|---|---|
| `litellm_overflow_local_inflight` (gauge) | current `len(_active)` |
| `litellm_overflow_decisions_total{decision,model}` (counter) | `local`, `overflow_max_local`, `overflow_forced`, `local_spark_down` |
| `litellm_greedy_sampling_requests_total{model,reason}` (counter) | client-set cold sampling detected |
| `litellm_greedy_coerce_total{model,reason}` (counter) | …and rewritten to the anti-loop profile |

Metrics must never endanger routing: if the client library is missing or the
port is taken, the gate runs without them and says so once in the log.

## The semaphores

Three environment variables on the [Deployment](deployment.yaml). Because
they are GitOps, every flip is a commit, and the git log of that file is the
record of when a box was handed to research and when it came back.

| | Meaning | Effect on a local request |
|---|---|---|
| `SPARK_DOWN=1` | capacity tier out of the pool | counted locally, queues on the RTX; **no** overflow to a dead box |
| `RTX_DOWN=1` | RTX out of the pool | rewritten to `-overflow` immediately |
| both | nothing local | **HTTP 503 with `Retry-After`**, immediately |
| `SPARK_FORCE=1` | test window | everything to the Spark regardless of load |

Precedence: both-down, then `RTX_DOWN`, then `SPARK_DOWN` before
`SPARK_FORCE` before `MAX_LOCAL`. The 503 is a design decision: when both
boxes are out, agents get told, immediately, instead of hanging on a dead
backend or being silently sent to a cloud model. There is no silent cloud
path in this gateway.

**Backpressure downstream.** The semaphores are also read by pipelines that
do not talk to the gateway at all. A document/OCR worker in another namespace
gets a read-only `Role` on the litellm Deployment
([`rbac-semaphore-reader.yaml`](rbac-semaphore-reader.yaml)), reads the
three variables through the Kubernetes API, and queues its jobs while both
are set — releasing them on its own once one is lifted. No polling of a dead
endpoint, no retry storms, no coordination service. The Deployment's
environment *is* the coordination.

## What a request passes through

Four callbacks, in this order, all stateless, all defensive (an unexpected
error passes the request through unchanged; the one intentional exception is
the 503).

**[`overflow_notice.py`](overflow_notice.py)** — the gate and the semaphores
above, plus the greedy-sampling guard: the local model loops under
near-greedy sampling (tool-call relays, chunk loops), profiles and server
defaults are set correctly, but a client that sends `temperature` explicitly
overrides both. Cold sampling on the local model is counted and, with
`GREEDY_COERCE=1`, rewritten to the anti-loop profile (0.7 / 0.8 / presence
penalty 1.5).

**[`loop_breaker.py`](loop_breaker.py)** — force-progress for tool-call
loops. The harness sends the full history every turn, so the loop is entirely
inside the request. On an identical run (≥3) or a cycle (period 2–5, ≥2
repetitions) at the tail of the assistant tool-call turns: the repeated pairs
are cut from the forwarded messages (first occurrence kept, only complete
tool-call/result pairs), a directive is appended, `tool_choice` is set to
`none` so the model *must* produce text, and the intervention is signalled
out of band in a response header. The guard that took an incident to learn:
a loop is only a loop if the *most recent* assistant turn is part of it.
Without that check the breaker fired on every subsequent request — the
harness kept replaying the old loop region — and the agent could never call a
tool again.

**[`system_message_normalizer.py`](system_message_normalizer.py)** — the
Qwen template rejects a system message that is not at index 0; some clients
inject them mid-history and resend the full history on every turn, so every
retry fails identically. Non-leading system messages become user messages
with a `[System note]` prefix, in place. Order is meaning; nothing is moved.

**[`custom_auth.py`](custom_auth.py)** — keyless for every model served in
the house, 401 for everything else, and no key database: it cost resources
and every DB hiccup was an incident, so one master key (admin API only) under
GitOps is the decided steady state. A client that insists on sending *some*
key is treated as keyless.

## Lessons that became invariants

Each of these is in a code comment where it applies; collected here because
they generalise beyond this gateway.

- **Never inject into `msg.content`.** An earlier version prepended an
  "overflow mode" banner to responses. Agents replay their context every
  turn; the banner saturated it; the model repeated the banner forever.
  Signals go out of band — header, metadata — or not at all.
- **The in-flight TTL must equal the request timeout**, and the two must
  live next to each other so they cannot drift.
- **Metrics must never endanger routing.** Import failure, port conflict:
  log once, route anyway.
- **A loop is only a loop if it is running now.** Detection on a replayed
  history needs a liveness guard or it fires forever.
- **Streaming hooks must handle both iterable kinds.** An async-only
  pass-through swallowed the stream when the response object was not
  async-iterable; the client saw zero tokens until timeout.
- **Do not patch `request._receive`.** Auth may be called more than once per
  request; `request.body()` is cached and idempotent, a patched receive is
  not, and the SSE consumer hangs on the second call.
- **A dead backend gets an honest 503**, not a silent fallback.

## What is generalised here

Box addresses are `spark.example.internal` and `jetson.example.internal`
(the cluster uses node-local IPs); the LoadBalancer VIP and the ingress host
are placeholders; the harness gateway alias and its secret reference are kept
because they show how the [harness track](../../harness/README.md) is
published behind the same endpoint. Model names in the callbacks come from
`LOCAL_MODEL_PREFIXES` instead of being hard-coded. Everything else is as it
runs.
