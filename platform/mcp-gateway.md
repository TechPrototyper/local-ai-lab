# MCP gateway: why nothing fit, and what we shipped

*September 2026.* The [agent platform track](README.md) needs one front door for
many agents. This note is about the door itself — thirteen MCP servers behind a
four-hop chain, a field of fourteen gateway projects, and why the only thing that
helped in the end was a twelve-line change we had to write ourselves.

It is written the way the [inference track](../README.md) is: what broke, what we
measured, what the numbers said.

## The chain, and the seven ways it fails

```
agent → API gateway → cluster aggregator → Mac adapter → stdio MCP servers
```

Ten of the thirteen servers live on a Mac and speak stdio: OmniFocus, Mail,
Photos, Calendar, Messages, and friends. They are child processes with AppleScript
access and no network port at all, so something on that Mac has to spawn them and
put an HTTP face on them. That adapter is not optional and never was. Both proxy
hops run [`tbxark/mcp-proxy`](https://github.com/TBXark/mcp-proxy), v0.58.0 in the
cluster and v0.43.2 on the Mac.

Seven failure modes, all measured on our own chain between 6 and 9 September:

| # | Failure | Evidence |
|---|---|---|
| 1 | No reconnect after upstream loss | A pod restart dropped the `ttp` route out of the table. `404` until the proxy itself was restarted. |
| 2 | No initialize timeout | A synchronous reindex tool blocked the upstream's event loop. The proxy logged `Connecting`, never `Connected`. The route was never mounted, and a rollout restart would not have helped. |
| 3 | Health lies in both directions | 7 Sep: `status: ok`, serverCount 12, with the route dead. 9 Sep: `degraded, unhealthy: ["ttp"]` with the same route answering 200 and listing 18 tools, upstream idle at 2 millicores. |
| 4 | No SSE keep-alive | Two sessions held open for 300 s, one through the gateway and one direct. **124 bytes each** — the endpoint event and nothing else. |
| 5 | No authentication | `KEY_LESS`, zero policies. All thirteen collections open to anything on the network. |
| 6 | No per-tool rules | DEVONthink alone exposes 59 tools, Monkey Office 55. All or nothing. |
| 7 | Chained callback URLs | Each hop carries the next hop's public address in its config. A client connecting straight to the Mac is handed a cluster-internal DNS name it cannot resolve — the stream opens with 200 and the first tool call goes nowhere. |

Number 7 is worth dwelling on, because it explains a behaviour we kept seeing:
agents that hit a `404` at the gateway would "find another way" to the Mac, and
that way looks healthy right up until it isn't. The bypass was broken by
construction.

## What we already owned, and why it did not help

Two commercial gateways were already deployed. Neither could speak MCP for free.

**Gravitee APIM 4.11.25.** The MCP plugins are installed and sitting there:
`entrypoint-mcp-proxy` 2.2.3, `endpoint-mcp-proxy` 2.2.3, `mcp-tool-server` 2.0.2,
and a `policy-mcp-acl` 1.0.4 that does exactly the per-tool rules we wanted. All of
them declare `feature=apim-mcp-proxy-reactor` in their manifests, with classes under
`com.graviteesource`. The free plugins carry no `feature` line and live under
`io.gravitee`. The gateway logs `0 licenses synchronized`. Our MCP routes therefore
run as plain HTTP pass-through — Gravitee is not acting as an MCP gateway at all,
it is a reverse proxy with a good name.

**Kong 3.9.3, community image.** The AI plugins are present (`ai-proxy`,
`ai-prompt-guard`, the transformers). There is no MCP plugin. Kong's
[`ai-mcp-proxy`](https://developer.konghq.com/plugins/ai-mcp-proxy/) needs Kong 3.12
or newer and AI Gateway Enterprise. Moving to Kong OSS would also have cost us the
admin console and developer portal, which Gravitee ships for free.

So: two gateways, both with the capability visibly installed or advertised, both
gating it. That is the state of the commercial market as of September 2026.

## The open-source field

Fourteen projects examined, nine seriously. Star counts and last-push dates queried
against the GitHub API on 9 September 2026, not taken from blog posts.

| Project | Why it did not fit |
|---|---|
| [MCPJungle](https://github.com/mcpjungle/MCPJungle) | MPL-2.0, but per-client access control is gated behind an "Enterprise Mode". SSE backend support described as not mature in its own docs. |
| [Obot](https://github.com/obot-platform/obot) | MIT, and genuinely capable — but the free edition authenticates against Google and GitHub only. SAML and OIDC are Enterprise. With Keycloak as our IdP, that is the one gate we could not accept. |
| [MetaMCP](https://github.com/metatool-ai/metamcp) | Nicest UI model of the lot. No commit since 22 June 2026, PostgreSQL required, 2–4 GB RAM suggested, no Kubernetes documentation. |
| [Docker MCP Gateway](https://github.com/docker/mcp-gateway) | Its own documentation scopes it to local development, not server deployment. Config lives in a local database. |
| [Agent Router](https://github.com/theagentrouter/agent-router) (ex Envoy AI Gateway) | Streamable HTTP only, no documented stdio backend — the Mac adapter would have stayed regardless — and a full Envoy substrate underneath. |
| [ToolHive vMCP](https://github.com/stacklok/toolhive) | The cleanest Kubernetes model here. Its vMCP docs are silent on health checks *and* transports, which are two of our seven. A Squid sidecar per server does not scale down to a home lab. |
| [IBM ContextForge](https://github.com/IBM/mcp-context-forge) | Apache-2.0, a real registry, RBAC to tool level, and the only project that documents **SSE keep-alive** explicitly. Config lives in a database plus 100+ environment variables, which takes it out of Git. Kept as the fallback. |
| [`tbxark/mcp-proxy`](https://github.com/TBXark/mcp-proxy) | What we run. MIT, small, honest. Its reconnect PR [#63](https://github.com/TBXark/mcp-proxy/pull/63) has been open since 14 June 2026; the tool-filter bug [#53](https://github.com/TBXark/mcp-proxy/issues/53) since 31 January. |
| [agentgateway](https://github.com/agentgateway/agentgateway) | Apache-2.0, Linux Foundation, Rust, single binary. Answered six of seven. |

The pattern across the discards is worth naming, because it is not "these are bad
projects". Two are open-core at precisely the feature a self-hosted operator needs.
One is dormant. One is honest about being a desktop tool. Two are architecturally
mismatched. And the one we run is fine code whose maintenance has stalled at exactly
the two issues that hurt us.

## agentgateway, and the one gap

agentgateway answered six of the seven with a documented mechanism: retries and
outlier detection for #1, `failureMode: FailOpen` for #2, OTel and Prometheus instead
of a hand-rolled health route for #3, JWT/OAuth/API keys for #5, CEL rules per tool
that also drop denied tools out of `tools/list` for #6, and one config surface —
a flat YAML standalone or Gateway API objects in Kubernetes — for #7. Standalone
mode matters here: it means the Mac adapter and the cluster aggregator become the
same binary with two configs, and the chained-baseURL problem disappears.

The gap was #4. Nothing in the documentation mentioned SSE keep-alive.

So we read the source at `3d5f59f`.

```rust
// crates/agentgateway/src/mcp/session.rs
pub(crate) fn sse_stream_response(
    stream: impl futures::Stream<Item = ServerSseMessage> + Send + 'static,
    keep_alive: Option<Duration>,
) -> Response {
```

It was there. The mechanism was built and correct. The only caller passed `None`,
and no config field existed anywhere to set it. A second stream — the legacy SSE
`GET` path in `mcp/sse.rs` — constructed its response with `Sse::new(stream)` and
never called axum's `keep_alive` at all.

The gap was not a missing feature. It was a missing switch.

That reframes the build-or-buy question entirely. We had been asking whether to
write our own gateway: Kubernetes-native with an operator, MCP and A2A outward,
Keycloak identity, YAML config, no database. That description is agentgateway,
minus a switch. The expensive part of an MCP gateway is not the routing — it is
session handling across two channels, resumable streams, OAuth resource-server
semantics, and chasing a spec that has moved four times since November 2024. The
cheap part is the operator, and the cheap part does not make a product.

## What we shipped

[**agentgateway#3393**](https://github.com/agentgateway/agentgateway/pull/3393) —
`sseKeepAlive` on the MCP backend, wired into both stream construction sites,
plumbed through local config, the xDS path, `McpBackendGroup` and `UpstreamGroup`
following the existing `sessionIdleTtl` pattern. Unset preserves current behaviour,
so it is opt-in and no existing deployment changes.

The motivating case is inside agentgateway itself, not just our chain. When all
upstream GET streams are gone, `FailOpen` deliberately hands the response
`Messages::pending()` so legacy clients do not reconnect in a tight loop. That
connection then carries zero bytes for its entire life — the exact shape a load
balancer reaps.

Measured against a real MCP SSE upstream, two 70-second sessions:

| Config | Bytes | Keep-alive comments |
|---|---|---|
| `sseKeepAlive: 10s` | 93 | 6 |
| unset | 75 | 0 |

`cargo test -p agentgateway mcp::` — 295 passed, 0 failed, including a new unit test
that covers the `pending()` case directly: a stream that never yields must still
produce comment frames.

Two more contributions are scoped from the same reading, both marked in the code by
its own maintainers or absent by design: free per-tool renaming (`McpPrefixMode` does
prefixes per target, nothing finer), and caching of list results (`TODO cache list
results` sits in `handler.rs`).

## Why contribute instead of fork

The honest answer is that we were already living in the alternative. `tbxark/mcp-proxy`
is MIT and small enough to fork comfortably. Its reconnect fix has been proposed and
unmerged for three months, and its virtual-server request has been open for fourteen.
Forking would have handed us a permanent maintenance line for a protocol whose spec
keeps moving, in exchange for solving two problems once.

agentgateway is set up for the other path: a contribution guide, a development guide,
a Linux Foundation charter, PR and issue templates, automated dependency upkeep. The
MCP module is about 8,300 lines excluding tests, which is small enough to work in
confidently. A twelve-line switch that closes an obvious gap costs us one afternoon
and costs us nothing afterwards, because we do not maintain the engine.

There is also a straightforward self-interest argument. Every operator running an MCP
gateway behind a load balancer has this bug and most have not measured it yet. The
fix being upstream means we get it back with every release, tested by people who know
the codebase better than we do.

## What this does not solve

The API gateway stays. Of the ten APIs published on ours, only two are MCP — the other
eight include LiteLLM and four geo services already running Keycloak-backed JWT plans.
agentgateway replaces the *aggregator*, not the gateway, and anyone reading this note
as "one tool replaced everything" is reading it wrong.

Three of our seven are still open on our side: honest health reporting depends on
metrics we have not wired, per-tool rules depend on a CEL policy we have not written,
and the Mac adapter is still v0.43.2 with a 49 MB unrotated error log and a launch
agent that retries a doomed second start every thirty seconds.

Those are ours to fix. This one was everyone's.
