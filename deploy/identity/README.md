# identity — one cell, end to end

The [platform track](../../platform/README.md) decides that an agent's
context, skills and permissions depend on two dimensions — *what kind of
agent* and *for whom* — and that the combination, a **cell**, is one Hermes
profile behind one multiplex gateway, resolved from the caller's token. This
directory is the declarative half of that chain as it runs, verified end to
end with a first cell since 2026-08-20.

## The chain

```
(1) caller ─── OAuth2 ───► Keycloak            human: Authorization Code + PKCE
    (app, CLI, agent)                            service: client_credentials
                                                 → JWT with claim hermes_user
        │
(2) Bearer JWT ─────────► API gateway           agents-hermes-api.yaml
                          POST /agents/hermes/…  JWT plan: JWKS of the IdP,
                                                 extractClaims, no keyless
        │
(3) two flow policies:    transform-headers      Authorization := Bearer
                                                   dictionary cell-keys[hermes_user]
                          dynamic-routing        → http://hermes-gateway:8000
                                                   /p/{hermes_user}/<rest>
        │
(4) Hermes multiplex ──► checks the profile key, loads the profile, runs the turn
                          hermes-gateway-*.yaml   /opt/data/profiles/<cell>/
```

Two sentences carry the design: **the path is the specialist** (static: one
API definition per specialist), **the token is the identity** (checked: one
user attribute, `hermes_user`, is the only identity-to-cell mapping there
is). The one API definition serves every cell of the specialist; a new user
requires no gateway change.

Fail-closed at every hop. No JWT, or an invalid one: the plan rejects with
401. A valid JWT without the claim: the expression-language lookup fails,
and there is deliberately no default-profile fallback. A claim for a cell
that was never provisioned: the multiplex gateway answers 404. A token
without a cell can never land in the default profile.

## Declarative vs. live-only — the honest ledger

| Artifact | Where it lives | Here |
|---|---|---|
| API definition (plan, flow policies) | git, reconciled by the API gateway's operator | [`agents-hermes-api.yaml`](agents-hermes-api.yaml) |
| Multiplex gateway (Deployment, default-profile config, Service, PVC) | git, Flux | `hermes-gateway-*.yaml`, trimmed to the multiplex core |
| Grants (what a cell may reach) | git — a declarative file per realm, synced to Keycloak by provisioning | pattern below; not the file |
| Keycloak realm: users, the `hermes_user` attribute, the client scope, one confidential client per cell | **live-only** (the IdP's own store) | — |
| API-gateway dictionary `cell-keys`, the Application + Subscription per cell | **live-only** (management API) | — |
| The profile itself: `config.yaml`, `.env` with `API_SERVER_KEY`, `SOUL.md` | **live-only**, in-pod on the volume | — |

Secrets are never in git and never in an agent transcript. Cell keys are
generated in-pod; IdP passwords are set by the admin in the console. The
live-only rows are exactly the provisioner's job — the open item in the
platform page.

## Grants: the pattern

One file per realm is the source of truth for what a cell may reach; the
sync to Keycloak roles and scopes is the provisioner's. Scope convention
`<service>:<resource>:<action>`, and `subagent:<name>:invoke` for the
escalation rungs. Wildcards are allowed in the file only — they are expanded
at sync time and never appear in a token.

```yaml
version: 2026-08-19
realm: example
users:
  alice:
    hermes_user: alice
    grants:
      - "subagent:*:invoke"      # may climb the escalation ladder (own subscriptions)
      - "calendar:alice:*"       # her own calendar, read and write
      - "documents:*:read"       # the shared document pipeline, read-only
```

Subagent rungs are *user-bound* resources: the cloud subscriptions behind
them belong to a person. Another user climbing them needs their own
subscription, or an explicit grant here.

## Provisioning a cell today: the five manual steps

Automating these is the provisioner. Until then, this is the runbook.

1. **IdP user** (if new): attribute `hermes_user=<user>`, roles per the
   grants file.
2. **Cell client**: `cell-hermes-<user>`, confidential, service account on,
   the `hermes` scope as default; the service-account user gets the same
   `hermes_user` attribute.
3. **Profile** in-pod at `/opt/data/profiles/<user>/`: `config.yaml` (with
   `platforms.api_server.enabled: false` — mandatory in the multiplex, or the
   profile is skipped fail-closed), `.env` with an in-pod generated
   `API_SERVER_KEY` (mode 600), `SOUL.md`. Then restart the gateway
   deployment: the multiplexer enumerates profiles at start.
4. **Dictionary entry** `cell-keys[<user>] = <API_SERVER_KEY>` via the
   management API — read from the profile's `.env`, never echoed.
5. **Application + Subscription** on the API gateway: `client_id =
   cell-hermes-<user>`, subscribed to the JWT plan.

## The 401 matrix

From the verification tests. Which hop said no tells you what to fix.

| Symptom | Hop | Cause |
|---|---|---|
| 401, body `{"message":"Unauthorized"}` | API gateway | JWT missing/invalid, **or** the Application has no subscription (check `client_id`) |
| 401 only after routing | Hermes | dictionary entry missing or stale — profile key rotated without a dictionary update |
| 404 at Hermes | Hermes | cell not provisioned, or profile not enumerated (gateway restart after creating it?) |
| log `Skipping secondary profile … port-binding` | Hermes | `platforms.api_server.enabled: false` missing in the profile's config |
| IdP logins fail globally, `unknown_error` | IdP | its volume went read-only once; check volume health, a pod restart remounts rw |
| the `hermes_user` attribute vanishes | IdP | declarative user profiles drop unknown attributes; `unmanagedAttributePolicy` must allow admin edits |

`/health` and `/p/<cell>/health` are unauthenticated; test auth against
`/p/<cell>/v1/models`.

## What is generalized

The realm is `example`; the JWKS URL and the gateway target keep their
cluster-internal DNS names because those *are* the reference. The image tag
points at the public carry-patch fork ([`harness/`](../../harness/README.md));
the cluster builds it in-cluster and pulls from a private registry. The
default profile's config is trimmed to model, gateway, and guardrails; the
escalation-ladder rungs are placeholders because they name subscription
products.
