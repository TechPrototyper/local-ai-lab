# Deploy — reference manifests

The lab's production serving line, as Kubernetes objects. Everything in
[`nodes/`](../nodes/), [`recipes/`](../recipes/) and
[`containers/`](../containers/) describes *what* runs and *why*; this
directory is *how it is wired*: the Flux-managed manifests for the RTX 5090
node (`neo26`) and the two-tier inference gateway in front of it and the GB10.

These are extracted from the private GitOps repository that Flux reconciles
against the live cluster — not a mirror of it. The private repo carries dozens
of namespaces, an operational diary in every comment, and the network. This
directory carries one readable reference: the objects that make local
inference production-grade, with the reasoning kept and the diary cut.

## What is generalized

| In the cluster | Here | Why |
|---|---|---|
| Node-local IPs (the GB10, the Jetson, the LoadBalancer VIP) | DNS-style placeholders under `example.internal`; the VIP commented out | network layout is not the point |
| Private registry image tags | the public GHCR tags from [`containers/`](../containers/) — the same serving line | anyone can pull these |
| Internal domain and CA | `example.internal`, `internal-ca` | yours will differ |
| Secret *names* | kept (`hf-token`, `litellm-secrets`, …) | the pattern matters; no value ever appears |
| Date-stamped operational comments (who flipped what, when) | replaced by the technical rationale | the *why* is reference material; the *when* is the private repo's git log |
| Downstream consumers by name | `downstream-worker` | the backpressure pattern is the point, not the tenant |
| Model names hard-coded in callbacks | env-configurable, same defaults | so the gate is reusable without editing Python |

`neo26` stays. It is the node's hostname label, it is the name used
throughout this notebook, and a `nodeSelector` without a real key is not a
reference.

## Trust boundary

This is a single-operator LAN. The inference gateway in `inference-gateway/`
is **keyless on purpose**: it is reachable only from inside the cluster
network, and every caller that can reach it has already been admitted
somewhere else. Authentication lives at the two places where an identity
actually enters the system — the API gateway (JWT plan against the IdP) and
the cell identity chain in `identity/`, which turns a token into exactly one
profile. Nothing between those hops and the model server re-checks who is
calling; the model server sees a request, not a person. Keyless-internal is
therefore a scoping decision, not an oversight: the boundary is drawn at the
entrance, and the reference documents where. What a deployment with a
regulated perimeter adds on top of this is spelled out in
[`identity/README.md`](identity/README.md#what-a-regulated-deployment-adds).

## What is deliberately not here

- The other namespaces: identity, storage guardrails, monitoring, the
  document pipelines, the tenants' applications. Those are the cluster, not
  the serving reference.
- Talos machine config, the Flux bootstrap, the vault and secret-operator
  wiring.
- The llama.cpp fallback stack (passive, 0 replicas). The `llm-active`
  switch in `neo26/` shows where it plugs in.
- Anything with a value that should not leave the house. Every file here was
  reviewed line by line for that before it landed.

## Layout

```
deploy/
├── flux-kustomization.yaml     the Flux Kustomization that reconciles this tree
├── kustomization.yaml          → neo26 + inference-gateway
├── neo26/                      the RTX 5090 serving line (vLLM on Kubernetes)
├── inference-gateway/          LiteLLM: two boxes, one endpoint, measured routing
└── identity/                   the cell identity chain, verified with a first cell
                                (other namespaces; built alone, not from the root)
```

Apply order is Flux's business (`dependsOn`, `wait: true`). By hand it is
`neo26` first — namespace, PVC, power-cap DaemonSet, serve — then the
gateway. `identity/` lives in the platform's namespaces and is reconciled
by a different Flux Kustomization in the cluster; it is here because the
[platform page](../platform/README.md) claims the chain works, and a claim
without the objects is what this directory exists to end. Each
subdirectory's README says what every object does and which incident put it
there.
