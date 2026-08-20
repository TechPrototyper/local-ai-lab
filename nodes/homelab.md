# The home lab, exposed

The topology the [node profiles](dgx-spark-gb10.md) live in. Everything
below hangs off one high-speed switch; the interesting part is the role
split.

```mermaid
graph TB
    CLIENTS["Clients<br/>agent harnesses, laptops, phones"]

    subgraph CLUSTER["Kubernetes cluster — Talos Linux, GitOps via Flux"]
        MAIN["Main node — RTX 5090<br/>latency tier · LiteLLM gateway<br/>(nodes/rtx-5090.md)"]
        W1["Old Mac, now running Talos<br/>worker node"]
        W2["Old Mac, now running Talos<br/>worker node"]
    end

    SW(("High-speed<br/>switch"))

    SPARK["DGX Spark — GB10<br/>capacity tier · research testbed<br/>(nodes/dgx-spark-gb10.md)"]

    MAC["Mac<br/>bridge into the Apple ecosystem<br/>(MCP gateway)"]

    CLIENTS --> MAIN
    MAIN <--> SW
    W1 <--> SW
    W2 <--> SW
    SW <--> SPARK
    SW <--> MAC
```

## How the pieces relate

- **Main node (RTX 5090)** — the cluster's workhorse: fast interactive
  inference plus the LiteLLM gateway that makes both inference boxes one
  endpoint ([profile](rtx-5090.md)). Also carries the cluster's ingress,
  API management, and the multi-user agent gateway.
- **Worker nodes** — two retired Macs, repurposed as Talos workers: the
  cluster's services don't all need a GPU, and three nodes make it a
  cluster rather than a box with opinions.
- **DGX Spark (GB10)** — capacity tier and research testbed
  ([profile](dgx-spark-gb10.md)). Deliberately *not* a cluster node: one
  box, explicit memory budgeting, no scheduler moving things mid-benchmark.
- **Mac** — the bridge into the Apple ecosystem: exposes Mail, Calendar,
  Notes & Co. to agents via MCP
  ([mac-mcp-gateway](https://github.com/TechPrototyper/mac-mcp-gateway)),
  so cluster-side agents can act on Apple-side data without anything
  Apple running in the cluster.
- **High-speed switch** — the reason two-stage retrieval (embed on one
  box, rerank on another) and gateway overflow routing are latency-free
  in practice.

The pattern in one sentence: **Kubernetes for everything that should be
declarative and multi-user; bare Docker for the one box where memory
layout is the experiment; a bridge host for the ecosystem that won't
containerize.**
