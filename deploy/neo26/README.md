# neo26 — the RTX 5090 serving line on Kubernetes

The production vLLM serve from [`nodes/rtx-5090.md`](../../nodes/rtx-5090.md),
as the Flux-managed objects that actually run it. The serve *command* and its
flags are explained in
[`recipes/rtx-5090-sm120-gridbook-mtp.md`](../../recipes/rtx-5090-sm120-gridbook-mtp.md);
this page is about everything around the command that lets it survive a node
reboot, a runaway job, and a first cold start.

## The objects

| File | Object | What it is for |
|---|---|---|
| `namespace.yaml` | Namespace `inference` | PSA privileged (device access, memory-backed shm), warn at baseline |
| `limitrange.yaml` | LimitRange | 16Gi default cap for any container that forgets its own — the backstop after an uncapped quant job starved the node |
| `gpu-power-limit.yaml` | DaemonSet (privileged, `gpu-operator` ns) | `nvidia-smi -pl 550` at node start — the fix for months of "driver crashes" that were PSU over-current trips |
| `vllm-cache-pvc.yaml` | PVC, 150Gi, prune-disabled | weights, HF cache, FlashInfer/Torch JIT caches; never garbage-collected |
| `vllm-deployment.yaml` | Deployment `vllm-qwen38` | the serve, with a power-cap gate in front of it |
| `vllm-service.yaml` | Service (ClusterIP) | the serve's own endpoint |
| `llm-active-service.yaml` | Service `llm-active` | the engine switch — what the gateway talks to |

## Three things worth reading closely

**Cap before load.** The DaemonSet sets the power cap, but after a reboot it
only gets to run once CNI and the GPU operator are up — and the serve might be
scheduled first. So the serve carries an initContainer that polls
`nvidia-smi --query-gpu=power.limit` and blocks until it reads 550 W. It fails
open after about three and a half minutes with a loud warning, because a
serve that starts uncapped is a risk while a serve that never starts is an
outage. This is the Kubernetes form of the power lesson in the node profile:
the crash that looked like a driver fault was the PSU.

**One GPU, one serve, changed in git.** `replicas` is only ever changed by
commit. `kubectl scale` is reverted by Flux at the next interval, and that is
the point: the git log of the deployment file is the record of when the card
was handed to research and when it came back. The passive llama.cpp fallback
and any experiment pod share the same card; the rule is never more than one
at replicas 1, and cutover is old-to-0 before new-to-1 (`strategy: Recreate`
enforces the same thing at pod level).

**The switch is a selector.** `llm-active` is the only Service the gateway
knows. Which engine answers is its selector. Engine cutover and rollback are
a one-object change with the port fixed at 8000 for every client; the
llama.cpp fallback only differs in `targetPort`. The gateway's resilience
semaphores are a separate layer on top of this
([`../inference-gateway/`](../inference-gateway/README.md)).

## Reproducing this

- Pull the public image named in the deployment (roster and provenance:
  [`containers/`](../../containers/)). The cluster pulls the same build from a
  private mirror; nothing else differs.
- Prepare the checkpoint per the recipe onto the PVC under `/cache/models/`.
- Check whether your image needs the three FlashInfer submodule mounts at the
  bottom of the deployment. They exist because one image line was built
  without `--recurse-submodules`; if `flashinfer/data/cutlass` is populated
  in your image, delete the three mounts.
- First start on a cold cache compiles the sm120 kernels: the startupProbe
  allows 25 minutes for a reason. A warm start takes about three.
- `nodeSelector: kubernetes.io/hostname: neo26` — rename to your node.

## What is not in this directory

The GPU operator, Longhorn, Flux itself, the private registry, the monitoring
that scrapes `:8000/metrics`. Those are the cluster.
[`../inference-gateway/`](../inference-gateway/) next door is the other half
of what a client sees.
