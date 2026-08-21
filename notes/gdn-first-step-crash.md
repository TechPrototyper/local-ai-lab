# GDN first-step crash: concurrent prefills in the very first engine step (2026-08-20)

**TL;DR:** On the GB10 production engine (27B hybrid linear-attention model,
NVFP4 KV, CUDA graphs `FULL_AND_PIECEWISE`), the **very first engine step
after boot** crashes if it contains **two or more concurrent prefills** —
`cudaErrorNotPermitted` inside the GDN linear-attention state write, engine
dead, container restart. After **one single request** has gone through, the
same concurrency is stable. Mitigation: a warmup request after every boot,
before any concurrent traffic.

## Symptom

First traffic after a boot happened to be two concurrent greedy requests
(scheduler dump: `step_counter=0`, two `NewRequestData` prefills). The
engine died in the model forward:

```
File ".../mamba/gdn/qwen_gdn_linear_attn.py", line 1452, in _forward_core
    ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)
torch.AcceleratorError: CUDA error: operation not permitted
```

→ `EngineDeadError`, HTTP 500 on in-flight requests, API server shutdown,
container restarted by policy (~2 min outage with a warm compile cache).

## What makes it fire

- It must be the **first executed step** since engine start (`step_counter=0`).
- That step must contain **≥2 concurrent prefills** hitting the GDN
  (linear-attention) path of the hybrid model.
- A single request first, then the same concurrent load: no crash
  (verified immediately after the incident — single OK, then 3/3
  concurrent OK, engine stable since).

`cudaErrorNotPermitted` is the signature of an operation that is illegal
while a CUDA stream capture is in progress — the working hypothesis is a
lazy capture/compile window in the first step that a multi-prefill batch
shape enters on the GDN path. Reproduction count so far: **n=1** (plus the
positive control that single-first is stable). A controlled reproduction is
queued before this becomes an upstream issue.

## Why this matters operationally

A restart policy turns the crash into a loop hazard: engine boots → agents
immediately send concurrent traffic → first step crashes → restart → same
traffic → crash. Anything that reboots the service into live traffic (node
reboot, rollback after an experiment window, autostart after power loss)
walks into this window.

## Mitigation (deployed)

Every boot path now ends with a **warmup**: wait for the health endpoint,
then send exactly one small completion request before the service sees
parallel traffic. Cost: one request. This closes the first-step window
deterministically, whatever the root cause turns out to be.

**Update 2026-08-21:** the controlled repro came back **negative** —
restart without warmup + immediate 2-concurrent prefills did *not*
crash; restart with warmup was stable as expected
([`results/RESULT_gdn_repro.json`](../results/RESULT_gdn_repro.json)).
"First step × concurrent prefills" alone is not sufficient; the
narrowed suspect is a co-factor present at the original incident: a
short-lived **co-resident container with GPU access** during the first
step. Repro count stays n=1, the upstream report stays on hold, and the
warmup stays deployed — it is free and closes the window regardless of
the exact trigger combination.
