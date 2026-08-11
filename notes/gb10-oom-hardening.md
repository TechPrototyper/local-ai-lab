# GB10 OOM-hardening (2026-08-11)

**Trigger:** repeated thrash-wedge — sshd starved of CPU/IO, physical
reboot required each time. Not an sshd OOM-kill (`oom_score_adj` was
already `-1000`); the kernel thrashed into swap under memory pressure
instead of killing promptly, which starves everything of CPU/IO time
including sshd's banner exchange (observed: SSH ping ~90 ms, timeouts).

**Root cause:** containers ran with no `--memory` limit (`docker inspect`
showed `Memory=0`) combined with `swappiness=60` + 16 GB swap.

## Key gotcha — unified memory

On GB10, **GPU allocation (weights, KV cache) does not count toward the
container cgroup.** Per-container `--memory` binds only CPU-side RSS —
measured via `docker stats`: the aura container showed ~10 GB CPU-side vs.
~50 GB real GPU footprint. Per-container caps are a leak-guard against a
CPU-side runaway; they are **not** the real safety net for the GPU-side
memory that actually dominates this box.

## The real host guards

1. **`swappiness=10` + `min_free_kbytes=1G`**
   (`/etc/sysctl.d/99-spark-oom.conf`) — primary protection, prevents
   swap-thrash. Verified: a 110 GB memory hog left sshd reachable on
   155/155 SSH polls across 3 runs, down to 5% memory available.
2. **earlyoom** (`-m 6,3 -s 100,100 --avoid '^(sshd|systemd|containerd|
   dockerd)$' --prefer '^(python3|vllm|pt_main_thread)$'`) — active
   backstop, SIGKILLs the largest process below 3% RAM free. `-s 100` is
   mandatory: with swappiness=10 keeping swap empty, earlyoom would
   otherwise never fire on its default swap-based trigger. SIGKILL (not
   SIGTERM) is required because container PID 1 — including vLLM — ignores
   SIGTERM. Verified: killed a 103 GB runaway (exit 137).
3. **Per-container `--memory` == `--memory-swap`** (in the `run-*.sh`
   scripts) — CPU-side leak-guard only (see gotcha above); sized from real
   RSS + headroom, summing to 82 GB across the fleet (≤109 GB host, ≥12 GB
   OS reserve kept free). Verified: a 4g-capped container showed
   `OOMKilled=true` on over-allocation.
4. **`oom_score_adj=-1000`** for sshd/containerd/dockerd — never an OOM
   target.

GPU-side budget (the load the caps above don't cover) is governed instead
by `--gpu-memory-utilization` + `--kv-cache-memory-bytes` per instance.

## Test protocol (mandatory, for every memory-hungry test going forward)

1. All services down (`docker rm -f` on every container).
2. Run the test alone, `--gpu-memory-utilization 0.8`.
3. Bring the fleet back up — only after verifying that over-allocation
   kills the test container cleanly and sshd stays reachable throughout.

Never run a full 27B test alongside the running fleet — that combination
was the original wedge trigger.
