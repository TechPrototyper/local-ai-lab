# The edge node: Jetson Orin Nano

The lab's smallest box — 8 GB, ARM64, JetPack R36 — with a deliberately
narrow job: embeddings, always on, off the main GPUs' budgets.

## The box

- **Jetson Orin Nano (8 GB)**, aarch64, JetPack R36.5. Hangs off the same
  high-speed switch as everything else ([topology](homelab.md)).
- Not a cluster node — like the Spark, it runs bare: two `llama-server`
  instances, GPU-offloaded, nothing else competing for its 8 GB.

## Serving

- **BGE-M3** (F16 GGUF) — dedicated embeddings server, 8k context,
  full GPU offload.
- **nomic-embed-text-v1.5** (F16 GGUF) — second embedder, served from a
  multi-model preset that also keeps **Gemma-4-E4B** loadable as a small
  edge LLM (q4 KV cache on both).

The division of labor this buys: embedding queries never touch the RTX or
the GB10, so retrieval stays warm while the big boxes benchmark, reboot,
or serve at full memory budget. The switch makes the hop free in practice.

## Next role

The Orin is the designated platform for the lab's **out-of-band management
tool** — a controller that watches and manages the inference boxes from
*outside* their own failure domain. The lab has had exactly the incidents
that motivate one (PSU transients presenting as driver crashes, earlyoom
loops, a shutdown fired 14 minutes early); a box that never benchmarks is
the right place to put the thing that must survive the benchmark going
wrong. Design and build: not started.
