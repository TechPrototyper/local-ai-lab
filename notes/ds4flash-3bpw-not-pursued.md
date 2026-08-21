# Looked at, liked, not pursued: DeepSeek V4 Flash @3.0bpw on one Spark (2026-08-21)

[MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
runs DeepSeek V4 Flash 0731 as a 3.0-bpw EXL3 build (~107 GB) on exactly
this lab's hardware — one GB10 — via the ExLlamaV3/sparkinfer stack with
its own speculative decoding and an MLA-compressed NVFP4 KV cache. Their
reported numbers (theirs, not verified here): 44–47 tok/s decode at 330k
context, 384k max length, exact needle recall at 370k tokens, a
439k-token KV pool. Impressive engineering, and the launcher itself
reviews cleanly (pinned revisions and image digests, local-only
download, checksum verification).

We considered a quality-focused A/B against the production 27B and
decided **not to run it**. The reasons are about *fit for this lab's
operation*, not about the project:

1. **Structural mismatch.** The config is single-sequence
   (`MAX_NUM_SEQS=1`) at 94% memory — a one-session instrument. This
   lab's workload is the opposite: concurrent multi-agent traffic with
   an embedder/reranker/audio fleet resident on the same box. The
   candidate wouldn't replace the production model; it would displace
   the entire node. The only sensible role would be a dedicated
   deep-context specialist — and no current workload needs one.
2. **Quality at 3.0 bpw is the open question, and it's fully open.**
   Needle recall is retrieval, not reasoning; the repo reports no
   reasoning or tool-calling benchmarks. Answering that would be
   original measurement work — for a candidate without a role.
3. **A second serving world** (EXL3/sparkinfer beside vLLM) with its own
   maintenance surface, plus operational requirements (earlyoom off,
   ~10-minute full-context prefills) that conflict with this lab's
   hardening.

Filed with revisit triggers: a genuine long-context workload, a quality
gap in the production model, a ≥4-bpw build with published quality
numbers, or multi-sequence serving in that stack. The A/B harness that
would decide it (tool-conformance K.O. gate → paired GSM →
role-specific set → McNemar) sits ready either way. One datapoint we
keep regardless: MLA's KV economics — ~440k tokens of cache beside
107 GB of weights — is a benchmark for our own KV roadmap.
