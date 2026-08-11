# MTP × tool-calling — sm121-specific failure mode

**Finding (2026-08-09):** `qwen3_5_mtp` speculative decoding, combined with
tool-calling, produced empty answers and aborting tool-call chains in
agent workloads — on the DGX Spark (sm121) only. RTX 5090 (sm120) is not
affected: it has run without MTP since 2026-07-12, which serves as
counter-evidence that MTP itself is not inherently broken — the failure is
specific to this box/build combination. Suspected cause: speculative-
decoding premature-EOS, or a desync between spec-decode and the
`qwen3_coder` tool-call grammar; not further root-caused (MTP is simply
off).

**Action:** MTP disabled on sm121. The flag is kept reversible, commented
out in the source script:
`--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'`.
Reverting costs only decode speed, not prefix-cache reuse.

**Cross-reference:** this is the finding behind the correction applied to
[`recipes/dgx-spark-sm121.md`](../recipes/dgx-spark-sm121.md) on
2026-08-11 — the recipe had drifted (still showed MTP on) after the
on-box script was already updated.
