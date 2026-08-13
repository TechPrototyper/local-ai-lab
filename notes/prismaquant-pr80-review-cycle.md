# PrismaQuant PR #80: two bugs, two review rounds, approved for production

Record of an upstream fix cycle worth keeping — both for the substance
(hybrid linear-attention masking under transformers ≥5.13) and for two
generalizable engineering lessons it produced.

**PR:** [RobTand/prismaquant#80](https://github.com/RobTand/prismaquant/pull/80)
— "Fix hybrid linear-attention masking in layer streaming for Qwen3.5/3.6 on
transformers >= 5.15". Status: **MERGED** (2026-08-13T15:27Z, after [review](https://github.com/RobTand/prismaquant/pull/80#pullrequestreview-approved)),
head `b09f968`); announced follow-up: a fresh PrismaQuant release.

## The underlying problem

PrismaQuant's layer-streaming forward fed hybrid models a single dense
`[1,1,T,T]` causal mask. Qwen3.5/3.6 (GatedDeltaNet) `linear_attention`
layers expect a 2D padding signal instead — the dense mask broadcasts
against `hidden_states` and crashes on transformers ≥5.15 (which removed a
shape guard that had silently absorbed the mismatch before). A second bug
sat behind the first: the transformers 5.13 rename of the hybrid layer-type
attribute (`.layer_type` → `.block_type`) left recurrent layers unresolvable,
tripping the streaming code's fail-closed guard. Full trace in the quant-run
notes ([`qwen36-aura-head-to-head.md`](qwen36-aura-head-to-head.md) covers
what the fixed pipeline then produced).

## Review round 1 (maintainer: Rob Tand)

Three requested changes, all sustained on inspection:

1. Route `linear_attention` through transformers' recurrent-mask contract
   (trim growing continuation masks, `None` for single-token/all-ones/non-2D)
   rather than passing the raw incoming mask — the raw mask can itself
   mismatch on cache continuation.
2. No structural `hasattr` guessing for layer types; resolve the recurrent
   child through the existing attribute/index/config lookups and keep
   unknown layers fail-closed.
3. Regression tests for every contract point.

Rework: contract mirrored from the transformers v5.15.0 source, lookup-only
resolution, two test files (contract fakes + integration against real
`Qwen3_5` modules on CPU), verified across transformers 5.8/5.15 on
macOS/arm64 and GB10 (aarch64), plus a fresh end-to-end run of the AURA
quantization start on GB10 against the real 27B: the previously-crashing
calibration forward walked the full 64-layer hybrid stack (48 linear + 16
full) cleanly, phase-2 complete, phase-3 underway when bounded.

## Review round 2 — the deeper catch

The rework gated on *helper import*: use upstream
`create_recurrent_attention_mask` when importable, else a local fallback.
The maintainer caught what that misses: **the helper exists from
transformers 5.13, but 5.13.0–5.14.1 ship it with the pre-fix contract**
(returns `None` whenever the cache has previous state — silently dropping
padding from multi-token continuations, the exact recurrent-state
corruption 5.15 fixed). He reproduced our suite failing in his own 5.13
environment. Fix: the local shim is now used on **every** version
(implementing the 5.15/current observable contract), with a booby-trapped
regression proving the upstream helper is never consulted.

Also in round 2: declared non-attention block types (`moe`/`mlp`,
Nemotron-H-style schedules) now receive `None` via an explicit allowlist —
mirroring upstream's `.get(block_type)` dispatch without weakening
fail-closed handling — and the CPU integration tests neutralize optional
CUDA kernel extensions. That last fixture the maintainer refined **himself**
with a commit onto the PR branch
([`85d40d5`](https://github.com/RobTand/prismaquant/pull/80/commits)),
validated against a GPU-visible CUDA/FLA matrix across 5.8/5.13/5.14.1/5.15.
While pinning the moe/mlp case we found one adjacent gap the review hadn't
flagged (non-attention schedules *without* recurrent layers still took the
single-dense-mask path) and fixed it in the same revision.

## The lessons (generalized)

1. **API existence is not a compatibility gate.** `try: import X` proves a
   symbol exists, not which contract it honors. Contracts are versioned;
   when in doubt, implement the current contract locally and prove with a
   test that the versioned symbol is never consulted.
2. **A green screening run only certifies the paths it exercises.** The
   original fix looked fully validated — GSM8K parity, needle, determinism —
   because the quantization pipeline runs unpadded single-request batches,
   which structurally cannot reach the padded/continuation code paths the
   review targeted. Tests belong on the failure surface, not the happy path.

Final state: four-version test matrix green (5.8: 22✓, 5.13: 23✓, 5.14.1:
23✓, 5.15: 24✓, version-gated skips as expected), full suite 3661✓, the
maintainer's independent GPU matrix green, approval with the n=1319
head-to-head explicitly welcomed as follow-up evidence rather than a
blocker.
