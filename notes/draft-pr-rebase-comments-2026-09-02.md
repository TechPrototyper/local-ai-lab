# ENTWÜRFE — PR-Kommentare nach Rebase (2026-09-02) — POSTEN NUR MIT TIMS GO

## #53979 (Antwort auf jethacs Ping in #46329, dort ODER auf #53979 posten — Empfehlung: #53979, mit Verweis)

> Rebased per your recipe (`git rebase --onto 9590d2464 7a5cf1431`) — thanks for
> pre-resolving the pinned-memory block in your layer; the wrapper commit applied
> **clean**, `git range-diff` shows `=` (byte-identical patch on the new base).
> New head `71a1133e8`, mergeable again; `ruff check` / `ruff format --check`
> clean on the touched file. sm12x runtime re-validation on this head queues
> next on our side — same battery as the 08-29 runs.

## #53977

> Rebased onto current `main` (`7894394b0`), new head `d726b9a15`. The rebase is
> not purely textual: upstream's VocabParallelEmbedding rework (fused-embedding
> path, FP8 int8-SUM all-reduce, and a tp==1 early return) means the tp==1 OOB
> masking now lives in that early return rather than the shared masked branch —
> `git range-diff` shows exactly that relocation, semantics unchanged (mask
> ids <0 / >= num_embeddings_per_partition to 0, zero their output rows). The
> regression test is unchanged. `ruff check` / `format --check` clean; conflict
> label should clear.

## Optional (SPÄTER, nach GPU-Validierung): Fold-in-Angebot an ZJY0516 auf #50897

> We've carried this branch in production since 08-28 and have now re-ported it
> onto current main (through the #46329 NVFP4 base): 57 conflict hunks against
> three weeks of upstream drift, resolved and documented — including the
> interaction with the new store-job ledger, `num_prefill_lookahead`, and the
> `use_eagle_block_drop` split. Happy to hand the rebase over (branch:
> TechPrototyper/vllm `integrate/v5-9590d246`, merge commit documents every
> decision + three review points), or to open it as a PR into your branch —
> whichever is easier for you.

**NICHT posten ohne:** (a) GPU-Batterie grün auf v5, (b) Tims Go.
