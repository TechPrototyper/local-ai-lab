# DFlash2 + NVFP4-KV speculative decoding serves end-to-end on the RTX 5090 (sm120)

**Run window:** 2026-08-22 (branch record §17/§18) · **written up:** 2026-08-27

**TL;DR:** Block-diffusion speculative decoding (DFlash2, draft length 7) and a
4-bit NVFP4 KV cache now run **together** on consumer Blackwell — the
combination that previously raised
`NotImplementedError: FlashInfer non-causal attention is not supported with
NVFP4 KV cache`. The full delta between "raises" and "serves" is **two small
fixes** on top of two spec-decode warmup fixes. To our knowledge this
combination did not exist anywhere in the stack before: MTP-style drafters are
causal and pass; DFlash's verify step needs a **non-causal** query block over
NVFP4 pages, and that path simply wasn't wired.

## The four-commit stack (branch `kernel/dflash2-nvfp4-sm120-v2` @ `571a65e92`)

| Commit | What | Class |
|---|---|---|
| `952ed2fe2` | Open the non-causal prefill wrapper for the FA2-NVFP4 paged reader (`backend='fa2'` iff `use_fa2_nvfp4_kv`); trtllm-gen stays causal-only. Byte-identical for every causal and every non-NVFP4 path. | the seam |
| `714255f46` | Mask out-of-range input ids in `VocabParallelEmbedding` at `tp_size==1` — spec-decode warmup passes `-1` padding sentinels unmasked into `F.embedding` (device-side OOB). The tp>1 branch masks; the single-GPU branch didn't. **Upstream bug, not sm12x-specific.** | warmup OOB #1 |
| `2608f81d3` | Embed DFlash draft ids outside the compiled forward (the embed was inside an inductor region; graph-mode unblock). | warmup OOB #2 |
| `96d4568e7` | Clamp DFlash2 `CandidateSelector._score_edges` codebook-gather ids — a **third** `-1`-sentinel gather site (`successor_table[candidate_ids]` etc., bound = vocab 248320) the first two fixes don't reach. Identity on the serving path. | warmup OOB #3 |

Base: jethac's FA2-NVFP4 line (flashinfer#3684 merged, vLLM#46329) — which
natively carries the `causal` metadata plumbing, `plan(causal=...)`, and the
V-scale handling. **The kernel needed no changes**; a numerics probe on real
(production-layout) KV pages shows non-causal tracking the production-validated
causal path to within Δcos ≤ 0.0013 across two regimes, with the residual gap
at fp4-quantization magnitude and identical on the causal anchor.

## Spec battery (first live DFlash2+NVFP4-KV spec serve, graph mode)

Target Qwen3.8-27B (PrismaAQUA 5.5-bit), drafter DFlash2 fp8 line,
`num_speculative_tokens=7`, `--kv-cache-dtype nvfp4`, 3 GiB KV pinned,
max_model_len 32768 for the battery:

| Metric | Result | vs no-spec (same target, §16) |
|---|---|---|
| Coherence | PASS | — |
| Greedy determinism 2× (temp0/seed0) | PASS, byte-identical | — |
| tok/s single-stream prose | **82.5** | 55.6 → **+48%** |
| tok/s count-to-200 | **176.2** | 55.6 |
| tok/s c=2 aggregate | **126.9** | 107.1 → +19% |
| Acceptance | rate **0.539** (1151/2135 over 305 drafts), mean accept ≈ **4.77**/step incl. bonus | — |

Speculation is quality-neutral under greedy (the drafter shifts only
acceptance/speed, never the output distribution) — confirmed here by the
byte-identical determinism gate.

## Honest limits

- **Acceptance is content-dependent** (prose ≈0.54; structured/reasoning-heavy
  output accepts better — hence count-200's 176.2).
- **Speculation inflates KV demand ~5.9×** (drafter aux layers + 7 verify slots
  + hybrid GDN state): 47,304 tokens / 1.44× concurrency @ 3 GiB with spec at
  32k window. On a 32 GB card, spec pays in the single-stream/latency regime;
  high concurrency and very long context remain better served without it.
- **Prefix caching does not hit under DFlash2-spec + hybrid GDN** (upstream
  [#52244](https://github.com/vllm-project/vllm/pull/52244); confirmed live,
  0% hit on identical prompts). This — not the seam — is why the RTX
  production config currently runs no-spec with prefix caching.
- Single config, one card; v1-line references (101.3/373.8 tok/s, Qwen3.6 +
  DFlash-v1) are a different model/drafter and not directly comparable.

## Layout footnote (corrects an earlier tooling claim)

Our earlier dequant tooling needed a 4×4 block-scale unswizzle to reach
cos = 1.0 — that turned out to be specific to the Option-B **dequant shim's**
read path, not the native FA2-NVFP4 reader, and upstream has since pinned the
store contract (`7a5cf1431`): K scales always linear, V scales swizzled only on
CC < 12 — **on sm12x both are linear**. Anyone touching pages directly should
follow that arch-aware contract, not an unconditional unswizzle.

## Where this sits

Node-safe by construction (Guaranteed 56Gi, warmup autotune off, PVC AOT
caches; peak host RAM 32.04 GiB; control plane healthy throughout; production
untouched at 0/0). Raw record:
[`results/RESULT_e2e_dflash2_nvfp4_sm120.json`](../results/RESULT_e2e_dflash2_nvfp4_sm120.json)
(§17/§18 sections).

## Upstream (2026-08-27, night)

The stack was rebased onto #46329's current head (`7a5cf14`), revalidated
end-to-end on **both** arches (fresh sm120 image = main ∪ #46329 ∪ the fixes;
sm121 on the production lineage), paired spec vs. no-spec at identical
settings, byte-identical greedy determinism throughout — and filed as three
reviewable PRs:

- [vllm#53977](https://github.com/vllm-project/vllm/pull/53977) —
  `VocabParallelEmbedding` tp==1 OOB mask (+ regression test, 2 passed)
- [vllm#53978](https://github.com/vllm-project/vllm/pull/53978) — DFlash2
  warmup OOBs (embed-outside-compile; selector-gather clamp)
- [vllm#53979](https://github.com/vllm-project/vllm/pull/53979) — the
  non-causal FA2-NVFP4 seam, stacked on #46329 (fold-in offered)

Revalidation headlines (paired, same config): sm120 count-to-200 **190.9 vs
55.9 tok/s (3.4×)**, step-by-step reasoning **135.0 tok/s** (accept 0.653),
essay prose at parity (accept ≈0.20 — speculation does not regress when
drafts miss); sm121 prose **+72%**, count-to-200 **5.9×**, c=2 **+70%** —
the first DFlash2+NVFP4-KV spec serve on sm121. Acceptance is strongly
content-dependent; the earlier 82.5-tok/s prose figure reflects a more
acceptance-friendly prose/thinking mix, not a stack difference. Raw:
[`results/RESULT_nvfp4_spec_crossarch_revalidation.json`](../results/RESULT_nvfp4_spec_crossarch_revalidation.json).
