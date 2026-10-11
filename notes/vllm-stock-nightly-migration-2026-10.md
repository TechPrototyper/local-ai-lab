# Off the fork: both boxes on stock vLLM nightly + one cherry-pick (Oct 2026)

*2026-10-09 → 2026-10-11. For four months both serving boxes ran a
hand-assembled vLLM: a fork carrying up to thirteen patch lines ("carries"),
plus a FlashInfer fork, built from source per architecture. On 2026-10-11 both
moved to an **unmodified vLLM nightly** with FlashInfer 0.7.1, **one** vLLM
cherry-pick, and two plugins. It went well on paper, badly in production for
an afternoon, and well again by evening — this note has all three parts.*

## Why now

[vllm#46963](https://github.com/vllm-project/vllm/pull/46963) (lesj0610,
"NVFP4 KV cache on SM8x and SM12x with FlashInfer") was merged on
2026-10-07. That is the feature this lab's whole consumer-Blackwell line was
built around — previously only available through
[@jethac](https://github.com/jethac)'s [#46329](https://github.com/vllm-project/vllm/pull/46329)
stack plus a FlashInfer fork. The 10-10 nightly
(`vllm/vllm-openai:nightly-7d0b4e57…`, arm64 + amd64) carries it, with
torch 2.13 and CUDA 13.

The operator's guardrail for the whole migration, verbatim: *"What matters to
me are performance and KV-cache numbers. If anything changes with the new
software — prefill, decode, KV size (NVFP4 mandatory) — it must be
transparent. I won't give up a path that's advantageous for my use."* So:
KV measured only in NVFP4, a mandatory old-vs-new table per box, and every
regression stated, never averaged away.

## The carry audit

| Carry | What | Fate on the nightly |
|---|---|---|
| C1 | #46329 base stack (NVFP4-KV write/read, per-page, VO split, mm-prefix mask, …) | **gone** — every part has a main equivalent via #46963 |
| C2 | FlashInfer fork | **gone** — stock FlashInfer |
| C3 | Gemma-4 head_dim-512 VO split | **gone** — FlashInfer splits VO ≥ 512 incl. FP4 in-kernel now (confirmed by a bf16-KV parity arm, below). Trap: Gemma4Config routes to Triton without FA4, and Triton has no NVFP4 → set the backend explicitly. |
| C4 | Gemma image spans (bidirectional mm-prefix) on NVFP4/FA2 | **still needed, solved differently** — see "the vision rule" |
| C5 | split-KV gate lift for NVFP4 prefill | kept as a tiny vLLM general plugin (the old one-argument patch would be a `TypeError` on FlashInfer 0.7). Measured on the GB10: **no effect** (±1 %). |
| C6–C8 | our [#53977](https://github.com/vllm-project/vllm/pull/53977)/[#53978](https://github.com/vllm-project/vllm/pull/53978)/[#53979](https://github.com/vllm-project/vllm/pull/53979) (DFlash × NVFP4) | not needed — Gemma uses MTP, the RTX runs no-spec; main still blocks DFlash × NVFP4 |
| C9 | [#50897](https://github.com/vllm-project/vllm/pull/50897) port (EAGLE prefix hashing) | not needed — the nightly got prefix hits under MTP without it (below) |
| C10 | DFlash2 line + sm121 embeddings shim | gone — upstream |
| C11 | GridBook plugin 0.8.8 | **kept as a plugin**; all 31 vLLM symbols it imports exist on the nightly |
| C12/13 | qutlass off (torch-2.11 ABI); PrismaQuant pipeline patches | gone (torch 2.13); out of scope |

Also found: `VLLM_ATTENTION_BACKEND` and `VLLM_NVFP4_GEMM_BACKEND` had not been
read by anything for a while — dead config on both boxes.

## The vision rule

The first GB10 trial (10-10) ran the nightly *text-only*: main refuses
Gemma-4 multimodal on NVFP4 at sm12x (FlashInfer reports
`supports_mm_prefix() = False`; the upstream composite backend
[#56305](https://github.com/vllm-project/vllm/pull/56305) is SM100-only and its
Triton child can't read NVFP4). The text numbers were great — and the
operator's answer was a rule: **a migration without the vision tower is not
an option, not even as an interim.** `--language-model-only` exists only in
the lab, and computing image tokens causally instead of bidirectionally (the
server would run, image answers would silently degrade) is excluded too.

Four ways were weighed; the chosen one:
[FlashInfer PR 5189](https://github.com/flashinfer-ai/flashinfer/pull/5189)
(in **v0.7.1**, released 10-09) adds
`BatchPrefillWithCausalBidirectionalRangesWrapper` — FA2 paged prefill that
takes bidirectional ranges as a compact `int32 [total_q, 2]` tensor instead of
a dense O(q·kv) mask (16k queries at 262k context: 128 KiB instead of 4 GiB),
explicitly with packed NVFP4 KV in mind; and
[vllm#46558](https://github.com/vllm-project/vllm/pull/46558) (lesj0610, still
open) wires it into vLLM's FlashInfer backend. Tested upstream only on sm80 —
our runs are the sm120/sm121 datapoints.

(The RTX's Qwen3.5-hybrid architecture is causal over images, so #46558 is
inert there; the RTX image carries it only to keep both boxes on one base.)

## The candidate

One Dockerfile, both architectures: `FROM` the nightly → FlashInfer 0.7.1 →
apply #46558 (5 files, clean `git apply --3way`) → the split-KV plugin →
GridBook 0.8.8 (RTX) → a `verify.py` that fails the build if the FlashInfer
wrapper or the patch is missing. No source build of vLLM or FlashInfer
anymore — the RTX image builds in-cluster in ~11 minutes instead of 1–2 h.

## Results

**GB10, Gemma-4-31B, vision on, NVFP4 KV, MTP n=4** (old = fork, new = candidate):

| Metric | old | new | Δ |
|---|---|---|---|
| Vision gate / vision-refusal gate | 3/3 / 9/9 | 3/3 (identical answers) / 9/9 | = |
| KV pool | 524,903 | 524,903 | 0 |
| Prefill 8k / 32k / 128k / 240k (tok/s) | 1066 / 700 / 278 / 161 | 1198 / 902 / 438 / 271 | +12 / +29 / +58 / +68 % |
| Time to first token at 240k | ~1525 s | ~908 s | −40 % |
| Decode short, 1 / 2 streams | 16.6 / 33.7 | 16.8 / 32.5 | +1 / **−4 %** |
| Decode at 130k, 2 streams | 1.0 (prefix miss) | 15.1 (prefix hit) | new |
| MTP mean acceptance length | 2.91 | 2.85 | −2 % |
| Needle 32k/128k · determinism · German | 3/3 · 5/5 · = | 3/3 · 5/5 · = | = |
| bf16-KV control arm (quality only) | — | vision 3/3, refusal 9/9, needle 3/3 | NVFP4 ≡ bf16 |

Without MTP decode drops to 8.0 tok/s short — MTP stays. The "two long
sessions" row is the practical headline: on the fork, a second 130k session
recomputed its prefix and the first fell to ~1 tok/s; on the nightly both
stay in the prefix cache.

**RTX 5090, Qwen3.8 GridBook-13GB + vision, NVFP4 KV, no spec, 8 seqs**
(A = fork, measured live under foreign load; B = candidate, idle):

| Metric | A | B | Δ |
|---|---|---|---|
| Vision · refusal · needle · determinism · German | = | = | = |
| KV pool | 760,477 | 765,668 | +0.7 % |
| Prefill 8k / 32k / 128k / 240k (tok/s) | 5292 / 4311 / 2713 / 1806 | 5200 / 4659 / 2895 / 1952 | −2 / +8 / +7 / +8 % |
| Decode short, 1 / 4 / 8 streams | 48.8 / 150.0 / 186.5 | 59.1 / 153.0 / 179.1 | +21 / +2 / **−4 %** |
| Decode at 130k, 1 / 4 streams | 45.3 / 93.1 | 46.6 / 96.6 | +3 / +4 % |

The +21 % at one stream is partly A's foreign load; the −4 % at eight
streams is probably real.

## What went wrong on the way

1. **A FlashInfer leftover broke the first build.** Uninstalling
   `flashinfer-jit-cache` left a `flashinfer_jit_cache/` directory behind
   (the kernels live in per-arch sub-packages); Python imported it as a
   namespace package without `__version__`, and `import flashinfer` died.
   Fix: remove all `flashinfer*` packages *and* their directories, install
   0.7.1 from the FlashInfer wheel index, then `flashinfer download-kernels`.
2. **The nightly has no CUDA development headers.** GridBook JIT-compiles a
   small CUDA extension at start; it needs `cusparse.h`, `cublas*`,
   `cusolverDn.h`. Putting the whole pip `nvidia/cu13/include` on `CPATH`
   breaks nvcc (its CRT headers clash: `__cudaLaunch` macro mismatch); copying
   only the *missing* library headers works. To be baked into the image.
3. **Measurements on an empty box are not acceptance.** The GB10 cutover at
   12:10 passed every gate in the lab — and the first image request in
   production, with the embedder/reranker/transcriber fleet running, killed
   the engine (`[FP4 gemm Runner StreamK] Failed to initialize cutlass FP4
   gemm`). Rolled back in five minutes. Later the *fork* failed the same way
   under the same conditions (`CUBLAS_STATUS_INTERNAL_ERROR`) — it was never
   the new vLLM. It is memory: CUDA allocates from physically *free* memory,
   which was ~4 GB while "available" said 11 GB; a lazily loaded kernel or a
   workspace allocation at the first image request fails with an internal
   error. The September painting crash was the same thing.
4. **Two wedges in one afternoon.** Two agent sessions worked the box in
   parallel (a mistake in itself). One ran a variant sweep and left the last
   variant (`--linear-backend flashinfer_cutedsl`) with an auto-restart
   policy; that variant ate memory from 29 GB to 0 in six minutes. After a
   reboot Docker revived it together with the whole fleet, and the box
   wedged again. Three defects lined up:
   - **earlyoom had been inert since mid-August**: configured `-s 10,5`
     instead of `-s 100,100`. It only acts when RAM *and* swap are low, and
     with swappiness 10 the swap stays 97 % free. Journal:
     `mem avail: 0 of 124546 MiB (0.00%), swap free 97.64%` — no kill.
   - **cgroup caps don't see GPU memory on GB10**: the main LLM container
     showed 49.6 GB in `nvidia-smi` and 10 GB in `memory.current` against a
     40 GiB cap.
   - **Docker autostart bypassed the stagger doctrine** after a reboot.

## What runs now (2026-10-11 evening)

**RTX 5090** — the candidate is production since 18:11; KV pool 765,668,
engine start 29 s, vision gate 3/3.

**GB10** — the candidate is production since 18:03, after a stress test
*with the fleet running*: three rounds of vision gate 3/3, paintings 9/9,
photos 9/9 serial and 9 + 9 with two parallel streams, needle 128k 3/3 — no
crash. What made the difference, and what now makes a wedge structurally
impossible:

- **earlyoom fixed** (`-s 100,100`) and lowered to `-m 4,2`.
- **`vm.min_free_kbytes` 1 → 3 GiB**: the kernel keeps 3 GiB *truly* free and
  reclaims page cache earlier, so CUDA allocations at request time succeed.
- **The main LLM no longer auto-restarts via Docker.** A watchdog (cron, every
  2 min) starts it after boot or crash: only outside maintenance windows,
  only after 5 min uptime, only with ≥ 45 GB available, at most 3 starts per
  hour — no crash loops, no simultaneous boot load.
- **Experiment containers never get an auto-restart policy**; the bring-up
  script replaces a leftover experiment container instead of reviving it.
- The drift checker verifies all of the above.
- **Memory budget made explicit:** with both transcribers back (an
  application depends on both), the Gemma KV pool is temporarily
  **398,934 tokens** (1.52 × 256k) instead of 524,903 — still one full 256k
  session, two sessions at ~200k each. Restored once that application moves.

## Upstream state (read-only, as of 10-11)

- [#46963](https://github.com/vllm-project/vllm/pull/46963) merged — the reason
  this was possible.
- [#46558](https://github.com/vllm-project/vllm/pull/46558) open (changes
  requested, merge conflicts) — our one cherry-pick; sm120/sm121 datapoints
  now exist.
- [#60948](https://github.com/vllm-project/vllm/pull/60948) open — FlashInfer
  0.7.1 pin; once merged the image drops its FlashInfer swap.
- [#50539](https://github.com/vllm-project/vllm/pull/50539) (Gemma-4 image
  spans within the sliding window, V2 runner) and
  [#47118](https://github.com/vllm-project/vllm/pull/47118) (FP8-KV text path)
  watched.
- Our [#53977](https://github.com/vllm-project/vllm/pull/53977)–[#53979](https://github.com/vllm-project/vllm/pull/53979)
  are superseded for our use; their fate is an open item.

## Open

- RTX variants not yet measured: CUDA-graph capture [1,2,4,8], without the
  split-KV plugin, XQA decode ([#60452](https://github.com/vllm-project/vllm/pull/60452)); the −4 % at eight streams re-measured idle.
- GB10: an image at 200k context, several images per prompt, MTP acceptance
  on image answers.
- Speculation re-evaluation on the new base (RTX MTP k=1/2; GB10 draft length).
- CUDA headers into the RTX image; GridBook 0.9.1 measured separately.
