# The 13 GB endgame on the RTX 5090 — GridBook at verdict tier, with the full serving map (2026-08-29)

*The memory-track question this lab has been walking toward: can a 32 GB
consumer card serve a 27B model with **near-BF16 quality**, a **~1M-token
KV pool**, **multiple parallel sessions**, and a **working prefix cache** —
at once? Today the pieces were measured together. Everything below ran on
one RTX 5090 (sm120), image `v4@2cf8b8a` ∪ vllm#50897 ∪ the #53979 SWA
guard ∪ `gridbook==0.8.8` (PyPI), target
[`rdtand/…gridbook-13GB…`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm).*

## 1 · Quality: the 13 GB artifact holds at verdict tier

GSM8K n=1319, greedy, same items/config/node as the 08-29 verdict runs:

| Pairing | acc | discordants | McNemar exact |
|---|---|---|---|
| **GridBook-13GB** vs AQUA-23.6GB | **0.9757 vs 0.9757** | 9:9 | **p = 1.0000** |
| **GridBook-13GB** vs Scout-20GB | **0.9757 vs 0.9757** | 9:9 | **p = 1.0000** |

Plus needle 6/6 (12k/24k-word grid), determinism 5/5 (identical hashes),
0 errors. The discordance is symmetric and sits inside the same-config
noise floor measured this morning (15 items, 10:5, p=0.30). All three
artifacts — 23.6, 20, and 13 GB — land on **the same 1287/1319**. At this
tier the codebook quant appears to cost nothing measurable.

*(Honesty: the reference arms ran yesterday on the `v4@2cf8b8a` image;
today's GridBook arm adds the #50897 python overlay — kernels identical,
and the noise-floor control brackets exactly this kind of delta.)*

## 2 · The serving map — two configs, measured

### Config M — "max pool": GridBook + NVFP4-KV, no spec, 262k, util 0.97

| Metric | Value |
|---|---|
| **GPU KV cache pool** | **898,037 tokens** |
| vs AQUA prod shape (342,604) | **2.62×** |
| vs Scout shape (477,569) | 1.88× |
| Single-stream decode (warm, greedy prose) | see §3 |
| Sessions c=8 | 191 tok/s aggregate, ~24 tok/s each |
| Cold prefill | ~4,500–6,500 tok/s (19.7k–88k-token prompts) |
| Prefix-cache replay (88k-token prompt) | 17.7 s → **1.36 s (13×)** |

**Nine hundred thousand tokens of KV on a 32 GB consumer card**, with
verdict-tier quality. That is the number the memory track existed for.

### Config T — "turbo": GridBook + DFlash2 + fp8-KV + prefix cache, 131k

| Metric | Value |
|---|---|
| Draft acceptance vs the **GridBook** target — greedy prose | **24.0% / mean 2.68** |
| …retrieval-style completions | 54.3% / mean 4.80 |
| …the AQUA-target prose reference (08-22) | 53.9% / 4.77 |
| Single-stream, prose (warm) | 56.5 tok/s — vs 60.0 no-spec: **spec does not pay on prose here** |
| Single-stream, **structured** (count-200 / JSON, warm, 2× each) | **149–156 tok/s** — acceptance 72.3% / mean 6.06: **the turbo is intact off-prose** |
| Per-session decode, c=5–7 | **~37 tok/s** (no-spec drops to ~24 at c≥5) — the regime where spec wins |
| Sessions c=7 | 258.8 tok/s aggregate (c=8 collapses to 176.7 — KV pressure) |
| KV pool at this shape | 213,071 tokens (fp8 + spec) |
| Prefix-cache replay (88k-token prompt) | 15.97 s → **0.38 s (42×)** |
| Prefix hit rates under spec | 65–73% on this workload |

Two honest findings here. First, the good one: the 08-24 rollback reason
is gone — prefix caching **hits under speculation** (the #50897 dividend),
so a spec config no longer pays the cache for the speed. Second, the
sobering one: **draft acceptance against the GridBook target is
workload-dependent and roughly halves on prose** (24% vs the ~54% the same
drafter reaches on the AQUA target). The drafter was trained against the
AQUA/BF16 distribution; GridBook's codebook reconstruction appears to
shift next-token distributions enough on open prose to cost acceptance,
while constrained output keeps acceptance high — measured directly:
count-200 and JSON generation run at **149–156 tok/s** (2.5× the no-spec
baseline) on this same GridBook target. Net: on this target, speculation
is a **large win on structured/predictable output and at mid concurrency
(c≈5–7)**, and roughly neutral single-stream on open prose. The fix
directions are known: a drafter tuned on the GridBook distribution (speed
question only — spec stays lossless by construction), or accepting the
spec win only where acceptance carries it.

## 3 · Single-stream, measured warm

| Config | tok/s (256-tok greedy prose, warm, median of 3) |
|---|---|
| M (nvfp4-KV, no spec) | **60.0** |
| T (fp8-KV + DFlash2) | 56.5 — see §2-T: prose acceptance too low for spec to pay single-stream |

*(The first sweep's c=1 rows were taken seconds after boot and are
warmup-confounded; these are the clean numbers.)*

## 4 · What's still guarded — and the path

**NVFP4-KV + DFlash2 together** (the last composition) is blocked by the
SWA guard this lab itself contributed today: the DFlash2 drafter is
all-layers sliding-window (`sliding_window: 2048`), and non-causal +
window on the FA2-NVFP4 path has no kernel-parity test yet (jethac's
#53979 finding). The same combination **serves in production on sm121**
with byte-identical greedy gates — strong empirical evidence — but the
formal test is the commitment made upstream, so the guard stays until the
parity probe exists. That probe is the next work item; jethac's
`dequant_nvfp4_kv_cache` reference reader is the tool for it. Until then:
turbo = fp8-KV (quality-equivalent per the 08-29 cutover verdict, half the
pool), max-pool = nvfp4-KV no-spec.

## Scope / honesty

- GSM8K is one task; needle/determinism are sanity, not verdict axes.
  Tool-calling was not exercised in these configs (no parser flags).
- Sessions/prefill numbers are one workload shape (256-tok greedy prose,
  synthetic fact-grid prompts) on one card — indicative, not a benchmark
  suite.
- The 898k pool is the engine's reported allocation at 262k max-len /
  util 0.97; long-context *quality* beyond the needle grid (far-window
  pass at several hundred k) remains unmeasured.
- Spec remains lossless by construction (verified rejection sampling);
  its speed contribution varies with acceptance, which is workload-dependent.

## Provenance

- Verdict: `results/RESULT_sm120-gridbook13-n1319.json` (paired vs the
  08-29 Scout/AQUA arms)
- Sweeps: `results/SWEEP_TURBO.json`, `results/SWEEP_MAXPOOL.json`
- Companions: [`verdict-scout20gb-vs-aqua55-n1319-2026-08-29.md`](verdict-scout20gb-vs-aqua55-n1319-2026-08-29.md) ·
  [`verdict-prod50897-nachlauf-noisefloor-n1319-2026-08-29.md`](verdict-prod50897-nachlauf-noisefloor-n1319-2026-08-29.md) ·
  [`pc50897-sm120-cache-under-spec-2026-08-29.md`](pc50897-sm120-cache-under-spec-2026-08-29.md) ·
  [`gridbook-13gb-quality-holds.md`](gridbook-13gb-quality-holds.md)
