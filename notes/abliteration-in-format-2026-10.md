# Abliteration *in format*: both production models, without re-quantizing (Oct 2026)

*2026-09-14 → 2026-10-04. Both serving models — Gemma-4-31B on the GB10 and
Qwen3.8-27B GridBook on the RTX 5090 — were abliterated **inside their
existing production quantization**: same format, same bit budget, same
codebooks, same file size. Published on Hugging Face (links at the end).*

## Why not just take an abliterated model and quantize it?

Two reasons, both measured:

1. **A fresh quantization is a different model.** PrismaQuant's allocator,
   given the abliterated BF16 weights and the same 6.000 bpp budget, picks a
   different per-tensor format assignment (246/116/48 instead of 234/135/41
   modules per format class). Every quality gate we had passed for the
   production artifact would have to be re-earned.
2. **The production artifacts are not ours to regenerate.** Both come from
   [Robert Tand (rdtand)](https://huggingface.co/rdtand)'s PrismaQuant line;
   the GridBook export lane was even removed upstream on 09-02. What we can
   do is *edit bytes* inside his format.

So the binding we set ourselves (2026-10-02): **only format, bit budget,
codebooks and renderer are fixed; the encoder is free.**

## The method

1. **Donor:** an existing abliterated BF16 model of the same family, made
   with [Heretic](https://github.com/p-e-w/heretic) in its ARA mode
   (Arbitrary-Rank Ablation, an unmerged branch — PR #211). ARA is *not*
   the classic rank-1 directional ablation; it optimizes the weight matrices
   directly. Donors:
   - Gemma: [trohrbaugh/gemma-4-31b-it-heretic-ara](https://huggingface.co/trohrbaugh/gemma-4-31b-it-heretic-ara)
   - Qwen: [heretic-org/Qwen3.8-27B-heretic-ara](https://huggingface.co/heretic-org/Qwen3.8-27B-heretic-ara)
2. **Only the matrices that write into the residual stream** take the
   donor's weights — that is where abliteration acts:
   - Gemma: `o_proj` + `down_proj` (120 matrices)
   - Qwen (hybrid): `o_proj` 16 + `out_proj` 48 (the linear-attention/SSM
     layers) + `down_proj` 64 = 128
3. **Re-encode onto the frozen production grid** with the production encoder:
   same codebooks, same format and code width per tensor, frozen scales, the
   same number of bytes per superblock.
4. **Write only those byte ranges into a copy of the artifact.** The
   safetensors header and every other tensor are bit-identical by
   construction. The vision tower is not even in the edited file.
5. **Null test:** re-encoding the *unchanged* production weights through the
   same path yields bit-identical bytes — the transplant changes exactly what
   it is supposed to change and nothing else.

## The measurement protocol (and the three mistakes it took to get there)

- 100 prompts from `mlabonne/harmful_behaviors`, one API path, one
  classifier (HARD / SOFT / EMPTY / NONE refusal), **thinking off**.
- Tool suite **serial, 5×**, German probe (12), needle, determinism,
  vision gate.

Mistakes along the way, kept here because they nearly decided the outcome:

- **Thinking mode** counted 70 empty answers as refusals.
- **Heretic's own counter** scores caution as refusal (09-17); we switched to
  a strict marker set.
- **Parallel tool-suite repeats** under `--async-scheduling` showed
  "regressions" on the *unmodified* model too — serial runs only.
- And the big one: an apparent result "quantization destroys abliteration"
  (a series 7 → 54 → 33) came from **three different measurement setups**.
  Measured paired on one setup (10-02): BF16 donor 33 vs quantized 24 hard
  refusals — quantization does not undo it.

## Results

| | Gemma-4-31B (GB10) | Qwen3.8-27B GridBook (RTX 5090) |
|---|---|---|
| Tensors changed | 118 of 2,025 (0 in the vision tower) | 113 of 1,363; 15 coarse-grid modules deliberately left bit-identical |
| Hard refusals (of 100) | **100 → 36** (BF16 donor: 33) | **100 → 29** (candidate); the deployed card reports 32 |
| Primary gate | German 9/12 = production | Tool suite 5/5 ×5 = production |
| Tool suite | 4/5 (one case: `exec_command` instead of `apply_patch`) | 5/5 |
| Needle / determinism / vision | 3/3 · 5/5 · 3/3 | 3/3 (32k, 128k) · 5/5 · 2/3 (no baseline yet) |
| Speed | 19.0 tok/s (production 18.2) | 57.9 tok/s single, 152.2 at 4 streams |
| Weights / KV pool | 26.8 GiB / 524,903 = production | 14.07 GiB / 760,477 = production |
| German | 9/12 | 8/12 (production 9/12) |
| In production since | 2026-10-03 03:03 | 2026-10-04 00:19 |

### The Qwen detour: dose is not the lever

The first Qwen attempts scaled the whole intervention by a dose α
(`W = W_prod + α·(W_donor − W_prod)`):

| Dose | Hard refusals | Tool suite | Vector indices changed |
|---|---|---|---|
| α = 1.0 | 25/100 | **2/5** (called a tool that was never offered) | 13.15 % |
| α = 0.75 | 88/100 | **3/5** | 3.42 % |

Both broke Qwen's primary gate — tool calling is its job — and α = 0.75
barely abliterated at all. The tool damage appears at 3.4 % changed indices,
the refusal drop only at 13 %: the two effects live in *different modules*,
and a global dose scales them together. Rolled back the same evening (10-03).

**What worked was selection, not dose.** The coarse grids represent the
transplanted weights worst (excess over the sibling's quantization error:
K12 0.149, K14 0.105, K32 0.102 vs K28 0.042); with frozen scales the
encoder clips outliers there (2.13 % in one NVFP4 `down_proj`) — and those
outliers carry precise format-following, i.e. tool behaviour. Leaving the 15
coarse-grid modules untouched (13 of them `down_proj` in layers 0, 9–17, 30,
36, 63) and giving the other 113 the **full** dose: 29/100 hard refusals —
*below* the full BF16 abliteration of the same family (33) — with tools 5/5.

## Limits

- Neither model is "uncensored": 29–36 of 100 prompts are still refused hard.
- Gemma loses one tool case; Qwen's `simple_exec` flickers under async
  scheduling; Qwen's vision gate (2/3) still lacks a matched baseline.
- Free scales (re-fitting the frozen scale plane) are untried; on two
  modules they would move the angle 0.426 → 0.511.
- LoRA-style shortcuts don't apply: the intervention has no low-rank
  structure.

## Published

- **Gemma-4-31B, abliterated in format:**
  [TechPrototyper/Gemma4-31B-IT-PrismaQuant-6bit-abliterated-vllm](https://huggingface.co/TechPrototyper/Gemma4-31B-IT-PrismaQuant-6bit-abliterated-vllm)
  (license: gemma)
- **Qwen3.8-27B GridBook 13 GB + vision, abliterated in format:**
  [TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm](https://huggingface.co/TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm)
  (license: apache-2.0)

Built on: [rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm](https://huggingface.co/rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm),
[rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm)
(quantization, GridBook format and the vLLM plugin: [RobTand/gridbook](https://github.com/RobTand/gridbook)),
the Heretic-ARA donors above, and the base models
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) and
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). This lab's
contribution is the in-format transplant, the vision-tower reintegration,
and the measurements.
