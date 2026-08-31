# Recipe: Qwen3.8-27B drafter-free MTP-Turbo on RTX 5090 (sm120)

**Was:** Qwen3.8-27B als GridBook-13GB-Quant, mit nativer **MTP** (Multi-Token-
Prediction) statt externem Drafter, NVFP4-KV, Prefix-Caching unter Spekulation.
Ergebnis: drafter-frei, **~624k Token KV-Pool**, agentisch tauglich (Tool-Calling +
Prefix-Cache-Hits unter Spec), qualitäts-äquivalent zu no-spec (n=1319, p=0,83).

**Hardware:** 1× RTX 5090 (sm120, 32 GB), exklusiv (kein GPU-Split).

---

## 1. Serving-Image
`vllm-nvfp4:sm120-pc50897-gridbook-088` — vLLM V1 (sm120-NVFP4-Linie) mit:
- **vllm#50897** vorab integriert (lookahead-aware Prefix-Cache-Hashing → Prefix-Cache
  greift unter Spekulation; ohne das ~0% Hits, vllm#52244).
- **GridBook-Plugin 0.8.8** (`vllm.general_plugins`-Entry-Point → `quantization=gridbook`).
- non-causal-NVFP4-Seam-Härtung (vllm#53977/#53978/#53979).
- FlashInfer-Backend (`VLLM_ATTENTION_BACKEND=FLASHINFER`), warmer JIT/Autotune-Cache.

## 2. Modell-Artefakt: `gridbook-13gb-mtpfix`
GridBook-13GB behält native MTP-Köpfe, quantisiert aber die geteilte Body-Embedding —
was vLLMs MTP-Draft (teilt die Target-Embedding) bricht. Fix (bit-exakt, self-serve):
1. Basis: `rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB` (NVFP4-CB, ~13 GB).
2. Die 3 gepackten `model.language_model.embed_tokens.{weight_packed,weight_scale,
   weight_global_scale}` droppen; den **BF16 `embed_tokens.weight` aus dem AQUA-Quant**
   (`Qwen3.8-27B-PrismaAQUA-5.5bit`, identische Qwen3.8-Basis) verbatim einsetzen.
3. `quant_config.json`: `quantized_embedding`-Key entfernen, Unit in `ignore`.
   (Skript: `surgical_embed_swap.py` im Lab-Repo.)
Korrektheit über n=1319-Verdikt bewiesen (nicht über Byte-Gleichheit — MTP-Spec ist
FP-bedingt nicht byte-identisch zu no-spec, aber distributions-äquivalent).

## 3. Serve-Kommando (die maßgebliche Config)
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/gridbook-13gb-mtpfix \
  --served-model-name qwen3.8-27b \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}' \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --async-scheduling --enable-prefix-caching --enable-chunked-prefill
```
Env: `VLLM_ATTENTION_BACKEND=FLASHINFER`, HF-Cache/FlashInfer-Workspace gemountet.

**Kritisch:**
- **`--gpu-memory-utilization 0.90`**, NICHT 0.97. Bei 0.97 crasht die Engine
  reproduzierbar (Fragmentierungs-OOM unter Long-Context-Last, z.B. Needle 12k–64k):
  vLLMs statische KV-Buchung lässt zu wenig Slack für Aktivierungen + Allocator-
  Fragmentierung über die Profiling-Schätzung hinaus. 0.90 ist stabil.
- **GPU exklusiv** — bei geteilter GPU profiliert vLLM gegen weniger Speicher (534k statt 624k).
- **`--enable-prefix-caching` behalten** — für agentische Dialoge essenziell; greift dank
  #50897 auch unter MTP (gemessen ~77% Hit-Rate).

## 4. Erwartete Werte (gemessen, RTX 5090)
| Achse | Wert |
|---|---|
| KV-Pool | **624.152 Tokens** (2,38× Concurrency @ 262.144/Request), util 0.90, exklusiv |
| Qualität (n=1319, MTP vs no-spec, greedy, gepaart) | Acc 0,9704 vs 0,9719 · McNemar p=0,8318 (äquivalent) |
| Batterie | Tools 6/6 · GSM8K-250 0,9840 · Needle 12k/24k/64k 9/9 · Determinismus 5/5 |
| MTP-Turbo (AQUA-gemessen, überträgt sich) | Prosa +49% @ 54,3% Acceptance; count200 +112%; JSON +72% |
| Prefix-Cache (unter MTP) | ~77% Hit-Rate (queries 887k / hits 684k) |

## 5. Bekannte Hebel / offen
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` könnte höheren util stabil machen
  → Pool-Rückgewinnung über die 624k hinaus. Noch nicht gegengetestet.
- Optimum Sampling/Thinking für agentisches Tool-Calling: thinking-ON (Sweep-Befund),
  Loop-Fix liegt in der Agent-Schicht (Guardrails/Steering), nicht im Sampling.

Siehe auch die volle Cutover-Doku: `notes/rtx-gridbook-mtp-cutover-2026-08-31.md`.
