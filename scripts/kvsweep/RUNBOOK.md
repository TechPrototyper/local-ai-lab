# KV-Sweep — Decode over Context Size (Reply to maxpla3, vllm#46329)

As of 2026-09-02 evening. Supervised run, do NOT start unattended.

## Why
maxpla3 (#46329, 09-02 20:16 UTC) measures a "harsh drop" in decode rate on an
RTX 5090 as context grows under `nvfp4 + mtp@3` (old PR state
`7a5cf1431`, vanilla HF NVFP4 model). That is our MTP KV-drain axis.
We deliver the same sweep on OUR stack: v5 line (`fe3247505`,
jethac rebase + our patches + pr-50897) + GridBook-13GB + MTP@2 +
NVFP4 KV — Tim's line: share, don't compete; brief in the thread, depth in the
notebook.

## Order (rule: take the RTX down first, then work)
1. Spark battery is back + `SPARK_DOWN=0` verified (completion via
   LiteLLM with Spark fingerprint).
2. `kubectl -n inference set env deploy/litellm RTX_DOWN=1` → failover check
   (completion must come from the Spark).
3. Run the sweep directly against the RTX pod (pod IP via
   `kubectl -n inference get pod -l app=llm-inference -o wide`), executed
   inside the cluster (pod network!), e.g. via `kubectl exec` into the vLLM pod:
   the script is in this folder; copy it in with `kubectl cp`.
   ```bash
   python3 decode_context_sweep.py \
     --base http://localhost:8000 --model qwen3.8-27b \
     --runs 8 --decode-tokens 512 \
     --contexts 1024,8192,32768,65536,98304,131072,196608,262144 \
     --out /tmp/kvsweep-rtx-v5.json
   ```
   Rough duration: the 8×262k prefills dominate (~minutes per run) — budget
   1–2 h. If time is short: `--runs 4` and measure 262144 first.
4. Optional second arm: Spark/AQUA+DFlash (same script against
   <spark-host>:8000 — but then plan the SPARK flag logic in reverse; only if
   there is time and both sides can each carry the load alone once).
5. Collect the JSON (`kubectl cp`), put it in `results/`, teardown:
   `RTX_DOWN=0`, verification completion.

## Evaluation / Use
- Curve of decode_tok_s_mean over prompt_tokens; reference point against maxpla3's
  plot (he: drop at nvfp4+mtp@3 vanilla; we: expected to be flatter thanks to
  775k pool economics + MTP@2 + GridBook KV footprint).
- metrics_delta contains spec acceptance (accepted/drafted) per context level —
  THE explanatory building block for the drain mechanism.
- Brief comment draft: notes/draft-maxpla3-kvsweep-comment.md
  (POST ONLY WITH TIM'S GO). Depth: notebook update (task #8 vicinity).
