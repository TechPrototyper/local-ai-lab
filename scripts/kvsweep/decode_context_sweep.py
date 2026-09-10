#!/usr/bin/env python3
"""Decode-Speed über Kontextgröße — Antwort-Harness auf maxpla3s #46329-Plot.

Spiegelt seine Methode: pro Kontextgröße N unabhängige Runs, Decode-Rate aus
dem Streaming-Timing (erster→letzter Token), Prefill-Rate aus Zeit bis zum
ersten Token; zusätzlich Spec-Decode-Akzeptanz aus /metrics als Delta.

Läuft gegen einen DIREKTEN vLLM-Endpoint (nie über LiteLLM; RTX vorher in
LiteLLM flaggen — Regel [[rtx-erst-runter-dann-arbeiten]] beachten, hier:
RTX_DOWN=1 + Spark trägt Prod, dann messen).

Verwendung:
  python3 decode_context_sweep.py --base http://<pod>:8000 \
      --model qwen3.8-27b --runs 8 --decode-tokens 512 \
      --contexts 1024,8192,32768,65536,98304,131072,196608,262144 \
      --out results/kvsweep-<stand>.json
"""
import argparse
import json
import time
import urllib.request

FILLER_SENTENCE = (
    "Der Archivar nummeriert die Regale, prüft die Etiketten, notiert die "
    "Abweichungen und heftet den Bericht in den Tagesordner. "
)


def approx_tokens(text: str) -> float:
    # Grobheuristik nur fürs Prompt-Sizing; die echte Zahl kommt aus usage.
    return len(text) / 3.6


def build_prompt(target_tokens: int) -> str:
    reps = max(1, int(target_tokens / approx_tokens(FILLER_SENTENCE)))
    body = FILLER_SENTENCE * reps
    return (
        body
        + "\n\nFasse den obigen Ablauf in einem fortlaufenden Text zusammen "
        "und beschreibe ausführlich, welche Schritte der Archivar wiederholt."
    )


def metrics_snapshot(base: str) -> dict:
    out = {}
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=20) as r:
            for line in r.read().decode().splitlines():
                for key in (
                    "vllm:spec_decode_num_accepted_tokens_total",
                    "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:generation_tokens_total",
                    "vllm:prompt_tokens_total",
                ):
                    if line.startswith(key):
                        out[key] = float(line.rsplit(" ", 1)[-1])
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


def one_run(base: str, model: str, prompt: str, decode_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t_send = time.time()
    t_first = None
    n_chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices"):
                delta = obj["choices"][0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.time()
                    n_chunks += 1
    t_end = time.time()
    completion = usage.get("completion_tokens", n_chunks)
    prompt_toks = usage.get("prompt_tokens", 0)
    prefill_s = (t_first or t_end) - t_send
    decode_s = t_end - (t_first or t_send)
    return {
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion,
        "prefill_s": round(prefill_s, 3),
        "decode_s": round(decode_s, 3),
        "prefill_tok_s": round(prompt_toks / prefill_s, 1) if prefill_s > 0 else None,
        "decode_tok_s": round(completion / decode_s, 1) if decode_s > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--contexts", default="1024,8192,32768,65536,98304,131072,196608,262144")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {"base": args.base, "model": args.model, "runs": args.runs,
               "decode_tokens": args.decode_tokens, "sweep": []}
    for ctx in [int(x) for x in args.contexts.split(",")]:
        prompt = build_prompt(ctx)
        m0 = metrics_snapshot(args.base)
        runs = []
        for i in range(args.runs):
            r = one_run(args.base, args.model, prompt, args.decode_tokens)
            runs.append(r)
            print(f"ctx~{ctx}: run {i+1}/{args.runs}: "
                  f"prompt={r['prompt_tokens']} decode={r['decode_tok_s']} tok/s", flush=True)
        m1 = metrics_snapshot(args.base)
        decode_rates = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
        entry = {
            "target_ctx": ctx,
            "prompt_tokens": runs[0]["prompt_tokens"],
            "decode_tok_s_mean": round(sum(decode_rates) / len(decode_rates), 1) if decode_rates else None,
            "decode_tok_s_min": min(decode_rates) if decode_rates else None,
            "decode_tok_s_max": max(decode_rates) if decode_rates else None,
            "prefill_tok_s_first_run": runs[0]["prefill_tok_s"],
            "runs": runs,
            "metrics_delta": {k: round(m1.get(k, 0) - m0.get(k, 0), 1)
                              for k in m0 if k != "error"},
        }
        results["sweep"].append(entry)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"ctx~{ctx}: mean decode {entry['decode_tok_s_mean']} tok/s "
              f"(gespeichert nach {args.out})", flush=True)


if __name__ == "__main__":
    main()
