#!/usr/bin/env python3
"""
Extended-window needle + determinism probe (FIRST-PASS, not the verdict-tier
far-window quality gate).

Scope / honesty:
- This measures long-context *retrieval* (synthetic needle-in-haystack at depth
  fractions beyond the prod needle grid) and *determinism* (byte-identical greedy
  on a repeated far-window prompt). Synthetic-by-design; disclosed as such.
- This is NOT a graded long-document quality verdict. That needs a real long-doc
  set + a reference artifact + paired scoring and stays a supervised task.
- Greedy only (temperature 0). Raw JSON out for morning review.

Hits an OpenAI-compatible vLLM server (/v1/completions). No external deps beyond
`requests` (falls back to urllib if requests missing).
"""
import argparse, json, sys, time, hashlib

try:
    import requests
    def _post(url, payload, timeout):
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
except Exception:
    import urllib.request
    def _post(url, payload, timeout):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

FILLER = ("The archival log records routine maintenance of the orbital relay "
          "station; nothing of note occurred during this interval. ")

def build_context(approx_tokens, depth_frac, magic):
    # ~ approx_tokens via filler (heuristic: ~0.75 tok/word); plant the needle at depth.
    # crude token->char: assume ~4 chars/token.
    target_chars = int(approx_tokens * 4)
    needle = f"\n\n>>> IMPORTANT: The vault access code is {magic}. Remember it. <<<\n\n"
    body = ""
    while len(body) < target_chars:
        body += FILLER
    body = body[:target_chars]
    cut = max(0, min(len(body), int(len(body) * depth_frac)))
    ctx = body[:cut] + needle + body[cut:]
    prompt = (ctx + "\n\nQuestion: What is the vault access code? "
              "Answer with the code only.\nAnswer:")
    return prompt

def score(text, magic):
    return magic in (text or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[131072, 262144, 393216, 524288],
                    help="approx context sizes in tokens to probe")
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    ap.add_argument("--determinism-ctx", type=int, default=262144)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="results/RESULT_farwindow_firstpass.json")
    ap.add_argument("--dry", action="store_true",
                    help="tiny run (ctx=2048, depths=[0.5]) to validate harness")
    args = ap.parse_args()

    if args.dry:
        args.ctx = [2048]
        args.depths = [0.5]
        args.determinism_ctx = 2048

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    SYS = ("You are given a long document followed by a question. "
           "Answer with only the requested value, nothing else.")
    results = {"meta": {"model": args.model, "base_url": args.base_url,
                        "endpoint": "chat", "scope": "first-pass needle+determinism, NOT verdict-tier",
                        "ctx": args.ctx, "depths": args.depths},
               "needle": [], "determinism": None}

    for ctx in args.ctx:
        for d in args.depths:
            magic = str(1000000 + ctx + int(d * 997))
            prompt = build_context(ctx, d, magic)
            payload = {"model": args.model,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": prompt}],
                       "max_tokens": 800, "temperature": 0.0, "seed": 0}
            t0 = time.time()
            try:
                resp = _post(url, payload, args.timeout)
                out = resp["choices"][0]["message"]["content"]
                ok = score(out, magic)
                usage = resp.get("usage", {})
                row = {"ctx": ctx, "depth": d, "hit": ok,
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "latency_s": round(time.time() - t0, 1),
                       "answer_head": (out or "")[:60]}
            except Exception as e:
                row = {"ctx": ctx, "depth": d, "hit": None, "error": str(e)[:200],
                       "latency_s": round(time.time() - t0, 1)}
            results["needle"].append(row)
            print(json.dumps(row), flush=True)

    # determinism: same far prompt twice, compare greedy output hash
    magic = "424242"
    prompt = build_context(args.determinism_ctx, 0.5, magic)
    payload = {"model": args.model,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": prompt}],
               "max_tokens": 256, "temperature": 0.0, "seed": 0}
    outs = []
    for i in range(2):
        try:
            resp = _post(url, payload, args.timeout)
            outs.append(resp["choices"][0]["message"]["content"])
        except Exception as e:
            outs.append(f"__ERR__{e}")
    if all(o.startswith("__ERR__") for o in outs):
        results["determinism_note"] = "both requests errored; identical flag not meaningful"
    h = [hashlib.sha256((o or "").encode()).hexdigest()[:16] for o in outs]
    results["determinism"] = {"ctx": args.determinism_ctx,
                              "identical": outs[0] == outs[1],
                              "hashes": h}
    print(json.dumps(results["determinism"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {args.out}", flush=True)

    hits = [r for r in results["needle"] if r.get("hit") is True]
    tot = [r for r in results["needle"] if r.get("hit") is not None]
    print(f"SUMMARY needle {len(hits)}/{len(tot)} hit; "
          f"determinism identical={results['determinism']['identical']}", flush=True)

if __name__ == "__main__":
    main()
