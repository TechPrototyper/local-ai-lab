#!/usr/bin/env python3
# DGX Spark benchmark: TTFT + tokens/s, single & concurrent sessions.
# Stdlib-only; runs on the serving host against a local vLLM endpoint.
import json
import random
import threading
import time
import urllib.request

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen3.6-27b"
WORDS = ("Anlage Bericht Kunde Vertrag Analyse Modell Speicher Rechnung Projekt "
         "Systeme Wartung Prozess Freigabe Antwort Zeitplan Budget Messung Daten "
         "Prüfung Konzept Entwurf Lieferung Qualität Verfahren Ergebnis").split()
rng = random.Random(20260806)


def filler(n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def one_session(prompt, max_tokens, results, idx):
    req = {"model": MODEL,
           "messages": [{"role": "user", "content": prompt}],
           "max_tokens": max_tokens, "stream": True,
           "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft = None
    usage = None
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            BASE, json.dumps(req).encode(),
            {"Content-Type": "application/json"}), timeout=1800)
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ttft is None:
                delta = ch[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                    ttft = time.time() - t0
        total = time.time() - t0
        results[idx] = {"ok": True, "ttft_s": ttft, "total_s": total,
                        "prompt_tokens": usage and usage.get("prompt_tokens"),
                        "completion_tokens": usage and usage.get("completion_tokens")}
    except Exception as e:  # noqa: BLE001
        results[idx] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}


def run_round(label, n_sessions, prompt_words, max_tokens):
    prompts = [f"Sitzung {i}: Fasse den folgenden Text in zwei Sätzen zusammen "
               f"und bewerte seine Struktur. {filler(prompt_words)}"
               for i in range(n_sessions)]
    results = [None] * n_sessions
    threads = [threading.Thread(target=one_session,
                                args=(prompts[i], max_tokens, results, i))
               for i in range(n_sessions)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in results if r and r.get("ok")]
    row = {"label": label, "sessions": n_sessions, "ok": len(ok),
           "wall_s": round(wall, 2)}
    if ok:
        ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
        decs = [(r["completion_tokens"] or 0) / (r["total_s"] - r["ttft_s"])
                for r in ok if r["ttft_s"] and r["completion_tokens"]
                and r["total_s"] > r["ttft_s"]]
        agg = sum(r["completion_tokens"] or 0 for r in ok) / wall
        row.update({
            "prompt_tokens_actual": ok[0]["prompt_tokens"],
            "ttft_mean_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "ttft_max_s": round(max(ttfts), 2) if ttfts else None,
            "decode_toks_per_session_mean": round(sum(decs) / len(decs), 1) if decs else None,
            "aggregate_toks": round(agg, 1),
        })
    errs = [r["error"] for r in results if r and not r.get("ok")]
    if errs:
        row["errors"] = errs[:2]
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = []
    # Warmlauf (JIT/Autotune-Reste, nicht gewertet)
    run_round("warmup", 1, 60, 64)
    # A) Kurz-Prompt, Concurrency-Treppe
    for n in (1, 1, 1, 2, 4, 8):
        rows.append(run_round(f"A-kurz-c{n}", n, 60, 256))
    # B) Lang-Prompt-TTFT nur kurz+mittel (128k/240k aus Lauf 1 abgeleitet)
    for label, words in (("B-8k", 6200), ("B-32k", 24600)):
        rows.append(run_round(label, 1, words, 128))
    json.dump(rows, open("RESULT_spark_bench.json", "w"),
              indent=1)
    print("written RESULT_spark_bench.json", flush=True)


if __name__ == "__main__":
    main()
