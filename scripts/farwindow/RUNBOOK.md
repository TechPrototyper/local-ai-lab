# Far-Window First-Pass — Morning Runbook (supervised)

**Scope:** Extended-window **needle + determinism** first pass on GridBook-13GB,
exclusively on <rtx-node>, `max-model-len 524288`, single-seq. **NOT** the verdict-tier
quality gate (that needs a real long-doc set + reference + paired scoring).

**Rule (memory `rtx-erst-runter-dann-arbeiten`): FIRST take the RTX down, then work.**

## 0) Precondition — verify DGX/Spark failover (BEFORE flipping!)
```
curl -s http://<gateway-host>:4000/health -H "Authorization: Bearer $LLKEY" | jq '.healthy_endpoints|length'
# make sure: there is a healthy qwen3.8-27b endpoint that is NOT <rtx-node>/RTX.
# (Could NOT be verified reliably last night — the cluster is single-node <rtx-node>,
#  DGX runs externally; clarify the topology with Tim before prod goes to 0.)
```

## 1) Take the RTX out of the pool (failover → DGX/Spark)
```
kubectl -n inference scale deploy vllm-qwen38-mtp-gridbook --replicas=0
kubectl -n inference rollout status deploy vllm-qwen38-mtp-gridbook --timeout=200s || true
# verify: inference keeps running via DGX (short test call to LiteLLM <gateway-host>:4000)
```

## 2) Set the safety-net restore (fallback net, independent of the session)
One-shot cron/reminder ~90 min: `scale deploy vllm-qwen38-gridbook-farwindow=0` +
`scale deploy vllm-qwen38-mtp-gridbook=1` — in case the run hangs.

## 3) Bring up the far-window serve (the card is free now)
```
kubectl apply -f scripts/farwindow/serve-farwindow.json
kubectl -n inference scale deploy vllm-qwen38-gridbook-farwindow --replicas=1
kubectl -n inference rollout status deploy vllm-qwen38-gridbook-farwindow --timeout=1500s
# Not ready after booting for ~25min? -> scale=0 + prod back (step 6), abort.
```

## 4) Dry-run the harness against the RESEARCH serve (NEVER against prod)
```
POD=$(kubectl -n inference get pod -l app=llm-farwindow -o jsonpath='{.items[0].metadata.name}')
kubectl -n inference port-forward pod/$POD 18900:8000 &
python3 scripts/farwindow/needle_determinism_probe.py --base-url http://127.0.0.1:18900 \
  --dry --out results/RESULT_farwindow_DRYRUN.json          # must show hit=true
```
> The harness is already patched to the chat endpoint + max_tokens=800 (prod dry run
> 2026-09-01 showed: `<think>` mode needs a token budget, otherwise a false miss).

## 5) Full first pass
```
python3 scripts/farwindow/needle_determinism_probe.py --base-url http://127.0.0.1:18900 \
  --ctx 131072 262144 393216 524288 --out results/RESULT_farwindow_firstpass.json
```

## 6) Teardown + prod restore (ALWAYS)
```
kubectl -n inference scale deploy vllm-qwen38-gridbook-farwindow --replicas=0
kubectl -n inference scale deploy vllm-qwen38-mtp-gridbook --replicas=1
kubectl -n inference rollout status deploy vllm-qwen38-mtp-gridbook --timeout=200s
# Delete the safety-net cron. Review the raw JSON; do NOT auto-publish (manual).
```
