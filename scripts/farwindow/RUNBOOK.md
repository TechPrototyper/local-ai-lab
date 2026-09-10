# Far-Window First-Pass — Morgen-Runbook (supervised)

**Scope:** Extended-Window **Needle + Determinismus** First-Pass auf GridBook-13GB,
exklusiv neo26, `max-model-len 524288`, single-seq. **NICHT** das Verdikt-tier
Qualitäts-Gate (das braucht echtes Long-Doc-Set + Referenz + gepaarte Wertung).

**Regel (Memory `rtx-erst-runter-dann-arbeiten`): ZUERST RTX runter, dann arbeiten.**

## 0) Vorbedingung — DGX/Spark-Failover verifizieren (VOR dem Kippen!)
```
curl -s http://10.1.0.240:4000/health -H "Authorization: Bearer $LLKEY" | jq '.healthy_endpoints|length'
# sicherstellen: es gibt einen gesunden qwen3.8-27b-Endpunkt, der NICHT neo26/RTX ist.
# (Heute Nacht NICHT sicher verifizierbar gewesen — Cluster ist single-node neo26,
#  DGX läuft extern; Topologie mit Tim klären, bevor Prod auf 0 geht.)
```

## 1) RTX aus dem Pool nehmen (Failover → DGX/Spark)
```
kubectl -n inference scale deploy vllm-qwen38-mtp-gridbook --replicas=0
kubectl -n inference rollout status deploy vllm-qwen38-mtp-gridbook --timeout=200s || true
# verifizieren: Inferenz läuft weiter über DGX (kurzer Test-Call an LiteLLM 10.1.0.240:4000)
```

## 2) Safety-Net-Restore setzen (Rückfallnetz, unabhängig von der Session)
Ein-Shot-Cron/Reminder ~90 min: `scale deploy vllm-qwen38-gridbook-farwindow=0` +
`scale deploy vllm-qwen38-mtp-gridbook=1` — falls der Lauf hängt.

## 3) Far-Window-Serve hoch (Card ist jetzt frei)
```
kubectl apply -f scripts/farwindow/serve-farwindow.json
kubectl -n inference scale deploy vllm-qwen38-gridbook-farwindow --replicas=1
kubectl -n inference rollout status deploy vllm-qwen38-gridbook-farwindow --timeout=1500s
# Bootet nicht ready in ~25min? -> scale=0 + Prod zurück (Schritt 6), abbrechen.
```

## 4) Harness gegen den RESEARCH-Serve dry-runnen (NIE gegen Prod)
```
POD=$(kubectl -n inference get pod -l app=llm-farwindow -o jsonpath='{.items[0].metadata.name}')
kubectl -n inference port-forward pod/$POD 18900:8000 &
python3 scripts/farwindow/needle_determinism_probe.py --base-url http://127.0.0.1:18900 \
  --dry --out results/RESULT_farwindow_DRYRUN.json          # muss hit=true zeigen
```
> Harness ist bereits auf Chat-Endpoint + max_tokens=800 gepatcht (Prod-Dry-Run
> 2026-09-01 zeigte: `<think>`-Modus braucht Token-Budget, sonst falscher Miss).

## 5) Voller First-Pass
```
python3 scripts/farwindow/needle_determinism_probe.py --base-url http://127.0.0.1:18900 \
  --ctx 131072 262144 393216 524288 --out results/RESULT_farwindow_firstpass.json
```

## 6) Teardown + Prod restore (IMMER)
```
kubectl -n inference scale deploy vllm-qwen38-gridbook-farwindow --replicas=0
kubectl -n inference scale deploy vllm-qwen38-mtp-gridbook --replicas=1
kubectl -n inference rollout status deploy vllm-qwen38-mtp-gridbook --timeout=200s
# Safety-Net-Cron löschen. Roh-JSON sichten; NICHT auto-publizieren (manuell).
```
