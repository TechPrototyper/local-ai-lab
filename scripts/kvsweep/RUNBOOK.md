# KV-Sweep — Decode über Kontextgröße (Antwort auf maxpla3, vllm#46329)

Stand 2026-09-02 abends. Supervised-Lauf, NICHT unbeaufsichtigt starten.

## Warum
maxpla3 (#46329, 09-02 20:16 UTC) misst auf RTX 5090 einen „harsh drop" der
Decode-Rate mit wachsendem Kontext unter `nvfp4 + mtp@3` (alter PR-Stand
`7a5cf1431`, vanilla HF-NVFP4-Modell). Das ist unsere MTP-KV-Drain-Achse.
Wir liefern denselben Sweep auf UNSEREM Stack: v5-Linie (`fe3247505`,
jethac-Rebase + unsere Patches + pr-50897) + GridBook-13GB + MTP@2 +
NVFP4-KV — Tims Linie: teilen, nicht antreten; knapp im Strang, Tiefe im
Notebook.

## Reihenfolge (Regel: RTX erst runter, dann arbeiten)
1. Spark-Batterie ist zurück + `SPARK_DOWN=0` verifiziert (Completion via
   LiteLLM mit Spark-Fingerprint).
2. `kubectl -n inference set env deploy/litellm RTX_DOWN=1` → Failover-Check
   (Completion muss vom Spark kommen).
3. Sweep direkt gegen den RTX-Pod (Pod-IP via
   `kubectl -n inference get pod -l app=llm-inference -o wide`), im Cluster
   ausführen (Pod-Netz!), z. B. via `kubectl exec` in den vLLM-Pod:
   Script liegt in diesem Ordner; hineinkopieren mit `kubectl cp`.
   ```bash
   python3 decode_context_sweep.py \
     --base http://localhost:8000 --model qwen3.8-27b \
     --runs 8 --decode-tokens 512 \
     --contexts 1024,8192,32768,65536,98304,131072,196608,262144 \
     --out /tmp/kvsweep-rtx-v5.json
   ```
   Dauer grob: die 8×262k-Prefills dominieren (~Minuten pro Run) — Budget
   1–2 h. Bei Zeitdruck: `--runs 4` und 262144 zuerst messen.
4. Optional zweiter Arm: Spark/AQUA+DFlash (gleiches Script gegen
   10.1.0.229:8000 — dann aber SPARK-Flag-Logik umgekehrt planen; nur wenn
   Zeit und beide Seiten je einmal einzeln tragen können).
5. JSON einsammeln (`kubectl cp`), nach `results/` legen, Teardown:
   `RTX_DOWN=0`, Verifikations-Completion.

## Auswertung / Verwendung
- Kurve decode_tok_s_mean über prompt_tokens; Vergleichspunkt zu maxpla3s
  Plot (er: Drop bei nvfp4+mtp@3 vanilla; wir: erwartet flacher dank
  775k-Pool-Ökonomie + MTP@2 + GridBook-KV-Fußabdruck).
- metrics_delta enthält Spec-Akzeptanz (accepted/drafted) je Kontextstufe —
  DER Erklärbaustein für den Drain-Mechanismus.
- Knapper Kommentar-Entwurf: notes/draft-maxpla3-kvsweep-comment.md
  (POSTEN NUR MIT TIMS GO). Tiefe: Notebook-Update (Task #8-Umfeld).
