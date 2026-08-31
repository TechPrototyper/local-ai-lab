# CP-3 · Hermes Loop-Signal-Cleanup — apply-ready Patch-Spec

Zweiter Teil des Force-Progress-Loop-Breakers (Ergänzung zu CP-2 in LiteLLM). Hermes
liest das Out-of-Band-Signal, das LiteLLM bei Loop-Erkennung setzt, und **kollabiert
seine EIGENE Konversations-Historie** (die wiederholten Tool-Call-Turns) → Bloat an der
Wurzel weg. Semantisch beschrieben (nicht linien-nummern-fragil), da das Repo mitten in
der Consolidation ist. Fail-open, consume-once, nie die tool_call↔tool_result-Paarung brechen.

## Signal-Kontrakt (von CP-2/LiteLLM)
Response-**Header** `x-litellm-loop-intervention` = JSON:
`{"kind": "identical|cycle", "period": int, "reps": int, "dropped_turns": int, "signature": str}`
(Primärkanal Header, weil Hermes streamt und dort der rohe httpx-Response mit Headern
vorliegt; Body-Feld `x_loop_intervention` als Fallback auf dem Non-Streaming-Pfad.)

## Edit 1 — Signal lesen + stashen (run_agent.py, beim `_capture_rate_limits`-Cluster ~4041)
Init neben `self._rate_limit_state = None`:
```python
self._loop_intervention_state = None
```
Neue Methode (Muster: `_capture_rate_limits`):
```python
def _capture_loop_intervention(self, http_response) -> None:
    """Parse x-litellm-loop-intervention header (out-of-band loop-breaker signal)."""
    if http_response is None:
        return
    headers = getattr(http_response, "headers", None)
    if not headers:
        return
    try:
        raw = headers.get("x-litellm-loop-intervention")
        if raw:
            import json
            self._loop_intervention_state = json.loads(raw)
    except Exception:
        pass  # never let header parsing break the loop
```

## Edit 2 — im Streaming-Capture verdrahten (chat_completion_helpers.py, `_stream_created` ~3931)
Neben den bestehenden Captures einfügen:
```python
agent._capture_rate_limits(response)
agent._capture_credits(response)
agent._capture_loop_intervention(response)          # <-- NEU (CP-3)
```
(Non-Streaming-Fallback optional: in `normalize_response` das Body-Feld
`x_loop_intervention` via `model_extra` nach `provider_data["loop_intervention"]` lesen.)

## Edit 3 — eigene Historie kollabieren (conversation_loop.py, im Post-Tool-Prune-Block ~7378-7404)
Nach dem bestehenden `prune_tool_results_only`-Commit (`messages = _pruned_msgs`), NEU:
```python
_li = getattr(agent, "_loop_intervention_state", None)
if _li:
    try:
        messages = _collapse_loop_turns(messages, int(_li.get("dropped_turns", 0)))
        agent._session_messages = messages
        agent._persist_session(messages, conversation_history)  # oder die atomare Prune-Persist-Disziplin
    except Exception:
        logger.debug("loop-collapse skipped", exc_info=True)  # fail-open
    finally:
        agent._loop_intervention_state = None   # consume-once
```
Helfer (neben den anderen History-Helfern, z.B. message_sanitization.py):
```python
def _collapse_loop_turns(messages, dropped_turns):
    """Entfernt die letzten `dropped_turns` KOMPLETTEN assistant-tool_call-Gruppen
    (assistant-Turn mit tool_calls + alle zugehoerigen role:tool-Rows) vom Tail,
    behaelt das erste Vorkommen + einen Marker. Bricht nie die Paarung."""
    if not dropped_turns or dropped_turns < 1:
        return messages
    # Indizes der assistant-tool_call-Turns (in Reihenfolge)
    a_idx = [i for i, m in enumerate(messages)
             if m.get("role") == "assistant" and m.get("tool_calls")]
    if len(a_idx) <= dropped_turns:
        return messages  # zu wenig -> defensiv nichts tun
    # erster zu entfernender assistant-Turn = der (len-dropped_turns)-te
    cut = a_idx[len(a_idx) - dropped_turns]
    # ACHTUNG: falls der Force-Progress-Response (Text, ohne tool_calls) schon
    # angehaengt ist, liegt er HINTER dem letzten tool-Row -> bleibt erhalten,
    # da wir nur bis zum letzten role:tool des Loops schneiden. Marker + Tail-nach-Loop behalten.
    # Finde das Ende des Loops (letzter role:tool nach dem letzten Loop-assistant)
    last_a = a_idx[-1]
    end = last_a + 1
    while end < len(messages) and messages[end].get("role") == "tool":
        end += 1
    kept_head = messages[:cut]
    marker = {"role": "user", "content": "[Loop-Breaker: wiederholte Tool-Aktion aus der Historie entfernt (Kontext-Rettung).]"}
    tail_after_loop = messages[end:]   # z.B. der Force-Progress-Text-Response
    return kept_head + [marker] + tail_after_loop
```

## Invarianten
- **Nur komplette Gruppen** entfernen (jedes role:tool hat seinen assistant-Parent im entfernten Span) → `close_interrupted_tool_sequence` (message_sanitization.py:296) bleibt konsistent.
- **Consume-once** (`_loop_intervention_state=None` nach Anwendung), gated auf `signature` falls doppelte Signale möglich.
- **Fail-open** (try/except → no-op), wie der bestehende Prune-Block.
- **Persistenz** über die bestehende atomare Prune-Persist-Disziplin (archive_and_compact + _DB_PERSISTED_MARKER) statt naivem Doppel-Write.

## Test vor Deploy
Offline: eine Loop-`messages`-Sequenz durch `_collapse_loop_turns` (identisch + Zyklus)
→ prüfen: N Gruppen weg, erste + Marker + Post-Loop-Tail bleiben, keine dangling role:tool.
Dann live gegen einen echten Loop-Mitschnitt (mit deployter CP-2), Header-Empfang verifizieren.
