"""LiteLLM callback: loop breaker — force progress and debloat the context.

PROBLEM: an agent on a mid-size local model falls into tool-call loops —
either byte-identical repeats (`git checkout current` × 50) or identical
chains (A, B, C, A, B, C, …). A plain counter that aborts with an error is a
dead end; the model must be forced *forward*, not stopped. And the context
the loop inflated has to come out, or the session tips over.

WHY A STATELESS PROXY CAN DO THIS: the harness sends the full `messages`
history with every turn, so the loop is entirely inside the single request.
The proxy reads it out of the payload; no cross-call state needed.

THREE ACTIONS on detection (in the tail of the assistant tool-call turns):
  1. FORCE PROGRESS: `tool_choice="none"` — the model cannot emit a tool call
     and must produce text (synthesis or an answer), plus a hard directive as
     the final turn.
  2. DEBLOAT: the repeated (assistant tool_call + tool result) pairs are removed
     from the *forwarded* messages; the first occurrence stays, with a marker.
     Only complete pairs — never a tool_call without its result, which breaks
     the template and the parser.
  3. SIGNAL, out of band: detection info goes into a response header (and, on
     the non-streaming path, a response attribute) so the harness can clean
     its own persistent history. Never into msg.content (see overflow_notice.py
     for why).

THE GUARD THAT MATTERS: a loop is only a loop if it is running *now*, i.e. the
most recent assistant turn is part of it. Without that check the breaker
fires forever: after an intervention the model answers with text, the harness
keeps replaying the old loop region, the filtered tool-call list still ends
with the old identical run, and every subsequent request gets
tool_choice="none". The agent can never call a tool again and only produces
status summaries. `_loop_is_active` is that guard.

Defensive: any unexpected situation -> request passed through unchanged.
Scoped to the local model prefixes. Off via LOOP_BREAKER=0.

Registration (config.yaml): litellm_settings.callbacks:
  ["overflow_notice.proxy_handler_instance", "loop_breaker.proxy_handler_instance"]
"""
import json
import os

from litellm.integrations.custom_logger import CustomLogger

LOOP_BREAKER_ON = os.environ.get("LOOP_BREAKER", "1") == "1"
K_IDENTICAL = int(os.environ.get("LOOP_K_IDENTICAL", "3"))        # >=3 identical calls => loop
CYCLE_MIN_PERIOD = int(os.environ.get("LOOP_CYCLE_MIN_PERIOD", "2"))
CYCLE_MAX_PERIOD = int(os.environ.get("LOOP_CYCLE_MAX_PERIOD", "5"))
CYCLE_MIN_REPS = int(os.environ.get("LOOP_CYCLE_MIN_REPS", "2"))  # cycle >=2× => loop
LOCAL_MODEL_PREFIXES = tuple(
    p.strip() for p in os.environ.get("LOCAL_MODEL_PREFIXES", "qwen3.8-27b").split(",") if p.strip()
)

_DIRECTIVE = (
    "SYSTEM/LOOP-BREAKER: You have repeated the same action or chain of actions "
    "several times without a new result. Do NOT call a tool now. Summarize what "
    "you already know and either give the final answer or name the ONE clearly "
    "different next step."
)


def _sig(msg):
    """Signature of an assistant message with tool_calls: a tuple of
    (name, arguments) per call. None if it has no tool calls."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None
    parts = []
    for tc in msg.get("tool_calls") or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        parts.append((fn.get("name"), fn.get("arguments")))
    return tuple(parts) if parts else None


def _loop_is_active(messages):
    """True only if the most recent assistant message is a tool-call turn."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return bool(m.get("tool_calls"))
    return False


def _assistant_toolcall_turns(messages):
    """[(msg_index, signature), ...] for every assistant tool-call turn, in order."""
    out = []
    for i, m in enumerate(messages):
        s = _sig(m)
        if s is not None:
            out.append((i, s))
    return out


def _detect_loop(sigs):
    """sigs: signatures of assistant tool-call turns, in order.
    Returns (kind, period, reps, start_sig_idx) or None; start_sig_idx is the
    index in sigs where the loop region begins (first occurrence of the block)."""
    n = len(sigs)
    if n < K_IDENTICAL:
        return None
    # (a) identical run at the tail
    last = sigs[-1]
    run = 0
    for s in reversed(sigs):
        if s == last:
            run += 1
        else:
            break
    if run >= K_IDENTICAL:
        return ("identical", 1, run, n - run)
    # (b) cycle at the tail (period L, >= CYCLE_MIN_REPS repetitions)
    for L in range(CYCLE_MIN_PERIOD, CYCLE_MAX_PERIOD + 1):
        if n < L * CYCLE_MIN_REPS:
            continue
        block = sigs[-L:]
        reps = 0
        idx = n
        while idx - L >= 0 and sigs[idx - L:idx] == block:
            reps += 1
            idx -= L
        if reps >= CYCLE_MIN_REPS:
            return ("cycle", L, reps, idx)
    return None


def _prune_and_force(data, messages, atc, kind, period, reps, start_sig_idx):
    """Rewrite messages: keep the first occurrence of the block, drop the
    repeated tail (including tool results), append marker + directive, set
    tool_choice=none. Returns the number of assistant turns dropped."""
    first_dropped = start_sig_idx + period
    if first_dropped >= len(atc):
        return 0
    cut_msg_idx = atc[first_dropped][0]
    dropped_turns = len(atc) - first_dropped
    kept = messages[:cut_msg_idx]
    marker = {
        "role": "user",
        "content": (
            f"[Loop breaker: the previous {'action' if kind == 'identical' else 'chain of actions'} "
            f"was repeated {dropped_turns}x without a new result and removed to protect the "
            f"context. Do NOT repeat it.]\n\n{_DIRECTIVE}"
        ),
    }
    data["messages"] = kept + [marker]
    data["tool_choice"] = "none"
    return dropped_turns


class LoopBreaker(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            if not LOOP_BREAKER_ON:
                return data
            model = data.get("model")
            if not (isinstance(model, str) and model.startswith(LOCAL_MODEL_PREFIXES)):
                return data
            messages = data.get("messages")
            if not isinstance(messages, list) or len(messages) < 4:
                return data
            if not _loop_is_active(messages):
                return data
            atc = _assistant_toolcall_turns(messages)
            sigs = [s for _, s in atc]
            det = _detect_loop(sigs)
            if not det:
                return data
            kind, period, reps, start = det
            dropped = _prune_and_force(data, messages, atc, kind, period, reps, start)
            if dropped > 0:
                meta = data.setdefault("metadata", {})
                meta["loop_intervention"] = {
                    "kind": kind, "period": period, "reps": reps,
                    "dropped_turns": dropped, "signature": str(sigs[-1])[:200],
                }
                print(
                    f"[loop_breaker] INTERVENTION model={model} kind={kind} period={period} "
                    f"reps={reps} dropped_turns={dropped} -> tool_choice=none + debloat",
                    flush=True,
                )
            return data
        except Exception as e:
            print(f"[loop_breaker] skip (err): {str(e)[:160]}", flush=True)
            return data

    async def async_post_call_response_headers_hook(
        self, data, user_api_key_dict, response, request_headers=None, litellm_call_info=None
    ):
        # Primary signal path: a response header. Harnesses stream almost
        # always, and on the streaming path the header is trivially readable
        # while body fields are discarded by the finalizer.
        try:
            li = (data.get("metadata") or {}).get("loop_intervention")
            if li:
                return {"x-litellm-loop-intervention": json.dumps(li)}
        except Exception:
            pass
        return None

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        # Fallback for the non-streaming path (subagents, quiet mode): an
        # extra top-level response attribute. Never msg.content.
        try:
            li = (data.get("metadata") or {}).get("loop_intervention")
            if li:
                try:
                    setattr(response, "x_loop_intervention", li)
                except Exception:
                    if isinstance(response, dict):
                        response["x_loop_intervention"] = li
        except Exception:
            pass
        return response


proxy_handler_instance = LoopBreaker()
