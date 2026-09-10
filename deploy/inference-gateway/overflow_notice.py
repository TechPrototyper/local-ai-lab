"""LiteLLM callback: routing gate + resilience semaphores for two local inference boxes.

Sits in front of the RTX 5090 (primary, latency tier) and the DGX Spark
(capacity tier). Three jobs, in priority order:

1. RESILIENCE SEMAPHORES — env on the Deployment, i.e. GitOps with a git-log
   audit trail:
     SPARK_DOWN=1   capacity tier out of the pool: no overflow; local requests
                    queue on the RTX rather than going to a dead box.
     RTX_DOWN=1     RTX out of the pool: every local request to the capacity tier.
     both           honest HTTP 503 with Retry-After, immediately. Never a
                    silent cloud fallback.
     SPARK_FORCE=1  everything to the capacity tier regardless of load.

2. CONCURRENCY GATE — more than MAX_LOCAL local requests in flight and the next
   one is rewritten to its "-overflow" alias (config.yaml maps that to the
   capacity tier). The counter is a process-local dict; that is correct only
   because the Deployment runs `--num_workers=1`. Multiple workers means
   moving this state to Redis. Entries expire after ACTIVE_TTL seconds, which
   MUST equal litellm's request_timeout (see the lesson below).

3. GREEDY-SAMPLING GUARD — the local model loops under near-greedy sampling
   (tool-call relays, chunk loops). Profiles and server defaults are set
   correctly, but a client that sends `temperature` explicitly overrides both.
   Cold sampling on the local model is counted, logged and, with
   GREEDY_COERCE=1, rewritten to the anti-loop profile.

Two lessons this file carries:

- Never put a notice into msg.content. An earlier version prepended an
  "overflow mode" banner to the response; agents replay their context every
  turn, the banner saturated it, and the model repeated the banner forever.
  Signals go out of band (headers, metadata) or not at all.
- ACTIVE_TTL and request_timeout drifted apart once (600 vs 3600 s) and the
  gate silently under-counted: a request running longer than the TTL fell out
  of the counter while still holding a vLLM slot. Both are env-driven now so
  they can be set together.

Defensive throughout: an unexpected error must never break a request
(try/except -> pass it through unchanged). The intentional 503 is the one
exception and is re-raised on purpose.

Registration: litellm_settings.callbacks: ["overflow_notice.proxy_handler_instance"]
"""
import os
import threading
import time

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

MAX_LOCAL = int(os.environ.get("MAX_LOCAL", "2"))
ACTIVE_TTL = float(os.environ.get("ACTIVE_TTL", "3600"))  # == request_timeout
SPARK_FORCE = os.environ.get("SPARK_FORCE", "0") == "1"
SPARK_DOWN = os.environ.get("SPARK_DOWN", "0") == "1"
RTX_DOWN = os.environ.get("RTX_DOWN", "0") == "1"

# Which model names count as "local" for the gate. Comma-separated prefixes;
# "-overflow" and "-spark" suffixed aliases are never local.
LOCAL_MODEL_PREFIXES = tuple(
    p.strip() for p in os.environ.get("LOCAL_MODEL_PREFIXES", "qwen3.8-27b").split(",") if p.strip()
)

GREEDY_COERCE = os.environ.get("GREEDY_COERCE", "1") == "1"
SAFE_TEMPERATURE = float(os.environ.get("GREEDY_SAFE_TEMPERATURE", "0.7"))
SAFE_TOP_P = float(os.environ.get("GREEDY_SAFE_TOP_P", "0.8"))
SAFE_PRESENCE_PENALTY = float(os.environ.get("GREEDY_SAFE_PRESENCE_PENALTY", "1.5"))

_lock = threading.Lock()
_active = {}  # call_id -> monotonic start time; the local in-flight counter

# Metrics must never endanger routing: if prometheus_client is missing or the
# port is taken, the gate runs without them.
try:
    from prometheus_client import Counter, Gauge, start_http_server

    _inflight_gauge = Gauge(
        "litellm_overflow_local_inflight",
        "Locally counted in-flight requests in the overflow gate (basis of the MAX_LOCAL decision)",
    )
    _decision_counter = Counter(
        "litellm_overflow_decisions_total",
        "Routing decisions of the overflow gate",
        ["decision", "model"],
    )
    _greedy_counter = Counter(
        "litellm_greedy_sampling_requests_total",
        "Local-model requests with client-set greedy sampling (loop risk)",
        ["model", "reason"],
    )
    _greedy_coerce_counter = Counter(
        "litellm_greedy_coerce_total",
        "Local-model requests whose cold sampling was coerced to the anti-loop profile",
        ["model", "reason"],
    )
    start_http_server(int(os.environ.get("GUARD_METRICS_PORT", "9105")))
    _METRICS_ON = True
except Exception as _e:
    print(f"[overflow_notice] guard metrics disabled: {_e}", flush=True)
    _METRICS_ON = False


def _purge_stale_locked() -> None:
    """Drop entries older than ACTIVE_TTL (lock must be held). After
    request_timeout the request is guaranteed finished; a surviving entry is
    a leak (client abort or similar)."""
    now = time.monotonic()
    for cid in [cid for cid, t0 in _active.items() if now - t0 > ACTIVE_TTL]:
        _active.pop(cid, None)


def _is_local_base(model) -> bool:
    """A directly selectable local model (not an -overflow twin, not a direct
    -spark alias). Direct capacity-tier aliases are neither counted nor
    rewritten."""
    return (
        isinstance(model, str)
        and model.startswith(LOCAL_MODEL_PREFIXES)
        and not model.endswith("-overflow")
        and not model.endswith("-spark")
    )


def _overflow_name(model: str) -> str:
    return model + "-overflow"


def _call_id(data) -> str:
    d = data or {}
    return d.get("litellm_call_id") or d.get("litellm_trace_id") or str(id(d))


def _observe(decision: str, model: str, inflight: int) -> None:
    """Export the gate decision and the in-flight level. Without these two
    metrics, "does the overflow trigger at the right moment?" can only be
    answered indirectly from the vLLM queue — which is exactly what one
    analysis cost."""
    if not _METRICS_ON:
        return
    try:
        _decision_counter.labels(decision=decision, model=str(model)).inc()
        _inflight_gauge.set(inflight)
    except Exception:
        pass


def _release(data) -> None:
    try:
        with _lock:
            _active.pop(_call_id(data), None)
            n = len(_active)
        if _METRICS_ON:
            try:
                _inflight_gauge.set(n)
            except Exception:
                pass
    except Exception:
        pass


def _greedy_check(model, data):
    """Detect client-set cold sampling on a local model. Logs, counts, and
    returns the reason (str) or None so the hook can optionally coerce.
    Heuristic: no profile in config.yaml ever yields temperature <= 0.1 for
    the local model; if it shows up in the merged params, the client set it."""
    try:
        if not (isinstance(model, str) and model.startswith(LOCAL_MODEL_PREFIXES)):
            return None
        temp = data.get("temperature")
        pp = data.get("presence_penalty")
        reason = None
        if temp is not None and float(temp) <= 0.1:
            reason = "temperature_le_0.1"
        elif pp is not None and float(pp) == 0.0 and temp is not None and float(temp) <= 0.3:
            reason = "cold_no_presence_penalty"
        if reason:
            meta = data.get("metadata") or {}
            print(
                f"[overflow_notice] GREEDY-SAMPLING model={model} temp={temp} "
                f"top_p={data.get('top_p')} presence_penalty={pp} reason={reason} "
                f"key_alias={meta.get('user_api_key_alias')}",
                flush=True,
            )
            if _METRICS_ON:
                _greedy_counter.labels(model=model, reason=reason).inc()
        return reason
    except Exception:
        return None


class OverflowNotice(CustomLogger):
    # Routing gate: semaphores first, then concurrency.
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            model = data.get("model")

            greedy_reason = _greedy_check(model, data)
            if greedy_reason and GREEDY_COERCE and _is_local_base(model):
                data["temperature"] = SAFE_TEMPERATURE
                data["top_p"] = SAFE_TOP_P
                data["presence_penalty"] = SAFE_PRESENCE_PENALTY
                print(
                    f"[overflow_notice] GREEDY-COERCE model={model} -> "
                    f"temp={SAFE_TEMPERATURE}/top_p={SAFE_TOP_P}/"
                    f"presence_penalty={SAFE_PRESENCE_PENALTY} (reason={greedy_reason})",
                    flush=True,
                )
                if _METRICS_ON:
                    _greedy_coerce_counter.labels(model=model, reason=greedy_reason).inc()

            if not _is_local_base(model):
                return data

            # (a) both boxes down -> honest 503, immediately
            if SPARK_DOWN and RTX_DOWN:
                raise HTTPException(
                    status_code=503,
                    detail="Local inference temporarily unavailable (both inference tiers are down).",
                    headers={"Retry-After": "30"},
                )

            # (b) RTX down -> everything to the capacity tier
            if RTX_DOWN:
                data["model"] = _overflow_name(model)
                return data

            # (c) normal path: SPARK_DOWN (no overflow) before SPARK_FORCE before MAX_LOCAL
            cid = _call_id(data)
            with _lock:
                _purge_stale_locked()
                if SPARK_DOWN:
                    _active[cid] = time.monotonic()
                    decision = "local_spark_down"
                elif SPARK_FORCE or len(_active) >= MAX_LOCAL:
                    data["model"] = _overflow_name(model)
                    decision = "overflow_forced" if SPARK_FORCE else "overflow_max_local"
                else:
                    _active[cid] = time.monotonic()
                    decision = "local"
                n = len(_active)
            _observe(decision, model, n)
            return data
        except HTTPException:
            raise  # the intentional 503
        except Exception:
            return data  # unexpected -> pass through unchanged

    # Release the counter (idempotent). No banner, ever.
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        _release(data)
        return response

    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        _release(request_data)

    # Streaming: pure pass-through, release at the end. Handle both iterable
    # kinds — an earlier async-only pass-through swallowed the stream when
    # the response object was not async-iterable, and the client never saw a
    # token.
    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        try:
            if hasattr(response, "__aiter__"):
                async for chunk in response:
                    yield chunk
            else:
                for chunk in response:
                    yield chunk
        finally:
            _release(request_data)


proxy_handler_instance = OverflowNotice()
