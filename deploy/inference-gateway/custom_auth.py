"""LiteLLM custom auth: keyless for local models, nothing else exposed.

Policy: every model served by a box in the house is reachable without a key
on the internal network. Anything that is not in LOCAL_MODELS answers 401.
There is deliberately no key database — it cost resources and every DB
hiccup was an incident; one master key (admin API only) under GitOps is the
decided steady state.

Two behaviours worth knowing:

- A client that sends *some* key (OpenAI SDKs insist on a non-empty one) is
  treated like a keyless client: local models as guest, everything else 401.
  Without a connected key DB, LiteLLM's standard auth would otherwise end
  every such request with "No connected db."
- The request body is read with `request.body()`, which Starlette caches and
  which is idempotent across repeated calls. An earlier version patched
  `request._receive` to re-read the body; LiteLLM may call this function more
  than once per request, the second patch overwrote a stream the SSE consumer
  was still reading from, and streaming requests hung with zero bytes. Do
  not patch `_receive`.
"""
import json
import logging

from fastapi import HTTPException, Request
from litellm.proxy._types import UserAPIKeyAuth

logger = logging.getLogger("litellm.custom_auth")
logger.setLevel(logging.INFO)

# Everything in config.yaml's model_list, plus the harness aliases.
LOCAL_MODELS = {
    "qwen3.8-27b",
    "qwen3.8-27b-no-think",
    "qwen3.8-27b-tools",
    "qwen3.8-27b-overflow",
    "qwen3.8-27b-no-think-overflow",
    "qwen3.8-27b-tools-overflow",
    "hermes",
    "hermes-agent",
    "hermes-embed",
    "hermes-overflow",
    "hermes-primary",
    "nomic-embed-text-v1.5.f16",
    "bge-m3",
    "qwen3-embed-8b",
    "harrier-embed",
    "qwen3-reranker",
    "bge-reranker-v2-m3",
}

PUBLIC_PATHS = {
    "/v1/models",
    "/models",
    "/health/liveliness",
    "/health/readiness",
    "/metrics",
}

KEYLESS_VALUES = {
    "none",
    "keyless",
    "no-key-required",
    "Bearer none",
    "Bearer ",
    "Bearer no-key-required",
}


def _db_connected() -> bool:
    """True if LiteLLM initialised a Prisma client (DATABASE_URL set)."""
    try:
        from litellm.proxy import proxy_server  # lazy: avoid a circular import at load
        return getattr(proxy_server, "prisma_client", None) is not None
    except Exception as e:
        logger.warning(f"Could not determine DB state, assuming disconnected: {e}")
        return False


def _guest_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="keyless_guest",
        user_id="anonymous",
        models=list(LOCAL_MODELS),
        max_budget=0.0,
    )


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    path = request.url.path
    if path in PUBLIC_PATHS:
        return _guest_auth()

    requested_model = None
    try:
        body_bytes = await request.body()
        if body_bytes:
            requested_model = json.loads(body_bytes).get("model")
    except Exception as e:
        logger.warning(f"Could not parse request body: {e}")

    is_keyless = not api_key or api_key.strip() in KEYLESS_VALUES
    if is_keyless:
        if requested_model in LOCAL_MODELS:
            logger.info(f"Allowing keyless access to local model: {requested_model}")
            return _guest_auth()
        raise HTTPException(
            status_code=401,
            detail=f"Authentication required for model '{requested_model}'.",
        )

    # Keyed request without a key DB: local models as guest, everything else 401.
    if not _db_connected():
        if requested_model in LOCAL_MODELS:
            logger.info(f"DB decoupled: treating keyed request as guest for local model: {requested_model}")
            return _guest_auth()
        raise HTTPException(
            status_code=401,
            detail=f"Model '{requested_model}' requires a virtual key, but the key database is not connected.",
        )

    raise Exception("Bypass to standard DB key authentication")
