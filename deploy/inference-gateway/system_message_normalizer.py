"""LiteLLM callback: system-message normalizer.

PROBLEM: the Qwen3.8 chat template throws `System message must be at the
beginning.` as soon as a system message is not at index 0. Some clients
(workflow builders, notably) inject system messages mid-history and resend
the full history every turn — so every retry fails identically and the
session becomes unusable.

FIX, stateless and in the request: every system message with index > 0
becomes a user message prefixed "[System note]". Position and content are
preserved; the template accepts it. A late system message is never moved to
the front (order is meaning), it is rewritten in place.

Applies to all models; for OpenAI-compatible backends the rewrite is
harmless. Off via SYSMSG_NORMALIZE=0. Defensive: error -> request unchanged.

Registration (config.yaml): litellm_settings.callbacks:
  - system_message_normalizer.proxy_handler_instance
"""
import os

from litellm.integrations.custom_logger import CustomLogger

NORMALIZE_ON = os.environ.get("SYSMSG_NORMALIZE", "1") == "1"
PREFIX = os.environ.get("SYSMSG_PREFIX", "[System note]\n")


def _prefix_content(content):
    """Prefix string or parts content; unknown types unchanged."""
    if isinstance(content, str):
        return PREFIX + content
    if isinstance(content, list):
        return [{"type": "text", "text": PREFIX.rstrip("\n")}] + content
    return content


def normalize_messages(messages):
    """Returns (new_messages, count_rewritten). Index 0 is never touched."""
    if not isinstance(messages, list):
        return messages, 0
    changed = 0
    out = []
    for i, m in enumerate(messages):
        if i > 0 and isinstance(m, dict) and m.get("role") == "system":
            nm = dict(m)
            nm["role"] = "user"
            nm["content"] = _prefix_content(m.get("content", ""))
            nm.pop("name", None)
            out.append(nm)
            changed += 1
        else:
            out.append(m)
    return out, changed


class SystemMessageNormalizer(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            if not NORMALIZE_ON:
                return data
            if call_type not in ("completion", "acompletion", "text_completion", None):
                return data
            messages = data.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                return data
            new_messages, changed = normalize_messages(messages)
            if changed:
                data["messages"] = new_messages
                print(
                    f"[sysmsg-normalizer] model={data.get('model')} rewrote {changed} "
                    f"non-leading system message(s) to user",
                    flush=True,
                )
            return data
        except Exception as exc:
            print(f"[sysmsg-normalizer] skipped: {exc!r}", flush=True)
            return data


proxy_handler_instance = SystemMessageNormalizer()
