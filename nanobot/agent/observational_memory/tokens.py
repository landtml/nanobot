"""Token counting for Observational Memory thresholds (Mastra 1.1.0 ``TokenCounter``).

Thresholds are what decide when the Observer and Reflector run, so they are
counted exactly as in the benchmarked version: ``o200k_base`` via tiktoken,
3.8 tokens of framing per message and 24 per conversation. When the encoding
cannot be loaded (for example offline on first run) counting falls back to a
conservative bytes/4 estimate and logs once; thresholds then trigger slightly
early rather than late.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from loguru import logger

from nanobot.agent.observational_memory.text import ObserverMessage, ToolInvocation, js_stringify

TOKENS_PER_MESSAGE = 3.8
TOKENS_PER_CONVERSATION = 24


class _Encoding(Protocol):
    def encode(self, text: str, *, allowed_special: str) -> list[int]: ...


@functools.cache
def _encoding() -> _Encoding | None:
    try:
        import tiktoken

        return cast(_Encoding, tiktoken.get_encoding("o200k_base"))
    except Exception as exc:  # pragma: no cover - depends on network and cache state
        logger.warning(
            "Observational memory: o200k_base unavailable ({}); estimating tokens from bytes",
            exc,
        )
        return None


def _js_truthy(value: object) -> bool:
    """JavaScript truthiness: empty objects and arrays are truthy."""
    if isinstance(value, (dict, list, tuple)):
        return True
    return bool(value)


def count_string(text: str) -> int:
    """Tokens in *text*, treating special-token text as ordinary text."""
    if not text:
        return 0
    encoding = _encoding()
    if encoding is None:
        return (len(text.encode("utf-8")) + 3) // 4
    return len(encoding.encode(text, allowed_special="all"))


def count_message(message: ObserverMessage) -> float:
    """``TokenCounter.countMessage``: content tokens plus framing overhead."""
    token_string = message["role"]
    overhead = TOKENS_PER_MESSAGE
    tool_results = 0
    content = message["content"]
    if isinstance(content, str):
        token_string += content
    elif content.get("content") and not isinstance(content.get("parts"), list):
        token_string += content.get("content", "")
    elif isinstance(content.get("parts"), list):
        for part in content.get("parts") or []:
            kind = part.get("type")
            if kind == "text":
                token_string += str(part.get("text", ""))
            elif kind == "tool-invocation":
                invocation = cast(ToolInvocation, part.get("toolInvocation") or {})
                state = invocation.get("state")
                if state in ("call", "partial-call"):
                    token_string += invocation.get("toolName") or ""
                    args = invocation.get("args")
                    if _js_truthy(args):
                        if isinstance(args, str):
                            token_string += args
                        else:
                            token_string += js_stringify(args)
                            overhead -= 12
                elif state == "result":
                    tool_results += 1
                    if "result" in invocation:
                        result = invocation.get("result")
                        if isinstance(result, str):
                            token_string += result
                        else:
                            token_string += js_stringify(result)
                            overhead -= 12
                else:
                    raise ValueError(
                        f"Unhandled tool-invocation state {state!r} in token counting"
                    )
            else:
                token_string += js_stringify(cast(Mapping[str, Any], part))
    if tool_results:
        overhead += tool_results * TOKENS_PER_MESSAGE
    return count_string(token_string) + overhead


def count_messages(messages: Sequence[ObserverMessage]) -> float:
    """``TokenCounter.countMessages``."""
    if not messages:
        return 0
    return TOKENS_PER_CONVERSATION + sum(count_message(message) for message in messages)
