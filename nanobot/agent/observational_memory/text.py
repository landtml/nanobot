"""Formatting and parsing for Observational Memory, ported from Mastra 1.1.0.

Everything here is pure and deterministic. The functions mirror the upstream
TypeScript closely enough that the golden tests can compare them byte for byte;
where JavaScript semantics differ from Python's (string trimming, ``Date``
parsing, ``JSON.stringify``, UTF-16 string lengths) a small helper reproduces
the JavaScript behavior instead of silently diverging.

Upstream: ``observer-agent.ts``, ``reflector-agent.ts`` and the helpers at the
top of ``observational-memory.ts`` at mastra-ai/mastra@dc7ea18 (Apache-2.0).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Any, Literal, NotRequired, TypedDict, cast

# ---------------------------------------------------------------------------
# Message shape
# ---------------------------------------------------------------------------


class ToolInvocation(TypedDict, total=False):
    state: Literal["call", "partial-call", "result"]
    toolCallId: str
    toolName: str
    args: Any
    result: Any


class MessageContent(TypedDict, total=False):
    parts: list[dict[str, Any]]
    content: str


class ObserverMessage(TypedDict):
    """One message as the Observer sees it (Mastra's ``MastraDBMessage`` subset).

    ``content`` is either a plain string or Mastra's v2 content object whose
    ``parts`` hold ``text`` and ``tool-invocation`` parts. nanobot adds one
    part type, ``tool-exchange``, for a tool call paired with its result; see
    :func:`format_messages_for_observer`.
    """

    id: str
    role: str
    created_at: datetime | None
    content: str | MessageContent
    thread_id: NotRequired[str]


# Tool payloads are not part of the LongMemEval path Mastra benchmarked. nanobot
# runs shell and MCP tools whose output can be very large, so the Observer
# reads at most this much of any single tool argument or result.
TOOL_PAYLOAD_MAX_CHARS = 20_000

# ---------------------------------------------------------------------------
# JavaScript semantics
# ---------------------------------------------------------------------------

# ECMAScript WhiteSpace and LineTerminator code points, used by String#trim.
_JS_WHITESPACE = (
    "\t\n\v\f\r          "
    "        　﻿"
)


def js_trim(value: str) -> str:
    """``String.prototype.trim``."""
    return value.strip(_JS_WHITESPACE)


def js_trim_end(value: str) -> str:
    """``String.prototype.trimEnd``."""
    return value.rstrip(_JS_WHITESPACE)


def _utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _utf16_slice(value: str, end: int) -> str:
    """``value.slice(0, end)`` measured in UTF-16 code units.

    A surrogate pair split by the cut is dropped rather than emitted as a lone
    surrogate, which JavaScript would do but UTF-8 cannot carry.
    """
    raw = value.encode("utf-16-le")[: end * 2]
    return raw.decode("utf-16-le", errors="ignore")


def _js_number(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return "null"
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    return repr(value)


def js_stringify(value: Any, *, indent: int | None = None, _depth: int = 0) -> str:
    """``JSON.stringify(value, null, indent)`` for JSON-like Python data."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _js_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        items = [
            (json.dumps(str(k), ensure_ascii=False), v)
            for k, v in cast(Mapping[object, Any], value).items()
        ]
        rendered = [
            f"{key}:{' ' if indent else ''}{js_stringify(v, indent=indent, _depth=_depth + 1)}"
            for key, v in items
        ]
        return _js_container("{", "}", rendered, indent, _depth)
    if isinstance(value, (list, tuple)):
        rendered = [
            js_stringify(v, indent=indent, _depth=_depth + 1)
            for v in cast(Iterable[Any], value)
        ]
        return _js_container("[", "]", rendered, indent, _depth)
    return json.dumps(str(value), ensure_ascii=False)


def _js_container(open_: str, close: str, items: list[str], indent: int | None, depth: int) -> str:
    if not items:
        return open_ + close
    if not indent:
        return open_ + ",".join(items) + close
    inner = "\n" + " " * (indent * (depth + 1))
    return open_ + inner + ("," + inner).join(items) + "\n" + " " * (indent * depth) + close


# V8's legacy date parser matches month names by their first three letters.
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def js_date(month_name: str, day: int, year: int, tz: tzinfo) -> datetime | None:
    """``new Date("<Month> <day>, <year>")`` as V8 parses it: local midnight in *tz*.

    Months match on their first three letters, case-insensitively; days from 1
    to 31 roll over into the next month the way V8 does ("Feb 30" is Mar 2);
    anything else is invalid.
    """
    key = month_name[:3].lower()
    if len(month_name) < 3 or key not in _MONTHS or not 1 <= day <= 31:
        return None
    if year < 50:
        year += 2000
    elif year < 100:
        year += 1900
    first = datetime(year, _MONTHS.index(key) + 1, 1, tzinfo=tz)
    return first + timedelta(days=day - 1)


def _js_floor_days(start: datetime, end: datetime) -> int:
    """Whole days from *start* to *end* over absolute time, as JavaScript computes them.

    Python subtracts datetimes that share a tzinfo by wall clock, which skips
    the hour a DST change adds or removes; JavaScript never does.
    """
    return math.floor((end.timestamp() - start.timestamp()) * 1000 / 86_400_000)


# ---------------------------------------------------------------------------
# Message formatting (formatMessagesForObserver)
# ---------------------------------------------------------------------------


def format_timestamp(value: datetime, tz: tzinfo) -> str:
    """``toLocaleString('en-US', {year, month: 'short', day, hour, minute, hour12})``."""
    local = value.astimezone(tz)
    hour = local.hour % 12 or 12
    meridiem = "AM" if local.hour < 12 else "PM"
    return f"{local:%b} {local.day}, {local.year}, {hour}:{local.minute:02d} {meridiem}"


def _maybe_truncate(value: str, max_len: int | None) -> str:
    if not max_len or _utf16_units(value) <= max_len:
        return value
    remaining = _utf16_units(value) - max_len
    return f"{_utf16_slice(value, max_len)}\n... [truncated {remaining} characters]"


def _tool_payload(value: Any) -> str:
    text = value if isinstance(value, str) else js_stringify(value, indent=2)
    return _maybe_truncate(text, TOOL_PAYLOAD_MAX_CHARS)


def _format_part(part: Mapping[str, Any], max_len: int | None) -> str:
    kind = part.get("type")
    if kind == "text":
        return _maybe_truncate(str(part.get("text", "")), max_len)
    if kind == "tool-invocation":
        inv = cast(ToolInvocation, part.get("toolInvocation") or {})
        name = inv.get("toolName", "")
        if inv.get("state") == "result":
            return f"[Tool Result: {name}]\n{_maybe_truncate(js_stringify(inv.get('result'), indent=2), max_len)}"
        return f"[Tool Call: {name}]\n{_maybe_truncate(js_stringify(inv.get('args'), indent=2), max_len)}"
    if kind == "tool-exchange":
        # nanobot extension: a call together with its result, both bounded.
        name = str(part.get("toolName", ""))
        lines = [f"[Tool Call: {name}]\n{_maybe_truncate(_tool_payload(part.get('args')), max_len)}"]
        if "result" in part:
            lines.append(
                f"[Tool Result: {name}]\n{_maybe_truncate(_tool_payload(part.get('result')), max_len)}"
            )
        return "\n".join(lines)
    return ""


def format_messages_for_observer(
    messages: Sequence[ObserverMessage],
    tz: tzinfo,
    *,
    max_part_length: int | None = None,
) -> str:
    """Render messages as the Observer reads them (``formatMessagesForObserver``)."""
    rendered: list[str] = []
    for message in messages:
        created_at = message["created_at"]
        timestamp = f" ({format_timestamp(created_at, tz)})" if created_at else ""
        role = message["role"]
        role = role[:1].upper() + role[1:]
        content = message["content"]
        if isinstance(content, str):
            body = _maybe_truncate(content, max_part_length)
        elif content.get("parts"):
            body = "\n".join(
                text
                for part in content.get("parts") or []
                if (text := _format_part(part, max_part_length))
            )
        elif content.get("content"):
            body = _maybe_truncate(content.get("content", ""), max_part_length)
        else:
            body = ""
        rendered.append(f"**{role}{timestamp}:**\n{body}")
    return "\n\n---\n\n".join(rendered)


def format_multi_thread_messages(
    messages_by_thread: Mapping[str, Sequence[ObserverMessage]],
    thread_order: Sequence[str],
    tz: tzinfo,
) -> str:
    """``formatMultiThreadMessagesForObserver``."""
    sections: list[str] = []
    for thread_id in thread_order:
        messages = messages_by_thread.get(thread_id)
        if not messages:
            continue
        formatted = format_messages_for_observer(messages, tz)
        sections.append(f'<thread id="{thread_id}">\n{formatted}\n</thread>')
    return "\n\n".join(sections)


def format_other_conversations(
    messages_by_thread: Mapping[str, Sequence[ObserverMessage]],
    current_thread_id: str,
    tz: tzinfo,
    represent_thread_id: Callable[[str], str],
) -> str:
    """``formatUnobservedContextBlocks``: other threads' unobserved messages, truncated."""
    blocks: list[str] = []
    for thread_id, messages in messages_by_thread.items():
        if thread_id == current_thread_id or not messages:
            continue
        formatted = format_messages_for_observer(messages, tz, max_part_length=500)
        if formatted:
            blocks.append(
                f'<other-conversation id="{represent_thread_id(thread_id)}">\n'
                f"{formatted}\n</other-conversation>"
            )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObserverResult:
    observations: str
    current_task: str | None = None
    suggested_response: str | None = None


_OBSERVATIONS_BLOCK_RE = re.compile(
    r"^[ \t]*<observations>([\s\S]*?)^[ \t]*</observations>", re.IGNORECASE | re.MULTILINE
)
_ANCHORED_TASK_RE = re.compile(
    r"^[ \t]*<current-task>([\s\S]*?)^[ \t]*</current-task>", re.IGNORECASE | re.MULTILINE
)
_ANCHORED_SUGGESTED_RE = re.compile(
    r"^[ \t]*<suggested-response>([\s\S]*?)^[ \t]*</suggested-response>",
    re.IGNORECASE | re.MULTILINE,
)
_TASK_RE = re.compile(r"<current-task>([\s\S]*?)</current-task>", re.IGNORECASE)
_SUGGESTED_RE = re.compile(r"<suggested-response>([\s\S]*?)</suggested-response>", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^\s*[-*]\s|^\s*\d+\.\s")
_THREAD_BLOCK_RE = re.compile(r'<thread\s+id="([^"]+)">([\s\S]*?)</thread>', re.IGNORECASE)


def _list_items(content: str) -> str:
    return js_trim("\n".join(line for line in content.split("\n") if _LIST_ITEM_RE.match(line)))


def _observation_blocks(content: str) -> str | None:
    matches = _OBSERVATIONS_BLOCK_RE.findall(content)
    if not matches:
        return None
    return "\n".join(text for block in matches if (text := js_trim(block)))


def _first(pattern: re.Pattern[str], content: str) -> str | None:
    match = pattern.search(content)
    if match is None or not match.group(1):
        return None
    return js_trim(match.group(1)) or None


def parse_observer_output(output: str) -> ObserverResult:
    """``parseObserverOutput`` / ``parseMemorySectionXml``."""
    observations = _observation_blocks(output)
    if observations is None:
        observations = _list_items(output)
    return ObserverResult(
        observations=observations,
        current_task=_first(_ANCHORED_TASK_RE, output),
        suggested_response=_first(_ANCHORED_SUGGESTED_RE, output),
    )


def parse_reflector_output(output: str) -> ObserverResult:
    """``parseReflectorOutput``. The Reflector's current task is intentionally dropped."""
    observations = _observation_blocks(output)
    if observations is None:
        observations = _list_items(output) or js_trim(output)
    return ObserverResult(
        observations=observations,
        suggested_response=_first(_SUGGESTED_RE, output),
    )


def parse_multi_thread_output(output: str) -> dict[str, ObserverResult]:
    """``parseMultiThreadObserverOutput``: per-thread results keyed by thread id."""
    block = _OBSERVATIONS_BLOCK_RE.search(output)
    content = block.group(1) if block is not None else output
    results: dict[str, ObserverResult] = {}
    for match in _THREAD_BLOCK_RE.finditer(content):
        thread_id, thread_content = match.group(1), match.group(2)
        if not thread_id or not thread_content:
            continue
        observations = thread_content
        current_task = _first(_TASK_RE, thread_content)
        if (task := _TASK_RE.search(thread_content)) is not None and task.group(1):
            observations = _TASK_RE.sub("", observations, count=1)
        suggested = _first(_SUGGESTED_RE, thread_content)
        if (hint := _SUGGESTED_RE.search(thread_content)) is not None and hint.group(1):
            observations = _SUGGESTED_RE.sub("", observations, count=1)
        results[thread_id] = ObserverResult(
            observations=js_trim(observations),
            current_task=current_task,
            suggested_response=suggested,
        )
    return results


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------

_SEMANTIC_TAG_RE = re.compile(r"\[(?![\d\s]*items collapsed)[^\]]+\]")


def optimize_observations_for_context(observations: str) -> str:
    """``optimizeObservationsForContext``: drop low-priority emojis, tags and arrows."""
    optimized = re.sub(r"🟡\s*", "", observations)
    optimized = re.sub(r"🟢\s*", "", optimized)
    optimized = _SEMANTIC_TAG_RE.sub("", optimized)
    optimized = re.sub(r"\s*->\s*", " ", optimized)
    optimized = re.sub(r"  +", " ", optimized)
    optimized = re.sub(r"\n{3,}", "\n\n", optimized)
    return js_trim(optimized)


def format_relative_time(date: datetime, current: datetime) -> str:
    """``formatRelativeTime``."""
    days = _js_floor_days(date, current)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 14:
        return "1 week ago"
    if days < 30:
        return f"{days // 7} weeks ago"
    if days < 60:
        return "1 month ago"
    if days < 365:
        return f"{days // 30} months ago"
    years = days // 365
    return f"{years} year{'s' if years > 1 else ''} ago"


def _format_gap(previous: datetime, current: datetime) -> str | None:
    days = _js_floor_days(previous, current)
    if days <= 1:
        return None
    if days < 7:
        return f"[{days} days later]"
    if days < 14:
        return "[1 week later]"
    if days < 30:
        return f"[{days // 7} weeks later]"
    if days < 60:
        return "[1 month later]"
    return f"[{days // 30} months later]"


_SIMPLE_DATE_RE = re.compile(r"([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})")
_RANGE_DATE_RE = re.compile(r"([A-Z][a-z]+)\s+(\d{1,2})-\d{1,2},?\s+(\d{4})")
_VAGUE_DATE_RE = re.compile(
    r"(late|early|mid)[- ]?(?:to[- ]?(?:late|early|mid)[- ]?)?([A-Z][a-z]+)\s+(\d{4})",
    re.IGNORECASE,
)
_CROSS_MONTH_RE = re.compile(r"([A-Z][a-z]+)\s+to\s+(?:early\s+)?([A-Z][a-z]+)\s+(\d{4})", re.IGNORECASE)
_FUTURE_INTENT_RES = tuple(
    re.compile(pattern, re.IGNORECASE | re.ASCII)
    for pattern in (
        r"\bwill\s+(?:be\s+)?(?:\w+ing|\w+)\b",
        r"\bplans?\s+to\b",
        r"\bplanning\s+to\b",
        r"\blooking\s+forward\s+to\b",
        r"\bgoing\s+to\b",
        r"\bintends?\s+to\b",
        r"\bwants?\s+to\b",
        r"\bneeds?\s+to\b",
        r"\babout\s+to\b",
    )
)
_INLINE_DATE_RE = re.compile(r"\((estimated|meaning)\s+([^)]+\d{4})\)", re.IGNORECASE)
_DATE_HEADER_RE = re.compile(r"^(Date:\s*)([A-Z][a-z]+ \d{1,2}, \d{4})$", re.MULTILINE)


def _parse_date_content(content: str, tz: tzinfo) -> datetime | None:
    """``parseDateFromContent``."""
    for pattern in (_SIMPLE_DATE_RE, _RANGE_DATE_RE):
        match = pattern.search(content)
        if match is not None:
            parsed = js_date(match.group(1), int(match.group(2)), int(match.group(3)), tz)
            if parsed is not None:
                return parsed
    match = _VAGUE_DATE_RE.search(content)
    if match is not None:
        day = {"early": 7, "late": 23}.get(match.group(1).lower(), 15)
        parsed = js_date(match.group(2), day, int(match.group(3)), tz)
        if parsed is not None:
            return parsed
    match = _CROSS_MONTH_RE.search(content)
    if match is not None:
        return js_date(match.group(2), 1, int(match.group(3)), tz)
    return None


def _expand_inline_dates(observations: str, current: datetime, tz: tzinfo) -> str:
    """``expandInlineEstimatedDates``."""

    def replace(match: re.Match[str]) -> str:
        prefix, content = match.group(1), match.group(2)
        target = _parse_date_content(content, tz)
        if target is None:
            return match.group(0)
        relative = format_relative_time(target, current)
        # Upstream locates the line with indexOf(match), i.e. the first occurrence.
        index = observations.find(match.group(0))
        line_before = observations[observations.rfind("\n", 0, index) + 1:index]
        if target < current and any(p.search(line_before) for p in _FUTURE_INTENT_RES):
            return f"({prefix} {content} - {relative}, likely already happened)"
        return f"({prefix} {content} - {relative})"

    return _INLINE_DATE_RE.sub(replace, observations)


def add_relative_time(observations: str, current: datetime, tz: tzinfo) -> str:
    """``addRelativeTimeToObservations``: relative dates and gap markers for the Actor."""
    expanded = _expand_inline_dates(observations, current, tz)
    headers: list[tuple[re.Match[str], datetime]] = []
    for match in _DATE_HEADER_RE.finditer(expanded):
        month, day, year = match.group(2).replace(",", "").split(" ")
        parsed = js_date(month, int(day), int(year), tz)
        if parsed is not None:
            headers.append((match, parsed))
    if not headers:
        return expanded
    result: list[str] = []
    last = 0
    previous: datetime | None = None
    for match, date in headers:
        result.append(expanded[last:match.start()])
        if previous is not None and (gap := _format_gap(previous, date)):
            result.append(f"\n{gap}\n\n")
        result.append(f"{match.group(1)}{match.group(2)} ({format_relative_time(date, current)})")
        last = match.end()
        previous = date
    result.append(expanded[last:])
    return "".join(result)


def render_observations(observations: str, current: datetime | None, tz: tzinfo) -> str:
    """Optimize observations and annotate dates, as the Actor sees them."""
    optimized = optimize_observations_for_context(observations)
    return add_relative_time(optimized, current, tz) if current is not None else optimized


# ---------------------------------------------------------------------------
# Resource-scope thread sections
# ---------------------------------------------------------------------------

_THREAD_TAG_RE = re.compile(r"<thread[^>]*>|</thread>", re.IGNORECASE)
_SECTION_THREAD_RE = re.compile(r'<thread id="([^"]+)">')
_SECTION_DATE_RE = re.compile(r"Date:\s*([A-Za-z]+\s+\d+,\s+\d+)")
_SECTION_BODY_RE = re.compile(r'<thread id="[^"]+">[\s\S]*?Date:[^\n]*\n([\s\S]*?)\n</thread>')


def strip_thread_tags(observations: str) -> str:
    """``stripThreadTags``."""
    return js_trim(_THREAD_TAG_RE.sub("", observations))


def wrap_with_thread_tag(thread_id: str, observations: str) -> str:
    """``wrapWithThreadTag``; *thread_id* is the already obscured id."""
    return f'<thread id="{thread_id}">\n{strip_thread_tags(observations)}\n</thread>'


def replace_or_append_thread_section(existing: str, section: str) -> str:
    """``replaceOrAppendThreadSection``: merge into a same-thread, same-date section."""
    if not existing:
        return section
    thread_match = _SECTION_THREAD_RE.search(section)
    date_match = _SECTION_DATE_RE.search(section)
    if thread_match is None or date_match is None:
        return f"{existing}\n\n{section}"
    pattern = re.compile(
        f'<thread id="{thread_match.group(1)}">\\s*Date:\\s*{re.escape(date_match.group(1))}'
        r"([\s\S]*?)</thread>"
    )
    existing_match = pattern.search(existing)
    if existing_match is not None:
        body = _SECTION_BODY_RE.search(section)
        if body is not None and body.group(1):
            addition = js_trim(body.group(1))
            merged = js_trim_end(re.sub(r"</thread>$", "", existing_match.group(0)))
            return (
                existing[: existing_match.start()]
                + f"{merged}\n{addition}\n</thread>"
                + existing[existing_match.end():]
            )
    return f"{existing}\n\n{section}"


# ---------------------------------------------------------------------------
# Thread id obscuring (xxhash32, as xxhash-wasm's h32ToString)
# ---------------------------------------------------------------------------

_P1, _P2, _P3, _P4, _P5 = 2654435761, 2246822519, 3266489917, 668265263, 374761393
_MASK = 0xFFFFFFFF


def _rotl(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (32 - bits))) & _MASK


def _round(acc: int, lane: int) -> int:
    return (_rotl((acc + lane * _P2) & _MASK, 13) * _P1) & _MASK


def xxh32(data: bytes, seed: int = 0) -> int:
    """The 32-bit xxHash of *data*."""
    length = len(data)
    index = 0
    if length >= 16:
        v1 = (seed + _P1 + _P2) & _MASK
        v2 = (seed + _P2) & _MASK
        v3 = seed & _MASK
        v4 = (seed - _P1) & _MASK
        while index <= length - 16:
            v1 = _round(v1, int.from_bytes(data[index:index + 4], "little"))
            v2 = _round(v2, int.from_bytes(data[index + 4:index + 8], "little"))
            v3 = _round(v3, int.from_bytes(data[index + 8:index + 12], "little"))
            v4 = _round(v4, int.from_bytes(data[index + 12:index + 16], "little"))
            index += 16
        acc = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _MASK
    else:
        acc = (seed + _P5) & _MASK
    acc = (acc + length) & _MASK
    while index <= length - 4:
        lane = int.from_bytes(data[index:index + 4], "little")
        acc = (_rotl((acc + lane * _P3) & _MASK, 17) * _P4) & _MASK
        index += 4
    while index < length:
        acc = (_rotl((acc + data[index] * _P5) & _MASK, 11) * _P1) & _MASK
        index += 1
    acc ^= acc >> 15
    acc = (acc * _P2) & _MASK
    acc ^= acc >> 13
    acc = (acc * _P3) & _MASK
    acc ^= acc >> 16
    return acc


def obscure_thread_id(thread_id: str) -> str:
    """Short, opaque id for a session key (``representThreadIDInContext``)."""
    return f"{xxh32(thread_id.encode('utf-8')):08x}"
