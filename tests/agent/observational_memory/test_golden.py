"""Byte-for-byte equivalence with Mastra Observational Memory 1.1.0.

The golden files were produced by running the upstream TypeScript (see
``scripts/om_golden``) on ``golden/fixtures.json`` in three timezones. Every
prompt, formatter, parser and token count the Observer, Reflector and Actor
depend on must reproduce them exactly: that is what lets nanobot's port claim
the behavior Mastra benchmarked on LongMemEval.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from nanobot.agent.observational_memory import prompts, text, tokens
from nanobot.agent.observational_memory.engine import select_and_batch
from nanobot.agent.observational_memory.text import ObserverMessage

GOLDEN = Path(__file__).parent / "golden"
FIXTURES: dict[str, Any] = json.loads((GOLDEN / "fixtures.json").read_text(encoding="utf-8"))
GOLDEN_FILES = sorted(GOLDEN.glob("golden.*.json"))


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _message(raw: dict[str, Any]) -> ObserverMessage:
    return {
        "id": raw["id"],
        "role": raw["role"],
        "created_at": _instant(raw["createdAt"]) if raw["createdAt"] else None,
        "content": raw["content"],
    }


def _conversation(name: str) -> list[ObserverMessage]:
    return [_message(raw) for raw in FIXTURES["conversations"][name]]


def _threads(mapping: dict[str, str]) -> dict[str, list[ObserverMessage]]:
    return {thread_id: _conversation(name) for thread_id, name in mapping.items()}


@pytest.fixture(params=GOLDEN_FILES, ids=lambda path: path.stem.removeprefix("golden."))
def golden(request: pytest.FixtureRequest) -> tuple[dict[str, Any], ZoneInfo]:
    data = json.loads(request.param.read_text(encoding="utf-8"))
    return data, ZoneInfo(data["timezone"])


def test_golden_files_cover_three_timezones() -> None:
    assert len(GOLDEN_FILES) == 3


def test_system_prompts(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, _ = golden
    assert prompts.observer_system_prompt() == data["system_prompts"]["observer"]
    assert (
        prompts.observer_system_prompt(multi_thread=True)
        == data["system_prompts"]["observer_multi_thread"]
    )
    assert prompts.reflector_system_prompt() == data["system_prompts"]["reflector"]
    assert prompts.CONTINUATION_REMINDER == data["continuation_reminder"]


def test_message_formatting(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, tz = golden
    for name in FIXTURES["conversations"]:
        assert text.format_messages_for_observer(_conversation(name), tz) == data["formatted"][name], name
    for case, expected in zip(FIXTURES["max_part_length"], data["formatted_truncated"], strict=True):
        formatted = text.format_messages_for_observer(
            _conversation(case["conversation"]), tz, max_part_length=case["max"],
        )
        assert formatted == expected


def test_observer_prompts(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, tz = golden
    for case, expected in zip(FIXTURES["observer_prompts"], data["observer_prompts"], strict=True):
        formatted = text.format_messages_for_observer(_conversation(case["conversation"]), tz)
        assert prompts.observer_prompt(case["existing"], formatted) == expected
    for case, expected in zip(
        FIXTURES["multi_thread_prompts"], data["multi_thread_prompts"], strict=True,
    ):
        threads = _threads(case["threads"])
        formatted = text.format_multi_thread_messages(threads, case["order"], tz)
        prompt = prompts.multi_thread_observer_prompt(case["existing"], formatted, len(case["order"]))
        assert prompt == expected


def test_reflector_prompts(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, _ = golden
    for case, expected in zip(FIXTURES["reflector_prompts"], data["reflector_prompts"], strict=True):
        prompt = prompts.reflector_prompt(
            case["observations"], case["manual"], compression_retry=case["retry"],
        )
        assert prompt == expected


def test_output_parsing(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, _ = golden
    for raw, expected in zip(FIXTURES["observer_outputs"], data["observer_outputs"], strict=True):
        parsed = text.parse_observer_output(raw)
        assert (parsed.observations, parsed.current_task, parsed.suggested_response) == (
            expected["observations"], expected["currentTask"], expected["suggested"],
        )
    for raw, expected in zip(FIXTURES["reflector_outputs"], data["reflector_outputs"], strict=True):
        parsed = text.parse_reflector_output(raw)
        assert (parsed.observations, parsed.suggested_response) == (
            expected["observations"], expected["suggested"],
        )
    for raw, expected in zip(
        FIXTURES["multi_thread_outputs"], data["multi_thread_outputs"], strict=True,
    ):
        parsed = text.parse_multi_thread_output(raw)
        assert [
            {
                "id": thread_id,
                "observations": result.observations,
                "currentTask": result.current_task,
                "suggested": result.suggested_response,
            }
            for thread_id, result in parsed.items()
        ] == expected


def test_context_rendering(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, tz = golden
    for raw, expected in zip(FIXTURES["optimize"], data["optimize"], strict=True):
        assert text.optimize_observations_for_context(raw) == expected
    for case, expected in zip(FIXTURES["relative_time"], data["relative_time"], strict=True):
        assert text.add_relative_time(case["observations"], _instant(case["now"]), tz) == expected
    for case, expected in zip(FIXTURES["context"], data["context"], strict=True):
        now = _instant(case["now"]) if case["now"] else None
        block = prompts.observations_context(
            text.render_observations(case["observations"], now, tz),
            current_task=case["current_task"],
            suggested_response=case["suggested"],
            other_conversations=case["unobserved"],
        )
        assert block == expected


def test_thread_sections(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, _ = golden
    assert [text.obscure_thread_id(tid) for tid in FIXTURES["thread_ids"]] == data["thread_ids"]
    for case, expected in zip(FIXTURES["thread_sections"], data["thread_sections"], strict=True):
        obscured = text.obscure_thread_id(case["thread"])
        section = text.wrap_with_thread_tag(obscured, case["observations"])
        existing = case["existing"].replace("THREAD", obscured)
        assert section == expected["section"]
        assert text.replace_or_append_thread_section(existing, section) == expected["merged"]


def test_other_conversation_blocks(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, tz = golden
    for case, expected in zip(
        FIXTURES["unobserved_blocks"], data["unobserved_blocks"], strict=True,
    ):
        block = text.format_other_conversations(
            _threads(case["threads"]), case["current"], tz, text.obscure_thread_id,
        )
        assert block == expected


def test_token_counts(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, _ = golden
    assert [tokens.count_string(s) for s in FIXTURES["token_strings"]] == data["token_strings"]
    for name, expected in data["token_messages"].items():
        conversation = _conversation(name)
        assert [tokens.count_message(m) for m in conversation] == pytest.approx(expected["each"])
        assert tokens.count_messages(conversation) == pytest.approx(expected["total"])


def test_thread_selection_and_batching(golden: tuple[dict[str, Any], ZoneInfo]) -> None:
    data, _ = golden
    for case, expected in zip(FIXTURES["selection"], data["selection"], strict=True):
        plan = select_and_batch(
            _threads(case["threads"]),
            message_tokens=case["messageTokens"],
            max_tokens_per_batch=case["maxTokensPerBatch"],
        )
        assert plan is not None
        assert {"threadOrder": plan.thread_order, "batches": plan.batches} == expected
