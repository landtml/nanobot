"""``/new``: the session clears at once and memory observes what it had not seen."""

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse

OBSERVED = (
    "<observations>\nDate: Sep 25, 2026\n* 🔴 (10:00) User discussed msg4\n</observations>\n"
    "<current-task>\nPrimary: answer msg4\n</current-task>"
)


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.estimate_prompt_tokens.return_value = (10_000, "test")
    provider.generation = GenerationSettings(max_tokens=100)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=128_000,
    )
    loop.provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content=OBSERVED, tool_calls=[])
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


def _conversation(loop: AgentLoop, turns: int):
    session = loop.sessions.get_or_create("cli:test")
    for i in range(turns):
        session.add_message("user", f"msg{i}")
        session.add_message("assistant", f"resp{i}")
    loop.sessions.save(session)
    return session


NEW = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")


@pytest.mark.asyncio
async def test_new_clears_session_immediately_even_if_observation_fails(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _conversation(loop, 5)
    loop.provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="boom", finish_reason="error")
    )

    response = await loop._process_message(NEW, runtime=loop.llm_runtime())

    assert response is not None
    assert "new session started" in response.content.lower()
    assert loop.sessions.get_or_create("cli:test").messages == []
    await loop.aclose()
    assert loop.memory.observations() == ""


@pytest.mark.asyncio
async def test_new_observes_only_unobserved_messages(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    session = _conversation(loop, 5)
    session.last_archived = len(session.messages) - 2
    loop.sessions.save(session)
    scheduled: list[Coroutine[Any, Any, object]] = []
    loop.schedule_background = scheduled.append  # type: ignore[method-assign]

    response = await loop._process_message(NEW, runtime=loop.llm_runtime())

    assert response is not None
    assert len(scheduled) == 1
    await scheduled[0]
    prompt = loop.provider.chat_stream_with_retry.call_args.kwargs["messages"][1]["content"]
    assert "msg4" in prompt and "resp4" in prompt
    assert "msg3" not in prompt
    assert "User discussed msg4" in loop.memory.observations()
    # The finished conversation leaves no stale task behind for the next one.
    assert loop.memory.om.thread_state("cli:test") is None


@pytest.mark.asyncio
async def test_new_on_fully_observed_session_schedules_nothing(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    session = _conversation(loop, 2)
    session.last_archived = len(session.messages)
    loop.sessions.save(session)
    scheduled: list[Coroutine[Any, Any, object]] = []
    loop.schedule_background = scheduled.append  # type: ignore[method-assign]

    await loop._process_message(NEW, runtime=loop.llm_runtime())

    assert scheduled == []
    assert loop.sessions.get_or_create("cli:test").messages == []


@pytest.mark.asyncio
async def test_aclose_drains_background_observation(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _conversation(loop, 3)
    release = asyncio.Event()
    observed = asyncio.Event()

    async def slow_observer(**_kwargs: Any) -> LLMResponse:
        await release.wait()
        observed.set()
        return LLMResponse(content=OBSERVED)

    loop.provider.chat_stream_with_retry = slow_observer
    await loop._process_message(NEW, runtime=loop.llm_runtime())

    assert not observed.is_set()
    release.set()
    await loop.aclose()
    assert observed.is_set()
    assert "User discussed msg4" in loop.memory.observations()
