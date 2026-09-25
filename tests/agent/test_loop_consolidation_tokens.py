"""Context pressure inside a turn: the loop observes and continues from memory."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    ProviderConversationState,
    ToolCallRequest,
)
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT


def _observation(text: str) -> LLMResponse:
    return LLMResponse(
        content=f"<observations>\nDate: Sep 25, 2026\n* 🔴 (10:00) {text}\n</observations>",
        tool_calls=[],
    )


def _is_observer(messages: list[dict[str, Any]]) -> bool:
    return str(messages[0].get("content")).startswith("You are the memory consciousness")


def _make_loop(
    tmp_path,
    *,
    estimated_tokens: int,
    context_window_tokens: int,
    max_tokens: int = 0,
) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=max_tokens)
    provider.estimate_prompt_tokens.return_value = (estimated_tokens, "test-counter")
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=context_window_tokens,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]
    return loop


def _old_session(loop: AgentLoop, key: str) -> list[dict[str, Any]]:
    session = loop.sessions.get_or_create(key)
    session.messages = [
        {"role": role, "content": f"old-{role}-{turn}", "timestamp": f"2026-09-20T10:{turn:02d}:00"}
        for turn in range(6)
        for role in ("user", "assistant")
    ]
    loop.sessions.save(session)
    return [dict(message) for message in session.messages]


def _pressure_while_old_history_is_sent(messages, _tools, _model):
    if _is_observer(messages):
        return 100, "test-counter"
    if any(str(message.get("content")).startswith("old-") for message in messages):
        return 600, "test-counter"
    return 100, "test-counter"


@pytest.mark.asyncio
async def test_runner_pressure_observes_history_and_keeps_the_current_turn(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=1_624, max_tokens=100)
    loop.provider.can_resume_conversation_state.return_value = False
    _old_session(loop, "cli:test")
    loop.provider.estimate_prompt_tokens.side_effect = _pressure_while_old_history_is_sent
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=[
        _observation("User worked through six old turns"),
        LLMResponse(content="done", tool_calls=[]),
    ])

    result = await loop.process_direct("continue the task", session_key="cli:test")

    assert result.content == "done"
    observer_request, model_request = (
        call.kwargs["messages"] for call in loop.provider.chat_stream_with_retry.await_args_list
    )
    assert _is_observer(observer_request)
    assert "old-user-0" in observer_request[1]["content"]
    assert "continue the task" in observer_request[1]["content"]
    assert "User worked through six old turns" in model_request[0]["content"]
    assert [message["role"] for message in model_request] == ["system", "user"]
    assert model_request[1]["content"] == "continue the task"

    reloaded = loop.sessions.get_or_create("cli:test")
    assert reloaded.messages[0]["content"] == "old-user-0"
    assert reloaded.messages[reloaded.last_archived]["content"] == SUMMARY_CONTINUATION_TEXT
    assert [message["content"] for message in reloaded.get_history()] == [
        "continue the task",
        "done",
    ]
    assert "User worked through six old turns" in loop.memory.observations()


@pytest.mark.asyncio
async def test_ephemeral_runner_pressure_observes_without_writing_memory(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=1_624, max_tokens=100)
    loop.provider.can_resume_conversation_state.return_value = False
    original_messages = _old_session(loop, "cli:ephemeral")
    loop.provider.estimate_prompt_tokens.side_effect = _pressure_while_old_history_is_sent
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=[
        _observation("Transient observation"),
        LLMResponse(content="done", tool_calls=[]),
    ])

    result = await loop.process_direct(
        "continue the task", session_key="cli:ephemeral", ephemeral=True,
    )

    assert result.content == "done"
    model_request = loop.provider.chat_stream_with_retry.await_args_list[1].kwargs["messages"]
    assert "Transient observation" in model_request[0]["content"]
    assert model_request[1]["content"] == "continue the task"
    reloaded = loop.sessions.get_or_create("cli:ephemeral")
    assert reloaded.messages[: len(original_messages)] == original_messages
    assert reloaded.last_archived == 0
    assert loop.memory.observations() == ""
    # The one-off run's observations do not follow the session into its next turn.
    assert loop.memory.history_prefix("cli:ephemeral") == []
    assert loop.memory.system_prompt_block("cli:ephemeral") is None


@pytest.mark.asyncio
async def test_ephemeral_tool_loop_pressure_observes_before_next_model_call(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=1_624, max_tokens=100)
    loop.provider.can_resume_conversation_state.return_value = False
    loop.tools.execute = AsyncMock(return_value="large-result")

    def estimate(messages, _tools, _model):
        if not _is_observer(messages) and sum(m.get("role") == "tool" for m in messages) >= 2:
            return 600, "test-counter"
        return 100, "test-counter"

    normal_calls = 0

    async def respond(*, messages, **_kwargs):
        nonlocal normal_calls
        if _is_observer(messages):
            return _observation("Agent listed the workspace twice")
        normal_calls += 1
        if normal_calls <= 2:
            return LLMResponse(
                content="checking",
                tool_calls=[ToolCallRequest(
                    id=f"call-{normal_calls}", name="list_dir", arguments={"path": "."},
                )],
            )
        return LLMResponse(content="done", tool_calls=[])

    loop.provider.estimate_prompt_tokens.side_effect = estimate
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=respond)

    result = await loop.process_direct(
        "inspect the workspace", session_key="cli:ephemeral-tool-loop", ephemeral=True,
    )

    assert result.content == "done"
    assert loop.provider.chat_stream_with_retry.await_count == 4
    observer_request = loop.provider.chat_stream_with_retry.await_args_list[2].kwargs["messages"]
    assert "[Tool Call: list_dir]" in observer_request[1]["content"]
    assert "[Tool Result: list_dir]" in observer_request[1]["content"]
    model_request = loop.provider.chat_stream_with_retry.await_args_list[3].kwargs["messages"]
    assert "Agent listed the workspace twice" in model_request[0]["content"]
    assert sum(message.get("role") == "tool" for message in model_request) == 1
    assert loop.memory.observations() == ""
    assert loop.memory.system_prompt_block("cli:ephemeral-tool-loop") is None


@pytest.mark.asyncio
async def test_native_provider_compaction_commits_portable_terminal_checkpoint(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=2_000)
    session = loop.sessions.get_or_create("cli:native")
    session.messages = [
        {"role": "user", "content": "accepted history"},
        {"role": "assistant", "content": "accepted answer"},
    ]
    loop.sessions.save(session)
    compacted_state = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="test-model",
        version=1,
        payload={"items": [{"type": "compaction", "encrypted_content": "opaque"}]},
    )
    loop.provider.can_resume_conversation_state.return_value = True
    done = LLMResponse(
        content="done",
        provider_state=compacted_state,
        provider_compaction_applied=True,
        provider_compaction_state=compacted_state,
        provider_compaction_scope="current_request",
    )

    async def respond(*, messages, **_kwargs):
        return _observation("User continued the task") if _is_observer(messages) else done

    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=respond)

    result = await loop.process_direct("continue", session_key="cli:native")

    assert result.content == "done"
    observer_prompt = next(
        call.kwargs["messages"][1]["content"]
        for call in loop.provider.chat_stream_with_retry.await_args_list
        if _is_observer(call.kwargs["messages"])
    )
    assert "accepted history" in observer_prompt
    assert "accepted answer" in observer_prompt
    assert "continue" in observer_prompt
    reloaded = loop.sessions.get_or_create("cli:native")
    assert reloaded.provider_state is None
    assert reloaded.messages[reloaded.last_archived]["content"] == SUMMARY_CONTINUATION_TEXT
    assert [message["content"] for message in reloaded.get_history()] == ["done"]
    assert "User continued the task" in loop.memory.observations()


@pytest.mark.asyncio
async def test_private_session_never_sees_workspace_memory_but_keeps_its_own(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=1_624, max_tokens=100)
    loop.provider.can_resume_conversation_state.return_value = False
    loop.memory.replace_observations("Date: Sep 25, 2026\n* 🔴 (09:00) workspace secret")
    key = "websocket:private"
    session = loop.sessions.get_or_create_transient(key)
    session.messages = [
        {"role": role, "content": f"old-{role}-{turn}", "timestamp": f"2026-09-20T10:{turn:02d}:00"}
        for turn in range(6)
        for role in ("user", "assistant")
    ]
    loop.provider.estimate_prompt_tokens.side_effect = _pressure_while_old_history_is_sent
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=[
        _observation("Private chat covered six turns"),
        LLMResponse(content="done", tool_calls=[]),
        LLMResponse(content="again", tool_calls=[]),
    ])

    await loop.process_direct("continue", session_key=key)
    await loop.process_direct("and more", session_key=key)

    requests = [c.kwargs["messages"] for c in loop.provider.chat_stream_with_retry.await_args_list]
    assert all("workspace secret" not in str(request) for request in requests)
    assert "Private chat covered six turns" in requests[1][0]["content"]
    assert "Private chat covered six turns" in requests[2][0]["content"]
    assert "Private chat covered six turns" not in loop.memory.observations()
    assert loop.sessions.read_session_file(key) is None
