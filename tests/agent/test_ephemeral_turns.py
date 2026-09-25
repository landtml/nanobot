"""Ephemeral direct turns, and the memory model preset.

Ephemeral turns run without persisting the session or writing memory; they were
Dream's execution path and remain part of the SDK and cron surface.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.hook import AgentHook
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import MemoryConfig, ModelPresetConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.supports_tools = True
    provider.generation = MagicMock(max_tokens=4096)
    provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="done", tool_calls=[], finish_reason="stop", usage=None)
    )
    return provider


def _loop(tmp_path, **kwargs) -> AgentLoop:
    with (
        patch("nanobot.agent.loop.SessionManager"),
        patch("nanobot.agent.loop.Memory", autospec=True),
        patch("nanobot.agent.loop.SubagentManager") as mock_sub,
    ):
        mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        return AgentLoop(
            bus=MessageBus(),
            provider=_provider(),
            workspace=tmp_path,
            context_window_tokens=32_000,
            **kwargs,
        )


def test_memory_runtime_uses_preset_without_changing_default(tmp_path) -> None:
    loop = _loop(tmp_path, memory_config=MemoryConfig(model_override="cheap"))
    loop.runtime_resolver._model_presets = {"cheap": ModelPresetConfig(model="cheap-model")}

    runtime = loop.memory_runtime()

    assert runtime.model == "cheap-model"
    assert runtime.model_preset == "cheap"
    assert loop.model == "test-model"
    assert loop.model_preset is None


def test_memory_runtime_defaults_to_the_agent_model(tmp_path) -> None:
    loop = _loop(tmp_path)

    assert loop.memory_runtime().model == "test-model"


async def test_ephemeral_turn_does_not_schedule_observation(tmp_path) -> None:
    loop = _loop(tmp_path)

    await loop.process_direct("test", session_key="cli:temp", ephemeral=True)

    loop.memory.after_turn.assert_not_called()


async def test_persisted_turn_schedules_observation_after_saving(tmp_path) -> None:
    loop = _loop(tmp_path)

    response = await loop.process_direct("test", session_key="cli:normal")
    await asyncio.gather(*loop._background_tasks)

    assert response is not None
    assert response.content == "done"
    loop.memory.after_turn.assert_awaited_once_with("cli:normal")


async def test_ephemeral_flag_reaches_the_turn_context(tmp_path) -> None:
    loop = _loop(tmp_path)
    captured: dict[str, bool] = {}
    original_save = loop._persist_turn

    async def patched_save(ctx):
        captured["ephemeral"] = ctx.ephemeral
        return await original_save(ctx)

    with patch.object(loop, "_persist_turn", side_effect=patched_save):
        await loop.process_direct("test", session_key="cli:check", ephemeral=True)
        assert captured == {"ephemeral": True}
        await loop.process_direct("test", session_key="cli:normal")
        assert captured == {"ephemeral": False}


async def test_ephemeral_response_reports_stop_reason(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(
        content="provider error",
        finish_reason="error",
    )

    response = await loop.process_direct("test", session_key="cli:error", ephemeral=True)

    assert response is not None
    assert response.metadata["_stop_reason"] == "error"


async def test_completed_response_after_tool_error_is_success(tmp_path) -> None:
    """A soft tool error is model input, not a second run-level failure state."""
    loop = _loop(tmp_path)
    (tmp_path / "SOUL.md").write_text("# Soul", encoding="utf-8")
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="trying an edit",
            finish_reason="tool_calls",
            tool_calls=[ToolCallRequest(
                id="call_edit",
                name="edit_file",
                arguments={"path": "SOUL.md", "old_text": "absent", "new_text": "x"},
            )],
            usage=None,
        ),
        LLMResponse(content="done", finish_reason="stop", tool_calls=[], usage=None),
    ])

    response = await loop.process_direct("test", session_key="cli:tool-error", ephemeral=True)

    assert response is not None
    assert response.metadata["_stop_reason"] == "completed"
    second_request = loop.provider.chat_stream_with_retry.await_args_list[1].kwargs["messages"]
    tool_result = next(message for message in second_request if message["role"] == "tool")
    assert "Error" in tool_result["content"]


@pytest.fixture
def loop_with_spy(tmp_path):
    spy = MagicMock(spec=AgentHook)
    spy.wants_streaming.return_value = False
    spy.before_iteration = AsyncMock()
    spy.after_iteration = AsyncMock()
    return _loop(tmp_path, hooks=[spy]), spy


async def test_extra_hooks_skipped_when_ephemeral(loop_with_spy) -> None:
    loop, spy = loop_with_spy

    await loop.process_direct("test", session_key="cli:hook-test", ephemeral=True)

    spy.before_iteration.assert_not_called()
    spy.after_iteration.assert_not_called()


async def test_extra_hooks_fire_for_normal_sessions(loop_with_spy) -> None:
    loop, spy = loop_with_spy

    await loop.process_direct("test", session_key="cli:normal")

    spy.before_iteration.assert_called()
