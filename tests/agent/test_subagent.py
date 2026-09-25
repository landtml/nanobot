"""Tests for SubagentManager."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.agent.tools.filesystem import FileToolsConfig
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.llm_usage.context import llm_usage_source
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime(provider: LLMProvider) -> LLMRuntime:
    provider.generation = GenerationSettings()
    return LLMRuntime.capture(provider, "test", context_window_tokens=128_000)


@pytest.mark.asyncio
async def test_subagent_uses_tool_loader():
    """Verify subagent registers tools via ToolLoader, not hard-coded imports."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=Path("/tmp"),
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    tools = sm._build_tools()
    assert tools.has("read_file")
    assert tools.has("write_file")
    assert not tools.has("message")
    assert not tools.has("spawn")


@pytest.mark.asyncio
async def test_subagent_build_tools_isolates_file_read_state(tmp_path):
    """Each spawned subagent needs a fresh file-state cache."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    first_read = sm._build_tools().get("read_file")
    second_read = sm._build_tools().get("read_file")

    assert first_read is not second_read
    assert (await first_read.execute(path="note.txt")).startswith("1| hello")
    second_result = await second_read.execute(path="note.txt")
    assert second_result.startswith("1| hello")
    assert "File unchanged" not in second_result


def test_subagent_respects_file_tool_toggle(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
        tools_config=ToolsConfig(file=FileToolsConfig(enable=False)),
    )

    tools = sm._build_tools()

    file_tools = {
        "apply_patch",
        "edit_file",
        "find_files",
        "grep",
        "list_dir",
        "read_file",
        "write_file",
    }
    assert file_tools.isdisjoint(tools.tool_names)


def test_subagent_prompt_keeps_agent_paths_for_selected_project(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    global_skill = agent_workspace / "skills" / "global-custom" / "SKILL.md"
    project_skill = project / "skills" / "project-custom" / "SKILL.md"
    global_skill.parent.mkdir(parents=True)
    project_skill.parent.mkdir(parents=True)
    global_skill.write_text("---\ndescription: global skill\n---\nGlobal", encoding="utf-8")
    project_skill.write_text("---\ndescription: project skill\n---\nProject", encoding="utf-8")
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    prompt = manager._build_subagent_prompt(workspace=project)

    assert "one root and relative SKILL.md paths" in prompt
    assert "Join them when using `read_file`" in prompt
    assert str(project.resolve()) not in prompt
    assert f"Nanobot's agent workspace: {agent_workspace.resolve()}" in prompt
    assert f"Memory (read-only): {agent_workspace.resolve() / 'memory' / 'observations.md'}" in prompt
    assert "global-custom" in prompt
    assert "project-custom" not in prompt


def test_subagent_prompt_uses_relative_paths_in_agent_workspace(tmp_path):
    skill = tmp_path / "skills" / "custom" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\ndescription: custom skill\n---\nCustom", encoding="utf-8")
    manager = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    prompt = manager._build_subagent_prompt()

    assert str(tmp_path.resolve()) not in prompt
    assert "Memory (read-only): memory/observations.md" in prompt
    assert "### Workspace skills (`skills`)" in prompt


@pytest.mark.asyncio
async def test_subagent_keeps_project_runtime_scope_with_agent_owned_tools(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    agent_workspace.mkdir()
    project.mkdir()
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    manager.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    manager._announce_result = AsyncMock()
    status = SubagentStatus(
        task_id="t1",
        label="label",
        task_description="task",
        started_at=0.0,
    )

    await manager._run_subagent(
        "t1",
        "task",
        "label",
        {"channel": "websocket", "chat_id": "direct"},
        status,
        _runtime(provider),
        workspace_scope=build_workspace_scope(project, "restricted"),
    )

    spec = manager.runner.run.call_args.args[0]
    assert spec.workspace == project
    assert spec.tools.get("read_file")._workspace == agent_workspace.resolve()


@pytest.mark.asyncio
async def test_subagent_recovers_from_tool_error_in_same_run(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="reading",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="read_file",
                    arguments={"path": "missing.txt"},
                )
            ],
        ),
        LLMResponse(content="recovered without restarting", tool_calls=[]),
    ])
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
        memory=MagicMock(),
    )

    result = await sm.run_inline(
        task="recover after a missing file",
        session_key="test:direct",
        runtime=_runtime(provider),
    )

    assert result == "recovered without restarting"
    assert provider.chat_stream_with_retry.await_count == 2


@pytest.mark.asyncio
async def test_subagent_pressure_observes_into_a_transient_overlay(tmp_path):
    from nanobot.agent.memory import Memory
    from nanobot.session.manager import SessionManager

    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.can_resume_conversation_state.return_value = False
    provider.generation = GenerationSettings(max_tokens=100)

    def is_observer(messages):
        return str(messages[0].get("content")).startswith("You are the memory consciousness")

    def estimate(messages, _tools, _model):
        if not is_observer(messages) and sum(m.get("role") == "tool" for m in messages) >= 2:
            return 600, "test-counter"
        return 100, "test-counter"

    turns = 0

    async def respond(*, messages, **_kwargs):
        nonlocal turns
        if is_observer(messages):
            return LLMResponse(content=(
                "<observations>\nDate: Sep 25, 2026\n"
                "* 🔴 (10:00) Subagent listed the workspace twice\n</observations>"
            ))
        turns += 1
        if turns <= 2:
            return LLMResponse(
                content="checking",
                tool_calls=[ToolCallRequest(id=f"call-{turns}", name="list_dir",
                                            arguments={"path": "."})],
            )
        return LLMResponse(content="done", tool_calls=[])

    provider.estimate_prompt_tokens = MagicMock(side_effect=estimate)
    provider.chat_stream_with_retry = AsyncMock(side_effect=respond)
    runtime = LLMRuntime.capture(provider, "test", context_window_tokens=1_624)
    memory = Memory(tmp_path, SessionManager(tmp_path), runtime=lambda: runtime, timezone="UTC")
    manager = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
        memory=memory,
    )

    result = await manager.run_inline(
        task="inspect the workspace",
        session_key="test:direct",
        runtime=runtime,
    )

    assert result == "done"
    requests = [call.kwargs["messages"] for call in provider.chat_stream_with_retry.await_args_list]
    assert [is_observer(r) for r in requests] == [False, False, True, False]
    assert "inspect the workspace" in requests[2][1]["content"]
    assert "Subagent listed the workspace twice" in requests[3][0]["content"]
    assert sum(message.get("role") == "tool" for message in requests[3]) == 1
    # Subagent observations stay transient: nothing reaches the workspace log.
    assert memory.observations() == ""
    assert memory.om._ephemeral == {}


@pytest.mark.asyncio
async def test_spawned_subagent_inherits_llm_usage_source(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()

    with llm_usage_source("cron"):
        await sm.spawn(
            "automation task",
            session_key="websocket:bound-automation",
            runtime=_runtime(provider),
        )
    tasks = list(sm._running_tasks.values())
    await asyncio.gather(*tasks)

    spec = sm.runner.run.call_args.args[0]
    assert spec.llm_usage_source == "cron"


@pytest.mark.asyncio
async def test_subagent_runs_concurrency_safe_tools_in_parallel(tmp_path, monkeypatch):
    import asyncio
    from typing import Any

    from orchestration.sim import ScriptedProvider

    from nanobot.agent.tools.base import Tool
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.config.schema import ToolsConfig, WebToolsConfig
    from nanobot.providers.base import LLMResponse, ToolCallRequest

    started: set[str] = set()
    completed: set[str] = set()
    both_started = asyncio.Event()

    class ProbeTool(Tool):
        def __init__(self, tool_name: str) -> None:
            self._name = tool_name

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return "A concurrency probe."

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        @property
        def read_only(self) -> bool:
            return True

        async def execute(self, **_kwargs: Any) -> str:
            started.add(self.name)
            if len(started) == 2:
                both_started.set()
            try:
                await asyncio.wait_for(both_started.wait(), timeout=0.5)
            except TimeoutError:
                return "tools did not overlap"
            completed.add(self.name)
            return "complete"

    provider = ScriptedProvider(
        {
            "child": [
                LLMResponse(
                    content="run both tools",
                    tool_calls=[
                        ToolCallRequest(id="a", name="safe_a", arguments={}),
                        ToolCallRequest(id="b", name="safe_b", arguments={}),
                    ],
                ),
                LLMResponse(content="child complete"),
            ],
        },
        route=lambda _messages: "child",
    )
    manager = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=1000,
        tools_config=ToolsConfig(web=WebToolsConfig()),
        memory=MagicMock(),
    )
    tools = ToolRegistry()
    tools.register(ProbeTool("safe_a"))
    tools.register(ProbeTool("safe_b"))
    monkeypatch.setattr(manager, "_build_tools", lambda tools_config=None: tools)
    monkeypatch.setattr(manager, "_build_subagent_prompt", lambda workspace=None: "test prompt")

    result = await asyncio.wait_for(
        manager.run_inline(task="run the tools", runtime=_runtime(provider)),
        timeout=2.0,
    )

    assert result == "child complete"
    assert started == {"safe_a", "safe_b"}
    assert completed == {"safe_a", "safe_b"}


@pytest.mark.asyncio
async def test_subagent_mailbox_injects_queued_messages_before_next_model_call(tmp_path, monkeypatch):
    import asyncio
    from typing import Any

    from orchestration.sim import ScriptedProvider

    from nanobot.agent.subagent import SubagentStatus
    from nanobot.agent.tools.base import Tool
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.config.schema import ToolsConfig, WebToolsConfig
    from nanobot.providers.base import LLMResponse, ToolCallRequest

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class WaitingTool(Tool):
        @property
        def name(self) -> str:
            return "wait_for_input"

        @property
        def description(self) -> str:
            return "Waits while input arrives."

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **_kwargs: Any) -> str:
            tool_started.set()
            await release_tool.wait()
            return "tool finished"

    provider = ScriptedProvider(
        {
            "child": [
                LLMResponse(
                    content="start waiting",
                    tool_calls=[
                        ToolCallRequest(id="wait", name="wait_for_input", arguments={}),
                    ],
                ),
                LLMResponse(content="child resumed"),
            ],
        },
        route=lambda _messages: "child",
    )
    manager = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=1000,
        tools_config=ToolsConfig(web=WebToolsConfig()),
        memory=MagicMock(),
    )
    tools = ToolRegistry()
    tools.register(WaitingTool())
    monkeypatch.setattr(manager, "_build_tools", lambda tools_config=None: tools)
    monkeypatch.setattr(manager, "_build_subagent_prompt", lambda workspace=None: "test prompt")
    status = SubagentStatus(
        task_id="child-1",
        label="child",
        task_description="test mailbox",
        started_at=0.0,
    )
    child_task = asyncio.create_task(manager._run_admitted_subagent(
        "child-1",
        "test mailbox",
        "child",
        {"channel": "cli", "chat_id": "direct", "session_key": None},
        status,
        _runtime(provider),
        announce=False,
    ))

    try:
        await asyncio.wait_for(tool_started.wait(), timeout=1.0)
        await status.mailbox.put({"role": "user", "content": "first steer"})
        await status.mailbox.put({"role": "user", "content": "second steer"})
    finally:
        release_tool.set()
        await asyncio.wait_for(child_task, timeout=2.0)

    messages = provider.calls[1].messages
    assert {"role": "user", "content": "first steer\n\nsecond steer"} in messages
