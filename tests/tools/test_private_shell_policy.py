from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from nanobot.agent.tools import sandbox
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.exec_session import (
    ExecSessionManager,
    ExecSessionTool,
    ListExecSessionsTool,
)
from nanobot.agent.tools.registry import ToolRegistry, is_tool_error_result
from nanobot.agent.tools.shell import ExecTool

_SHELL_TOOLS = {"exec", "exec_session", "list_exec_sessions"}


def _make_registry(workspace: Path) -> ToolRegistry:
    manager = ExecSessionManager()
    registry = ToolRegistry()
    registry.register(ExecTool(working_dir=str(workspace), session_manager=manager))
    registry.register(ExecSessionTool(manager=manager))
    registry.register(ListExecSessionsTool(manager=manager))
    return registry


def _definition_names(registry: ToolRegistry) -> set[str]:
    return {
        schema["function"]["name"]
        for schema in registry.get_definitions()
    }


@pytest.mark.asyncio
async def test_private_registry_withholds_shell_tools_without_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sandbox,
        "private_session_sandbox_backend",
        lambda: None,
        raising=False,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = _make_registry(workspace)
    marker = workspace / "should-not-exist"

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        session_persist=False,
    )):
        names = _definition_names(registry)
        result = await registry.execute("exec", {"command": f"touch {marker}"})

    assert names.isdisjoint(_SHELL_TOOLS)
    assert is_tool_error_result(result)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_private_exec_fails_closed_if_called_directly_without_isolation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "private_session_sandbox_backend",
        lambda: None,
        raising=False,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "should-not-exist"

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        session_persist=False,
    )):
        result = await ExecTool(working_dir=str(workspace)).execute(
            command=f"touch {marker}"
        )

    assert is_tool_error_result(result)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_private_session_tools_fail_closed_when_isolation_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "private_session_sandbox_backend",
        lambda: None,
    )
    manager = ExecSessionManager()

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        session_persist=False,
    )):
        managed = await ExecSessionTool(manager=manager).execute(session_id="unknown")
        listed = await ListExecSessionsTool(manager=manager).execute()

    assert is_tool_error_result(managed)
    assert "isolation is unavailable" in managed
    assert is_tool_error_result(listed)
    assert "isolation is unavailable" in listed


@pytest.mark.asyncio
async def test_private_session_cannot_access_an_unisolated_prior_shell(tmp_path):
    manager = ExecSessionManager()
    owner_session_key = "websocket:private"
    command = (
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote('import time; time.sleep(30)')}"
    )
    session_id, _ = await manager.start(
        command=command,
        cwd=str(tmp_path),
        env=os.environ.copy(),
        timeout=None,
        shell_program=None,
        login=False,
        yield_time_ms=0,
        max_output_chars=1000,
        owner_session_key=owner_session_key,
    )

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key=owner_session_key,
        session_persist=False,
    )):
        managed = await ExecSessionTool(manager=manager).execute(
            session_id=session_id,
            timeout_ms=0,
        )
        listed = await ListExecSessionsTool(manager=manager).execute()

    await manager.close_all()

    assert is_tool_error_result(managed)
    assert session_id not in listed


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bwrap") is None,
    reason="Linux bubblewrap is required",
)
async def test_private_immediate_and_managed_exec_cannot_read_observations(tmp_path):
    workspace = tmp_path / "workspace"
    observations = workspace / "memory" / "observations.md"
    observations.parent.mkdir(parents=True)
    observations.write_text("private memory marker", encoding="utf-8")
    manager = ExecSessionManager()
    tool = ExecTool(
        working_dir=str(workspace),
        timeout=5,
        session_manager=manager,
    )

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        workspace=workspace,
        session_persist=False,
    )):
        immediate = await tool.execute(command=f"cat {observations}")
        managed = await tool.execute(command=f"cat {observations}", yield_time_ms=1000)

    await manager.close_all()

    assert "private memory marker" not in immediate
    assert "private memory marker" not in managed
