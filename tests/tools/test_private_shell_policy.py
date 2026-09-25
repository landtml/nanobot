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


def test_registry_tool_names_respect_private_availability(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "private_session_sandbox_backend", lambda: None)
    registry = _make_registry(tmp_path)

    assert _SHELL_TOOLS.issubset(registry.tool_names)

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        session_persist=False,
    )):
        names = registry.tool_names

    assert _SHELL_TOOLS.isdisjoint(names)


@pytest.mark.asyncio
async def test_private_exec_rejects_custom_shell_before_process_start(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "private_session_sandbox_backend", lambda: "bwrap")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observations = workspace / "memory" / "observations.md"
    observations.parent.mkdir()
    observations.write_text("private memory marker", encoding="utf-8")
    custom_shell = workspace / "bash"
    host_marker = tmp_path / "custom-shell-ran"
    custom_shell.write_text(
        "#!/bin/sh\n"
        f"cat {shlex.quote(str(observations))} > {shlex.quote(str(host_marker))}\n",
        encoding="utf-8",
    )
    custom_shell.chmod(0o755)
    tool = ExecTool(working_dir=str(workspace), timeout=5)

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        workspace=workspace,
        session_persist=False,
    )):
        result = await tool.execute(command="true", shell=str(custom_shell))

    assert is_tool_error_result(result)
    assert "custom shell" in result
    assert not host_marker.exists()


@pytest.mark.asyncio
async def test_private_exec_rejects_login_shell_before_startup_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "private_session_sandbox_backend", lambda: "bwrap")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    host_marker = tmp_path / "login-profile-ran"
    (fake_home / ".bash_profile").write_text(
        f"touch {shlex.quote(str(host_marker))}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(fake_home))
    tool = ExecTool(working_dir=str(workspace), timeout=5)

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        workspace=workspace,
        session_persist=False,
    )):
        result = await tool.execute(command="true", login=True)

    assert not host_marker.exists()
    assert is_tool_error_result(result)
    assert "login shells" in result


@pytest.mark.asyncio
async def test_private_exec_does_not_resolve_sandbox_from_path_prepend(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "private_session_sandbox_backend", lambda: "bwrap")
    workspace = tmp_path / "workspace"
    fake_bin = workspace / "bin"
    fake_bin.mkdir(parents=True)
    host_marker = tmp_path / "fake-bwrap-ran"
    fake_bwrap = fake_bin / "bwrap"
    fake_bwrap.write_text(
        "#!/bin/sh\n"
        f"touch {shlex.quote(str(host_marker))}\n",
        encoding="utf-8",
    )
    fake_bwrap.chmod(0o755)
    tool = ExecTool(
        working_dir=str(workspace),
        timeout=5,
        path_prepend=str(fake_bin),
    )

    with request_context(RequestContext(
        channel="websocket",
        chat_id="private",
        session_key="websocket:private",
        workspace=workspace,
        session_persist=False,
    )):
        result = await tool.execute(command="printf private-shell-ok")

    assert "private-shell-ok" in result
    assert "Exit code: 0" in result
    assert not host_marker.exists()


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
        immediate_control = await tool.execute(command="printf private-shell-ok")
        managed_control = await tool.execute(
            command="printf private-shell-ok",
            yield_time_ms=1000,
        )
        immediate = await tool.execute(
            command=f"grep -F 'private memory marker' {shlex.quote(str(observations))}"
        )
        managed = await tool.execute(
            command=f"grep -F 'private memory marker' {shlex.quote(str(observations))}",
            yield_time_ms=1000,
        )

    await manager.close_all()

    assert "private-shell-ok" in immediate_control
    assert "Exit code: 0" in immediate_control
    assert "private-shell-ok" in managed_control
    assert "Exit code: 0" in managed_control
    assert "private memory marker" not in immediate
    assert "private memory marker" not in managed
    assert "Permission denied" in immediate
    assert "Permission denied" in managed
    assert "Exit code: 2" in immediate
    assert "Exit code: 2" in managed
