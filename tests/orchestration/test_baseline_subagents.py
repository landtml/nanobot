"""Baseline measurements for the pre-kernel subagent path.

Existing focused tests continue to characterize injection, deduplication,
usage attribution, message limits, reply timeouts, and background spawning.
This module adds a deterministic three-child turn and /stop latency baseline.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from agent.session_helpers import run_session
from loguru import logger

from nanobot.agent.context import TranscriptInput
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult, AgentRunSpec
from nanobot.agent.tools.context import current_request_context
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.router import CommandContext
from nanobot.config.schema import Config, ToolsConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.utils.llm_runtime import LLMRuntime
from orchestration.sim import (
    Fault,
    FaultInjector,
    ScriptedProvider,
    SimulatedCrashError,
    SimulatedRateLimitError,
    SimulatedTimeoutError,
    VirtualClock,
)


def _resolved_tools_config() -> ToolsConfig:
    # Resolve tool-local forward refs before constructing AgentLoop in isolation.
    return Config().tools


def _route_request(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.startswith("child-"):
                return content
    return "parent"


@pytest.mark.asyncio
async def test_three_waiting_children_finish_in_one_parent_turn(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    child_calls = [
        ToolCallRequest(
            id=f"spawn-{index}",
            name="spawn",
            arguments={"task": f"child-{index}", "wait": True},
        )
        for index in range(1, 4)
    ]
    provider = ScriptedProvider(
        {
            "parent": [
                LLMResponse(content=None, tool_calls=child_calls, finish_reason="tool_calls"),
                LLMResponse(content="all children finished"),
            ],
            **{
                f"child-{index}": [LLMResponse(content=f"child-{index} finished")]
                for index in range(1, 4)
            },
        },
        route=_route_request,
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        max_concurrent_subagents=3,
        tools_config=_resolved_tools_config(),
    )

    started = perf_counter()
    try:
        await run_session(
            loop,
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="direct",
                content="delegate three children",
            ),
        )
    finally:
        await loop.aclose()
    elapsed = perf_counter() - started
    parent_calls = sum(call.route == "parent" for call in provider.calls)
    observed_children = {call.route for call in provider.calls} - {"parent"}
    record_property("p00_parent_model_calls_for_3_children", parent_calls)
    record_property("p00_3_child_turn_seconds", elapsed)

    assert parent_calls == 2
    assert observed_children == {"child-1", "child-2", "child-3"}
    assert elapsed < 1.0
    print(
        f"P00 baseline: 3 children, {parent_calls} parent model calls, "
        f"{elapsed * 1000:.1f} ms wall time"
    )


@pytest.mark.asyncio
async def test_stop_cancels_a_child_and_records_latency(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    provider = ScriptedProvider({}, route=_route_request)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=_resolved_tools_config(),
    )
    runtime: LLMRuntime = loop.llm_runtime()
    child_started = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def blocked_run(_spec: AgentRunSpec) -> AgentRunResult:
        child_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        raise AssertionError("cancelled child unexpectedly continued")

    loop.subagents.runner.run = blocked_run
    try:
        if os.name == "nt":
            pytest.skip("managed exec process fixture uses the POSIX sleep command")
        await loop.subagents.spawn(
            "blocked child",
            runtime=runtime,
            session_key="cli:direct",
        )
        generation_id = loop.run_registry.session_root_id("cli:direct")
        assert generation_id is not None
        await asyncio.wait_for(child_started.wait(), timeout=1.0)
        await loop.subagents._exec_session_manager.start(
            command="sleep 30",
            cwd=str(tmp_path),
            env={},
            timeout=None,
            shell_program=None,
            login=False,
            yield_time_ms=0,
            max_output_chars=100,
            owner_session_key="cli:direct",
        )
        started = perf_counter()
        message = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="/stop",
        )
        command_context = CommandContext(
            msg=message,
            session=loop.sessions.get_or_create("cli:direct"),
            key="cli:direct",
            raw="/stop",
            loop=loop,
        )
        stopped = await loop.commands.dispatch_priority(command_context)
        elapsed = perf_counter() - started
    finally:
        await loop.aclose()

    record_property("p00_stop_cancel_latency_seconds", elapsed)
    assert stopped is not None and "stopped 1 task" in stopped.content.lower()
    assert child_cancelled.is_set()
    assert loop.run_registry.live_tasks() == ()
    assert loop.subagents._exec_session_manager._sessions == {}
    assert loop.run_registry.session_root_id("cli:direct") is None
    next_generation = await loop.run_registry.ensure_session_root("cli:direct")
    assert next_generation.id != generation_id
    assert elapsed < 1.0
    print(f"P00 baseline: /stop cancelled a child in {elapsed * 1000:.1f} ms")


@pytest.mark.asyncio
async def test_stop_cancels_an_active_process_direct_turn(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider({}, route=_route_request),
        workspace=tmp_path,
        model="test-model",
        tools_config=_resolved_tools_config(),
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()
    continued_after_cancel = False

    async def blocked_process(*_args: object, **_kwargs: object) -> None:
        nonlocal continued_after_cancel
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        continued_after_cancel = True

    loop._process_message = blocked_process
    session_key = "cli:direct-stop"
    direct_task = asyncio.create_task(
        loop.process_direct("blocked", session_key=session_key)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert direct_task in loop._active_tasks[session_key]
        message = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct-stop",
            content="/stop",
        )
        context = CommandContext(
            msg=message,
            session=loop.sessions.get_or_create(session_key),
            key=session_key,
            raw="/stop",
            loop=loop,
        )

        await loop.commands.dispatch_priority(context)
        await asyncio.gather(direct_task, return_exceptions=True)
    finally:
        if not direct_task.done():
            direct_task.cancel()
            await asyncio.gather(direct_task, return_exceptions=True)
        await loop.aclose()

    assert cancelled.is_set()
    assert not continued_after_cancel
    assert loop.run_registry.live_tasks() == ()


@pytest.mark.asyncio
async def test_failed_session_dispatch_emits_failed_turn_state(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider({}, route=_route_request),
        workspace=tmp_path,
        model="test-model",
        tools_config=_resolved_tools_config(),
    )

    async def fail_process(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated turn failure")

    loop._process_message = fail_process
    session_key = "cli:failed-turn"
    try:
        await run_session(
            loop,
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="failed-turn",
                content="fail this turn",
            ),
        )
        root_id = loop.run_registry.session_root_id(session_key)
        assert root_id is not None
        turns = [
            record
            for record in loop.run_registry._runs.values()
            if record.parent_id == root_id
        ]
    finally:
        await loop.aclose()

    assert len(turns) == 1
    assert turns[0].state == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True], ids=["queued", "direct"])
async def test_provider_error_response_marks_turn_failed(tmp_path: Path, direct: bool) -> None:
    provider = ScriptedProvider(
        {"parent": [LLMResponse(content="provider error", finish_reason="error")]},
        route=_route_request,
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=_resolved_tools_config(),
    )
    try:
        if direct:
            await loop.process_direct("cause provider error", session_key="cli:failure")
        else:
            await run_session(
                loop,
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="failure",
                    content="cause provider error",
                ),
            )
        root_id = loop.run_registry.session_root_id("cli:failure")
        assert root_id is not None
        turns = [
            record
            for record in loop.run_registry._runs.values()
            if record.parent_id == root_id
        ]
        assert len(turns) == 1
        assert turns[0].state == "failed"
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_new_from_process_direct_does_not_cancel_its_caller(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider({}, route=_route_request),
        workspace=tmp_path,
        model="test-model",
        tools_config=_resolved_tools_config(),
    )
    try:
        response = await asyncio.wait_for(
            loop.process_direct("/new", session_key="cli:direct"),
            timeout=2,
        )
        assert response is not None
        assert "new session started" in response.content.lower()
        await asyncio.sleep(0)
        assert "cli:direct" not in loop._active_tasks
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_stop_snapshots_old_direct_tasks_before_waiting_for_cancellation(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider({}, route=_route_request),
        workspace=tmp_path,
        model="test-model",
        tools_config=_resolved_tools_config(),
    )
    old_started = asyncio.Event()
    old_cancelled = asyncio.Event()
    release_old = asyncio.Event()

    async def process(msg: InboundMessage, **_kwargs: Any) -> OutboundMessage:
        if msg.content == "old":
            old_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                old_cancelled.set()
                await release_old.wait()
                raise
        return OutboundMessage(channel="cli", chat_id="direct", content=msg.content)

    loop._process_message = process
    old_task = asyncio.create_task(loop.process_direct("old", session_key="cli:direct"))
    await asyncio.wait_for(old_started.wait(), timeout=1)
    stop_task = asyncio.create_task(loop._cancel_active_tasks("cli:direct"))
    await asyncio.wait_for(old_cancelled.wait(), timeout=1)
    new_task = asyncio.create_task(loop.process_direct("new", session_key="cli:direct"))
    try:
        await asyncio.sleep(0)
        release_old.set()
        await asyncio.wait_for(stop_task, timeout=1)
        response = await asyncio.wait_for(new_task, timeout=1)
        assert response is not None and response.content == "new"
    finally:
        release_old.set()
        for task in (old_task, stop_task, new_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(old_task, stop_task, new_task, return_exceptions=True)
        await loop.aclose()


@pytest.mark.asyncio
async def test_private_parent_child_cannot_read_observations_or_log_task_label(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "memory" / "observations.md"
    observations.parent.mkdir(parents=True)
    observations.write_text("synthetic private observation")
    child_policy: list[tuple[bool, bool]] = []

    def route(messages: list[dict[str, Any]]) -> str:
        selected = _route_request(messages)
        if selected.startswith("child-"):
            request = current_request_context()
            assert request is not None
            child_policy.append((request.session_persist, request.log_content))
        return selected

    provider = ScriptedProvider(
        {
            "parent": [
                LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id="spawn-private",
                    name="spawn",
                    arguments={
                        "task": "child-read-observations",
                        "label": "synthetic-private-label",
                        "wait": True,
                    },
                )], finish_reason="tool_calls"),
                LLMResponse(content="Private child finished."),
            ],
            "child-read-observations": [
                LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id="read-private-observations",
                    name="read_file",
                    arguments={"path": "memory/observations.md"},
                )], finish_reason="tool_calls"),
                LLMResponse(content="I could not read private observations."),
            ],
        },
        route=route,
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=_resolved_tools_config(),
    )
    session = loop.sessions.get_or_create_transient("cli:private-child")
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), format="{message}")
    try:
        result = await loop._run_agent_loop(
            TranscriptInput(history=[], current_message="delegate a private child"),
            runtime=loop.llm_runtime(),
            session=session,
        )
    finally:
        logger.remove(sink)
        await loop.aclose()

    child_calls = [call for call in provider.calls if call.route == "child-read-observations"]
    child_tool_result = next(
        message["content"]
        for message in child_calls[1].messages
        if message.get("role") == "tool"
    )
    assert result.final_content == "Private child finished."
    assert child_policy and all(policy == (False, False) for policy in child_policy)
    assert "Private sessions cannot access memory/observations.md" in child_tool_result
    assert "synthetic private observation" not in str(child_calls[1].messages)
    assert "synthetic-private-label" not in "\n".join(logs)


@pytest.mark.asyncio
async def test_fault_injector_covers_provider_failures_and_slow_tools() -> None:
    clock = VirtualClock()
    faults = FaultInjector(clock)
    faults.add("provider:crash", Fault("crash"))
    faults.add("provider:rate-limit", Fault("rate_limit"))
    faults.add("provider:timeout", Fault("timeout"))
    faults.add("tool:slow", Fault("slow_tool", delay=5.0))

    with pytest.raises(SimulatedCrashError):
        await faults.trip("provider:crash")
    with pytest.raises(SimulatedRateLimitError):
        await faults.trip("provider:rate-limit")
    with pytest.raises(SimulatedTimeoutError):
        await faults.trip("provider:timeout")

    slow_tool = asyncio.create_task(faults.trip("tool:slow"))
    await asyncio.sleep(0)
    assert not slow_tool.done()
    clock.advance(5.0)
    await slow_tool
    assert clock.time() == 5.0

    provider = ScriptedProvider(
        {"parent": [LLMResponse(content="recovered scripted response")]},
        route=lambda _messages: "parent",
        faults=FaultInjector(
            clock,
            {"provider:parent": [Fault("rate_limit")]},
        ),
    )
    provider._CHAT_RETRY_DELAYS = (0,)
    response = await provider.chat_stream_with_retry(messages=[])
    assert response.content == "recovered scripted response"
    assert len(provider.calls) == 2
