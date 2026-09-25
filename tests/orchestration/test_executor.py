from __future__ import annotations

import asyncio
from pathlib import Path
from typing import assert_type

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.orchestration.executor import RunExecutor
from nanobot.orchestration.scheduler import ProviderLane, Scheduler
from nanobot.orchestration.types import Budget, Capabilities, JsonValue, RunMessage, RunSpec
from nanobot.utils.llm_runtime import LLMRuntime
from orchestration.sim import ScriptedProvider


def _runtime() -> LLMRuntime:
    provider = ScriptedProvider({}, route=lambda _messages: "test")
    return LLMRuntime.capture(provider, "test-model", context_window_tokens=4096)


def test_executor_preserves_runner_default_error_message() -> None:
    spec = RunSpec(
        runtime=_runtime(),
        tools=ToolRegistry(),
        budget=Budget(iterations=7),
        max_tool_result_chars=1234,
    )

    runner_spec = RunExecutor.build_agent_run_spec(spec)

    assert runner_spec.runtime is spec.runtime
    assert runner_spec.tools is spec.tools
    assert runner_spec.max_iterations == 7
    assert runner_spec.max_tool_result_chars == 1234
    assert runner_spec.error_message is not None


def test_executor_maps_subagent_run_without_changing_runner_options() -> None:
    messages = [
        {"role": "system", "content": "subagent"},
        {"role": "user", "content": "inspect this"},
    ]
    workspace = Path("/workspace")
    capabilities = Capabilities(tools=frozenset({"read_file", "grep"}))
    spec = RunSpec(
        runtime=_runtime(),
        tools=ToolRegistry(),
        budget=Budget(iterations=11),
        max_tool_result_chars=4321,
        initial_messages=messages,
        error_message=None,
        max_iterations_message="Task completed.",
        workspace=workspace,
        session_key="websocket:child",
        provider_retry_mode="conservative",
        profile="general",
        task="inspect this",
        caps=capabilities,
        context="fresh",
        isolation="shared",
        durable=False,
        lifetime="scoped",
    )

    runner_spec = RunExecutor.build_agent_run_spec(spec)

    assert runner_spec.initial_messages is messages
    assert runner_spec.max_iterations == 11
    assert runner_spec.max_tool_result_chars == 4321
    assert runner_spec.error_message is None
    assert runner_spec.max_iterations_message == "Task completed."
    assert runner_spec.workspace == workspace
    assert runner_spec.session_key == "websocket:child"
    assert runner_spec.provider_retry_mode == "conservative"
    assert (spec.profile, spec.task, spec.caps, spec.durable) == (
        "general",
        "inspect this",
        capabilities,
        False,
    )


def test_run_spec_messages_have_concrete_json_types() -> None:
    spec = RunSpec(
        runtime=_runtime(),
        tools=ToolRegistry(),
        budget=Budget(iterations=1),
        max_tool_result_chars=100,
        initial_messages=[{
            "role": "assistant",
            "content": [{"type": "tool_call", "arguments": {"count": 3}}],
        }],
    )

    assert spec.initial_messages is not None
    assert_type(spec.initial_messages, list[RunMessage])
    assert_type(spec.initial_messages[0]["role"], JsonValue)
    assert spec.initial_messages[0]["role"] == "assistant"


def test_executor_binds_run_roots_for_fair_scheduler_admission() -> None:
    scheduler = Scheduler(default_limit=1)
    lane = ProviderLane("scripted", "model")
    admitted: list[str | None] = []

    class AdmissionRunner:
        async def run(self, runner_spec):
            lease = await scheduler.acquire(lane)
            admitted.append(runner_spec.session_key)
            await lease.release()
            return None

    def spec(root: str | None, session_key: str) -> RunSpec:
        return RunSpec(
            runtime=_runtime(),
            tools=ToolRegistry(),
            budget=Budget(iterations=1),
            max_tool_result_chars=100,
            root=root,
            session_key=session_key,
        )

    async def run() -> None:
        held = await scheduler.acquire(lane)
        tasks = [
            asyncio.create_task(RunExecutor.run(AdmissionRunner(), spec("a", "a1"))),
            asyncio.create_task(RunExecutor.run(AdmissionRunner(), spec("a", "a2"))),
            asyncio.create_task(RunExecutor.run(AdmissionRunner(), spec(None, "b"))),
        ]
        await asyncio.sleep(0)
        await held.release()
        await asyncio.gather(*tasks)

    asyncio.run(run())
    assert admitted == ["a1", "b", "a2"]
