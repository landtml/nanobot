"""Translate orchestration runs into the shared agent runner contract."""

from __future__ import annotations

import asyncio

from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec, InjectionCallback
from nanobot.orchestration.scheduler import scheduler_context
from nanobot.orchestration.types import RunMessage, RunnerDefaultErrorMessage, RunSpec


class RunExecutor:
    @staticmethod
    def _mailbox_callback(mailbox: asyncio.Queue[RunMessage]) -> InjectionCallback:
        async def drain() -> list[RunMessage]:
            messages: list[RunMessage] = []
            while True:
                try:
                    messages.append(mailbox.get_nowait())
                except asyncio.QueueEmpty:
                    return messages

        return drain

    @staticmethod
    def build_agent_run_spec(spec: RunSpec) -> AgentRunSpec:
        injection_callback = spec.injection_callback
        if injection_callback is None and spec.mailbox is not None:
            injection_callback = RunExecutor._mailbox_callback(spec.mailbox)
        runner_spec = AgentRunSpec(
            initial_messages=spec.initial_messages,
            tools=spec.tools,
            runtime=spec.runtime,
            max_iterations=spec.budget.iterations,
            max_tool_result_chars=spec.max_tool_result_chars,
            transcript_input=spec.transcript_input,
            transcript_builder=spec.transcript_builder,
            hook=spec.hook,
            max_iterations_message=spec.max_iterations_message,
            concurrent_tools=spec.concurrent_tools,
            workspace=spec.workspace,
            session_key=spec.session_key,
            provider_retry_mode=spec.provider_retry_mode,
            checkpoint_callback=spec.checkpoint_callback,
            consolidate_history=spec.consolidate_history,
            consolidate_provider_compaction=spec.consolidate_provider_compaction,
            injection_callback=injection_callback,
            terminal_injection_callback=spec.terminal_injection_callback,
            continuation_callback=spec.continuation_callback,
            finalize_on_max_iterations=spec.finalize_on_max_iterations,
            provider_state=spec.provider_state,
            llm_usage_source=spec.llm_usage_source,
            events=spec.events,
        )
        if not isinstance(spec.error_message, RunnerDefaultErrorMessage):
            runner_spec.error_message = spec.error_message
        return runner_spec

    @staticmethod
    async def run(runner: AgentRunner, spec: RunSpec) -> AgentRunResult:
        root = spec.root or spec.session_key
        if root is None and spec.workspace is not None:
            root = f"workspace:{spec.workspace.expanduser().resolve(strict=False)}"
        with scheduler_context(root=root, priority=spec.priority):
            return await runner.run(RunExecutor.build_agent_run_spec(spec))
