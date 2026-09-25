"""Typed inputs for orchestration runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from nanobot.agent.context import TranscriptInput
from nanobot.agent.context_governance import (
    HistoryConsolidator,
    ProviderCompactionConsolidator,
    TranscriptBuilder,
)
from nanobot.agent.hook import AgentHook
from nanobot.agent.runner import CheckpointCallback, ContinuationCallback, InjectionCallback
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.events import NO_EVENTS, EventSink
from nanobot.llm_usage.context import LLMUsageSource
from nanobot.providers.base import ProviderConversationState
from nanobot.utils.llm_runtime import LLMRuntime

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
RunMessage: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Capabilities:
    tools: frozenset[str] = frozenset()
    mcp_servers: frozenset[str] = frozenset()
    paths: frozenset[str] = frozenset()
    network: bool = False
    memory: bool = False


@dataclass(frozen=True, slots=True)
class Budget:
    iterations: int
    tokens: int | None = None
    cost: float | None = None
    wall_seconds: float | None = None
    fan_out: int | None = None


@dataclass(frozen=True, slots=True)
class RunnerDefaultErrorMessage:
    pass


RUNNER_DEFAULT_ERROR_MESSAGE = RunnerDefaultErrorMessage()


@dataclass(slots=True)
class RunSpec:
    runtime: LLMRuntime
    tools: ToolRegistry
    budget: Budget
    max_tool_result_chars: int
    initial_messages: list[RunMessage] | None = None
    transcript_input: TranscriptInput | None = None
    transcript_builder: TranscriptBuilder | None = None
    hook: AgentHook | None = None
    error_message: str | None | RunnerDefaultErrorMessage = RUNNER_DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    workspace: Path | None = None
    session_key: str | None = None
    provider_retry_mode: str = "standard"
    checkpoint_callback: CheckpointCallback | None = None
    consolidate_history: HistoryConsolidator | None = None
    consolidate_provider_compaction: ProviderCompactionConsolidator | None = None
    injection_callback: InjectionCallback | None = None
    terminal_injection_callback: InjectionCallback | None = None
    continuation_callback: ContinuationCallback | None = None
    finalize_on_max_iterations: bool = True
    provider_state: ProviderConversationState | None = None
    llm_usage_source: LLMUsageSource | None = None
    events: EventSink = NO_EVENTS
    mailbox: asyncio.Queue[RunMessage] | None = None
    id: str | None = None
    root: str | None = None
    parent: str | None = None
    profile: str = "general"
    task: str = ""
    caps: Capabilities = field(default_factory=Capabilities)
    context: Literal["fork", "brief", "fresh"] = "fresh"
    isolation: Literal["shared", "worktree"] = "shared"
    durable: bool = True
    lifetime: Literal["scoped", "detached"] = "scoped"
