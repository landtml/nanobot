"""Small deterministic test primitives for orchestration runs."""

from __future__ import annotations

import asyncio
import heapq
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse

FaultKind = Literal["crash", "rate_limit", "timeout", "slow_tool"]
FaultRoute = Callable[[list[dict[str, Any]]], str]


@dataclass(frozen=True, slots=True)
class Fault:
    kind: FaultKind
    delay: float = 0.0


class SimulatedFaultError(RuntimeError):
    def __init__(self, kind: Literal["crash", "rate_limit", "timeout"]) -> None:
        super().__init__(f"simulated {kind}")
        self.kind = kind


class SimulatedCrashError(SimulatedFaultError):
    def __init__(self) -> None:
        super().__init__("crash")


class SimulatedRateLimitError(SimulatedFaultError):
    def __init__(self) -> None:
        super().__init__("rate_limit")


class SimulatedTimeoutError(SimulatedFaultError):
    def __init__(self) -> None:
        super().__init__("timeout")


@dataclass(slots=True)
class _TimerHandle:
    callback: Callable[[], None] = field(repr=False)
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class VirtualClock:
    """Manual monotonic clock with cancellable timers and deterministic sleeps."""

    def __init__(self) -> None:
        self._now = 0.0
        self._sequence = 0
        self._timers: list[tuple[float, int, _TimerHandle]] = []

    def time(self) -> float:
        return self._now

    def call_later(self, delay: float, callback: Callable[[], None]) -> _TimerHandle:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        handle = _TimerHandle(callback)
        self._sequence += 1
        heapq.heappush(self._timers, (self._now + delay, self._sequence, handle))
        return handle

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time cannot move backwards")
        target = self._now + seconds
        while self._timers and self._timers[0][0] <= target:
            deadline, _, handle = heapq.heappop(self._timers)
            self._now = deadline
            if not handle.cancelled:
                handle.callback()
        self._now = target

    async def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("delay must be non-negative")
        if seconds == 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        handle = self.call_later(
            seconds,
            lambda: None if future.done() else future.set_result(None),
        )
        try:
            await future
        finally:
            handle.cancel()


class FaultInjector:
    """Apply named faults to provider calls or tools under test."""

    def __init__(
        self,
        clock: VirtualClock,
        faults: Mapping[str, Iterable[Fault]] | None = None,
    ) -> None:
        self._clock = clock
        self._faults = {
            route: deque(script)
            for route, script in (faults or {}).items()
        }

    def add(self, route: str, fault: Fault) -> None:
        self._faults.setdefault(route, deque()).append(fault)

    async def trip(self, route: str) -> None:
        script = self._faults.get(route)
        if not script:
            return
        fault = script.popleft()
        if fault.kind == "slow_tool":
            await self._clock.sleep(fault.delay)
            return
        if fault.kind == "crash":
            raise SimulatedCrashError()
        if fault.kind == "rate_limit":
            raise SimulatedRateLimitError()
        raise SimulatedTimeoutError()


@dataclass(frozen=True, slots=True)
class RecordedCall:
    route: str
    model: str | None
    messages: list[dict[str, Any]]
    tool_names: tuple[str, ...]


class ScriptedProvider(LLMProvider):
    """Provider that consumes a separate response script for each route."""

    def __init__(
        self,
        scripts: Mapping[str, Iterable[LLMResponse]],
        *,
        route: FaultRoute,
        default_model: str = "test-model",
        faults: FaultInjector | None = None,
    ) -> None:
        super().__init__(provider_name="scripted")
        self.generation = GenerationSettings()
        self._scripts = {key: deque(script) for key, script in scripts.items()}
        self._route = route
        self._default_model = default_model
        self.faults = faults
        self.calls: list[RecordedCall] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        _ = max_tokens, temperature, reasoning_effort, tool_choice
        route = self._route(messages)
        self.calls.append(RecordedCall(
            route=route,
            model=model,
            messages=deepcopy(messages),
            tool_names=tuple(
                tool.get("function", {}).get("name", "")
                for tool in (tools or [])
            ),
        ))
        if self.faults is not None:
            try:
                await self.faults.trip(f"provider:{route}")
            except SimulatedFaultError as exc:
                return _fault_response(exc)
        script = self._scripts.get(route)
        if not script:
            raise AssertionError(f"no scripted response remains for route {route!r}")
        response = script.popleft()
        return response

    def get_default_model(self) -> str:
        return self._default_model


def _fault_response(fault: SimulatedFaultError) -> LLMResponse:
    response = LLMResponse(
        content=str(fault),
        finish_reason="error",
        error_should_retry=fault.kind in ("rate_limit", "timeout"),
    )
    if fault.kind == "rate_limit":
        response.error_status_code = 429
        response.error_kind = "rate_limit"
        response.error_code = "rate_limit_exceeded"
    elif fault.kind == "timeout":
        response.error_kind = "timeout"
    else:
        response.error_type = "simulated_crash"
    return response
