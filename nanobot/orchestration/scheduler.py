"""Fair, adaptive admission for model calls."""

# pyright: reportIncompatibleVariableOverride=false

from __future__ import annotations

import asyncio
import contextvars
import math
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from nanobot.providers.base import (
    GenerationSettings,
    LLMCallObserver,
    LLMProvider,
    LLMResponse,
    ProviderAdmissionError,
    ProviderCallContext,
    ProviderConversationState,
)

Priority: TypeAlias = Literal[
    "interactive", "awaited", "background", "automation", "maintenance"
]
_PRIORITIES: dict[Priority, int] = {
    "interactive": 0,
    "awaited": 1,
    "background": 2,
    "automation": 3,
    "maintenance": 4,
}
_PRIORITY_ORDER: tuple[Priority, ...] = (
    "interactive", "awaited", "background", "automation", "maintenance"
)
_root_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nanobot_scheduler_root", default="workspace"
)
_priority_context: contextvars.ContextVar[Priority] = contextvars.ContextVar(
    "nanobot_scheduler_priority", default="interactive"
)
_fallback_candidate_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "nanobot_scheduler_fallback_candidate", default=False
)
_fallback_candidate_admitted: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "nanobot_scheduler_fallback_candidate_admitted", default=False
)
_fallback_admitted_callback: contextvars.ContextVar[
    Callable[[], Awaitable[None]] | None
] = contextvars.ContextVar("nanobot_scheduler_fallback_admitted_callback", default=None)


@dataclass(slots=True)
class _ReservedLease:
    lease: LaneLease
    claimed: bool = False


_reserved_lease_context: contextvars.ContextVar[_ReservedLease | None] = (
    contextvars.ContextVar("nanobot_scheduler_reserved_lease", default=None)
)


@dataclass(frozen=True, slots=True)
class ProviderLane:
    provider: str
    model: str


@dataclass(slots=True)
class _Waiter:
    root: str
    priority: Priority
    future: asyncio.Future["LaneLease"]


@dataclass(slots=True)
class _Lane:
    limit: int | None
    ceiling: int | None
    in_flight: int = 0
    blocked_until: float = 0.0
    queues: dict[Priority, dict[str, deque[_Waiter]]] = field(default_factory=dict)
    roots: dict[Priority, deque[str]] = field(default_factory=dict)
    wake_task: asyncio.Task[None] | None = None


class LaneLease:
    def __init__(self, scheduler: Scheduler, lane: ProviderLane) -> None:
        self._scheduler = scheduler
        self.lane = lane
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    async def release(self, response: LLMResponse | None = None) -> None:
        self.release_now(response)

    def release_now(self, response: LLMResponse | None = None) -> None:
        if self._released:
            return
        self._released = True
        self._scheduler.release(self.lane, response)


class LaneSaturatedError(ProviderAdmissionError):
    """Internal signal that a fallback candidate was busy before provider I/O."""

    def __init__(self, scheduler: Scheduler, lane: ProviderLane) -> None:
        super().__init__("provider lane is locally saturated")
        self.scheduler = scheduler
        self.lane = lane


class Scheduler:
    """Per-provider/model lanes with priority-aware, root-fair waiting."""

    def __init__(
        self,
        *,
        default_limit: int | None = None,
        lane_limits: Mapping[str, int] | None = None,
        minimum_limit: int = 1,
        maximum_limit: int | None = None,
        decrease_factor: float = 0.5,
    ) -> None:
        if default_limit is not None and default_limit < 1:
            raise ValueError("default_limit must be positive or None")
        if minimum_limit < 1:
            raise ValueError("minimum_limit must be positive")
        if maximum_limit is not None and maximum_limit < minimum_limit:
            raise ValueError("maximum_limit must be at least minimum_limit")
        if not 0 < decrease_factor < 1:
            raise ValueError("decrease_factor must be between zero and one")
        self._default_limit = default_limit
        self._lane_limits = dict(lane_limits or {})
        self._minimum_limit = minimum_limit
        self._maximum_limit = maximum_limit
        self._decrease_factor = decrease_factor
        self._lanes: dict[ProviderLane, _Lane] = {}

    def _lane(self, key: ProviderLane) -> _Lane:
        lane = self._lanes.get(key)
        if lane is None:
            configured = self._lane_limits.get(f"{key.provider}/{key.model}")
            limit = configured if configured is not None else self._default_limit
            if limit is not None and self._maximum_limit is not None:
                limit = min(limit, self._maximum_limit)
            ceiling = self._maximum_limit if self._maximum_limit is not None else limit
            lane = _Lane(limit=limit, ceiling=ceiling)
            self._lanes[key] = lane
        return lane

    def limit_for(self, key: ProviderLane) -> int | None:
        return self._lane(key).limit

    async def acquire(self, key: ProviderLane) -> LaneLease:
        lane = self._lane(key)
        if (
            not self._has_waiters(lane)
            and self._available(lane)
        ):
            lane.in_flight += 1
            return LaneLease(self, key)

        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            _root_context.get(),
            _priority_context.get(),
            loop.create_future(),
        )
        root_queues = lane.queues.setdefault(waiter.priority, {})
        queue = root_queues.setdefault(waiter.root, deque())
        if not queue:
            lane.roots.setdefault(waiter.priority, deque()).append(waiter.root)
        queue.append(waiter)
        self._drain(key, lane)
        try:
            return await waiter.future
        except asyncio.CancelledError:
            if waiter.future.done() and not waiter.future.cancelled():
                waiter.future.result().release_now()
            self._remove_waiter(lane, waiter)
            self._drain(key, lane)
            raise

    async def acquire_any(self, keys: list[ProviderLane]) -> LaneLease:
        """Fairly wait for capacity in any candidate lane, cancelling losing queues."""
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise ValueError("at least one provider lane is required")
        tasks = [asyncio.create_task(self.acquire(key)) for key in unique_keys]
        selector = asyncio.create_task(self._select_lease(tasks))
        try:
            return await asyncio.shield(selector)
        except asyncio.CancelledError:
            selector.cancel()
            try:
                await asyncio.shield(selector)
            except BaseException:
                pass
            if selector.done() and not selector.cancelled():
                try:
                    selector.result().release_now()
                except BaseException:
                    pass
            raise

    async def _select_lease(
        self,
        tasks: list[asyncio.Task[LaneLease]],
    ) -> LaneLease:
        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            winner = next(iter(done))
            lease = winner.result()
            for task in tasks:
                if task is not winner and not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                if task is winner or task.cancelled():
                    continue
                try:
                    other_lease = task.result()
                except BaseException:
                    continue
                other_lease.release_now()
            return lease
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                if task.cancelled():
                    continue
                try:
                    lease = task.result()
                except BaseException:
                    continue
                lease.release_now()
            raise

    def try_acquire(self, key: ProviderLane) -> LaneLease | None:
        lane = self._lane(key)
        if self._has_waiters(lane) or not self._available(lane):
            return None
        lane.in_flight += 1
        return LaneLease(self, key)

    async def run(
        self,
        key: ProviderLane,
        call: Callable[[], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        lease = await self.acquire(key)
        try:
            response = await call()
        except BaseException:
            await lease.release()
            raise
        await lease.release(response)
        return response

    def observe(
        self,
        key: ProviderLane,
        *,
        error_kind: str | None,
        retry_after: float | None = None,
    ) -> None:
        lane = self._lane(key)
        if lane.limit is None:
            return
        if (error_kind or "").lower() in {"rate_limit", "overloaded"}:
            lane.limit = max(
                self._minimum_limit,
                math.floor(lane.limit * self._decrease_factor),
            )
            if retry_after is not None and retry_after > 0:
                lane.blocked_until = max(
                    lane.blocked_until,
                    asyncio.get_running_loop().time() + retry_after,
                )
                self._schedule_wake(key, lane)
        elif error_kind is None and (lane.ceiling is None or lane.limit < lane.ceiling):
            lane.limit += 1
        self._drain(key, lane)

    def release(self, key: ProviderLane, response: LLMResponse | None = None) -> None:
        lane = self._lane(key)
        if lane.in_flight <= 0:
            raise RuntimeError("provider lane lease released without admission")
        lane.in_flight -= 1
        if response is not None:
            error_kind = (
                response.error_kind
                if response.finish_reason == "error"
                else None
            )
            status = response.error_status_code
            if response.finish_reason == "error" and (
                status == 429
                or (
                    (error_kind or "").lower() == "http"
                    and "429" in (response.content or "")
                )
            ):
                error_kind = "rate_limit"
            self.observe(
                key,
                error_kind=error_kind,
                retry_after=response.error_retry_after_s or response.retry_after,
            )
        self._drain(key, lane)

    def _available(self, lane: _Lane) -> bool:
        return (
            lane.limit is None or lane.in_flight < lane.limit
        ) and asyncio.get_running_loop().time() >= lane.blocked_until

    @staticmethod
    def _has_waiters(lane: _Lane) -> bool:
        return any(lane.roots.values())

    def _drain(self, key: ProviderLane, lane: _Lane) -> None:
        while self._has_waiters(lane) and self._available(lane):
            priority: Priority = "maintenance"
            for candidate_priority in _PRIORITY_ORDER:
                if lane.roots.get(candidate_priority):
                    priority = candidate_priority
                    break
            roots = lane.roots[priority]
            root = roots.popleft()
            root_queues = lane.queues[priority]
            queue = root_queues[root]
            waiter = queue.popleft()
            if queue:
                roots.append(root)
            else:
                del root_queues[root]
            if not roots:
                del lane.roots[priority]
                del lane.queues[priority]
            if waiter.future.cancelled():
                continue
            lane.in_flight += 1
            waiter.future.set_result(LaneLease(self, key))

    def _remove_waiter(self, lane: _Lane, waiter: _Waiter) -> None:
        root_queues = lane.queues.get(waiter.priority)
        queue = root_queues.get(waiter.root) if root_queues else None
        if queue is None:
            return
        try:
            queue.remove(waiter)
        except ValueError:
            return
        if not queue:
            if root_queues is None:
                return
            del root_queues[waiter.root]
            roots = lane.roots[waiter.priority]
            roots.remove(waiter.root)
            if not roots:
                del lane.roots[waiter.priority]
                del lane.queues[waiter.priority]

    def _schedule_wake(self, key: ProviderLane, lane: _Lane) -> None:
        if lane.wake_task is not None and not lane.wake_task.done():
            lane.wake_task.cancel()

        async def wake() -> None:
            delay = max(0.0, lane.blocked_until - asyncio.get_running_loop().time())
            await asyncio.sleep(delay)
            self._drain(key, lane)

        lane.wake_task = asyncio.create_task(wake())


class ScheduledProvider(LLMProvider):
    """Transparent single-leaf provider wrapper that schedules each call."""

    def __init__(self, provider: LLMProvider, scheduler: Scheduler) -> None:
        self._provider = provider
        self._scheduler = scheduler
        generation = provider.generation
        super().__init__(provider_name=provider.provider_name)
        self.generation = generation

    @property
    def generation(self) -> GenerationSettings:
        return self._provider.generation

    @generation.setter
    def generation(self, value: GenerationSettings) -> None:
        self._provider.generation = value

    def get_default_model(self) -> str:
        return self._provider.get_default_model()

    def lane_for(self, model: str | None = None) -> ProviderLane:
        return ProviderLane(self.provider_name, model or self.get_default_model())

    def set_llm_call_observer(self, observer: LLMCallObserver | None) -> None:
        super().set_llm_call_observer(observer)
        self._provider.set_llm_call_observer(observer)

    def can_resume_conversation_state(
        self,
        state: ProviderConversationState,
        model: str | None = None,
    ) -> bool:
        return self._provider.can_resume_conversation_state(state, model)

    def supports_native_compaction(self, model: str | None = None) -> bool:
        return self._provider.supports_native_compaction(model)

    def supports_pre_request_compaction(self, model: str | None = None) -> bool:
        return self._provider.supports_pre_request_compaction(model)

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
        selected_model = model or self.get_default_model()
        return await self._run(
            selected_model,
            lambda: self._provider.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            ),
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        selected_model = model or self.get_default_model()
        return await self._run(
            selected_model,
            lambda: self._provider.chat_stream(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
                on_content_delta=on_content_delta,
                on_thinking_delta=on_thinking_delta,
                on_tool_call_delta=on_tool_call_delta,
            ),
        )

    async def chat_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: object,
    ) -> LLMResponse:
        model = kwargs.get("model")
        selected_model = model if isinstance(model, str) else self.get_default_model()
        return await self._run(
            selected_model,
            lambda: self._provider.chat_with_context(
                provider_context=provider_context, **kwargs
            ),
        )

    async def chat_stream_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: object,
    ) -> LLMResponse:
        model = kwargs.get("model")
        selected_model = model if isinstance(model, str) else self.get_default_model()
        return await self._run(
            selected_model,
            lambda: self._provider.chat_stream_with_context(
                provider_context=provider_context, **kwargs
            ),
        )

    async def _run(
        self,
        model: str,
        call: Callable[[], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        lane = self.lane_for(model)
        if _fallback_candidate_context.get():
            reserved = _reserved_lease_context.get()
            if (
                reserved is not None
                and reserved.lease.lane == lane
                and not reserved.claimed
                and not reserved.lease.released
            ):
                reserved.claimed = True
                lease = reserved.lease
                _fallback_candidate_admitted.set(True)
            elif _fallback_candidate_admitted.get():
                lease = await self._scheduler.acquire(lane)
            else:
                lease = self._scheduler.try_acquire(lane)
                if lease is None:
                    raise LaneSaturatedError(self._scheduler, lane)
                _fallback_candidate_admitted.set(True)
        else:
            lease = await self._scheduler.acquire(lane)
        try:
            admitted_callback = _fallback_admitted_callback.get()
            if admitted_callback is not None:
                await admitted_callback()
            response = await call()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError | ProviderAdmissionError):
                await lease.release()
            else:
                await lease.release(_response_for_exception(exc))
            raise
        await lease.release(response)
        return response


@contextmanager
def fallback_candidate_admission(
    on_admitted: Callable[[], Awaitable[None]] | None = None,
):
    """Make wrapped leaves report busy lanes to the fallback selector."""
    token = _fallback_candidate_context.set(True)
    admitted_token = _fallback_candidate_admitted.set(False)
    callback_token = _fallback_admitted_callback.set(on_admitted)
    try:
        yield
    finally:
        _fallback_admitted_callback.reset(callback_token)
        _fallback_candidate_admitted.reset(admitted_token)
        _fallback_candidate_context.reset(token)


@contextmanager
def reserved_lane_lease(lease: LaneLease):
    """Offer an acquired lane lease to the matching fallback candidate once."""
    reservation = _ReservedLease(lease)
    token = _reserved_lease_context.set(reservation)
    try:
        yield reservation
    finally:
        _reserved_lease_context.reset(token)
        if not reservation.claimed:
            lease.release_now()


@contextmanager
def scheduler_context(*, root: str | None = None, priority: Priority = "interactive"):
    """Bind per-call fairness identity and priority for the current task."""
    if priority not in _PRIORITIES:
        raise ValueError(f"unknown scheduler priority: {priority!r}")
    root_token = _root_context.set(root or "workspace")
    priority_token = _priority_context.set(priority)
    try:
        yield
    finally:
        _priority_context.reset(priority_token)
        _root_context.reset(root_token)


def _response_for_exception(exc: BaseException) -> LLMResponse:
    if not isinstance(exc, Exception):
        return LLMResponse(content=None, finish_reason="error")
    response = LLMProvider.error_response_from_exception(exc)
    retry_after: float | None = None
    raw_retry_after = getattr(exc, "retry_after", None)
    try:
        retry_after = float(raw_retry_after) if raw_retry_after is not None else None
    except (TypeError, ValueError):
        retry_after = None
    if retry_after is None:
        error_response = getattr(exc, "response", None)
        headers = getattr(error_response, "headers", None)
        if headers is not None:
            retry_after = LLMProvider.retry_after_from_headers(headers)
    response.error_retry_after_s = retry_after
    if response.error_status_code == 429:
        response.error_kind = "rate_limit"
    return response
