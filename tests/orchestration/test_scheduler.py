from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.memory import Memory
from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.orchestration import scheduler as scheduler_module
from nanobot.orchestration.scheduler import (
    Priority,
    ProviderLane,
    ScheduledProvider,
    Scheduler,
    scheduler_context,
)
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.utils.llm_runtime import LLMRuntime
from orchestration.sim import ScriptedProvider


@pytest.mark.asyncio
async def test_lane_limits_are_independent_by_provider_and_model() -> None:
    scheduler = Scheduler(default_limit=1)
    first = ProviderLane("provider-a", "model-a")
    second = ProviderLane("provider-a", "model-b")
    held = await scheduler.acquire(first)

    other = await asyncio.wait_for(scheduler.acquire(second), timeout=0.1)
    await other.release()
    await held.release()


@pytest.mark.asyncio
async def test_queued_roots_take_turns_instead_of_one_root_flooding_lane() -> None:
    scheduler = Scheduler(default_limit=1)
    lane = ProviderLane("provider", "model")
    held = await scheduler.acquire(lane)
    order: list[str] = []

    async def request(root: str) -> None:
        with scheduler_context(root=root):
            lease = await scheduler.acquire(lane)
            order.append(root)
            await lease.release()

    tasks = [asyncio.create_task(request("a")) for _ in range(3)]
    await asyncio.sleep(0)
    tasks.insert(1, asyncio.create_task(request("b")))
    await asyncio.sleep(0)
    await held.release()
    await asyncio.gather(*tasks)

    assert order[:2] == ["a", "b"]


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed_without_leaking_capacity() -> None:
    scheduler = Scheduler(default_limit=1)
    lane = ProviderLane("provider", "model")
    held = await scheduler.acquire(lane)
    task = asyncio.create_task(scheduler.acquire(lane))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await held.release()

    lease = await asyncio.wait_for(scheduler.acquire(lane), timeout=0.1)
    await lease.release()


@pytest.mark.asyncio
async def test_rate_limit_reduces_and_success_recovers_lane_limit() -> None:
    scheduler = Scheduler(default_limit=4, minimum_limit=1, maximum_limit=4)
    lane = ProviderLane("provider", "model")
    scheduler.observe(lane, error_kind="rate_limit", retry_after=0)
    assert scheduler.limit_for(lane) == 2

    scheduler.observe(lane, error_kind=None)
    assert scheduler.limit_for(lane) == 3


@pytest.mark.asyncio
async def test_priority_order_applies_before_root_round_robin() -> None:
    scheduler = Scheduler(default_limit=1)
    lane = ProviderLane("provider", "model")
    held = await scheduler.acquire(lane)
    admitted: list[str] = []

    async def request(name: str, priority: Priority) -> None:
        with scheduler_context(root=name, priority=priority):
            lease = await scheduler.acquire(lane)
            admitted.append(name)
            await lease.release()

    low = asyncio.create_task(request("maintenance", "maintenance"))
    await asyncio.sleep(0)
    high = asyncio.create_task(request("interactive", "interactive"))
    await asyncio.sleep(0)
    await held.release()
    await asyncio.gather(low, high)

    assert admitted == ["interactive", "maintenance"]


@pytest.mark.asyncio
async def test_scheduled_provider_preserves_response_and_provider_identity() -> None:
    response = LLMResponse(content="ok", finish_reason="stop")
    provider = ScriptedProvider({"ok": [response]}, route=lambda _messages: "ok")
    provider.generation = GenerationSettings(temperature=0.2, max_tokens=88)
    scheduled = ScheduledProvider(provider, Scheduler())

    actual = await scheduled.chat(messages=[{"role": "user", "content": "hi"}])

    assert actual is response
    assert scheduled.provider_name == provider.provider_name
    assert scheduled.generation == provider.generation
    assert scheduled.generation.temperature == 0.2
    assert len(provider.calls) == 1


def test_legacy_request_limit_maps_to_scheduler_limit_without_claiming_parity() -> None:
    from nanobot.providers.factory import scheduler_limit_from_settings

    assert scheduler_limit_from_settings(0) is None
    assert scheduler_limit_from_settings(3) == 3
    assert scheduler_limit_from_settings(0, environment_limit=2) == 2
    assert scheduler_limit_from_settings(3, environment_limit=0) is None


def test_scheduler_config_accepts_camel_case_and_preserves_uncapped_default() -> None:
    default_config = Config()
    config = Config.model_validate({
        "orchestration": {
            "scheduler": {
                "maxConcurrentRequests": 3,
                "laneLimits": {"provider/model": 2},
            }
        }
    })
    assert config.orchestration.scheduler.max_concurrent_requests == 3
    assert config.orchestration.scheduler.lane_limits == {"provider/model": 2}
    assert default_config.orchestration.scheduler.max_concurrent_requests == 0


@pytest.mark.asyncio
async def test_scheduled_call_honors_retry_after_before_next_admission() -> None:
    scheduler = Scheduler(default_limit=2)
    lane = ProviderLane("provider", "model")
    scheduler.observe(lane, error_kind="overloaded", retry_after=0.03)
    start = asyncio.get_running_loop().time()
    lease = await asyncio.wait_for(scheduler.acquire(lane), timeout=0.2)
    elapsed = asyncio.get_running_loop().time() - start
    await lease.release()

    assert elapsed >= 0.02


@pytest.mark.asyncio
async def test_locally_saturated_primary_uses_fallback_without_poisoning_breaker() -> None:
    scheduler = Scheduler(default_limit=1)
    primary = ScriptedProvider({}, route=lambda _messages: "unused", default_model="primary")
    fallback_leaf = ScriptedProvider(
        {"ok": [LLMResponse(content="fallback", finish_reason="stop")]},
        route=lambda _messages: "ok",
        default_model="fallback",
    )
    fallback = FallbackProvider(
        ScheduledProvider(primary, scheduler),
        [ModelPresetConfig(model="fallback", provider="scripted")],
        lambda _preset: ScheduledProvider(fallback_leaf, scheduler),
    )
    primary_lane = ProviderLane("scripted", "primary")
    occupied = await scheduler.acquire(primary_lane)

    response = await fallback.chat(
        messages=[{"role": "user", "content": "hi"}], model="primary"
    )

    assert response.content == "fallback"
    assert fallback._primary_failures == 0
    assert primary.calls == []
    assert len(fallback_leaf.calls) == 1
    await occupied.release()


@pytest.mark.asyncio
async def test_all_saturated_candidates_queue_and_cancellation_releases_waiter() -> None:
    scheduler = Scheduler(default_limit=1)
    primary = ScriptedProvider({}, route=lambda _messages: "unused", default_model="primary")
    fallback_leaf = ScriptedProvider({}, route=lambda _messages: "unused", default_model="fallback")
    fallback = FallbackProvider(
        ScheduledProvider(primary, scheduler),
        [ModelPresetConfig(model="fallback", provider="scripted")],
        lambda _preset: ScheduledProvider(fallback_leaf, scheduler),
    )
    primary_lease = await scheduler.acquire(ProviderLane("scripted", "primary"))
    fallback_lease = await scheduler.acquire(ProviderLane("scripted", "fallback"))
    callback_calls: list[str] = []

    async def fallback_started() -> None:
        callback_calls.append("started")

    task = asyncio.create_task(
        fallback._try_with_fallback(
            lambda provider, kwargs: provider.chat(**kwargs),
            {"messages": [{"role": "user", "content": "hi"}], "model": "primary"},
            has_streamed=None,
            on_fallback_attempt=fallback_started,
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert callback_calls == []
    await primary_lease.release()
    await fallback_lease.release()

    available = await asyncio.wait_for(
        scheduler.acquire(ProviderLane("scripted", "primary")), timeout=0.1
    )
    await available.release()
    assert fallback._primary_failures == 0


@pytest.mark.asyncio
async def test_all_saturated_candidates_wait_then_use_the_lane_that_frees() -> None:
    scheduler = Scheduler(default_limit=1)
    primary = ScriptedProvider({}, route=lambda _messages: "unused", default_model="primary")
    fallback_leaf = ScriptedProvider(
        {"ok": [LLMResponse(content="after wait", finish_reason="stop")]},
        route=lambda _messages: "ok",
        default_model="fallback",
    )
    fallback = FallbackProvider(
        ScheduledProvider(primary, scheduler),
        [ModelPresetConfig(model="fallback", provider="scripted")],
        lambda _preset: ScheduledProvider(fallback_leaf, scheduler),
    )
    primary_lease = await scheduler.acquire(ProviderLane("scripted", "primary"))
    fallback_lease = await scheduler.acquire(ProviderLane("scripted", "fallback"))
    task = asyncio.create_task(
        fallback.chat(messages=[{"role": "user", "content": "hi"}], model="primary")
    )
    await asyncio.sleep(0.01)
    await fallback_lease.release()

    response = await asyncio.wait_for(task, timeout=0.2)
    await primary_lease.release()

    assert response.content == "after wait"
    assert primary.calls == []
    assert fallback._primary_failures == 0


@pytest.mark.asyncio
async def test_fair_queue_lease_is_used_by_selected_candidate_without_releasing_it() -> None:
    scheduler = Scheduler(default_limit=1)
    primary = ScriptedProvider({}, route=lambda _messages: "unused", default_model="primary")
    entered_fallback = asyncio.Event()
    finish_fallback = asyncio.Event()

    class BlockingFallback(ScriptedProvider):
        async def chat(self, **kwargs: Any) -> LLMResponse:
            entered_fallback.set()
            await finish_fallback.wait()
            return LLMResponse(content="fallback", finish_reason="stop")

    fallback_leaf = BlockingFallback({}, route=lambda _messages: "unused", default_model="fallback")
    fallback = FallbackProvider(
        ScheduledProvider(primary, scheduler),
        [ModelPresetConfig(model="fallback", provider="scripted")],
        lambda _preset: ScheduledProvider(fallback_leaf, scheduler),
    )
    primary_lease = await scheduler.acquire(ProviderLane("scripted", "primary"))
    fallback_lease = await scheduler.acquire(ProviderLane("scripted", "fallback"))
    request = asyncio.create_task(
        fallback.chat(messages=[{"role": "user", "content": "hi"}], model="primary")
    )
    await asyncio.sleep(0)
    competitor_started = asyncio.Event()

    async def competitor() -> None:
        lease = await scheduler.acquire(ProviderLane("scripted", "fallback"))
        competitor_started.set()
        await lease.release()

    competitor_task = asyncio.create_task(competitor())
    await asyncio.sleep(0)
    await fallback_lease.release()

    await asyncio.wait_for(entered_fallback.wait(), timeout=0.2)
    assert scheduler._lane(ProviderLane("scripted", "fallback")).in_flight == 1
    assert not competitor_started.is_set()

    finish_fallback.set()
    response = await asyncio.wait_for(request, timeout=0.2)
    await asyncio.wait_for(competitor_task, timeout=0.2)
    await primary_lease.release()
    assert response.content == "fallback"


@pytest.mark.asyncio
async def test_acquire_cancellation_after_grant_releases_assigned_lease() -> None:
    scheduler = Scheduler(default_limit=1)
    lane = ProviderLane("provider", "model")
    held = await scheduler.acquire(lane)
    task = asyncio.create_task(scheduler.acquire(lane))
    await asyncio.sleep(0)

    await held.release()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    lease = await asyncio.wait_for(scheduler.acquire(lane), timeout=0.1)
    await lease.release()


@pytest.mark.asyncio
async def test_acquire_any_cancellation_releases_completed_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = Scheduler(default_limit=1)
    real_wait = asyncio.wait
    owner: asyncio.Task[Any] | None = None

    async def cancel_after_a_winner(
        tasks: Any,
        *,
        return_when: Any,
    ) -> tuple[set[Any], set[Any]]:
        done, pending = await real_wait(tasks, return_when=return_when)
        assert owner is not None
        owner.cancel()
        return done, pending

    monkeypatch.setattr(scheduler_module.asyncio, "wait", cancel_after_a_winner)

    async def reserve() -> None:
        await scheduler.acquire_any([
            ProviderLane("provider", "first"),
            ProviderLane("provider", "second"),
        ])

    owner = asyncio.create_task(reserve())
    with pytest.raises(asyncio.CancelledError):
        await owner

    for model in ("first", "second"):
        lease = await asyncio.wait_for(
            scheduler.acquire(ProviderLane("provider", model)), timeout=0.1
        )
        await lease.release()


@pytest.mark.asyncio
async def test_scheduled_retry_paths_preserve_local_saturation_for_fallback() -> None:
    scheduler = Scheduler(default_limit=1)
    primary_leaf = ScriptedProvider({}, route=lambda _messages: "unused", default_model="primary")
    fallback_leaf = ScriptedProvider(
        {"ok": [LLMResponse(content="fallback", finish_reason="stop")]},
        route=lambda _messages: "ok",
        default_model="fallback",
    )
    fallback = FallbackProvider(
        ScheduledProvider(primary_leaf, scheduler),
        [ModelPresetConfig(model="fallback", provider="scripted")],
        lambda _preset: ScheduledProvider(fallback_leaf, scheduler),
    )
    primary_observations: list[Any] = []
    fallback_observations: list[Any] = []
    primary = fallback._primary
    assert isinstance(primary, ScheduledProvider)
    primary.set_llm_call_observer(primary_observations.append)
    fallback.set_llm_call_observer(fallback_observations.append)
    held = await scheduler.acquire(ProviderLane("scripted", "primary"))
    retry_statuses: list[Any] = []

    async def status(event: Any) -> None:
        retry_statuses.append(event)

    response = await fallback.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}],
        model="primary",
        on_retry_status=status,
    )
    await held.release()

    assert response.content == "fallback"
    assert fallback._primary_failures == 0
    assert primary_observations == []
    assert len(fallback_observations) == 1
    assert retry_statuses == []


@pytest.mark.asyncio
async def test_scheduled_stream_retry_paths_preserve_local_saturation_for_fallback() -> None:
    scheduler = Scheduler(default_limit=1)
    primary_leaf = ScriptedProvider({}, route=lambda _messages: "unused", default_model="primary")
    fallback_leaf = ScriptedProvider(
        {"ok": [LLMResponse(content="fallback", finish_reason="stop")]},
        route=lambda _messages: "ok",
        default_model="fallback",
    )
    fallback = FallbackProvider(
        ScheduledProvider(primary_leaf, scheduler),
        [ModelPresetConfig(model="fallback", provider="scripted")],
        lambda _preset: ScheduledProvider(fallback_leaf, scheduler),
    )
    primary_observations: list[Any] = []
    primary = fallback._primary
    assert isinstance(primary, ScheduledProvider)
    primary.set_llm_call_observer(primary_observations.append)
    held = await scheduler.acquire(ProviderLane("scripted", "primary"))

    response = await fallback.chat_stream_with_retry(
        messages=[{"role": "user", "content": "hi"}], model="primary"
    )
    await held.release()

    assert response.content == "fallback"
    assert fallback._primary_failures == 0
    assert primary_observations == []


@pytest.mark.asyncio
async def test_429_provider_responses_decrease_and_recover_without_losing_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanobot.providers.base as provider_base

    scheduler = Scheduler(default_limit=4, minimum_limit=1, maximum_limit=4)
    responses = [
        LLMResponse(
            content="HTTP 429 rate limit",
            finish_reason="error",
            error_status_code=429,
            error_kind="http",
            retry_after=0.02,
        ),
        LLMResponse(
            content="HTTP 429 rate limit",
            finish_reason="error",
            error_status_code=429,
            error_kind="http",
        ),
        LLMResponse(content="recovered", finish_reason="stop"),
    ]
    leaf = ScriptedProvider({"ok": responses}, route=lambda _messages: "ok", default_model="model")
    provider = ScheduledProvider(leaf, scheduler)
    provider._CHAT_RETRY_DELAYS = (0, 0, 0)
    monkeypatch.setattr(provider_base, "RETRY_AFTER_BUFFER", 0)
    started = asyncio.get_running_loop().time()
    response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}], model="model"
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert response.content == "recovered"
    assert len(leaf.calls) == 3
    assert scheduler.limit_for(ProviderLane("scripted", "model")) == 2
    assert elapsed >= 0.015


@pytest.mark.asyncio
async def test_http_429_exception_applies_aimd_and_retry_after() -> None:
    class RateLimitError(Exception):
        status_code = 429
        retry_after = 0.02

    class RaisingProvider(ScriptedProvider):
        async def chat(self, **kwargs: Any) -> LLMResponse:
            raise RateLimitError("too many requests")

    scheduler = Scheduler(default_limit=4)
    leaf = RaisingProvider({}, route=lambda _messages: "unused", default_model="model")
    provider = ScheduledProvider(leaf, scheduler)
    with pytest.raises(RateLimitError):
        await provider.chat(messages=[{"role": "user", "content": "hi"}], model="model")
    assert scheduler.limit_for(ProviderLane("scripted", "model")) == 2

    started = asyncio.get_running_loop().time()
    lease = await asyncio.wait_for(
        scheduler.acquire(ProviderLane("scripted", "model")), timeout=0.2
    )
    elapsed = asyncio.get_running_loop().time() - started
    await lease.release()
    assert elapsed >= 0.015


@pytest.mark.asyncio
async def test_root_fairness_limits_10_way_fanout_p95_impact_to_under_20_percent() -> None:
    async def p95_admission_ticket(*, include_fanout: bool) -> int:
        scheduler = Scheduler(default_limit=1)
        lane = ProviderLane("provider", "model")
        held = await scheduler.acquire(lane)
        tickets: dict[str, list[int]] = {"b": []}
        if include_fanout:
            tickets["a"] = []
        admission_ticket = 0

        async def request(root: str) -> None:
            nonlocal admission_ticket
            with scheduler_context(root=root):
                lease = await scheduler.acquire(lane)
            admission_ticket += 1
            tickets[root].append(admission_ticket)
            await lease.release()

        b_tasks = [asyncio.create_task(request("b")) for _ in range(100)]
        a_tasks = (
            [asyncio.create_task(request("a")) for _ in range(10)]
            if include_fanout else []
        )
        await asyncio.sleep(0)
        await held.release()
        await asyncio.gather(*b_tasks, *a_tasks)
        return tickets["b"][94]

    baseline_p95 = await p95_admission_ticket(include_fanout=False)
    fanout_p95 = await p95_admission_ticket(include_fanout=True)

    assert fanout_p95 <= baseline_p95 * 1.2




@pytest.mark.asyncio
async def test_memory_completion_uses_maintenance_priority() -> None:
    scheduler = Scheduler(default_limit=1)
    calls: list[str] = []

    def route(messages: list[dict[str, Any]]) -> str:
        system = str(messages[0].get("content"))
        calls.append(system)
        return system

    leaf = ScriptedProvider(
        {
            "memory": [LLMResponse(content="memory done", finish_reason="stop")],
            "interactive": [LLMResponse(content="interactive done", finish_reason="stop")],
        },
        route=route,
        default_model="model",
    )
    provider = ScheduledProvider(leaf, scheduler)
    runtime = LLMRuntime.capture(provider, "model", context_window_tokens=2048)

    class MemoryRuntime(Memory):
        def __init__(self) -> None:
            self.workspace = Path("/workspace")

        def _runtime(self) -> LLMRuntime:
            return runtime

    held = await scheduler.acquire(ProviderLane("scripted", "model"))
    memory_call = asyncio.create_task(
        Memory._complete(MemoryRuntime(), system="memory", prompt="compact", temperature=0)
    )
    await asyncio.sleep(0)
    interactive_call = asyncio.create_task(provider.chat(
        messages=[{"role": "system", "content": "interactive"}],
        model="model",
    ))
    await asyncio.sleep(0)
    await held.release()
    await asyncio.gather(memory_call, interactive_call)

    assert calls == ["interactive", "memory"]
