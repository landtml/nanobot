"""Behavior of the Observational Memory engine around its model calls.

Prompt text, formatting, parsing, token counts and thread selection are
checked byte for byte against Mastra 1.1.0 in ``test_golden.py``; these tests
cover what happens between those pieces: thresholds, cursors, batching,
reflection, concurrency and transient observations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nanobot.agent.memory import message_fingerprint, observer_messages
from nanobot.agent.observational_memory import (
    ObservationalMemory,
    ObservationalMemoryConfig,
    ObservationStore,
    PendingThread,
    prompts,
    text,
    tokens,
)
from nanobot.agent.observational_memory.engine import (
    OBSERVER_TEMPERATURE,
    REFLECTOR_TEMPERATURE,
)

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _memory(tmp_path: Path, **config: int) -> ObservationalMemory:
    return ObservationalMemory(
        ObservationStore(tmp_path),
        config=ObservationalMemoryConfig(**config),
        timezone=timezone.utc,
        clock=lambda: NOW,
    )


def _session(key: str, texts: list[str], *, day: int = 25) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": content,
            "timestamp": f"2026-09-{day:02d}T10:{i:02d}:00+00:00",
        }
        for i, content in enumerate(texts)
    ]


def _thread(key: str, texts: list[str], *, start: int = 0, day: int = 25) -> PendingThread:
    messages = _session(key, texts, day=day)[start:]
    return PendingThread(
        key=key,
        messages=tuple(observer_messages(key, messages, start=start, tz=timezone.utc)),
        end=len(texts),
        fingerprint=message_fingerprint(_session(key, texts, day=day)[-1]),
    )


def _observed(line: str, *, task: str | None = None) -> str:
    out = f"<observations>\nDate: Sep 25, 2026\n* 🔴 (10:00) {line}\n</observations>"
    return out + (f"\n<current-task>\n{task}\n</current-task>" if task else "")


@dataclass
class _Model:
    """Scripted Observer/Reflector: answers by temperature, records every call."""

    observer: list[str] = field(default_factory=list)
    reflector: list[str] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    async def __call__(self, *, system: str, prompt: str, temperature: float) -> str:
        self.calls.append({"system": system, "prompt": prompt, "temperature": temperature})
        if temperature == OBSERVER_TEMPERATURE:
            return self.observer.pop(0)
        assert temperature == REFLECTOR_TEMPERATURE
        return self.reflector.pop(0)

    def observer_calls(self) -> list[dict[str, object]]:
        return [c for c in self.calls if c["temperature"] == OBSERVER_TEMPERATURE]

    def reflector_calls(self) -> list[dict[str, object]]:
        return [c for c in self.calls if c["temperature"] == REFLECTOR_TEMPERATURE]


def _long(n: int, word: str = "detail") -> list[str]:
    return [f"{word} {i} " + "lorem ipsum dolor sit amet " * 20 for i in range(n)]


# ---------------------------------------------------------------------------
# Thresholds and cursors
# ---------------------------------------------------------------------------


async def test_backlog_below_the_threshold_is_not_observed(tmp_path) -> None:
    om = _memory(tmp_path, message_tokens=30_000)
    model = _Model()

    assert await om.maybe_observe("cli:a", [_thread("cli:a", ["hi", "hello"])], model) is None
    assert model.calls == []


async def test_backlog_at_the_threshold_is_observed_and_cursors_advance(tmp_path) -> None:
    om = _memory(tmp_path, message_tokens=100)
    texts = _long(4)
    model = _Model(observer=[_observed("User shared four details", task="Primary: list details")])

    outcome = await om.maybe_observe("cli:a", [_thread("cli:a", texts)], model)

    assert outcome is not None and outcome.threads == ("cli:a",)
    snapshot = om.snapshot()
    obscured = text.obscure_thread_id("cli:a")
    assert snapshot.observations.startswith(f'<thread id="{obscured}">')
    assert "User shared four details" in snapshot.observations
    state = snapshot.state.threads["cli:a"]
    assert state.observed_count == 4
    assert state.observed_fingerprint == message_fingerprint(_session("cli:a", texts)[-1])
    assert state.current_task == "Primary: list details"
    assert snapshot.state.revision == 1
    # The Actor's view carries the task of its own session only.
    assert "Primary: list details" in (om.context_block("cli:a") or "")
    assert "Primary: list details" not in (om.context_block("cli:b") or "")


async def test_other_sessions_count_toward_the_threshold(tmp_path) -> None:
    small = _thread("cli:a", ["hi", "hello"])
    other = _thread("cli:b", _long(4, "other"))
    om = _memory(tmp_path, message_tokens=int(om_tokens(small, other) - 1))
    model = _Model(observer=[_observed("B covered many details")])

    outcome = await om.maybe_observe("cli:a", [small, other], model)

    assert outcome is not None
    assert "cli:b" in outcome.threads


async def test_an_idle_session_larger_than_the_threshold_still_triggers(tmp_path) -> None:
    """The Actor's view of other sessions is bounded; the threshold count is not."""
    small = _thread("cli:a", ["hi", "hello"])
    idle = _thread("cli:idle", _long(8, "idle"))
    om = _memory(tmp_path, message_tokens=int(tokens.count_messages(idle.messages) / 2))

    assert om.other_conversations("cli:a", [small, idle]) == ""
    outcome = await om.maybe_observe(
        "cli:a", [small, idle], _Model(observer=[_observed("idle session observed")]),
    )

    assert outcome is not None and outcome.threads == ("cli:idle",)


def om_tokens(own: PendingThread, other: PendingThread) -> float:
    om = ObservationalMemory(ObservationStore(Path("/nonexistent")), timezone=timezone.utc)
    return om.pending_tokens(own.key, [own, other])


async def test_threads_are_batched_and_observed_in_parallel(tmp_path) -> None:
    a = _thread("cli:a", _long(4, "alpha"), day=24)
    b = _thread("cli:b", _long(4, "beta"))
    size = tokens.count_messages(a.messages)
    # Selection stops once the threshold is reached, so it must take both threads.
    om = _memory(tmp_path, message_tokens=int(size) + 1, max_tokens_per_batch=int(size))
    started = 0
    both_started = asyncio.Event()

    async def model(*, system: str, prompt: str, temperature: float) -> str:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        # Each call waits for the other: they only finish if run concurrently.
        await asyncio.wait_for(both_started.wait(), timeout=2)
        key = "cli:a" if "alpha 0" in prompt else "cli:b"
        return (
            f'<thread id="{text.obscure_thread_id(key)}">\n'
            f"{_observed(key + ' observed')}\n</thread>"
        )

    outcome = await om.maybe_observe("cli:a", [a, b], model)

    assert outcome is not None and set(outcome.threads) == {"cli:a", "cli:b"}
    log = om.snapshot().observations
    # Sections follow the batch order: oldest thread first.
    assert log.index("cli:a observed") < log.index("cli:b observed")
    assert {k: s.observed_count for k, s in om.snapshot().state.threads.items()} == {
        "cli:a": 4, "cli:b": 4,
    }


async def test_a_thread_observed_meanwhile_is_not_observed_again(tmp_path) -> None:
    om = _memory(tmp_path, message_tokens=100)
    thread = _thread("cli:a", _long(4))
    model = _Model(observer=[_observed("first")])

    assert await om.maybe_observe("cli:a", [thread], model) is not None
    assert await om.maybe_observe("cli:a", [thread], model) is None
    assert len(model.observer_calls()) == 1


async def test_observed_count_resets_when_the_session_was_rewritten(tmp_path) -> None:
    om = _memory(tmp_path, message_tokens=100)
    texts = _long(4)
    await om.maybe_observe("cli:a", [_thread("cli:a", texts)], _Model(observer=[_observed("x")]))
    session = _session("cli:a", texts)

    def fingerprint_at(i: int) -> str | None:
        return message_fingerprint(session[i]) if 0 <= i < len(session) else None

    assert om.observed_count("cli:a", fingerprint_at) == 4
    session[3] = {**session[3], "timestamp": "2026-09-26T08:00:00+00:00"}
    assert om.observed_count("cli:a", fingerprint_at) == 0
    assert om.observed_count("cli:unknown", fingerprint_at) == 0


async def test_new_observations_for_the_same_thread_and_date_merge(tmp_path) -> None:
    om = _memory(tmp_path)
    await om.observe_now(_thread("cli:a", ["one", "two"]), _Model(observer=[_observed("first")]))
    await om.observe_now(
        _thread("cli:a", ["one", "two", "three", "four"], start=2),
        _Model(observer=[_observed("second")]),
    )

    log = om.snapshot().observations
    assert log.count("<thread id=") == 1
    assert log.count("Date: Sep 25, 2026") == 1
    assert log.index("first") < log.index("second")


async def test_the_observer_sees_existing_observations(tmp_path) -> None:
    om = _memory(tmp_path)
    await om.observe_now(_thread("cli:a", ["one", "two"]), _Model(observer=[_observed("first")]))
    model = _Model(observer=[_observed("second")])

    await om.observe_now(_thread("cli:b", ["three", "four"]), model)

    prompt = str(model.calls[0]["prompt"])
    assert prompt.startswith("## Previous Observations")
    assert "first" in prompt


async def test_empty_observer_output_still_advances_the_cursor(tmp_path) -> None:
    om = _memory(tmp_path)

    outcome = await om.observe_now(_thread("cli:a", ["ok", "sure"]), _Model(observer=[""]))

    assert outcome is not None
    assert om.snapshot().observations == ""
    assert om.thread_state("cli:a").observed_count == 2  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_a_cycle_based_on_stale_state_is_discarded(tmp_path) -> None:
    om = _memory(tmp_path)
    other_writer = ObservationStore(tmp_path)

    async def model(*, system: str, prompt: str, temperature: float) -> str:
        # Another process commits while the Observer is running.
        other_writer.commit(
            other_writer.read().state.revision,
            lambda current: replace(current, observations="written elsewhere"),
        )
        return _observed("stale")

    assert await om.observe_now(_thread("cli:a", ["one", "two"]), model) is None
    snapshot = om.snapshot()
    assert snapshot.observations == "written elsewhere"
    assert "cli:a" not in snapshot.state.threads


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


async def test_a_log_over_the_threshold_is_reflected(tmp_path) -> None:
    om = _memory(tmp_path, message_tokens=10, observation_tokens=60)
    big = "\n".join(f"* 🔴 (10:{i:02d}) fact number {i} about the project" for i in range(20))
    model = _Model(
        observer=[f"<observations>\nDate: Sep 25, 2026\n{big}\n</observations>"],
        reflector=["<observations>\nDate: Sep 25, 2026\n* 🔴 (10:00) project facts\n</observations>"],
    )

    outcome = await om.maybe_observe("cli:a", [_thread("cli:a", _long(2))], model)

    assert outcome is not None and outcome.reflected
    assert len(model.reflector_calls()) == 1
    snapshot = om.snapshot()
    assert snapshot.observations == "Date: Sep 25, 2026\n* 🔴 (10:00) project facts"
    assert snapshot.state.generation == 1
    assert snapshot.state.threads["cli:a"].observed_count == 2


async def test_reflection_that_does_not_compress_retries_once(tmp_path) -> None:
    om = _memory(tmp_path, observation_tokens=20)
    long_log = "Date: Sep 25, 2026\n" + "\n".join(
        f"* 🔴 (10:{i:02d}) fact {i} with plenty of words to exceed the target" for i in range(10)
    )
    om.store.commit(0, lambda current: replace(current, observations=long_log))
    model = _Model(reflector=[
        f"<observations>\n{long_log}\n</observations>",
        "<observations>\nDate: Sep 25, 2026\n* 🔴 (10:00) facts\n</observations>",
    ])

    assert await om.reflect(model) is True

    first, retry = model.reflector_calls()
    assert prompts.COMPRESSION_RETRY_PROMPT not in str(first["prompt"])
    assert prompts.COMPRESSION_RETRY_PROMPT in str(retry["prompt"])
    assert om.snapshot().observations == "Date: Sep 25, 2026\n* 🔴 (10:00) facts"


async def test_an_empty_reflection_keeps_the_log(tmp_path) -> None:
    om = _memory(tmp_path)
    om.store.commit(0, lambda current: replace(current, observations="Date: x\n* keep me"))

    assert await om.reflect(_Model(reflector=[""])) is False

    snapshot = om.snapshot()
    assert snapshot.observations == "Date: x\n* keep me"
    assert snapshot.state.generation == 0


async def test_manual_reflection_passes_guidance_and_skips_an_empty_log(tmp_path) -> None:
    om = _memory(tmp_path)
    model = _Model(reflector=["<observations>\n* short\n</observations>"])

    assert await om.reflect(model, "drop greetings") is False
    assert model.calls == []

    om.store.commit(0, lambda current: replace(current, observations="* hello\n* fact"))
    assert await om.reflect(model, "drop greetings") is True
    assert "drop greetings" in str(model.calls[0]["prompt"])


# ---------------------------------------------------------------------------
# Transient observations
# ---------------------------------------------------------------------------


async def test_transient_observations_stay_with_their_session(tmp_path) -> None:
    om = _memory(tmp_path)
    om.store.commit(0, lambda current: replace(current, observations="Date: x\n* durable fact"))
    model = _Model(observer=[_observed("transient fact")])

    outcome = await om.observe_now(_thread("cli:a", ["one", "two"]), model, persist=False)

    assert outcome is not None
    assert "durable fact" in str(model.calls[0]["prompt"])
    snapshot = om.snapshot()
    assert "transient fact" not in snapshot.observations
    assert "cli:a" not in snapshot.state.threads
    assert "transient fact" in om.observations_for("cli:a")
    assert "durable fact" in om.observations_for("cli:a")
    assert "transient fact" not in om.observations_for("cli:b")

    om.forget_ephemeral("cli:a")
    assert "transient fact" not in om.observations_for("cli:a")


async def test_private_observation_never_shows_the_workspace_log(tmp_path) -> None:
    om = _memory(tmp_path)
    om.store.commit(0, lambda current: replace(current, observations="Date: x\n* secret"))
    model = _Model(observer=[_observed("private fact")])

    await om.observe_now(_thread("ws:p", ["one", "two"]), model, persist=True, durable=False)

    assert "secret" not in str(model.calls[0]["prompt"])
    assert "private fact" not in om.snapshot().observations
    block = om.context_block("ws:p", durable=False) or ""
    assert "private fact" in block and "secret" not in block
    assert om.context_block("ws:other", durable=False) is None


async def test_retiring_a_thread_keeps_its_observations(tmp_path) -> None:
    om = _memory(tmp_path)
    await om.observe_now(
        _thread("cli:a", ["one", "two"]), _Model(observer=[_observed("kept", task="Primary: x")]),
    )

    om.retire_thread("cli:a")

    assert om.thread_state("cli:a") is None
    assert "kept" in om.snapshot().observations
    assert "Primary: x" not in (om.context_block("cli:a") or "")


# ---------------------------------------------------------------------------
# What the Actor sees
# ---------------------------------------------------------------------------


def test_no_observations_means_no_memory_block(tmp_path) -> None:
    assert _memory(tmp_path).context_block("cli:a") is None


def test_other_conversations_drop_the_oldest_sessions_when_too_large(tmp_path) -> None:
    old = _thread("cli:old", _long(4, "ancient"), day=1)
    new = _thread("cli:new", _long(4, "recent"), day=24)
    budget = tokens.count_string(
        text.format_other_conversations(
            {"cli:new": list(new.messages)}, "cli:me", timezone.utc, text.obscure_thread_id,
        )
    )
    om = _memory(tmp_path, message_tokens=budget)

    block = om.other_conversations("cli:me", [old, new])

    assert "recent 0" in block
    assert "ancient 0" not in block


def test_relative_dates_are_added_for_the_actor(tmp_path) -> None:
    om = _memory(tmp_path)
    yesterday = NOW - timedelta(days=1)
    log = f"Date: {yesterday:%b} {yesterday.day}, {yesterday.year}\n* 🔴 (09:00) fact"
    om.store.commit(0, lambda current: replace(current, observations=log))

    block = om.context_block("cli:a") or ""

    assert "yesterday" in block
    assert "<observations>" in block


@pytest.mark.parametrize("key", ["cli:a", "telegram:123:thread"])
def test_thread_ids_are_obscured_consistently(key: str) -> None:
    obscured = text.obscure_thread_id(key)
    assert obscured == text.obscure_thread_id(key)
    assert key not in obscured
