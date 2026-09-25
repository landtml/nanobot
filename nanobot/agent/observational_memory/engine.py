"""Observational Memory engine: when to observe and reflect, and what the Actor sees.

This is the Python counterpart of Mastra's 1.1.0 ``ObservationalMemory``
processor in resource scope, the configuration Mastra benchmarked on
LongMemEval. One observation log is shared by every conversation in the
workspace; each nanobot session is a thread.

* **Observe.** After a turn, unobserved messages of the current session plus
  the formatted unobserved context of other sessions are counted. At
  ``message_tokens`` (30k) the largest threads are selected up to that budget,
  batched by ``max_tokens_per_batch`` (10k) and sent to the Observer in
  parallel. Results are appended per thread (``<thread id="…">`` sections,
  merged when thread and date match) and each thread's cursor advances.
* **Reflect.** When the log exceeds ``observation_tokens`` (40k) the Reflector
  rewrites it; if the rewrite is not smaller than the threshold it retries
  once with the compression prompt.
* **Recall.** The Actor gets the optimized, date-annotated log, the current
  session's task and suggested response, and the unobserved tail of other
  sessions, plus the upstream continuation reminder.

Deliberate differences from upstream, all outside what LongMemEval exercises:
thread ids are obscured in the Observer prompt too (upstream obscured them only
in the Actor's context); tool calls and results are rendered together and
capped (``TOOL_PAYLOAD_MAX_CHARS``); and observation runs after a turn rather
than between tool-loop steps, with context-window pressure handled by
:meth:`ObservationalMemory.observe_now`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, tzinfo
from typing import Protocol

from loguru import logger

from nanobot.agent.observational_memory import prompts, text, tokens
from nanobot.agent.observational_memory.store import (
    MemoryState,
    ObservationStore,
    Snapshot,
    StaleStateError,
    ThreadState,
)
from nanobot.agent.observational_memory.text import ObserverMessage, ObserverResult

OBSERVER_TEMPERATURE = 0.3
REFLECTOR_TEMPERATURE = 0.0


@dataclass(frozen=True, slots=True)
class ObservationalMemoryConfig:
    message_tokens: int = 30_000
    """Observe once this many unobserved tokens have accumulated."""
    observation_tokens: int = 40_000
    """Reflect once the observation log exceeds this many tokens."""
    max_tokens_per_batch: int = 10_000
    """Split one observation cycle into parallel Observer calls of about this size."""


class ModelCall(Protocol):
    """One Observer or Reflector completion: system prompt, user prompt, temperature."""

    def __call__(self, *, system: str, prompt: str, temperature: float) -> Awaitable[str]: ...


@dataclass(frozen=True, slots=True)
class PendingThread:
    """A session's messages that are not yet observed."""

    key: str
    messages: tuple[ObserverMessage, ...]
    end: int
    """Session message count once these messages are observed."""
    fingerprint: str
    """Fingerprint of the session message at ``end - 1``."""


@dataclass(frozen=True, slots=True)
class BatchPlan:
    thread_order: list[str]
    batches: list[list[str]]


@dataclass(frozen=True, slots=True)
class ObservationOutcome:
    threads: tuple[str, ...]
    observation_tokens: int
    reflected: bool


def select_and_batch(
    messages_by_thread: Mapping[str, Sequence[ObserverMessage]],
    *,
    message_tokens: int,
    max_tokens_per_batch: int,
) -> BatchPlan | None:
    """Pick threads for one cycle, largest first up to the threshold, then batch them.

    Mirrors the selection and batching in ``doResourceScopedObservation``:
    threads are chosen by size until ``message_tokens`` is reached, ordered by
    their oldest message, and packed into batches of ``max_tokens_per_batch``
    (a batch always takes at least one thread).
    """
    thread_tokens = {
        thread_id: sum(tokens.count_message(message) for message in messages)
        for thread_id, messages in messages_by_thread.items()
    }
    by_size = sorted(messages_by_thread, key=lambda thread_id: -thread_tokens[thread_id])
    selected: list[str] = []
    accumulated = 0.0
    for thread_id in by_size:
        if accumulated >= message_tokens:
            break
        selected.append(thread_id)
        accumulated += thread_tokens[thread_id]
    if not selected:
        return None

    def oldest(thread_id: str) -> float:
        stamps = [
            message["created_at"].timestamp() * 1000
            if message["created_at"] is not None
            else datetime.now().timestamp() * 1000
            for message in messages_by_thread[thread_id]
        ]
        return min(stamps) if stamps else float("inf")

    order = sorted(selected, key=oldest)
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0.0
    for thread_id in order:
        if not messages_by_thread[thread_id]:
            continue
        size = thread_tokens[thread_id]
        if current_tokens + size > max_tokens_per_batch and current:
            batches.append(current)
            current = []
            current_tokens = 0.0
        current.append(thread_id)
        current_tokens += size
    if current:
        batches.append(current)
    return BatchPlan(thread_order=order, batches=batches)


class ObservationalMemory:
    """Workspace-wide observational memory shared by all sessions."""

    def __init__(
        self,
        store: ObservationStore,
        *,
        config: ObservationalMemoryConfig | None = None,
        timezone: tzinfo,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.config = config or ObservationalMemoryConfig()
        self.timezone = timezone
        self._clock = clock or (lambda: datetime.now(self.timezone))
        self._lock = asyncio.Lock()
        # Observations made for sessions that must not write memory (temporary
        # chats); visible to that session's later requests only.
        self._ephemeral: dict[str, str] = {}

    # -- reading -------------------------------------------------------------

    def now(self) -> datetime:
        return self._clock()

    def snapshot(self) -> Snapshot:
        return self.store.read()

    def thread_state(self, key: str) -> ThreadState | None:
        return self.store.read().state.threads.get(key)

    def observed_count(self, key: str, fingerprint_at: Callable[[int], str | None]) -> int:
        """How many leading messages of session *key* are observed per the store.

        ``fingerprint_at(i)`` returns the fingerprint of the session's message
        ``i``; a mismatch means the session was cleared or rewritten since, and
        the stored cursor no longer applies.
        """
        thread = self.thread_state(key)
        if thread is None or thread.observed_count <= 0:
            return 0
        if fingerprint_at(thread.observed_count - 1) != thread.observed_fingerprint:
            return 0
        return thread.observed_count

    def observations_for(self, key: str | None) -> str:
        observations = self.store.read().observations
        extra = self._ephemeral.get(key or "")
        if extra:
            return f"{observations}\n\n{extra}" if observations else extra
        return observations

    def context_block(
        self,
        key: str | None,
        *,
        other_conversations: str | None = None,
    ) -> str | None:
        """The Actor's memory block for session *key*, or None before any observation."""
        observations = self.observations_for(key)
        if not observations:
            return None
        thread = self.thread_state(key) if key else None
        return prompts.observations_context(
            text.render_observations(observations, self.now(), self.timezone),
            current_task=thread.current_task if thread else None,
            suggested_response=thread.suggested_response if thread else None,
            other_conversations=other_conversations or None,
        )

    def other_conversations(self, key: str | None, pending: Sequence[PendingThread]) -> str:
        """Unobserved tails of other sessions, as ``<other-conversation>`` blocks.

        Bounded to ``message_tokens``: in normal operation observation keeps it
        far smaller, and when observation keeps failing the oldest sessions are
        dropped first instead of letting the block grow without limit.
        """
        others = {thread.key: list(thread.messages) for thread in pending if thread.key != key}
        while others:
            block = text.format_other_conversations(
                others, key or "", self.timezone, text.obscure_thread_id,
            )
            if tokens.count_string(block) <= self.config.message_tokens:
                return block
            oldest = min(
                others,
                key=lambda k: min(
                    (m["created_at"].timestamp() for m in others[k] if m["created_at"]),
                    default=0.0,
                ),
            )
            del others[oldest]
        return ""

    def pending_tokens(self, key: str | None, pending: Sequence[PendingThread]) -> float:
        """Unobserved tokens that count toward the observation threshold."""
        current = next((t for t in pending if t.key == key), None)
        own = tokens.count_messages(current.messages) if current else 0
        return own + tokens.count_string(self.other_conversations(key, pending))

    # -- observing -------------------------------------------------------------

    async def maybe_observe(
        self,
        key: str | None,
        pending: Sequence[PendingThread],
        model: ModelCall,
    ) -> ObservationOutcome | None:
        """Observe when the unobserved backlog reaches the threshold."""
        if self.pending_tokens(key, pending) < self.config.message_tokens:
            return None
        async with self._lock:
            # Another cycle may have finished while this one waited.
            live = [t for t in pending if self._still_pending(t)]
            if self.pending_tokens(key, live) < self.config.message_tokens:
                return None
            plan = select_and_batch(
                {t.key: t.messages for t in live},
                message_tokens=self.config.message_tokens,
                max_tokens_per_batch=self.config.max_tokens_per_batch,
            )
            if plan is None:
                return None
            by_key = {t.key: t for t in live}
            return await self._observe(
                [[by_key[k] for k in batch] for batch in plan.batches],
                model,
                persist=True,
            )

    async def observe_now(
        self,
        thread: PendingThread,
        model: ModelCall,
        *,
        persist: bool = True,
    ) -> str | None:
        """Observe one session immediately (context pressure or ``/compact``).

        Returns the new observations for this thread, or None when the Observer
        produced nothing. With ``persist=False`` nothing is written: the result
        only extends that session's own view of memory.
        """
        if not thread.messages:
            return None
        async with self._lock:
            outcome = await self._observe([[thread]], model, persist=persist)
        if outcome is None:
            return None
        if not persist:
            return self._ephemeral.get(thread.key)
        return self.store.read().observations or None

    def _still_pending(self, thread: PendingThread) -> bool:
        state = self.thread_state(thread.key)
        return state is None or state.observed_count < thread.end

    async def _observe(
        self,
        batches: list[list[PendingThread]],
        model: ModelCall,
        *,
        persist: bool,
    ) -> ObservationOutcome | None:
        snapshot = self.store.read()
        existing = self.observations_for(batches[0][0].key if not persist else None)
        if persist:
            existing = snapshot.observations
        results = await asyncio.gather(
            *(self._call_observer(existing, batch, model) for batch in batches)
        )
        merged: dict[str, ObserverResult] = {}
        for result in results:
            merged.update(result)
        threads = [thread for batch in batches for thread in batch]
        now = self.now()

        observations = existing
        for thread in threads:
            result = merged.get(thread.key)
            if result is None or not result.observations:
                continue
            section = text.wrap_with_thread_tag(text.obscure_thread_id(thread.key), result.observations)
            observations = text.replace_or_append_thread_section(observations, section)

        if not persist:
            key = threads[0].key
            if observations != existing:
                self._ephemeral[key] = observations
            return ObservationOutcome(threads=(key,), observation_tokens=0, reflected=False)

        def apply(current: Snapshot) -> Snapshot:
            thread_states = dict(current.state.threads)
            for thread in threads:
                result = merged.get(thread.key, ObserverResult(observations=""))
                previous = thread_states.get(thread.key, ThreadState())
                thread_states[thread.key] = ThreadState(
                    observed_count=thread.end,
                    observed_fingerprint=thread.fingerprint,
                    current_task=result.current_task or previous.current_task,
                    suggested_response=result.suggested_response or previous.suggested_response,
                    last_observed_at=now.isoformat(),
                )
            return Snapshot(
                observations=observations,
                state=replace(current.state, threads=thread_states, last_observed_at=now.isoformat()),
            )

        try:
            committed = self.store.commit(snapshot.state.revision, apply)
        except StaleStateError as exc:
            logger.info("Observational memory: discarding observation cycle ({})", exc)
            return None
        observation_tokens = tokens.count_string(committed.observations)
        logger.info(
            "Observational memory: observed {} session(s); log is {} tokens",
            len(threads),
            observation_tokens,
        )
        reflected = False
        if observation_tokens > self.config.observation_tokens:
            reflected = await self._reflect(committed, model) is not None
        return ObservationOutcome(
            threads=tuple(t.key for t in threads),
            observation_tokens=observation_tokens,
            reflected=reflected,
        )

    async def _call_observer(
        self,
        existing: str,
        batch: list[PendingThread],
        model: ModelCall,
    ) -> dict[str, ObserverResult]:
        ids = {text.obscure_thread_id(thread.key): thread.key for thread in batch}
        order = list(ids)
        formatted = text.format_multi_thread_messages(
            {obscured: batch[i].messages for i, obscured in enumerate(order)},
            order,
            self.timezone,
        )
        output = await model(
            system=prompts.observer_system_prompt(multi_thread=True),
            prompt=prompts.multi_thread_observer_prompt(existing or None, formatted, len(order)),
            temperature=OBSERVER_TEMPERATURE,
        )
        parsed = text.parse_multi_thread_output(output)
        results: dict[str, ObserverResult] = {}
        for obscured, key in ids.items():
            result = parsed.get(obscured)
            if result is None and len(ids) == 1 and not parsed:
                # A single-thread batch answered without thread tags: accept it.
                result = text.parse_observer_output(output)
            results[key] = result or ObserverResult(observations="")
        return results

    # -- reflecting ------------------------------------------------------------

    async def reflect(self, model: ModelCall, guidance: str | None = None) -> bool:
        """Reflect now, regardless of the threshold."""
        async with self._lock:
            snapshot = self.store.read()
            if not snapshot.observations:
                return False
            return await self._reflect(snapshot, model, guidance) is not None

    async def _reflect(
        self,
        snapshot: Snapshot,
        model: ModelCall,
        guidance: str | None = None,
    ) -> Snapshot | None:
        target = self.config.observation_tokens
        system = prompts.reflector_system_prompt()
        output = await model(
            system=system,
            prompt=prompts.reflector_prompt(snapshot.observations, guidance),
            temperature=REFLECTOR_TEMPERATURE,
        )
        result = text.parse_reflector_output(output)
        if tokens.count_string(result.observations) >= target:
            output = await model(
                system=system,
                prompt=prompts.reflector_prompt(snapshot.observations, guidance, compression_retry=True),
                temperature=REFLECTOR_TEMPERATURE,
            )
            result = text.parse_reflector_output(output)
        if not result.observations:
            logger.warning("Observational memory: Reflector returned no observations; keeping the log")
            return None
        now = self.now().isoformat()

        def apply(current: Snapshot) -> Snapshot:
            state: MemoryState = replace(
                current.state,
                generation=current.state.generation + 1,
                last_reflected_at=now,
            )
            return Snapshot(observations=result.observations, state=state)

        try:
            committed = self.store.commit(snapshot.state.revision, apply)
        except StaleStateError as exc:
            logger.info("Observational memory: discarding reflection ({})", exc)
            return None
        logger.info(
            "Observational memory: reflected {} -> {} tokens (generation {})",
            tokens.count_string(snapshot.observations),
            tokens.count_string(committed.observations),
            committed.state.generation,
        )
        return committed

    def forget_ephemeral(self, key: str) -> None:
        self._ephemeral.pop(key, None)
