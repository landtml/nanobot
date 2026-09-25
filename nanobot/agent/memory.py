"""nanobot's memory: Observational Memory wired into sessions, providers and storage.

:class:`Memory` is the one place the agent loop, context builder, commands and
SDK talk to for memory. It adapts :mod:`nanobot.agent.observational_memory`
(the port of Mastra's benchmarked 1.1.0 release) to nanobot:

* **Threads are sessions.** Every persisted user session in the workspace
  shares one observation log. A session's observed prefix drops out of replay
  through ``Session.last_archived``; the engine's per-session cursor is applied
  when that session next runs, so observing one session never writes another
  session's file.
* **After a turn** the backlog is checked and, at the threshold, observed in
  the background; the reply is never delayed.
* **Under context pressure** the governor calls :meth:`Memory.compactor`, which
  observes the current session synchronously so the request can shrink.
* **Every change to the log is a git commit** in the workspace memory store, so
  ``/memory-log`` can show it and ``/memory-restore`` can undo it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from nanobot.agent.observational_memory import (
    ObservationalMemory,
    ObservationalMemoryConfig,
    ObservationOutcome,
    ObservationStore,
    PendingThread,
)
from nanobot.agent.observational_memory import prompts as om_prompts
from nanobot.agent.observational_memory import tokens as om_tokens
from nanobot.agent.observational_memory.store import Snapshot, StaleStateError
from nanobot.agent.observational_memory.text import ObserverMessage
from nanobot.llm_usage.context import llm_usage_source
from nanobot.runtime_context import public_history_message
from nanobot.session.history_visibility import HIDDEN_HISTORY_META
from nanobot.session.keys import is_internal_session
from nanobot.session.summary import is_summary_checkpoint
from nanobot.utils.gitstore import GitStore, GitStoreError
from nanobot.utils.helpers import strip_think

if TYPE_CHECKING:
    from nanobot.config.schema import MemoryConfig
    from nanobot.providers.base import ProviderConversationState
    from nanobot.session.manager import Session, SessionManager
    from nanobot.utils.llm_runtime import LLMRuntime

MEMORY_TRACKED_FILES = ["SOUL.md", "USER.md", "memory/observations.md"]
MEMORY_COMMIT_PREFIX = "memory:"
# What a successful compaction reports when the Observer found nothing to keep.
NOTHING_OBSERVED = "(nothing observed)"
_LEGACY_MEMORY_FILE = "memory/MEMORY.md"
_IMAGE_BREADCRUMB_RE = re.compile(r"\[image: [^\]]+\]")


# ---------------------------------------------------------------------------
# Session messages as the Observer sees them
# ---------------------------------------------------------------------------


def message_fingerprint(message: Mapping[str, Any]) -> str:
    """Identify a persisted message well enough to detect a cleared or rewritten session."""
    return f"{message.get('timestamp', '')}|{message.get('role', '')}"


def _parse_timestamp(value: object, tz: tzinfo) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    # Persisted timestamps are naive local time (datetime.now().isoformat()).
    return parsed.astimezone(tz) if parsed.tzinfo else parsed.astimezone().astimezone(tz)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        blocks = [
            cast(dict[str, Any], block)
            for block in cast(list[object], content)
            if isinstance(block, dict)
        ]
        text = "\n".join(
            str(block.get("text", "")) for block in blocks if block.get("type") == "text"
        )
    else:
        text = ""
    # Local media paths must not become memory (see .agent/gotchas.md).
    return _IMAGE_BREADCRUMB_RE.sub("[image]", text)


def _tool_arguments(call: Mapping[str, Any]) -> tuple[str, Any]:
    function = call.get("function")
    if isinstance(function, Mapping):
        function_data = cast(Mapping[str, Any], function)
        name = str(function_data.get("name") or "")
        raw = function_data.get("arguments")
    else:
        name = str(call.get("name") or "")
        raw = call.get("arguments")
    if isinstance(raw, str):
        import json

        try:
            return name, json.loads(raw)
        except ValueError:
            return name, raw
    return name, raw


def observer_messages(
    key: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    start: int,
    tz: tzinfo,
    default_time: datetime | None = None,
) -> list[ObserverMessage]:
    """Convert OpenAI-style session messages to Observer messages.

    An assistant message and the tool results answering its calls become one
    message with ``tool-exchange`` parts. Command replies, summary markers and
    hidden bookkeeping messages are skipped; runtime context is removed.
    """
    results: dict[str, Any] = {
        str(message.get("tool_call_id")): message.get("content")
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    converted: list[ObserverMessage] = []
    for offset, raw in enumerate(messages):
        role = raw.get("role")
        if role not in ("user", "assistant"):
            continue
        if raw.get("_command") or is_summary_checkpoint(raw) or raw.get(HIDDEN_HISTORY_META):
            continue
        message = public_history_message(raw)
        created_at = _parse_timestamp(message.get("timestamp"), tz) or default_time
        text = _content_text(message.get("content"))
        if role == "assistant":
            text = strip_think(text)
        parts: list[dict[str, Any]] = []
        if text.strip():
            parts.append({"type": "text", "text": text})
        for call_value in cast(list[object], message.get("tool_calls") or []):
            if not isinstance(call_value, dict):
                continue
            call = cast(dict[str, Any], call_value)
            name, args = _tool_arguments(call)
            part: dict[str, Any] = {"type": "tool-exchange", "toolName": name, "args": args}
            call_id = str(call.get("id") or "")
            if call_id in results:
                part["result"] = _content_text(results[call_id])
            parts.append(part)
        if not parts:
            continue
        converted.append({
            "id": f"{key}#{start + offset}",
            "role": role,
            "created_at": created_at,
            "content": {"parts": parts},
        })
    return converted


# ---------------------------------------------------------------------------
# Memory service
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    observation_tokens: int
    observation_threshold: int
    pending_tokens: int
    message_threshold: int
    generation: int
    last_observed_at: str | None
    last_reflected_at: str | None
    sessions: int


def _zone(name: str | None) -> tzinfo:
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Unknown timezone {!r} for memory; using the system timezone", name)
    local = datetime.now().astimezone().tzinfo
    assert local is not None
    return local


class Memory:
    """Observational Memory for one workspace."""

    def __init__(
        self,
        workspace: Path,
        sessions: SessionManager,
        *,
        runtime: Callable[[], LLMRuntime],
        config: MemoryConfig | None = None,
        timezone: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.sessions = sessions
        self._runtime = runtime
        self.timezone = _zone(timezone)
        om_config = ObservationalMemoryConfig()
        if config is not None:
            om_config = ObservationalMemoryConfig(
                message_tokens=config.message_tokens,
                observation_tokens=config.observation_tokens,
                max_tokens_per_batch=config.max_tokens_per_batch,
            )
        self.om = ObservationalMemory(
            ObservationStore(workspace),
            config=om_config,
            timezone=self.timezone,
        )
        self.git = GitStore(workspace, tracked_files=MEMORY_TRACKED_FILES)
        self._checking = False
        self._migrate_legacy_memory()
        self.om.store.activate(self.om.now())

    @property
    def config(self) -> ObservationalMemoryConfig:
        return self.om.config

    # -- model -----------------------------------------------------------------

    async def _complete(self, *, system: str, prompt: str, temperature: float) -> str:
        runtime = self._runtime()
        with llm_usage_source("memory"):
            response = await runtime.provider.chat_stream_with_retry(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                model=runtime.model,
                temperature=temperature,
                max_tokens=runtime.generation.max_tokens,
                reasoning_effort=runtime.generation.reasoning_effort,
            )
        if response.finish_reason == "error":
            raise RuntimeError(f"memory model call failed: {(response.content or '')[:200]}")
        return strip_think(response.content or "")

    # -- threads ---------------------------------------------------------------

    def observed_count(self, session: Session) -> int:
        """Messages of *session* already covered by the observation log."""
        messages = session.messages

        def fingerprint_at(index: int) -> str | None:
            return message_fingerprint(messages[index]) if 0 <= index < len(messages) else None

        return max(session.last_archived, self.om.observed_count(session.key, fingerprint_at))

    def pending_thread(self, session: Session) -> PendingThread | None:
        """The unobserved tail of *session*, or None when there is nothing new."""
        start = min(self.observed_count(session), len(session.messages))
        tail = session.messages[start:]
        if not tail:
            return None
        messages = observer_messages(session.key, tail, start=start, tz=self.timezone)
        if not messages:
            return None
        return PendingThread(
            key=session.key,
            messages=tuple(messages),
            end=len(session.messages),
            fingerprint=message_fingerprint(session.messages[-1]),
        )

    def _participates(self, key: str) -> bool:
        return not is_internal_session(key)

    def pending_threads(self, current: Session | None = None) -> list[PendingThread]:
        """Unobserved tails of every session that belongs to this workspace's memory.

        A session joins once it changes after memory was activated, so an
        upgrade does not bill the model for years of dormant chats.
        """
        activated_at = self.om.snapshot().state.activated_at
        since = datetime.fromisoformat(activated_at).timestamp() if activated_at else 0.0
        threads: list[PendingThread] = []
        for key in self.sessions.session_keys_modified_since(since):
            if not self._participates(key) or (current is not None and key == current.key):
                continue
            session = self.sessions.get_cached(key) or self.sessions.read_session_snapshot(key)
            if session is None:
                continue
            if (thread := self.pending_thread(session)) is not None:
                threads.append(thread)
        if current is not None and current.policy.persist and self._participates(current.key):
            if (thread := self.pending_thread(current)) is not None:
                threads.append(thread)
        return threads

    # -- turn integration --------------------------------------------------------

    def restore(self, session: Session) -> bool:
        """Drop messages observed since *session* last ran from its replay."""
        if not session.policy.persist:
            return False
        observed = self.observed_count(session)
        if observed <= session.last_archived:
            return False
        session.last_archived = observed
        # A resumable provider conversation still holds the observed messages.
        session.provider_state = None
        return True

    def system_prompt_block(self, session_key: str | None, *, durable: bool = True) -> str | None:
        """The memory section of the system prompt for *session_key*.

        ``durable=False`` is for private sessions: they see only what their own
        run observed under context pressure, never the workspace's memory.
        """
        if not durable:
            return self.om.context_block(session_key, durable=False)
        session = self.sessions.get_cached(session_key) if session_key else None
        other = self.om.other_conversations(session_key, self.pending_threads(session))
        return self.om.context_block(session_key, other_conversations=other)

    def history_prefix(
        self, session_key: str | None, *, durable: bool = True,
    ) -> list[dict[str, Any]]:
        """The upstream continuation reminder, placed before replayed history."""
        if not self.om.observations_for(session_key, durable=durable):
            return []
        from nanobot.agent.context import MEMORY_PREFIX_META

        return [{
            "role": "user",
            "content": om_prompts.CONTINUATION_REMINDER,
            "_meta": {MEMORY_PREFIX_META: True},
        }]

    async def after_turn(self, session_key: str) -> ObservationOutcome | None:
        """Observe the workspace backlog if it reached the threshold. Never raises.

        Checks do not queue: while one is running, later turns skip theirs and
        the next turn after it finishes looks at the whole backlog again.
        """
        if self._checking:
            return None
        self._checking = True
        try:
            session = self.sessions.get_cached(session_key)
            outcome = await self.om.maybe_observe(
                session_key, self.pending_threads(session), self._complete,
            )
        except Exception:
            logger.exception("Observational memory: background observation failed")
            return None
        finally:
            self._checking = False
        if outcome is not None:
            self._commit_git(
                f"{MEMORY_COMMIT_PREFIX} observe {len(outcome.threads)} session(s)"
                + (" and reflect" if outcome.reflected else "")
            )
        return outcome

    def compactor(
        self,
        session: Session | None,
        *,
        session_key: str,
        history_length: int,
        persist: bool,
        durable: bool = True,
    ) -> Callable[..., Any]:
        """A ``consolidate_history`` callback for the context governor.

        The first compaction in a turn observes the session's unobserved
        messages (with their real timestamps) plus the turn so far; later
        compactions in the same turn observe only what was added since.
        ``persist=False`` keeps new observations in the run's transient overlay;
        ``durable=False`` additionally hides the workspace log (private runs).
        """
        observed_session = False

        async def compact(
            accepted_messages: list[dict[str, Any]],
            previous_summary: str | None,
            **_: Any,
        ) -> str | None:
            nonlocal observed_session
            now = self.om.now()
            if not observed_session:
                base = self.pending_thread(session) if session is not None else None
                in_flight = accepted_messages[1 + history_length:]
            else:
                base = None
                in_flight = [
                    m for m in accepted_messages[1:]
                    if not (m.get("role") == "user" and _is_continuation(m))
                ]
            messages = [
                *(base.messages if base is not None else ()),
                *observer_messages(session_key, in_flight, start=10**9, tz=self.timezone,
                                   default_time=now),
            ]
            if not messages:
                return None
            thread = PendingThread(
                key=session_key,
                messages=tuple(messages),
                end=base.end if base is not None else 0,
                fingerprint=base.fingerprint if base is not None else "",
            )
            outcome = await self.om.observe_now(
                thread, self._complete, persist=persist, durable=durable,
            )
            observed_session = True
            if outcome is None:
                return None
            if persist:
                self._commit_git(f"{MEMORY_COMMIT_PREFIX} observe under context pressure")
            if session is None:
                # Runs without a session (subagents) render the summary themselves
                # and should see only what this run observed, not the whole log.
                return self.om.ephemeral_observations(session_key) or NOTHING_OBSERVED
            # Sessions re-read memory when their prompt is rebuilt; the text only
            # has to tell the governor that compaction succeeded.
            return self.om.observations_for(session_key, durable=durable) or NOTHING_OBSERVED

        return compact

    async def provider_compactor_adapter(
        self,
        compact: Callable[..., Any],
        state: ProviderConversationState,
        fallback_messages: list[dict[str, Any]],
        previous_summary: str | None,
        **kwargs: Any,
    ) -> str | None:
        """Adapt :meth:`compactor` to the provider-native compaction callback."""
        return await compact(fallback_messages, previous_summary, **kwargs)

    async def compact(self, session: Session) -> bool:
        """Observe *session* now and drop the observed messages from its replay."""
        thread = self.pending_thread(session)
        if thread is None:
            return False
        if await self.om.observe_now(thread, self._complete) is None:
            return False
        session.last_archived = thread.end
        session.provider_state = None
        self.sessions.save(session)
        self._commit_git(f"{MEMORY_COMMIT_PREFIX} compact one session")
        return True

    async def archive(self, snapshot: Session) -> None:
        """Observe what is left of a session that is being reset (``/new``). Never raises."""
        try:
            thread = self.pending_thread(snapshot)
            if thread is not None and await self.om.observe_now(thread, self._complete) is not None:
                self._commit_git(f"{MEMORY_COMMIT_PREFIX} observe a finished session")
            self.om.retire_thread(snapshot.key)
        except Exception:
            logger.exception("Observational memory: archiving {} failed", snapshot.key)

    async def reflect(self, guidance: str | None = None) -> bool:
        reflected = await self.om.reflect(self._complete, guidance)
        if reflected:
            self._commit_git(f"{MEMORY_COMMIT_PREFIX} reflect")
        return reflected

    def forget_session(self, session_key: str) -> None:
        self.om.forget_ephemeral(session_key)

    # -- reading ---------------------------------------------------------------

    def observations(self) -> str:
        return self.om.snapshot().observations

    def replace_observations(self, observations: str) -> None:
        """Overwrite the observation log (SDK and restore paths)."""
        snapshot = self.om.snapshot()
        self.om.store.commit(
            snapshot.state.revision,
            lambda current: replace(current, observations=observations.strip()),
        )
        self._commit_git(f"{MEMORY_COMMIT_PREFIX} replace observations")

    def status(self, session: Session | None = None) -> MemoryStatus:
        snapshot = self.om.snapshot()
        threads = self.pending_threads(session)
        key = session.key if session is not None else None
        return MemoryStatus(
            observation_tokens=om_tokens.count_string(snapshot.observations),
            observation_threshold=self.config.observation_tokens,
            pending_tokens=int(self.om.pending_tokens(key, threads)),
            message_threshold=self.config.message_tokens,
            generation=snapshot.state.generation,
            last_observed_at=snapshot.state.last_observed_at,
            last_reflected_at=snapshot.state.last_reflected_at,
            sessions=len(snapshot.state.threads),
        )

    # -- versioning ------------------------------------------------------------

    def restore_version(self, sha: str) -> str | None:
        """Undo memory commit *sha*; returns the new commit, or None if it could not.

        The store revision moves too, so an observation cycle that started from
        the log as it was before the restore is discarded instead of writing it back.
        """
        new_sha = self.git.revert(sha, message_prefix=MEMORY_COMMIT_PREFIX)
        if new_sha is not None:
            for _ in range(3):
                snapshot = self.om.store.read()
                try:
                    self.om.store.commit(snapshot.state.revision, lambda current: current)
                    break
                except StaleStateError:
                    continue
        return new_sha

    def _commit_git(self, message: str) -> str | None:
        if not self.git.is_initialized():
            return None
        try:
            self.git.ensure_tracked()
            return self.git.auto_commit(message)
        except GitStoreError:
            logger.exception("Observational memory: git commit failed")
            return None

    # -- migration -------------------------------------------------------------

    def _migrate_legacy_memory(self) -> None:
        """Carry ``memory/MEMORY.md`` from the Dream era into the observation log once."""
        snapshot = self.om.snapshot()
        if snapshot.state.revision > 0 or snapshot.observations:
            return
        legacy = self.workspace / _LEGACY_MEMORY_FILE
        try:
            content = legacy.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, UnicodeDecodeError):
            return
        from nanobot.utils.helpers import load_bundled_template

        template = (load_bundled_template("legacy/MEMORY.md") or "").strip()
        if not content or content == template:
            return
        now = self.om.now()
        lines = [f"Date: {now:%b} {now.day}, {now.year}"]
        lines.append(f"* 🔴 ({now:%H:%M}) Long-term memory carried over from memory/MEMORY.md:")
        lines.extend(f"  * {line.strip()}" for line in content.splitlines() if line.strip())
        observations = "\n".join(lines)

        def apply(current: Snapshot) -> Snapshot:
            return replace(current, observations=observations)

        try:
            self.om.store.commit(snapshot.state.revision, apply)
        except StaleStateError:
            return
        logger.info("Observational memory: imported memory/MEMORY.md into the observation log")


def _is_continuation(message: Mapping[str, Any]) -> bool:
    from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT

    return message.get("content") in (SUMMARY_CONTINUATION_TEXT, om_prompts.CONTINUATION_REMINDER)
