"""Durable state for Observational Memory.

Two files under ``<workspace>/memory/``:

``observations.md``
    The active observation log, exactly as the Actor's memory block renders it
    (before optimization). Plain text so people can read it and so the
    workspace git store can version it.

``observational_memory.json``
    Bookkeeping: per-session observation cursors, each session's current task
    and suggested response, the reflection generation, and a revision counter.

Both are replaced atomically. Every write goes through :meth:`ObservationStore.commit`,
which re-reads the state under a file lock and refuses the write when another
process committed in between (the revision moved), so a CLI and a gateway that
share a workspace cannot silently overwrite each other's observations.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from nanobot.utils.helpers import atomic_write_lines, ensure_dir

STATE_VERSION = 1
OBSERVATIONS_FILE = "observations.md"
STATE_FILE = "observational_memory.json"
_LOCK_FILE = ".observational_memory.lock"
_LOCK_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ThreadState:
    """What Observational Memory knows about one session."""

    observed_count: int = 0
    """How many of the session's messages have been observed."""
    observed_fingerprint: str = ""
    """Fingerprint of message ``observed_count - 1``; detects cleared or rewritten sessions."""
    current_task: str | None = None
    suggested_response: str | None = None
    last_observed_at: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryState:
    revision: int = 0
    generation: int = 0
    """Number of reflections applied to the observation log."""
    activated_at: str | None = None
    """Sessions last updated before this instant never join the resource."""
    last_observed_at: str | None = None
    last_reflected_at: str | None = None
    threads: Mapping[str, ThreadState] = field(default_factory=dict[str, ThreadState])


class StaleStateError(RuntimeError):
    """Another writer committed after this snapshot was read."""


def _thread_from_json(raw: object) -> ThreadState | None:
    if not isinstance(raw, dict):
        return None
    data = cast(dict[str, Any], raw)
    count = data.get("observed_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None

    def optional(key: str) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) and value else None

    fingerprint = data.get("observed_fingerprint")
    return ThreadState(
        observed_count=count,
        observed_fingerprint=fingerprint if isinstance(fingerprint, str) else "",
        current_task=optional("current_task"),
        suggested_response=optional("suggested_response"),
        last_observed_at=optional("last_observed_at"),
    )


def _state_from_json(raw: object) -> MemoryState:
    if not isinstance(raw, dict):
        return MemoryState()
    data = cast(dict[str, Any], raw)

    def integer(key: str) -> int:
        value = data.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    def optional(key: str) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) and value else None

    threads: dict[str, ThreadState] = {}
    raw_threads = data.get("threads")
    if isinstance(raw_threads, dict):
        for key, value in cast(dict[object, object], raw_threads).items():
            if isinstance(key, str) and (thread := _thread_from_json(value)) is not None:
                threads[key] = thread
    return MemoryState(
        revision=integer("revision"),
        generation=integer("generation"),
        activated_at=optional("activated_at"),
        last_observed_at=optional("last_observed_at"),
        last_reflected_at=optional("last_reflected_at"),
        threads=threads,
    )


def _state_to_json(state: MemoryState) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "revision": state.revision,
        "generation": state.generation,
        "activated_at": state.activated_at,
        "last_observed_at": state.last_observed_at,
        "last_reflected_at": state.last_reflected_at,
        "threads": {
            key: {
                "observed_count": thread.observed_count,
                "observed_fingerprint": thread.observed_fingerprint,
                "current_task": thread.current_task,
                "suggested_response": thread.suggested_response,
                "last_observed_at": thread.last_observed_at,
            }
            for key, thread in sorted(state.threads.items())
        },
    }


@dataclass(frozen=True, slots=True)
class Snapshot:
    observations: str
    state: MemoryState


class ObservationStore:
    """File-backed observation log and bookkeeping for one workspace."""

    def __init__(self, workspace: Path) -> None:
        self.memory_dir = workspace / "memory"
        self.observations_file = self.memory_dir / OBSERVATIONS_FILE
        self.state_file = self.memory_dir / STATE_FILE
        self._lock_file = self.memory_dir / _LOCK_FILE
        self._cache: tuple[tuple[int, int] | None, tuple[int, int] | None, Snapshot] | None = None

    @staticmethod
    def _stamp(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def read(self) -> Snapshot:
        """Current observations and state; cached until either file changes."""
        stamps = (self._stamp(self.observations_file), self._stamp(self.state_file))
        if self._cache is not None and self._cache[:2] == stamps:
            return self._cache[2]
        observations = ""
        with suppress(FileNotFoundError):
            observations = self.observations_file.read_text(encoding="utf-8").removesuffix("\n")
        raw: object = None
        with suppress(FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        snapshot = Snapshot(observations=observations, state=_state_from_json(raw))
        self._cache = (stamps[0], stamps[1], snapshot)
        return snapshot

    def commit(
        self,
        base_revision: int,
        update: Callable[[Snapshot], Snapshot],
    ) -> Snapshot:
        """Apply *update* to the current snapshot if nobody committed since *base_revision*.

        Raises :class:`StaleStateError` when the stored revision moved; the
        caller's LLM work was based on outdated observations and must be
        discarded (Mastra 1.1.0 drops such cycles the same way).
        """
        ensure_dir(self.memory_dir)
        with FileLock(str(self._lock_file), timeout=_LOCK_TIMEOUT_SECONDS):
            self._cache = None
            current = self.read()
            if current.state.revision != base_revision:
                raise StaleStateError(
                    f"observational memory moved from revision {base_revision} "
                    f"to {current.state.revision}"
                )
            updated = update(current)
            updated = replace(
                updated,
                state=replace(updated.state, revision=current.state.revision + 1),
            )
            if updated.observations != current.observations:
                atomic_write_lines(self.observations_file, [updated.observations])
            atomic_write_lines(
                self.state_file,
                [json.dumps(_state_to_json(updated.state), ensure_ascii=False, indent=2)],
            )
            self._cache = None
            return self.read()

    def activate(self, now: datetime) -> Snapshot:
        """Record the activation instant once; later calls are no-ops."""
        snapshot = self.read()
        if snapshot.state.activated_at is not None:
            return snapshot
        with suppress(StaleStateError):
            return self.commit(
                snapshot.state.revision,
                lambda current: current
                if current.state.activated_at is not None
                else replace(current, state=replace(current.state, activated_at=now.isoformat())),
            )
        return self.read()
