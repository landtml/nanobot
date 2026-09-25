"""In-memory run tree and structured cancellation."""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from nanobot.events import AgentEvent

RunState = Literal["queued", "running", "completed", "failed", "cancelled"]
RunLifetime = Literal["scoped", "detached"]
LifecyclePublisher = Callable[[AgentEvent], Awaitable[None]]
TerminalCleanup = Callable[[str], Awaitable[None]]
_CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("nanobot_current_run_id", default=None)


def current_run_id() -> str | None:
    return _CURRENT_RUN_ID.get()


def bind_current_run(run_id: str) -> Token[str | None]:
    return _CURRENT_RUN_ID.set(run_id)


def reset_current_run(token: Token[str | None]) -> None:
    _CURRENT_RUN_ID.reset(token)


@dataclass(frozen=True, slots=True)
class RunStarted(AgentEvent):
    run_id: str
    root_id: str
    parent_id: str | None
    lifetime: RunLifetime


@dataclass(frozen=True, slots=True)
class RunCompleted(AgentEvent):
    run_id: str
    root_id: str
    state: Literal["completed", "failed"]


@dataclass(frozen=True, slots=True)
class RunCancelled(AgentEvent):
    run_id: str
    root_id: str


@dataclass(slots=True)
class RunRecord:
    id: str
    root_id: str
    parent_id: str | None
    lifetime: RunLifetime = "scoped"
    state: RunState = "queued"
    task: asyncio.Task[object] | None = field(default=None, repr=False)
    legacy_status: object | None = None
    legacy_snapshot: Mapping[str, object] | None = None
    accepting_children: bool = True


class RunRegistry:
    """Own run identities, lifecycle state, task ownership, and cancellation."""

    def __init__(
        self,
        publish: LifecyclePublisher | None = None,
        *,
        max_terminal_records: int = 256,
    ) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._children: dict[str, set[str]] = {}
        self._session_roots: dict[str, str] = {}
        self._publish = publish
        self._terminal_cleanups: list[TerminalCleanup] = []
        self._terminal_order: deque[str] = deque()
        self._max_terminal_records = max_terminal_records

    def add_terminal_cleanup(self, cleanup: TerminalCleanup) -> None:
        self._terminal_cleanups.append(cleanup)

    async def ensure_session_root(self, session_key: str) -> RunRecord:
        """Return the active generation root for a session, creating it if needed."""
        root_id = self._session_roots.get(session_key)
        if root_id is not None and self.get(root_id).state == "running":
            return self.get(root_id)
        root_id = f"session-{uuid.uuid4()}"
        record = self.register(root_id, parent_id=None, root_id=root_id)
        self._session_roots[session_key] = root_id
        await self.start(root_id)
        return record

    def session_root_id(self, session_key: str) -> str | None:
        return self._session_roots.get(session_key)

    def register(
        self,
        run_id: str,
        *,
        parent_id: str | None,
        root_id: str,
        lifetime: RunLifetime = "scoped",
        legacy_status: object | None = None,
        legacy_snapshot: Mapping[str, object] | None = None,
    ) -> RunRecord:
        if run_id in self._runs:
            raise ValueError(f"run {run_id!r} is already registered")
        if parent_id is not None and parent_id not in self._runs:
            raise KeyError(f"parent run {parent_id!r} is not registered")
        if parent_id is not None:
            parent = self.get(parent_id)
            if parent.state in ("completed", "failed", "cancelled") or not parent.accepting_children:
                raise ValueError(f"parent run {parent_id!r} admission is closed")
        record = RunRecord(
            id=run_id,
            root_id=root_id,
            parent_id=parent_id,
            lifetime=lifetime,
            legacy_status=legacy_status,
            legacy_snapshot=legacy_snapshot,
        )
        self._runs[run_id] = record
        if parent_id is not None:
            self._children.setdefault(parent_id, set()).add(run_id)
        return record

    def get(self, run_id: str) -> RunRecord:
        return self._runs[run_id]

    def contains(self, run_id: str) -> bool:
        return run_id in self._runs

    async def start(self, run_id: str) -> None:
        record = self.get(run_id)
        if record.state != "queued":
            raise ValueError(f"cannot start run {run_id!r} from {record.state!r}")
        record.state = "running"
        await self._emit(RunStarted(
            run_id=record.id,
            root_id=record.root_id,
            parent_id=record.parent_id,
            lifetime=record.lifetime,
        ))

    def attach_task(self, run_id: str, task: asyncio.Task[object]) -> None:
        record = self.get(run_id)
        if record.task is not None:
            raise ValueError(f"run {run_id!r} already owns a task")
        record.task = task
        task.add_done_callback(lambda done: self.detach_task(run_id, done))

    def detach_task(self, run_id: str, task: asyncio.Task[object]) -> None:
        record = self._runs.get(run_id)
        if record is not None and record.task is task:
            record.task = None

    async def finish(self, run_id: str, state: Literal["completed", "failed"]) -> None:
        record = self.get(run_id)
        if record.state in ("completed", "failed", "cancelled"):
            return
        record.accepting_children = False
        await self._cancel_scoped_children(run_id)
        record.state = state
        await self._emit(RunCompleted(run_id=run_id, root_id=record.root_id, state=state))
        await self._terminalize(record)

    async def finish_cancelled(self, run_id: str) -> None:
        """Record cancellation once after a task observes CancelledError."""
        record = self.get(run_id)
        if record.state in ("completed", "failed", "cancelled"):
            return
        record.accepting_children = False
        await self._cancel_scoped_children(run_id)
        await self._mark_cancelled(record)

    async def cancel_tree(self, run_id: str) -> int:
        """Cancel descendants depth-first, then the requested run."""
        record = self.get(run_id)
        self._close_admission_tree(run_id)
        count = await self._cancel_descendants(run_id)
        if record.state not in ("completed", "failed", "cancelled"):
            current_task = asyncio.current_task()
            if (
                record.task is not None
                and record.task is not current_task
                and not record.task.done()
            ):
                record.task.cancel()
                count += 1
                await asyncio.gather(record.task, return_exceptions=True)
            await self._mark_cancelled(record)
        return count

    async def cancel_session(self, root_id: str) -> int:
        """End one session generation and cancel its whole run tree."""
        if root_id not in self._runs:
            return 0
        self._close_admission_tree(root_id)
        for session_key, candidate in tuple(self._session_roots.items()):
            if candidate == root_id:
                del self._session_roots[session_key]
        count = await self.cancel_tree(root_id)
        return count

    async def cancel_all(self) -> int:
        root_ids = tuple(
            record.id
            for record in self._runs.values()
            if record.parent_id is None and record.state in ("queued", "running")
        )
        count = 0
        for root_id in root_ids:
            count += await self.cancel_session(root_id)
        return count

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return active legacy status records for the self tool projection."""
        snapshot: dict[str, dict[str, object]] = {}
        for run_id, record in self._runs.items():
            status = record.legacy_snapshot
            if status is None or record.state not in ("queued", "running"):
                continue
            snapshot[run_id] = dict(status)
        return snapshot

    def legacy_statuses(self) -> dict[str, object]:
        return {
            run_id: record.legacy_status
            for run_id, record in self._runs.items()
            if record.legacy_status is not None
            and record.state in ("queued", "running")
        }

    def live_tasks(self, *, root_id: str | None = None) -> tuple[asyncio.Task[object], ...]:
        return tuple(
            record.task
            for record in self._runs.values()
            if record.task is not None
            and not record.task.done()
            and (root_id is None or record.root_id == root_id)
        )

    def terminal_ids(self) -> tuple[str, ...]:
        return tuple(self._terminal_order)

    def _close_admission_tree(self, run_id: str) -> None:
        record = self._runs.get(run_id)
        if record is None:
            return
        record.accepting_children = False
        for child_id in tuple(self._children.get(run_id, ())):
            self._close_admission_tree(child_id)

    async def _cancel_scoped_children(self, run_id: str) -> int:
        count = 0
        for child_id in tuple(self._children.get(run_id, ())):
            child = self.get(child_id)
            if child.lifetime == "scoped" and child.state not in (
                "completed", "failed", "cancelled",
            ):
                count += await self.cancel_tree(child_id)
        return count

    async def _cancel_descendants(self, run_id: str) -> int:
        count = 0
        for child_id in tuple(self._children.get(run_id, ())):
            child = self.get(child_id)
            count += await self._cancel_descendants(child_id)
            if child.state not in ("completed", "failed", "cancelled"):
                current_task = asyncio.current_task()
                if (
                    child.task is not None
                    and child.task is not current_task
                    and not child.task.done()
                ):
                    child.task.cancel()
                    count += 1
                    await asyncio.gather(child.task, return_exceptions=True)
                await self._mark_cancelled(child)
        return count

    async def _mark_cancelled(self, record: RunRecord) -> None:
        if record.state == "cancelled":
            return
        record.state = "cancelled"
        await self._emit(RunCancelled(run_id=record.id, root_id=record.root_id))
        await self._terminalize(record)

    async def _terminalize(self, record: RunRecord) -> None:
        for cleanup in self._terminal_cleanups:
            try:
                await cleanup(record.id)
            except Exception:
                logger.exception("Failed terminal cleanup for run {}", record.id)
        record.task = None
        record.legacy_status = None
        record.legacy_snapshot = None
        self._children.pop(record.id, None)
        if record.parent_id is not None and record.parent_id in self._children:
            self._children[record.parent_id].discard(record.id)
        self._terminal_order.append(record.id)
        while len(self._terminal_order) > self._max_terminal_records:
            expired = self._terminal_order.popleft()
            self._runs.pop(expired, None)

    async def _emit(self, event: AgentEvent) -> None:
        if self._publish is not None:
            try:
                await self._publish(event)
            except Exception:
                logger.exception("Failed to publish {}", type(event).__name__)
