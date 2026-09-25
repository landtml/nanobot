"""Lifecycle and structured cancellation tests for orchestration runs."""

from __future__ import annotations

import asyncio
from time import perf_counter

import pytest

from nanobot.events import AgentEvent
from nanobot.orchestration.supervisor import RunRegistry


@pytest.mark.asyncio
async def test_cancelling_parent_cancels_descendants_before_parent() -> None:
    registry = RunRegistry()
    order: list[str] = []

    async def wait(name: str) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            order.append(name)

    registry.register("root", parent_id=None, root_id="root")
    registry.register("turn", parent_id="root", root_id="root")
    registry.register("child", parent_id="turn", root_id="root")
    child = asyncio.create_task(wait("child"))
    turn = asyncio.create_task(wait("turn"))
    registry.attach_task("child", child)
    registry.attach_task("turn", turn)
    await asyncio.sleep(0)

    cancelled = await registry.cancel_tree("root")

    assert cancelled == 2
    assert order.index("child") < order.index("turn")
    assert registry.get("root").state == "cancelled"
    assert registry.get("turn").state == "cancelled"
    assert registry.get("child").state == "cancelled"
    assert registry.live_tasks() == ()


@pytest.mark.asyncio
async def test_run_cannot_finish_before_scoped_child_finishes() -> None:
    registry = RunRegistry()
    registry.register("root", parent_id=None, root_id="root")
    registry.register("turn", parent_id="root", root_id="root")
    registry.register("child", parent_id="turn", root_id="root")
    finished = asyncio.Event()

    async def child_work() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    task = asyncio.create_task(child_work())
    registry.attach_task("child", task)
    await asyncio.sleep(0)

    await registry.finish("turn", "completed")

    assert finished.is_set()
    assert registry.get("child").state == "cancelled"
    assert registry.get("turn").state == "completed"


@pytest.mark.asyncio
async def test_detached_child_outlives_ordinary_turn_completion() -> None:
    registry = RunRegistry()
    registry.register("root", parent_id=None, root_id="root")
    registry.register("turn", parent_id="root", root_id="root")
    registry.register("background", parent_id="root", root_id="root", lifetime="detached")
    await registry.start("root")
    await registry.start("turn")
    await registry.start("background")
    task = asyncio.create_task(asyncio.Event().wait())
    registry.attach_task("background", task)
    await asyncio.sleep(0)

    await registry.finish("turn", "completed")

    assert registry.get("background").state == "running"
    assert registry.live_tasks(root_id="root") == (task,)
    await registry.cancel_session("root")


@pytest.mark.asyncio
async def test_three_level_four_child_tree_stops_under_one_second() -> None:
    registry = RunRegistry()
    registry.register("root", parent_id=None, root_id="root")
    tasks: list[asyncio.Task[None]] = []

    async def wait() -> None:
        await asyncio.Event().wait()

    frontier = ["root"]
    for depth in range(3):
        next_frontier: list[str] = []
        for parent_id in frontier:
            for child_num in range(4):
                child_id = f"{depth}-{parent_id}-{child_num}"
                registry.register(child_id, parent_id=parent_id, root_id="root")
                task = asyncio.create_task(wait())
                tasks.append(task)
                registry.attach_task(child_id, task)
                next_frontier.append(child_id)
        frontier = next_frontier
    await asyncio.sleep(0)

    started = perf_counter()
    await registry.cancel_session("root")
    elapsed = perf_counter() - started

    assert elapsed < 1.0
    assert registry.live_tasks(root_id="root") == ()
    assert all(task.done() for task in tasks)


@pytest.mark.asyncio
async def test_concurrent_tree_stops_emit_one_cancellation_event_per_run() -> None:
    events: list[AgentEvent] = []

    async def publish(event: AgentEvent) -> None:
        events.append(event)

    registry = RunRegistry(publish)
    registry.register("root", parent_id=None, root_id="root")
    registry.register("turn", parent_id="root", root_id="root")
    task = asyncio.create_task(asyncio.Event().wait())
    registry.attach_task("turn", task)
    await asyncio.sleep(0)

    await asyncio.gather(
        registry.cancel_session("root"),
        registry.cancel_session("root"),
    )

    assert len([event for event in events if type(event).__name__ == "RunCancelled"]) == 2


@pytest.mark.asyncio
async def test_cancellation_closes_admission_before_waiting_for_children() -> None:
    registry = RunRegistry()
    root = await registry.ensure_session_root("session")
    child_started = asyncio.Event()
    release_child = asyncio.Event()

    async def blocked_child() -> None:
        child_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_child.wait()
            raise

    registry.register("child", parent_id=root.id, root_id=root.id)
    await registry.start("child")
    child_task = asyncio.create_task(blocked_child())
    registry.attach_task("child", child_task)
    await child_started.wait()
    stop_task = asyncio.create_task(registry.cancel_session(root.id))
    await asyncio.sleep(0)

    with pytest.raises(ValueError, match="admission is closed"):
        registry.register("late-child", parent_id=root.id, root_id=root.id)

    release_child.set()
    await stop_task


@pytest.mark.asyncio
async def test_terminal_runs_drop_task_status_and_child_references() -> None:
    registry = RunRegistry()
    registry.register("root", parent_id=None, root_id="root")
    registry.register(
        "turn",
        parent_id="root",
        root_id="root",
        legacy_status={"task_id": "turn", "phase": "running"},
    )
    child_task = asyncio.create_task(asyncio.sleep(0))
    await child_task
    registry.attach_task("turn", child_task)

    await registry.finish("turn", "completed")

    record = registry.get("turn")
    assert record.task is None
    assert record.legacy_status is None
    assert "turn" not in registry._children.get("root", set())
    assert "turn" not in registry._children
    assert child_task.done()


@pytest.mark.asyncio
async def test_terminal_run_history_is_bounded() -> None:
    registry = RunRegistry(max_terminal_records=8)
    registry.register("root", parent_id=None, root_id="root")
    for index in range(12):
        run_id = f"turn-{index}"
        registry.register(run_id, parent_id="root", root_id="root")
        await registry.finish(run_id, "completed")

    assert not registry.contains("turn-0")
    assert registry.contains("turn-11")
    assert len(registry.terminal_ids()) == 8


def test_snapshot_keeps_legacy_subagent_status_shape() -> None:
    registry = RunRegistry()
    registry.register("session-1", parent_id=None, root_id="session-1")
    registry.register("turn-1", parent_id="session-1", root_id="session-1")
    registry.register(
        "task-1",
        parent_id="turn-1",
        root_id="session-1",
        legacy_snapshot={"task_id": "task-1", "label": "research", "phase": "running"},
    )

    assert registry.snapshot() == {
        "task-1": {"task_id": "task-1", "label": "research", "phase": "running"}
    }
