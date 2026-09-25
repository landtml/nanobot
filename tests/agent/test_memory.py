"""The Memory service: Observational Memory wired into nanobot sessions and storage."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import (
    MEMORY_COMMIT_PREFIX,
    Memory,
    message_fingerprint,
    observer_messages,
)
from nanobot.agent.observational_memory import prompts
from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import cmd_memory, cmd_memory_log, cmd_memory_restore
from nanobot.command.router import CommandContext
from nanobot.config.schema import MemoryConfig
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import load_bundled_template


def _observed(line: str, *, task: str | None = None) -> str:
    out = f"<observations>\nDate: Sep 25, 2026\n* 🔴 (10:00) {line}\n</observations>"
    return out + (f"\n<current-task>\n{task}\n</current-task>" if task else "")


def _memory(workspace: Path, *, responses: list[str] | None = None, **config: int) -> Memory:
    memory = Memory(
        workspace,
        SessionManager(workspace),
        runtime=MagicMock(),
        config=MemoryConfig(**config) if config else None,
        timezone="UTC",
    )
    memory._complete = AsyncMock(side_effect=list(responses or []))  # type: ignore[method-assign]
    return memory


def _conversation(memory: Memory, key: str, turns: int, *, words: int = 5) -> Session:
    session = memory.sessions.get_or_create(key)
    for i in range(turns):
        session.add_message("user", f"question {i} " + "about the project " * words)
        session.add_message("assistant", f"answer {i} " + "with the details " * words)
    memory.sessions.save(session)
    return session


# ---------------------------------------------------------------------------
# Session messages as the Observer sees them
# ---------------------------------------------------------------------------


def test_observer_messages_merge_tool_exchanges_and_skip_bookkeeping() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "look at [image: /home/me/secret.png] please",
         "timestamp": "2026-09-25T10:00:00+00:00"},
        {"role": "assistant", "content": "<think>hmm</think>Checking.",
         "timestamp": "2026-09-25T10:01:00+00:00",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "file body"},
        {"role": "assistant", "content": "Status: ok", "_command": True},
        {"role": "user", "content": "hidden", "_hidden_history": True},
        {"role": "system", "content": "ignored"},
    ]

    converted = observer_messages("cli:a", messages, start=5, tz=timezone.utc)

    assert [m["id"] for m in converted] == ["cli:a#5", "cli:a#6"]
    assert converted[0]["content"]["parts"] == [{"type": "text", "text": "look at [image] please"}]
    assert converted[1]["content"]["parts"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool-exchange", "toolName": "read_file", "args": {"path": "a.txt"},
         "result": "file body"},
    ]
    assert converted[1]["created_at"].isoformat() == "2026-09-25T10:01:00+00:00"


def test_message_fingerprint_uses_timestamp_and_role() -> None:
    assert message_fingerprint({"timestamp": "t", "role": "user", "content": "x"}) == "t|user"
    assert message_fingerprint({}) == "|"


# ---------------------------------------------------------------------------
# Sessions as threads
# ---------------------------------------------------------------------------


async def test_background_observation_is_applied_when_the_session_next_runs(tmp_path) -> None:
    memory = _memory(tmp_path, responses=[_observed("User asked about the project")],
                     message_tokens=1_000)
    session = _conversation(memory, "cli:a", 6, words=30)
    session.provider_state = MagicMock()

    outcome = await memory.after_turn("cli:a")

    assert outcome is not None
    # The session file is not touched by observation...
    assert memory.sessions.read_session_snapshot("cli:a").last_archived == 0  # type: ignore[union-attr]
    # ...its cursor is applied the next time the session runs.
    assert memory.restore(session) is True
    assert session.last_archived == 12
    assert session.provider_state is None
    assert memory.restore(session) is False
    assert "User asked about the project" in (memory.system_prompt_block("cli:a") or "")
    assert memory.history_prefix("cli:a")[0]["content"] == prompts.CONTINUATION_REMINDER


async def test_after_turn_below_the_threshold_does_nothing(tmp_path) -> None:
    memory = _memory(tmp_path)
    _conversation(memory, "cli:a", 1)

    assert await memory.after_turn("cli:a") is None
    memory._complete.assert_not_awaited()  # type: ignore[attr-defined]


async def test_after_turn_is_single_flight(tmp_path) -> None:
    memory = _memory(tmp_path, message_tokens=1_000)
    _conversation(memory, "cli:a", 6, words=30)
    release = asyncio.Event()

    async def slow(**_kwargs: Any) -> str:
        await release.wait()
        return _observed("slow")

    memory._complete = slow  # type: ignore[method-assign]
    first = asyncio.create_task(memory.after_turn("cli:a"))
    await asyncio.sleep(0)

    assert await memory.after_turn("cli:a") is None
    release.set()
    assert await first is not None


async def test_after_turn_never_raises(tmp_path) -> None:
    memory = _memory(tmp_path, message_tokens=1_000)
    _conversation(memory, "cli:a", 6, words=30)
    memory._complete = AsyncMock(side_effect=RuntimeError("provider down"))  # type: ignore[method-assign]

    assert await memory.after_turn("cli:a") is None
    assert memory.observations() == ""


def test_sessions_idle_since_before_activation_do_not_join(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    old = sessions.get_or_create("cli:old")
    old.add_message("user", "an old conversation")
    sessions.save(old)
    past = time.time() - 3600
    for path in sessions.sessions_dir.glob("*.jsonl"):
        os.utime(path, (past, past))
    memory = Memory(tmp_path, SessionManager(tmp_path), runtime=MagicMock(), timezone="UTC")

    assert memory.pending_threads() == []

    _conversation(memory, "cli:new", 1)
    assert [t.key for t in memory.pending_threads()] == ["cli:new"]


def test_internal_and_private_sessions_never_join(tmp_path) -> None:
    memory = _memory(tmp_path)
    _conversation(memory, "heartbeat", 1)
    _conversation(memory, "dream:1234", 1)
    private = memory.sessions.get_or_create_transient("websocket:private")
    private.add_message("user", "secret")

    assert memory.pending_threads(private) == []


async def test_archive_observes_the_rest_and_retires_the_thread(tmp_path) -> None:
    memory = _memory(tmp_path, responses=[_observed("wrapped up", task="Primary: finish")])
    session = _conversation(memory, "cli:a", 2)

    await memory.archive(replace(session, messages=list(session.messages)))

    assert "wrapped up" in memory.observations()
    assert memory.om.thread_state("cli:a") is None


async def test_compact_observes_now_and_drops_observed_replay(tmp_path) -> None:
    memory = _memory(tmp_path, responses=[_observed("compacted")])
    session = _conversation(memory, "cli:a", 2)

    assert await memory.compact(session) is True
    assert session.get_history() == []
    assert memory.sessions.read_session_snapshot("cli:a").last_archived == 4  # type: ignore[union-attr]
    assert await memory.compact(session) is False


# ---------------------------------------------------------------------------
# Legacy memory and versioning
# ---------------------------------------------------------------------------


def test_a_dream_era_memory_file_is_imported_once(tmp_path) -> None:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "MEMORY.md").write_text(
        "# Long-term Memory\n\n- User prefers dark mode.\n", encoding="utf-8",
    )

    memory = _memory(tmp_path)
    log = memory.observations()

    assert "Long-term memory carried over from memory/MEMORY.md" in log
    assert "User prefers dark mode." in log
    _memory(tmp_path)
    assert memory.observations() == log


def test_the_untouched_memory_template_is_not_imported(tmp_path) -> None:
    (tmp_path / "memory").mkdir()
    template = load_bundled_template("legacy/MEMORY.md")
    assert template
    (tmp_path / "memory" / "MEMORY.md").write_text(template, encoding="utf-8")

    assert _memory(tmp_path).observations() == ""


async def test_observations_are_versioned_and_restorable(tmp_path) -> None:
    memory = _memory(tmp_path, responses=[_observed("first fact")])
    assert memory.git.init()
    session = _conversation(memory, "cli:a", 2)

    assert await memory.compact(session) is True
    commit = memory.git.log(max_entries=1, message_prefix=MEMORY_COMMIT_PREFIX)[0]
    assert commit.message.startswith("memory: compact")
    revision = memory.om.snapshot().state.revision

    assert memory.restore_version(commit.sha) is not None
    assert "first fact" not in memory.observations()
    # An observation cycle that read the log before the restore is discarded.
    assert memory.om.snapshot().state.revision == revision + 1


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _ctx(memory: Memory, raw: str, args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="u", chat_id="direct", content=raw)
    loop = SimpleNamespace(
        memory=memory,
        sessions=memory.sessions,
        bus=SimpleNamespace(publish_outbound=AsyncMock()),
        schedule_background=lambda coro: asyncio.ensure_future(coro),
    )
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


async def test_memory_command_shows_status_and_observations(tmp_path) -> None:
    memory = _memory(tmp_path)
    memory.replace_observations("Date: Sep 25, 2026\n* 🔴 (10:00) likes tea")

    out = await cmd_memory(_ctx(memory, "/memory"))

    assert "likes tea" in out.content
    assert "40,000" in out.content or "40000" in out.content


async def test_memory_reflect_runs_in_the_background(tmp_path) -> None:
    memory = _memory(tmp_path, responses=[_observed("tea")])
    memory.replace_observations("Date: Sep 25, 2026\n* 🔴 (10:00) likes tea\n* 🔴 (10:01) hi")
    ctx = _ctx(memory, "/memory reflect", "reflect")

    out = await cmd_memory(ctx)
    await asyncio.sleep(0.05)

    assert out.content == "Reflecting on memory..."
    published = ctx.loop.bus.publish_outbound.await_args.args[0]
    assert published.content.startswith("Memory condensed")
    assert memory.om.snapshot().state.generation == 1


async def test_memory_log_and_restore_commands(tmp_path) -> None:
    memory = _memory(tmp_path)
    assert memory.git.init()
    memory.replace_observations("Date: Sep 25, 2026\n* 🔴 (10:00) likes tea")
    sha = memory.git.log(max_entries=1, message_prefix=MEMORY_COMMIT_PREFIX)[0].sha

    log = await cmd_memory_log(_ctx(memory, "/memory-log"))
    listing = await cmd_memory_restore(_ctx(memory, "/memory-restore"))
    restored = await cmd_memory_restore(_ctx(memory, f"/memory-restore {sha}", sha))

    assert "likes tea" in log.content
    assert sha[:7] in listing.content
    assert "Restored memory" in restored.content
    assert "likes tea" not in memory.observations()


@pytest.mark.parametrize("command", [cmd_memory_log, cmd_memory_restore])
async def test_memory_versions_require_a_versioned_workspace(tmp_path, command) -> None:
    out = await command(_ctx(_memory(tmp_path), "/memory-log"))
    assert "version" in out.content.lower()
