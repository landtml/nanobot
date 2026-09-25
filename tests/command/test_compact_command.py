"""Manual context compaction (``/compact``): observe the session now."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent.session_helpers import run_session

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import ContextCompactionEvent
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import TurnCompleted
from nanobot.command.builtin import cmd_stop
from nanobot.command.router import CommandContext
from nanobot.providers.base import GenerationSettings, LLMResponse, ProviderConversationState

OBSERVED = (
    "<observations>\nDate: Sep 25, 2026\n* 🔴 (10:00) User asked an important question\n"
    "</observations>"
)


@pytest.fixture
async def loop(tmp_path):
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=100)
    provider.can_resume_conversation_state.return_value = True
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content=OBSERVED,
        finish_reason="stop",
    ))
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=128_000,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    try:
        yield loop
    finally:
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/compact", " /COMPACT@nanobot "])
async def test_compact_emits_one_lifecycle_and_keeps_the_session(loop, command) -> None:
    bus = loop.bus
    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "important question")
    session.add_message("assistant", "important answer")
    session.provider_state = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="test-model",
        version=1,
        payload={"items": []},
    )
    loop.sessions.save(session)

    msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content=command)
    response = await loop._process_message(msg, runtime=loop.llm_runtime())

    assert response is None
    assert bus.outbound_size == 2
    started = bus.outbound.get_nowait().event
    completed = bus.outbound.get_nowait().event
    assert isinstance(started, ContextCompactionEvent)
    assert isinstance(completed, ContextCompactionEvent)
    assert started.phase == "started"
    assert completed.phase == "succeeded"
    assert started.compaction_id == completed.compaction_id

    loop.sessions.invalidate("cli:test")
    reloaded = loop.sessions.get_or_create("cli:test")
    assert reloaded.provider_state is None
    assert reloaded.messages == session.messages
    assert reloaded.last_archived == 2
    assert reloaded.get_history() == []
    assert "User asked an important question" in loop.memory.observations()

    response = await loop._process_message(msg, runtime=loop.llm_runtime())
    assert response is None
    assert bus.outbound_size == 0
    loop.provider.chat_stream_with_retry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("observer_output", [OBSERVED, "Nothing worth keeping."])
async def test_compacted_session_resumes_from_memory_with_new_input(loop, observer_output) -> None:
    key = "cli:checkpoint-resume"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "Inspect the checkpoint")
    session.add_message("assistant", "Inspection complete.")
    loop.sessions.save(session)
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(content=observer_output)

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="checkpoint-resume",
                       content="/compact"),
        runtime=loop.llm_runtime(),
    )

    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.last_archived == 2
    assert reloaded.get_history() == []
    loop.provider.chat_stream_with_retry.assert_awaited_once()
    assert loop.bus.inbound_size == 0

    loop.provider.chat_stream_with_retry.reset_mock()
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(content="Hello!")
    response = await loop.process_direct("hi", session_key=key)
    assert response.content == "Hello!"
    sent = loop.provider.chat_stream_with_retry.await_args_list[0].kwargs["messages"]
    assert [message["role"] for message in sent] == ["system", "user"]
    assert "Inspect the checkpoint" not in str(sent)
    assert sent[1]["content"].endswith("hi")
    if observer_output == OBSERVED:
        assert "User asked an important question" in sent[0]["content"]
        assert sent[1]["content"].startswith("<system-reminder>")
    else:
        assert "<observations>" not in sent[0]["content"]
        assert sent[1]["content"] == "hi"

    loop.sessions.invalidate(key)
    resumed = loop.sessions.get_or_create(key)
    assert resumed.get_history() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_commands", [False, True])
async def test_empty_compact_finishes_silently(loop, legacy_commands) -> None:
    key = "websocket:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "already archived")
    session.add_message("assistant", "old answer")
    session.last_archived = 2
    if legacy_commands:
        session.add_message("user", "/compact", _command=True)
        session.add_message("assistant", "Nothing to compact.", _command=True)
    loop.sessions.save(session)
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)

    await run_session(loop, InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content="/compact",
        metadata={"webui_turn_id": "compact-turn"},
    ))

    assert loop.bus.outbound_size == 0
    assert len(completions) == 1
    assert completions[0].context.metadata["webui_turn_id"] == "compact-turn"
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.messages == session.messages
    assert reloaded.last_archived == 2
    assert loop.memory.observations() == ""
    loop.provider.chat_stream_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_compact_during_active_turn_waits_for_the_session_lock(loop) -> None:
    key = "websocket:test"
    msg = InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content="/COMPACT@nanobot",
    )
    lock = loop._get_session_lock(key)
    async with lock:
        await loop._dispatch_command_inline(msg, key, msg.content, loop.commands.dispatch)
        tasks = list(loop._active_tasks[key])
        assert len(tasks) == 1
        await asyncio.sleep(0)
        assert not tasks[0].done()
        assert loop.bus.outbound_size == 0
        session = loop.sessions.get_or_create(key)
        session.add_message("user", "active turn question")
        session.add_message("assistant", "active turn answer")
        loop.sessions.save(session)

    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert loop.bus.outbound_size == 2
    assert loop.sessions.get_or_create(key).last_archived == 2
    loop.provider.chat_stream_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_completes_a_compact_command_waiting_for_the_session_lock(loop) -> None:
    key = "websocket:test"
    msg = InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content="/compact",
        metadata={"webui_turn_id": "queued-compact"},
    )
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)
    async with loop._get_session_lock(key):
        await loop._dispatch_command_inline(msg, key, msg.content, loop.commands.dispatch)
        reply = await cmd_stop(CommandContext(
            msg=msg, session=None, key=key, raw="/stop", loop=loop,
        ))
    assert reply.content == "Stopped 1 task(s)."
    assert len(completions) == 1
    assert completions[0].context.metadata["webui_turn_id"] == "queued-compact"


@pytest.mark.asyncio
async def test_compact_is_a_fifo_barrier_during_an_active_turn(loop) -> None:
    key = "cli:test"
    started = asyncio.Event()
    release = asyncio.Event()
    requests = []
    compacted_history = []
    compact = loop.memory.compact

    async def capture_compaction(session):
        compacted_history.extend(dict(message) for message in session.messages)
        return await compact(session)

    loop.memory.compact = capture_compaction

    async def chat(*, messages, **kwargs):
        if messages[0]["content"].startswith("You are the memory consciousness"):
            return LLMResponse(content=OBSERVED, finish_reason="stop")
        requests.append([dict(message) for message in messages])
        if len(requests) == 1:
            started.set()
            await release.wait()
        return LLMResponse(content="answer", finish_reason="stop")

    loop.provider.chat_stream_with_retry = chat
    task = asyncio.create_task(run_session(loop, InboundMessage(
        channel="cli", sender_id="u", chat_id="test", content="initial question",
    )))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        loop._enqueue_session_message(InboundMessage(
            channel="cli", sender_id="u", chat_id="test", content="before compaction",
        ))
        command = InboundMessage(channel="cli", sender_id="u", chat_id="test", content="/compact")
        await loop._dispatch_command_inline(command, key, command.content, loop.commands.dispatch)
        loop._enqueue_session_message(InboundMessage(
            channel="cli", sender_id="u", chat_id="test", content="after compaction",
        ))
        release.set()
        await asyncio.wait_for(task, timeout=5)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert [message["content"] for message in compacted_history if message["role"] == "user"] == [
        "initial question", "before compaction",
    ]
    assert all("/compact" != message.get("content") for request in requests for message in request)
    assert "after compaction" in str(requests[-1])
    events = [loop.bus.outbound.get_nowait().event for _ in range(loop.bus.outbound_size)]
    assert [event.phase for event in events if isinstance(event, ContextCompactionEvent)] == [
        "started", "succeeded",
    ]


@pytest.mark.asyncio
async def test_stop_completes_compact_queued_behind_an_active_turn(loop) -> None:
    key = "websocket:test"
    started = asyncio.Event()

    async def chat(**kwargs):
        started.set()
        await asyncio.Event().wait()

    loop.provider.chat_stream_with_retry = chat
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)
    loop._enqueue_session_message(InboundMessage(
        channel="websocket", sender_id="u", chat_id="test", content="question",
    ))
    task = next(iter(loop._active_tasks[key]))
    await asyncio.wait_for(started.wait(), timeout=5)
    command = InboundMessage(
        channel="websocket", sender_id="u", chat_id="test", content="/compact",
        metadata={"webui_turn_id": "queued-compact"},
    )
    await loop._dispatch_command_inline(command, key, command.content, loop.commands.dispatch)
    await cmd_stop(CommandContext(msg=command, session=None, key=key, raw="/stop", loop=loop))

    assert task.cancelled()
    assert sum(
        event.context.metadata.get("webui_turn_id") == "queued-compact" for event in completions
    ) == 1
    assert key not in loop._pending_queues


@pytest.mark.asyncio
async def test_stop_finishes_inflight_compaction_as_cancelled(loop) -> None:
    key = "websocket:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "important question")
    session.add_message("assistant", "important answer")
    loop.sessions.save(session)
    entered = asyncio.Event()

    async def wait_for_cancel(**_kwargs):
        entered.set()
        await asyncio.Event().wait()

    loop.provider.chat_stream_with_retry.side_effect = wait_for_cancel
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)
    msg = InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content="/compact",
        metadata={"webui_turn_id": "compact-turn"},
    )
    loop._enqueue_session_message(msg)
    task = next(iter(loop._active_tasks[key]))
    await asyncio.wait_for(entered.wait(), timeout=5)

    reply = await cmd_stop(CommandContext(
        msg=msg, session=session, key=key, raw="/stop", loop=loop,
    ))

    assert reply.content == "Stopped 1 task(s)."
    assert task.cancelled()
    assert len(completions) == 1
    assert completions[0].context.metadata["webui_turn_id"] == "compact-turn"
    events = [loop.bus.outbound.get_nowait().event for _ in range(loop.bus.outbound_size)]
    assert all(isinstance(event, ContextCompactionEvent) for event in events)
    assert [event.phase for event in events] == ["started", "cancelled"]
    assert events[0].compaction_id == events[1].compaction_id
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.messages == session.messages
    assert reloaded.last_archived == 0
    assert reloaded.get_history() == session.get_history()


@pytest.mark.asyncio
async def test_compact_observes_tool_heavy_turns_once(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "large tool turn")
    for i in range(20):
        session.add_message("assistant", "", tool_calls=[{
            "id": f"tool-{i}", "type": "function",
            "function": {"name": "exec", "arguments": "{}"},
        }])
        session.add_message("tool", "x" * 10_000, tool_call_id=f"tool-{i}")
    session.add_message("assistant", "done")
    loop.sessions.save(session)
    command = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/compact")

    await loop._process_message(command, runtime=loop.llm_runtime())
    loop.provider.chat_stream_with_retry.assert_awaited_once()
    prompt = loop.provider.chat_stream_with_retry.call_args.kwargs["messages"][1]["content"]
    assert prompt.count("[Tool Call: exec]") == 20
    assert prompt.count("[Tool Result: exec]") == 20
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert len(reloaded.messages) == 42
    assert reloaded.last_archived == 42
    assert reloaded.get_history() == []

    # Nothing new: the second /compact is silent and makes no model call.
    await loop._process_message(command, runtime=loop.llm_runtime())
    loop.provider.chat_stream_with_retry.assert_awaited_once()
