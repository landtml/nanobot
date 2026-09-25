from dataclasses import replace

from nanobot.agent.memory import message_fingerprint
from nanobot.agent.observational_memory.store import ObservationStore, Snapshot, ThreadState
from nanobot.providers.base import LLMUsage
from nanobot.session import Session
from nanobot.utils.helpers import estimate_message_tokens
from nanobot.webui.session_context import session_context_payload


def _observed(tmp_path, session: Session, observations: str) -> ObservationStore:
    store = ObservationStore(tmp_path)
    fingerprint = message_fingerprint(session.messages[2])

    def apply(current: Snapshot) -> Snapshot:
        return replace(
            current,
            observations=observations,
            state=replace(
                current.state,
                last_observed_at="2026-08-13T10:00:00+00:00",
                threads={session.key: ThreadState(
                    observed_count=3, observed_fingerprint=fingerprint,
                )},
            ),
        )

    store.commit(0, apply)
    return store


def test_session_context_separates_observed_progress_from_replay(tmp_path) -> None:
    messages = [
        {"role": "user", "content": "old question", "timestamp": "2026-08-13T09:00:00"},
        {"role": "assistant", "content": "old answer", "timestamp": "2026-08-13T09:01:00"},
        {"role": "user", "content": "recent question", "timestamp": "2026-08-13T09:02:00"},
        {"role": "assistant", "content": "recent answer", "timestamp": "2026-08-13T09:03:00"},
    ]
    session = Session(key="websocket:context", messages=messages)
    session.last_archived = 2
    log = "Date: Aug 13, 2026\n* 🔴 (09:00) User settled the old question"
    store = _observed(tmp_path, session, log)

    replay = session.get_history(max_messages=0, include_runtime_context=False)
    replay_tokens = sum(estimate_message_tokens(message) for message in replay)
    observation_tokens = estimate_message_tokens({"role": "system", "content": log})
    payload = session_context_payload(session, store)

    assert payload == {
        "schema_version": 2,
        "session_key": "websocket:context",
        "total_messages": 4,
        # Observed in the background after the last turn, before replay caught up.
        "observed_messages": 3,
        "replay_messages": len(replay),
        "estimated_replay_tokens": replay_tokens,
        "estimated_observation_tokens": observation_tokens,
        "estimated_session_tokens": replay_tokens + observation_tokens,
        "observations": log,
        "observed_at": "2026-08-13T10:00:00+00:00",
        "last_usage": None,
    }


def test_session_context_ignores_a_cursor_from_a_cleared_session(tmp_path) -> None:
    session = Session(key="websocket:context", messages=[
        {"role": "user", "content": "a", "timestamp": "2026-08-13T09:00:00"},
        {"role": "assistant", "content": "b", "timestamp": "2026-08-13T09:01:00"},
        {"role": "user", "content": "c", "timestamp": "2026-08-13T09:02:00"},
    ])
    store = _observed(tmp_path, session, "Date: Aug 13, 2026\n* 🔴 (09:00) x")
    session.messages[2] = {"role": "user", "content": "new", "timestamp": "2026-08-14T09:00:00"}

    assert session_context_payload(session, store)["observed_messages"] == 0


def test_session_context_without_memory_reports_no_observations() -> None:
    session = Session(key="websocket:context", messages=[{"role": "user", "content": "hello"}])

    payload = session_context_payload(session)

    assert payload["observations"] is None
    assert payload["observed_at"] is None
    assert payload["estimated_observation_tokens"] == 0


def test_session_context_sanitizes_usage_metadata() -> None:
    usage = LLMUsage.reported(
        input_tokens=120,
        output_tokens=8,
        total_tokens=175,
        cache_read_tokens=48,
    ).with_timing(generation_ms=400, ttft_ms=75)
    session = Session(
        key="websocket:context",
        metadata={"_last_usage": usage.to_dict()},
    )

    payload = session_context_payload(session)

    assert payload["last_usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "total_tokens": 175,
        "context_tokens": 120,
        "cached_tokens": 48,
        "request_count": 1,
        "estimated_tokens": 0,
        "generation_ms": 400,
        "measured_completion_tokens": 8,
        "ttft_ms": 75,
        "timed_requests": 1,
    }
