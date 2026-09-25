"""Read-only projection of the session material available to the agent."""

from __future__ import annotations

from typing import Any

from nanobot.agent.memory import message_fingerprint
from nanobot.agent.observational_memory.store import ObservationStore, Snapshot
from nanobot.providers.base import LLMUsage
from nanobot.session.manager import Session
from nanobot.utils.helpers import estimate_message_tokens, truncate_text

_OBSERVATIONS_PREVIEW_CHARS = 4_000


def _observed_messages(session: Session, snapshot: Snapshot | None) -> int:
    """Messages memory has covered, including ones observed since the last turn."""
    observed = session.last_archived
    thread = snapshot.state.threads.get(session.key) if snapshot is not None else None
    if thread is not None and 0 < thread.observed_count <= len(session.messages):
        last = session.messages[thread.observed_count - 1]
        if message_fingerprint(last) == thread.observed_fingerprint:
            observed = max(observed, thread.observed_count)
    return min(observed, len(session.messages))


def session_context_payload(
    session: Session,
    observations: ObservationStore | None = None,
) -> dict[str, Any]:
    """Return an explainable view of session replay without building a model prompt.

    The final prompt also contains workspace instructions, skills, and a
    model-specific token budget. This projection reports the session's replay
    (messages not yet observed) and the observation log that memory shows the
    agent, which is shared by every session in the workspace.
    """
    replay = session.get_history(max_messages=0, include_runtime_context=False)
    replay_tokens = sum(estimate_message_tokens(message) for message in replay)
    snapshot = observations.read() if observations is not None else None
    log = snapshot.observations if snapshot is not None else ""
    observation_tokens = (
        estimate_message_tokens({"role": "system", "content": log}) if log else 0
    )
    stored_usage = LLMUsage.from_dict(session.metadata.get("_last_usage"))
    last_usage = stored_usage.to_turn_dict() if stored_usage is not None else None

    return {
        "schema_version": 2,
        "session_key": session.key,
        "total_messages": len(session.messages),
        "observed_messages": _observed_messages(session, snapshot),
        "replay_messages": len(replay),
        "estimated_replay_tokens": replay_tokens,
        "estimated_observation_tokens": observation_tokens,
        "estimated_session_tokens": replay_tokens + observation_tokens,
        "observations": truncate_text(log, _OBSERVATIONS_PREVIEW_CHARS) if log else None,
        "observed_at": snapshot.state.last_observed_at if snapshot is not None else None,
        "last_usage": last_usage,
    }
