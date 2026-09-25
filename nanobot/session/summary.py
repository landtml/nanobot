"""The replay boundary a context compaction leaves in a session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nanobot.session.history_visibility import is_hidden_history_message

SUMMARY_CONTINUATION_TEXT = (
    "Continue the active task from the working-memory checkpoint above."
)

def is_summary_checkpoint(message: Mapping[str, Any]) -> bool:
    """Identify the durable boundary of a replacement summary."""
    return (
        is_hidden_history_message(message)
        and message.get("content") == SUMMARY_CONTINUATION_TEXT
    )


@dataclass(frozen=True, slots=True)
class SessionSummaryCheckpoint:
    """A replacement summary and the raw transcript boundary it covers."""

    summary: str
    transcript_boundary: int
