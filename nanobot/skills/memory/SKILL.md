---
name: memory
description: Recall past conversations beyond the observations in your memory section.
---

# Memory

nanobot remembers automatically. When a conversation grows long, an observer
condenses it into dated observations; they appear in your memory section,
newest last, and earlier messages leave the conversation. You never need to
save anything yourself, and you must not edit the memory files.

## When the observations are not enough

Observations keep facts, decisions and outcomes, not exact wording. To recover
details from an earlier conversation:

1. `search_sessions(query="...")` finds past sessions by title or message text.
2. `read_session(...)` reads the matching session for exact wording, numbers or
   tool output.

To search the full observation log itself, `grep` the `Memory` path from the
system prompt, for example:
`grep(pattern="project-name", path="<memory-path>", output_mode="content", case_insensitive=true)`

## Reading dates

Observations are grouped under `Date:` headers with 24-hour times, and relative
annotations such as "(2 weeks ago)" are computed from today. When facts
conflict, the most recent observation wins.
