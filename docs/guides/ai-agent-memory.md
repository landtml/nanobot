# How AI Agent Memory Works in nanobot

This guide explains how to use nanobot's long-term AI agent memory:
Observational Memory, the observation log it shares across conversations, and
Git-backed memory changes.

## What you will build

- a workspace whose conversations are remembered across sessions and channels
- an observation log that condenses itself as it grows
- durable profile files such as `SOUL.md` and `USER.md`
- a versioned memory history you can inspect and restore

## When to use this

Memory is always on. It helps whenever an agent should remember preferences,
project facts, decisions and ongoing work across conversations. nanobot keeps
the full chat history in sessions and sends the model dated observations
instead of an ever-growing transcript.

## Install

```bash
python -m pip install nanobot-ai
nanobot onboard --wizard
nanobot agent -m "Hello!"
```

## Minimal working example

Talk to the agent as usual. Once a conversation grows long, nanobot observes it
in the background. To observe the current conversation right away:

```text
/compact
```

See what nanobot remembers:

```text
/memory
```

Inspect the latest memory change:

```text
/memory-log
```

The observation log lives in the active workspace, usually
`~/.nanobot/workspace/memory/observations.md`.

## Production notes

- Use one workspace per project or personal context; every conversation in a
  workspace shares its memory.
- Set `agents.defaults.memory.modelOverride` to a fast, inexpensive model
  preset: observation and reflection run often.
- Keep the default thresholds unless you have a reason; they are the
  configuration behind Observational Memory's LongMemEval result.
- Review Git-backed memory changes when memory affects important workflows.

## Security notes

- Memory files may contain sensitive user or project facts.
- Avoid sharing workspaces without reviewing `SOUL.md`, `USER.md`, and
  `memory/observations.md`.
- Use separate workspaces for personal and team contexts.
- Temporary chats never write memory and never see it.

## Troubleshooting

- If memory seems to miss something recent, it may not be observed yet: run
  `/compact` in that conversation, then `/memory`.
- If memory changed incorrectly, use `/memory-restore` to inspect and restore
  previous versions.
- If a new session lacks context, confirm it uses the same workspace.

## Related nanobot docs

- [AI Agent Memory in nanobot](../memory.md)
- [Concepts](../concepts.md)
- [Configuration](../configuration.md#memory)
- [Chat Commands](../chat-commands.md)
