# AI Agent Memory in nanobot

This page explains how nanobot remembers: Observational Memory, the observation
log it keeps for every conversation in a workspace, how that log is condensed,
what the agent sees, and how every change is versioned.

nanobot's memory is built on a simple belief: memory should feel alive, but it should not feel chaotic.

Good memory is not a pile of notes. It is a quiet system of attention. It notices what is worth keeping, lets go of what no longer needs the spotlight, and turns lived experience into something calm, durable, and useful.

That is the shape of memory in nanobot.

## The Design

nanobot's memory is **Observational Memory**, a faithful port of the system
[Mastra](https://mastra.ai/research/observational-memory) released as
`@mastra/memory@1.1.0`. In that configuration it scored 84.23% on
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) with `gpt-4o` and
94.87% with `gpt-5-mini`. nanobot does not approximate it: the prompts,
formatting, parsing, token counting, thresholds and selection rules are held to
that release byte for byte by tests (see [Faithful to the benchmark](#faithful-to-the-benchmark)).

The idea is the way people remember a working day. Nobody keeps a transcript.
You keep **observations**: dated, prioritized notes about what happened, what
was decided, and what matters next.

- An **Observer** reads conversation that has not been observed yet and writes
  observations about it.
- A **Reflector** condenses the observation log when it grows large, merging
  and dropping what no longer needs the spotlight.
- The agent (the **Actor**) sees the log in its system prompt, with dates
  annotated relative to today, and continues from it.

One observation log is shared by every conversation in the workspace. What you
tell nanobot on Telegram is remembered in the WebUI, and the other way round.

## The Flow

### Observe

After each reply, nanobot counts the tokens that are not observed yet: the rest
of the current conversation plus the unobserved part of every other
conversation in the workspace. Once that backlog reaches **30,000 tokens**, the
largest conversations are selected, split into batches of about 10,000 tokens,
and observed in parallel in the background. The reply is never delayed.

Each conversation gets its own section in the log, so observations stay
attributable. The Observer also records the conversation's **current task** and
a **suggested response**, which the agent sees the next time that conversation
continues.

Observed messages leave the model's context the next time their conversation
runs. Your saved chat history is never rewritten: the messages stay in the
session file and in the WebUI. They are just no longer sent to the model word
for word, because the observations carry them.

### Observe under pressure

If a single turn fills the context window before the threshold is reached (a
long tool loop, a huge file), nanobot observes the current conversation on the
spot, then continues the same turn from its observations. `/compact` does the
same on demand.

### Reflect

When the observation log grows past **40,000 tokens**, the Reflector rewrites
it into a denser log. If the first rewrite is not smaller than the threshold,
it tries once more with a stronger compression instruction. `/memory reflect`
runs a reflection immediately.

### Recall

Every request carries the log as the last section of the system prompt, so the
stable part of the prompt before it stays cacheable. The agent sees:

- the observation log, with relative dates such as "(yesterday)" or
  "(2 weeks ago)" added for the current day;
- the current conversation's task and suggested response;
- the recent, not-yet-observed messages of *other* conversations, so a thread
  started elsewhere is not lost while it waits to be observed.

## The Files

In this page, `workspace` means the configured **agent workspace** (the default
is `~/.nanobot/workspace/`, or the path passed with `--workspace`). Selecting a
different project in the WebUI changes that chat's project context and tool
working directory; it does not relocate the files below.

```text
workspace/
├── SOUL.md                          # The bot's long-term voice and communication style
├── USER.md                          # Stable knowledge about the user
└── memory/
    ├── observations.md              # The observation log (plain text)
    ├── observational_memory.json    # Per-conversation cursors, tasks, reflection count
    └── .git/                        # Version history for SOUL.md, USER.md, observations.md
```

A selected project may provide its own `AGENTS.md`, but project-local `SOUL.md`,
`USER.md`, and `memory/` do not replace the agent-owned files above. This keeps
one agent's profile and memory continuous while it works across projects. Use a
separate configured agent workspace when identity or memory must be isolated.

These files play different roles:

- `SOUL.md` remembers how nanobot should sound.
- `USER.md` remembers who the user is and what they prefer.
- `observations.md` remembers what happened and what remains true.

`observations.md` is ordinary text, grouped by conversation and date:

```text
<thread id="6f3a9c1e">
Date: Sep 24, 2026
* 🔴 (09:14) User is migrating the billing service to PostgreSQL 16
* 🟡 (09:20) Agent proposed a two-phase cutover; user approved phase one
</thread>
```

Read it, grep it, or show it with `/memory`. Do not edit it by hand while
nanobot runs: every write goes through a file lock and a revision check so a
CLI and a gateway sharing one workspace never overwrite each other, and the
agent's shell tool refuses direct writes to the memory files. To change
memory, use `/memory-restore`, or `bot.memory.write(...)` from the
[Python SDK](./python-sdk.md).

## Commands

Memory is not hidden behind the curtain. Users can inspect and guide it.

| Command | What it does |
|---------|--------------|
| `/memory` | Show memory status and the most recent observations |
| `/memory reflect` | Condense the observation log now |
| `/compact` | Observe this conversation now and continue from memory |
| `/memory-log` | Show the latest memory change |
| `/memory-log <sha>` | Show a specific memory change |
| `/memory-restore` | List recent memory versions |
| `/memory-restore <sha>` | Restore memory to the state before a specific change |

On Telegram, use `/memory_log` and `/memory_restore`.

These commands exist for a reason: automatic memory is powerful, but users should always retain the right to inspect, understand, and restore it.

## Versioned Memory

Every change to the observation log is a commit in the workspace's memory
repository, with a `memory:` message that says what happened (`memory: observe 2
session(s)`, `memory: reflect`, `memory: compact one session`).

This gives memory a history of its own:

- you can inspect what changed
- you can compare versions
- you can restore a previous state

That turns memory from a silent mutation into an auditable process.

## Private Conversations

Temporary chats and other sessions that are not saved never write memory.
They do not see the workspace's observations either, and when such a
conversation is condensed under pressure the Observer is not shown them. What a
private conversation observes about itself stays with that conversation and
disappears with it.

Subagents and one-off runs (for example SDK calls with `ephemeral=True`) never
change memory.

## Configuration

Memory is configured under `agents.defaults.memory`:

```json
{
  "agents": {
    "defaults": {
      "memory": {
        "modelOverride": null,
        "messageTokens": 30000,
        "observationTokens": 40000,
        "maxTokensPerBatch": 10000
      }
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `modelOverride` | Optional model preset for the Observer and Reflector |
| `messageTokens` | Unobserved tokens that trigger an observation |
| `observationTokens` | Observation log size that triggers a reflection |
| `maxTokensPerBatch` | Size of each parallel Observer call |

`modelOverride` selects a named entry from `modelPresets`; raw model
identifiers are not supported. If omitted, memory uses the agent's default
model. Observation and reflection run often and in the background, so a fast,
inexpensive model works well; Mastra's default for both is `gemini-2.5-flash`.

The defaults are the values behind the LongMemEval result. Change them only
with a reason: lower thresholds observe and reflect more often (more model
calls, a smaller context), higher ones do the opposite. The WebUI offers both
thresholds under **Settings → Capabilities → Memory**; changes apply after a
restart.

Memory calls are counted under **Memory** in token usage.

## Faithful to the Benchmark

`nanobot/agent/observational_memory/` is a Python port of the 1.1.0 processor.
`tests/agent/observational_memory/test_golden.py` compares it with the
TypeScript original on shared fixtures, in three time zones: every prompt, the
message formatting, the output parsers, the rendered context block, token
counts (`o200k_base`), and which conversations are selected and batched. How
to regenerate those fixtures is described in `scripts/om_golden/README.md`.

A few differences are deliberate, and none of them touch what LongMemEval
measures:

- Tool calls and their results are shown to the Observer together, and very
  large tool payloads are truncated.
- Conversation ids are obscured in the Observer prompt as well as in the
  Actor's context.
- Observation runs after a reply rather than between tool calls; context
  pressure inside a turn is handled by observing on the spot.

## Upgrading from Dream

Earlier versions of nanobot remembered with **Dream**, a scheduled job that
edited `memory/MEMORY.md` from summaries in `memory/history.jsonl`. On the
first start after upgrading:

- a customized `memory/MEMORY.md` is carried into the observation log once, as
  one dated entry, so nothing Dream learned is lost; the file itself is left
  in place;
- the Dream cron job is removed, and the `/dream*` commands no longer exist;
- `agents.defaults.dream.modelOverride` is used as the memory model preset
  unless `agents.defaults.memory` sets one; the other Dream and idle
  auto-compaction settings are ignored and dropped the next time the
  configuration is saved;
- conversations that have not changed since the upgrade are not observed
  retroactively, so an upgrade does not bill a model for old history. They
  join memory as soon as they continue.

`memory/history.jsonl` and `prompts/dream.md` are no longer read. You can
keep them for reference or delete them.

## In Practice

What this means in daily use is simple:

- conversations can stay fast without carrying infinite context
- what happened becomes dated, prioritized observations instead of a growing transcript
- every conversation benefits from what the others learned
- the user can inspect and restore memory when needed

Memory should not feel like a dump. It should feel like continuity.

That is what this design is trying to protect.
