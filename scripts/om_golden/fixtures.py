"""Write the fixture inputs shared by the TypeScript generator and the Python tests.

Run with ``python scripts/om_golden/fixtures.py``. The output is deterministic,
so a regenerated file only changes when the fixtures below change.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = (
    Path(__file__).resolve().parents[2]
    / "tests" / "agent" / "observational_memory" / "golden" / "fixtures.json"
)


def text(mid: str, role: str, created_at: str | None, body: str) -> dict:
    return {
        "id": mid,
        "role": role,
        "createdAt": created_at,
        "content": {"format": 2, "parts": [{"type": "text", "text": body}]},
    }


def conversations() -> dict[str, list[dict]]:
    return {
        "plain": [
            text("m1", "user", "2025-12-04T14:30:05.000Z",
                 "I have two kids and I work at Acme Corp."),
            text("m2", "assistant", "2025-12-04T14:30:40.000Z",
                 "Nice to meet you! How old are your kids?"),
            text("m3", "user", "2025-12-04T00:05:00.000Z",
                 "They are 4 and 7. I'm switching from Python to Rust next week."),
            text("m4", "assistant", "2025-12-04T12:00:00.000Z",
                 "Rust is a great choice.\n\nLet me know if you want a study plan."),
        ],
        "string_and_fallback": [
            {"id": "s1", "role": "user", "createdAt": "2024-02-29T23:59:59.000Z",
             "content": "Leap-day question: what's the weather like in Tórshavn? 🌧️"},
            {"id": "s2", "role": "assistant", "createdAt": None,
             "content": {"format": 2, "parts": [], "content": "Fallback content string."}},
            {"id": "s3", "role": "system", "createdAt": "2026-07-01T09:07:00.000Z",
             "content": {"format": 2, "parts": [
                 {"type": "step-start"},
                 {"type": "data-om-observation-start", "data": {"cycleId": "x"}},
                 {"type": "text", "text": "Only this text is visible."},
             ]}},
        ],
        "tools": [
            text("t1", "user", "2026-03-08T10:15:00.000Z", "List the files in src/ please."),
            {"id": "t2", "role": "assistant", "createdAt": "2026-03-08T10:15:03.000Z",
             "content": {"format": 2, "parts": [
                 {"type": "text", "text": "Let me look."},
                 {"type": "tool-invocation", "toolInvocation": {
                     "state": "call", "toolCallId": "c1", "toolName": "list_dir",
                     "args": {"path": "src", "recursive": False, "depth": 2}}},
                 {"type": "tool-invocation", "toolInvocation": {
                     "state": "result", "toolCallId": "c0", "toolName": "read_file",
                     "args": {"path": "README.md"},
                     "result": {"ok": True, "lines": ["# Title", "Body — ünïcödé"], "count": 2}}},
                 {"type": "tool-invocation", "toolInvocation": {
                     "state": "result", "toolCallId": "c2", "toolName": "exec",
                     "args": "raw-args", "result": "line one\nline two"}},
             ]}},
        ],
        "long": [
            text("l1", "user", "2026-11-02T08:00:00.000Z", "x" * 1200),
            text("l2", "assistant", "2026-11-02T08:01:00.000Z", "short reply"),
        ],
    }


def main() -> None:
    convs = conversations()
    fixtures = {
        "conversations": convs,
        "observer_prompts": [
            {"existing": None, "conversation": "plain"},
            {"existing": "Date: Dec 3, 2025\n* 🔴 (09:00) User likes tea", "conversation": "tools"},
            {"existing": "", "conversation": "string_and_fallback"},
        ],
        "multi_thread_prompts": [
            {"existing": None, "order": ["t-a", "t-b"],
             "threads": {"t-a": "plain", "t-b": "tools"}},
            {"existing": "<thread id=\"old\">\nDate: Jan 1, 2026\n* 🟡 (10:00) Old\n</thread>",
             "order": ["t-b", "missing", "t-a"],
             "threads": {"t-a": "long", "t-b": "string_and_fallback"}},
        ],
        "max_part_length": [{"conversation": "long", "max": 500}, {"conversation": "tools", "max": 20}],
        "reflector_prompts": [
            {"observations": "Date: Dec 4, 2025\n* 🔴 (14:30) A", "manual": None, "retry": False},
            {"observations": "obs", "manual": "focus on auth", "retry": True},
            {"observations": "obs", "manual": None, "retry": True},
        ],
        "observer_outputs": [
            "<observations>\nDate: Dec 4, 2025\n* 🔴 (14:30) User stated has two kids\n</observations>\n\n"
            "<current-task>\nPrimary: answer the question\n</current-task>\n\n"
            "<suggested-response>\nAsk about the kids.\n</suggested-response>",
            "Intro text\n  <observations>\n* one\n  </observations>\nmiddle\n<observations>\n* two\n"
            "</observations>\nUser discussed <observations> tags inline",
            "No tags here\n- first item\n  * nested item\n1. numbered\nplain line\n\t- tab item",
            "<observations>\n\n</observations>\n<current-task>inline task</current-task>\n"
            "<current-task>\n  second\n</current-task>",
            "",
        ],
        "reflector_outputs": [
            "<observations>\nDate: Dec 4, 2025\n* 🔴 merged\n</observations>\n"
            "<current-task>task</current-task>\n<suggested-response>say hi</suggested-response>",
            "no tags at all\njust text\n",
            "- only\n- list\nand prose",
        ],
        "multi_thread_outputs": [
            "<observations>\n<thread id=\"abc\">\nDate: Dec 4, 2025\n* 🔴 (14:30) Fact\n\n"
            "<current-task>\nTask A\n</current-task>\n\n<suggested-response>\nReply A\n"
            "</suggested-response>\n</thread>\n\n<thread id=\"def\">\nDate: Dec 5, 2025\n"
            "* 🟡 (09:15) Other\n</thread>\n</observations>",
            "<thread id=\"x1\">only thread</thread><thread id=\"x2\"></thread>",
            "garbage without threads",
        ],
        "optimize": [
            "* 🔴 (14:30) Keep critical\n* 🟡 (14:31) Medium [tag, other]\n* 🟢 Low -> arrow\n\n\n\n"
            "[72 items collapsed - ID: b1fa]  double  spaces\n  * -> nested",
            "   \n\n🟡🟢 only emojis -> [x]  \n",
        ],
        "relative_time": [
            {"now": "2026-01-15T12:00:00.000Z",
             "observations": "Date: Dec 4, 2025\n* 🔴 (14:30) User will start diet "
                             "(meaning Dec 8, 2025)\n* 🟡 trip (estimated May 27-28, 2026)\n\n"
                             "Date: Dec 20, 2025\n* note (estimated late December 2025)\n"
                             "Date: January 14, 2026\n* (meaning mid-to-late January 2026)\n"
                             "Date: Jan 15, 2026\n* plans to go (estimated June to early July 2025)\n"
                             "* unparseable (meaning sometime 2025)"},
            {"now": "2026-03-01T00:30:00.000Z",
             "observations": "no date headers (meaning Feb 28, 2026) here"},
            {"now": "2027-06-01T10:00:00.000Z",
             "observations": "Date: Feb 2, 2025\nx\nDate: Feb 3, 2025\ny\nDate: Feb 9, 2025\n"
                             "Date: Mar 1, 2025\nDate: Apr 30, 2025\nDate: Dec 31, 2025"},
        ],
        "context": [
            {"observations": "Date: Dec 4, 2025\n* 🔴 (14:30) Fact -> detail",
             "current_task": "Primary: help", "suggested": "Say hello",
             "unobserved": None, "now": "2025-12-06T08:00:00.000Z"},
            {"observations": "obs", "current_task": None, "suggested": None,
             "unobserved": "<other-conversation id=\"1\">\nx\n</other-conversation>",
             "now": None},
        ],
        "thread_sections": [
            {"existing": "", "thread": "telegram:1", "observations": "Date: Dec 4, 2025\n* a"},
            {"existing": "<thread id=\"THREAD\">\nDate: Dec 4, 2025\n* a\n</thread>",
             "thread": "telegram:1", "observations": "Date: Dec 4, 2025\n* b\n* c"},
            {"existing": "<thread id=\"THREAD\">\nDate: Dec 4, 2025\n* a\n</thread>",
             "thread": "telegram:1", "observations": "Date: Dec 5, 2025\n* d"},
            {"existing": "prior", "thread": "webui:x", "observations": "<thread id=\"leak\">* e</thread>"},
            {"existing": "prior", "thread": "cli:direct", "observations": "no date line"},
        ],
        "thread_ids": ["telegram:12345", "webui:abc-def", "unified:default", "", "cli:direct", "ü:ñ"],
        "token_strings": [
            "", "hello world", "Date: Dec 4, 2025\n* 🔴 (14:30) User stated has two kids",
            "<|endoftext|> special", "日本語のテキスト", "x" * 5000,
        ],
        "selection": [
            {"messageTokens": 30000, "maxTokensPerBatch": 10000,
             "threads": {"a": "plain", "b": "tools", "c": "long", "d": "string_and_fallback"}},
            {"messageTokens": 50, "maxTokensPerBatch": 60,
             "threads": {"a": "plain", "b": "tools", "c": "long", "d": "string_and_fallback"}},
        ],
        "unobserved_blocks": [
            {"current": "a", "threads": {"a": "plain", "b": "tools", "c": "long"}},
        ],
        "thresholds": [
            {"messageTokens": 30000, "observationTokens": 40000, "current": 0},
            {"messageTokens": {"min": 30000, "max": 70000}, "observationTokens": 40000, "current": 10000},
            {"messageTokens": {"min": 30000, "max": 70000}, "observationTokens": 40000, "current": 55000},
        ],
    }
    OUT.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
