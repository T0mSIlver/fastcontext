"""Compute eval metrics from a FastContext trajectory.

A trajectory is a JSONL file (one message per line) as written by
``fastcontext.agent.context.Context``. Each line is one of:

- ``{"role": "system", ...}``      the system prompt
- ``{"role": "user", ...}``        the query, a "max turns" nudge, or a
                                   self-correction request
- ``{"role": "assistant", "tool_calls": [...], "usage": {...}}``  an LLM step
- ``{"role": "tool", "tool_call_id": ..., "content": ...}``       a tool result

This module is intentionally self-contained (it re-implements the
observed-lines / citation-verification logic from
``fastcontext.agent.observed``) so the analyzer can score *any* trajectory --
including the pre-recorded example trajectories -- without importing or
matching the exact branch of FastContext that produced it.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tool-result parsing
# ---------------------------------------------------------------------------

# A failed tool call: ripgrep / read / glob surface errors as a system-reminder
# block. "No files found" (Glob) and empty results are *valid* answers, not
# failures, so we only count explicit Error / Permission error reminders.
_FAILED_TOOL_RE = re.compile(r"<system-reminder>\s*(?:[A-Za-z ]*?error|Error)\b", re.IGNORECASE)

# Read output header: ```/abs/path/to/file.py:12-40  (line content follows as "12|...")
_READ_HEADER_RE = re.compile(r"```(.+):(\d+)-(\d+)")
# Grep --heading content line: "12:match" or "12-context"; the digits are the line number.
_GREP_LINE_RE = re.compile(r"^(\d+)[:-]")

# The self-correction loop injects this user message when the final answer cites
# unopened line ranges (see fastcontext.agent.observed.correction_message).
_CORRECTION_MARKER = "cites line ranges you never opened"
# Forced-final nudge when max turns is hit (not a correction).
_MAX_TURNS_MARKER = "Max number of turns reached"

# The agent writes this into its answer/stdout when the LLM endpoint fails mid-run
# ("LLM API call failed. So stopping the agent."). Such a run is a harness failure,
# not a data point, and must be reported as an error rather than scored as zeros.
LLM_FAILURE_MARKER = "LLM API call failed"


def _is_failed_tool_result(content: str) -> bool:
    return bool(_FAILED_TOOL_RE.search(content or ""))


def _normalize(path: str, cwd: str | None) -> str:
    if cwd and not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.realpath(path)


def _record_read(observed: dict, output: str, cwd: str | None) -> None:
    m = _READ_HEADER_RE.search(output or "")
    if not m:
        return
    path, start, end = m.group(1).strip(), int(m.group(2)), int(m.group(3))
    observed.setdefault(_normalize(path, cwd), set()).update(range(start, end + 1))


def _record_grep(observed: dict, output: str, cwd: str | None) -> None:
    current: str | None = None
    for line in (output or "").splitlines():
        if not line or line == "--":
            continue
        m = _GREP_LINE_RE.match(line)
        if m and current is not None:
            observed.setdefault(current, set()).add(int(m.group(1)))
        elif not m:
            current = _normalize(line.strip(), cwd)


# ---------------------------------------------------------------------------
# Citation parsing (mirrors fastcontext.agent.utils.parse_citations)
# ---------------------------------------------------------------------------

_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_CITATION_RE = re.compile(r"(.+?):(\d+(?:-\d+)?)\s*(.*)")


def parse_citations(text: str) -> list[dict]:
    block = _FINAL_ANSWER_RE.search(text or "")
    if not block:
        return []
    citations = []
    for entry in block.group(1).strip().splitlines():
        entry = entry.strip()
        if not entry:
            continue
        m = _CITATION_RE.match(entry)
        if not m:
            continue
        line_range = m.group(2).strip()
        start, end = line_range.split("-") if "-" in line_range else (line_range, line_range)
        citations.append(
            {
                "path": m.group(1).strip(),
                "line_range": line_range,
                "start_line": int(start),
                "end_line": int(end),
                "explanation": m.group(3).strip(),
            }
        )
    return citations


def _citation_observed(observed: dict, citation: dict) -> bool:
    lines = observed.get(os.path.realpath(citation["path"]))
    if not lines:
        return False
    return any(n in lines for n in range(citation["start_line"], citation["end_line"] + 1))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    turns: int = 0  # number of LLM (assistant) steps
    tool_calls_total: int = 0
    tool_usage: dict = field(default_factory=dict)  # {Read: n, Glob: n, Grep: n}
    failed_tool_calls: int = 0
    duplicate_tool_calls: int = 0  # (name, args) tuples seen more than once
    corrections: int = 0  # self-correction retries triggered (cap 2)
    citations_total: int = 0  # citations in the final answer to existing files
    citations_missing_file: int = 0  # citations whose file does not exist
    unverified_citations: int = 0  # cited ranges the model never opened
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reached_final_answer: bool = False
    error: str | None = None  # populated when the run aborted (LLM/API error)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_trajectory(path: str) -> list[dict]:
    messages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def compute_metrics(messages: list[dict], cwd: str | None = None, final_answer: str | None = None) -> RunMetrics:
    """Score a single trajectory.

    ``cwd`` is the repo the run explored (used to resolve relative paths in
    tool output); when ``None`` we rely on the absolute paths already present.
    ``final_answer`` is the agent's returned ``<final_answer>`` block (citation
    mode stdout) when captured; otherwise the last assistant message is used.
    """
    m = RunMetrics()
    observed: dict = {}
    seen_calls: Counter = Counter()
    name_by_call_id: dict = {}
    last_assistant_content = ""

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            content = msg.get("content") or ""
            # A pure LLM step counts as a turn.
            m.turns += 1
            usage = msg.get("usage") or {}
            m.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            m.completion_tokens += usage.get("completion_tokens", 0) or 0
            m.total_tokens += usage.get("total_tokens", 0) or 0
            if LLM_FAILURE_MARKER in content:
                m.error = content.strip().splitlines()[0]
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                m.tool_calls_total += 1
                m.tool_usage[name] = m.tool_usage.get(name, 0) + 1
                key = (name, _canonical_args(args))
                if seen_calls[key]:
                    m.duplicate_tool_calls += 1
                seen_calls[key] += 1
                if tc.get("id"):
                    name_by_call_id[tc["id"]] = name
            if content and not tool_calls:
                last_assistant_content = content

        elif role == "tool":
            content = msg.get("content") or ""
            if _is_failed_tool_result(content):
                m.failed_tool_calls += 1
            name = name_by_call_id.get(msg.get("tool_call_id"))
            if name == "Read":
                _record_read(observed, content, cwd)
            elif name == "Grep":
                _record_grep(observed, content, cwd)
            elif name is None:
                # Fall back to content shape when ids don't line up.
                if _READ_HEADER_RE.search(content):
                    _record_read(observed, content, cwd)

        elif role == "user":
            content = msg.get("content") or ""
            if _CORRECTION_MARKER in content:
                m.corrections += 1

    answer_text = final_answer if final_answer is not None else last_assistant_content
    if m.error is None and LLM_FAILURE_MARKER in (answer_text or ""):
        m.error = LLM_FAILURE_MARKER
    citations = parse_citations(answer_text)
    for c in citations:
        if not os.path.isfile(c["path"]):
            m.citations_missing_file += 1
            continue
        m.citations_total += 1
        if not _citation_observed(observed, c):
            m.unverified_citations += 1

    m.reached_final_answer = bool(_FINAL_ANSWER_RE.search(answer_text or "")) and m.error is None
    return m


def _canonical_args(args) -> str:
    """Normalize a tool call's arguments for duplicate detection."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return args.strip()
    try:
        return json.dumps(args, sort_keys=True)
    except (TypeError, ValueError):
        return str(args)


def score_run(trajectory_path: str, cwd: str | None = None, final_answer: str | None = None) -> RunMetrics:
    return compute_metrics(load_trajectory(trajectory_path), cwd=cwd, final_answer=final_answer)
