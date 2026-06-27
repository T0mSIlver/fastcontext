"""Track which file line ranges the model has actually observed.

The exploration agent can "see" the contents of a file two ways:

- via the **Read** tool, whose output is fenced as ``` ```<path>:<start>-<end> ```;
- via the **Grep** tool, whose ``--heading`` output lists a file path on its
  own line followed by ``<n>:<match>`` and ``<n>-<context>`` lines.

We accumulate, per file, the set of line numbers the model has seen, so the
final-answer validator can flag citations that reference ranges the model never
actually opened (a common hallucination).
"""

import os
import re

# Read output header: ```/abs/path/to/file.py:12-40
_READ_HEADER_RE = re.compile(r"```(.+):(\d+)-(\d+)")
# Grep content line: "12:match" or "12-context"; the digits are the line number.
_GREP_LINE_RE = re.compile(r"^(\d+)[:-]")

# Type alias for readability: file realpath -> set of observed line numbers.
ObservedLines = dict


def _normalize(path: str, cwd: str | None) -> str:
    if cwd and not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.realpath(path)


def record_read(observed: ObservedLines, output: str, cwd: str | None = None) -> None:
    """Record the line range returned by a Read tool result."""
    m = _READ_HEADER_RE.search(output)
    if not m:
        return
    path, start, end = m.group(1).strip(), int(m.group(2)), int(m.group(3))
    observed.setdefault(_normalize(path, cwd), set()).update(range(start, end + 1))


def record_grep(observed: ObservedLines, output: str, cwd: str | None = None) -> None:
    """Record every line number revealed by a Grep tool result.

    Both match lines (``n:``) and context lines (``n-``) count as observed,
    because the model can read a file's content through Grep context instead of
    the Read tool.
    """
    current: str | None = None
    for line in output.splitlines():
        if not line or line == "--":
            continue
        m = _GREP_LINE_RE.match(line)
        if m and current is not None:
            observed.setdefault(current, set()).add(int(m.group(1)))
        elif not m:
            # A line that is not a numbered content line is a file heading.
            current = _normalize(line.strip(), cwd)


def record_tool_results(observed: ObservedLines, tool_calls, results, cwd: str | None = None) -> None:
    """Fold a turn's Read/Grep results into the observed-lines map.

    ``tool_calls`` is the assistant message's tool calls (each has ``id`` and
    ``name``); ``results`` is the list of tool result messages (each has
    ``tool_call_id`` and ``content``).
    """
    name_by_id = {c.id: c.name for c in tool_calls}
    for r in results:
        name = name_by_id.get(r.tool_call_id)
        output = r.content or ""
        if name == "Read":
            record_read(observed, output, cwd)
        elif name == "Grep":
            record_grep(observed, output, cwd)


def citation_observed(observed: ObservedLines, citation: dict) -> bool:
    """True if the cited range overlaps any observed line for that file.

    Uses zero-overlap semantics: a citation is considered verified if the model
    saw *at least one* line in the cited range. A file that was never observed
    (or a range with no overlap) is unverified.
    """
    lines = observed.get(os.path.realpath(citation["path"]))
    if not lines:
        return False
    return any(n in lines for n in range(citation["start_line"], citation["end_line"] + 1))


def unverified_citations(observed: ObservedLines, citations: list) -> list:
    """Return the citations whose ranges were never observed."""
    return [c for c in citations if not citation_observed(observed, c)]


def correction_message(unverified: list) -> str:
    """Build the user message asking the model to fix unverified citations."""
    listed = "\n".join(f"- {c['path']}:{c['line_range']}" for c in unverified)
    return (
        "Your final answer cites line ranges you never opened during exploration "
        "(they were not returned by the Read tool, nor shown in Grep output):\n"
        f"{listed}\n\n"
        "For each one, either open the exact lines with the Read or Grep tool to "
        "confirm them, or remove/replace the citation. Then provide a corrected "
        "final answer. Do not cite line ranges you have not actually seen."
    )
