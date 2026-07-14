"""Track which file line ranges the model has actually observed.

The exploration agent can "see" file contents two ways:

- via the **Read** tool, whose output is fenced as ``` ```<path>:<start>-<end> ```;
- via the **Grep** tool, whose ``--heading`` output lists a file path on its own line followed by
  ``<n>:<match>`` and ``<n>-<context>`` lines.

We accumulate, per file, the set of line numbers the model has seen, so the final answer can drop
citations that reference ranges the model never actually opened (a common hallucination).
"""

import json
import os
import re

# Read output header: ```/abs/path/to/file.py:12-40
_READ_HEADER_RE = re.compile(r"```(.+):(\d+)-(\d+)")
# Grep content line: "12:match" or "12-context"; the digits are the line number.
_GREP_LINE_RE = re.compile(r"^(\d+)[:-]")

# file realpath -> set of observed line numbers
type ObservedLines = dict[str, set[int]]


def _normalize(path: str, cwd: str | None) -> str:
    if cwd and not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.realpath(path)


def _heading_path(line: str, cwd: str | None) -> str | None:
    """Return the normalized path of a Grep file heading, or None if the line is not one.

    ripgrep only prints a heading when it searches more than one file, and the tool interleaves
    non-path notes ("No matches found", truncation notes, ``<system-reminder>`` blocks) into the
    same output. A line only counts as a heading if it resolves to an existing file inside the
    workspace; this also disambiguates a heading such as ``2024-01-01.log`` from a numbered
    content line.
    """
    candidate = line.strip()
    if not candidate:
        return None
    path = _normalize(candidate, cwd)
    if not os.path.isfile(path):
        return None
    if cwd and not path.startswith(os.path.realpath(cwd) + os.sep) and path != os.path.realpath(cwd):
        return None
    return path


def record_read(observed: ObservedLines, output: str, cwd: str | None = None) -> None:
    """Record the line range returned by a Read tool result."""
    m = _READ_HEADER_RE.search(output)
    if not m:
        return
    path, start, end = m.group(1).strip(), int(m.group(2)), int(m.group(3))
    observed.setdefault(_normalize(path, cwd), set()).update(range(start, end + 1))


def record_grep(observed: ObservedLines, output: str, cwd: str | None = None, path: str | None = None) -> None:
    """Record every line number revealed by a Grep tool result.

    Both match lines (``n:``) and context lines (``n-``) count as observed, because the model can
    read a file's content through Grep context instead of the Read tool.

    ``path`` is the Grep tool call's ``path`` argument. When it points at a single file, ripgrep
    prints no heading at all, so the numbered lines must be attributed to that file.
    """
    current: str | None = None
    if path:
        searched = _normalize(path, cwd)
        if os.path.isfile(searched):
            current = searched

    for line in output.splitlines():
        if not line or line == "--":
            continue
        heading = _heading_path(line, cwd)
        if heading is not None:
            current = heading
            continue
        m = _GREP_LINE_RE.match(line)
        if m and current is not None:
            observed.setdefault(current, set()).add(int(m.group(1)))


def _grep_path(arguments: str | None, cwd: str | None) -> str | None:
    try:
        params = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(params, dict):
        return None
    # The Grep tool defaults its search path to the working directory.
    return params.get("path") or cwd


def record_tool_results(observed: ObservedLines, tool_calls, results, cwd: str | None = None) -> None:
    """Fold a turn's Read/Grep results into the observed-lines map.

    ``tool_calls`` is the assistant message's tool calls (each has ``id``, ``name`` and JSON
    ``arguments``); ``results`` is the list of tool result messages (each has ``tool_call_id`` and
    ``content``).
    """
    call_by_id = {c.id: c for c in tool_calls}
    for r in results:
        call = call_by_id.get(r.tool_call_id)
        if call is None:
            continue
        output = r.content or ""
        if call.name == "Read":
            record_read(observed, output, cwd)
        elif call.name == "Grep":
            record_grep(observed, output, cwd, path=_grep_path(call.arguments, cwd))


def citation_observed(observed: ObservedLines, citation: dict, cwd: str | None = None) -> bool:
    """True if the cited range overlaps any observed line for that file.

    Uses zero-overlap semantics: a citation is verified if the model saw *at least one* line of the
    cited range. Paths are normalized against ``cwd`` exactly like the recording side, so relative
    citations key identically.
    """
    lines = observed.get(_normalize(citation["path"], cwd))
    if not lines:
        return False
    return any(n in lines for n in range(citation["start_line"], citation["end_line"] + 1))


def unverified_citations(observed: ObservedLines, citations: list, cwd: str | None = None) -> list:
    """Return the citations whose ranges were never observed."""
    return [c for c in citations if not citation_observed(observed, c, cwd)]


def correction_message(unverified: list) -> str:
    """Build the user message asking the model to fix unverified citations."""
    listed = "\n".join(f"- {c['path']}:{c['line_range']}" for c in unverified)
    return (
        "Your final answer cites line ranges you never opened during exploration "
        "(they were not returned by the Read tool, nor shown in Grep output):\n"
        f"{listed}\n\n"
        "For each one, either open the exact lines with the Read or Grep tool to confirm them, or "
        "remove/replace the citation. Then provide a corrected final answer. Citations you have not "
        "actually seen will be removed from your answer."
    )
