"""Tests for citation hallucination detection and the correction loop.

A citation is "verified" if the model actually opened at least one line of the cited range, either
through the Read tool or through Grep output (including context lines). Unverified citations
trigger up to MAX_CITATION_CORRECTIONS correction requests; whatever is still unverified after that
is dropped from the final answer.
"""

import asyncio
import json
import os
from pathlib import Path

from fastcontext.agent.agent import MAX_CITATION_CORRECTIONS, Agent
from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.observed import (
    citation_observed,
    correction_message,
    record_grep,
    record_read,
    record_tool_results,
    unverified_citations,
)
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.glob import GlobTool
from fastcontext.agent.tool.grep import GrepTool
from fastcontext.agent.tool.read import ReadTool
from fastcontext.agent.utils import format_citations, get_final_answer, parse_citations


def _cite(path, start, end=None):
    end = start if end is None else end
    line_range = str(start) if start == end else f"{start}-{end}"
    return {
        "path": str(path),
        "line_range": line_range,
        "start_line": start,
        "end_line": end,
        "explanation": "",
    }


def _repo(tmp_path):
    """A tiny real repo: two files with distinct content."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "alpha.py").write_text(
        "\n".join(["import os", "", "def target():", "    return 'needle'", "", "# tail"]),
        encoding="utf-8",
    )
    (repo / "beta.py").write_text(
        "\n".join(["# beta", "def other():", "    return 'needle'"]),
        encoding="utf-8",
    )
    return repo


# --- observed-lines tracking -------------------------------------------------


def test_record_read_captures_returned_range():
    observed: dict = {}
    record_read(observed, "```/repo/a.py:10-13\n10|x\n11|y\n12|z\n13|w\n```")
    assert observed[os.path.realpath("/repo/a.py")] == {10, 11, 12, 13}


def test_record_read_of_real_tool_output(tmp_path):
    repo = _repo(tmp_path)
    observed: dict = {}
    output = asyncio.run(ReadTool().call(json.dumps({"path": str(repo / "alpha.py")}), cwd=str(repo)))
    record_read(observed, output, cwd=str(repo))
    assert observed[os.path.realpath(str(repo / "alpha.py"))] == {1, 2, 3, 4, 5, 6}


def test_record_grep_single_file_has_no_heading(tmp_path):
    """Blocker regression: rg prints no heading when the search path is one file."""
    repo = _repo(tmp_path)
    target = repo / "alpha.py"
    args = {"pattern": "needle", "path": str(target), "output_mode": "content", "-C": 1}
    output = asyncio.run(GrepTool().call(json.dumps(args), cwd=str(repo)))
    assert str(target) not in output, "test premise: rg emits no heading for a single-file search"

    observed: dict = {}
    record_tool_results(
        observed,
        [FunctionCall(id="c1", name="Grep", arguments=json.dumps(args))],
        [Message(role="tool", tool_call_id="c1", content=output)],
        cwd=str(repo),
    )
    # match on line 4, context lines 3 and 5
    assert observed[os.path.realpath(str(target))] == {3, 4, 5}


def test_record_grep_directory_attributes_lines_to_right_files(tmp_path):
    repo = _repo(tmp_path)
    args = {"pattern": "needle", "path": str(repo), "output_mode": "content"}
    output = asyncio.run(GrepTool().call(json.dumps(args), cwd=str(repo)))
    assert str(repo / "alpha.py") in output, "test premise: rg emits headings for a directory search"

    observed: dict = {}
    record_tool_results(
        observed,
        [FunctionCall(id="c1", name="Grep", arguments=json.dumps(args))],
        [Message(role="tool", tool_call_id="c1", content=output)],
        cwd=str(repo),
    )
    assert observed[os.path.realpath(str(repo / "alpha.py"))] == {4}
    assert observed[os.path.realpath(str(repo / "beta.py"))] == {3}


def test_record_grep_defaults_path_to_cwd(tmp_path):
    """The Grep tool defaults `path` to the working directory, so an absent arg is not "no file"."""
    repo = _repo(tmp_path)
    args = {"pattern": "needle", "output_mode": "content"}
    output = asyncio.run(GrepTool().call(json.dumps(args), cwd=str(repo)))

    observed: dict = {}
    record_tool_results(
        observed,
        [FunctionCall(id="c1", name="Grep", arguments=json.dumps(args))],
        [Message(role="tool", tool_call_id="c1", content=output)],
        cwd=str(repo),
    )
    assert observed[os.path.realpath(str(repo / "alpha.py"))] == {4}


def test_record_grep_captures_match_and_context_lines(tmp_path):
    repo = _repo(tmp_path)
    a, b = repo / "alpha.py", repo / "beta.py"
    output = f"{a}\n2-line2\n3:line3 foo\n4-line4\n--\n7:line7 foo\n\n{b}\n1:foo here\n"
    observed: dict = {}
    record_grep(observed, output, cwd=str(repo))
    assert observed[os.path.realpath(str(a))] == {2, 3, 4, 7}
    assert observed[os.path.realpath(str(b))] == {1}


def test_record_grep_ignores_non_heading_lines(tmp_path):
    """Notes, reminders and no-match text are not file headings and must not be recorded."""
    repo = _repo(tmp_path)
    a = repo / "alpha.py"
    observed: dict = {}
    record_grep(observed, "No matches found", cwd=str(repo))
    record_grep(
        observed,
        f"<system-reminder>note</system-reminder>\n{a}\n1:x\nResults truncated to first 100 lines\n",
        cwd=str(repo),
    )
    assert list(observed) == [os.path.realpath(str(a))]
    assert observed[os.path.realpath(str(a))] == {1}


def test_record_grep_without_line_numbers_records_nothing(tmp_path):
    # output_mode content with "-n": false yields bare content lines, which are not observations.
    repo = _repo(tmp_path)
    args = {"pattern": "needle", "path": str(repo), "output_mode": "content", "-n": False}
    output = asyncio.run(GrepTool().call(json.dumps(args), cwd=str(repo)))
    observed: dict = {}
    record_tool_results(
        observed,
        [FunctionCall(id="c1", name="Grep", arguments=json.dumps(args))],
        [Message(role="tool", tool_call_id="c1", content=output)],
        cwd=str(repo),
    )
    assert all(not lines for lines in observed.values())


def test_record_grep_heading_that_looks_like_a_line_number(tmp_path):
    """A heading such as `2024-01-01.log` must not be parsed as line 2024 of the previous file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    log = repo / "2024-01-01.log"
    log.write_text("needle\n", encoding="utf-8")
    observed: dict = {}
    record_grep(observed, f"{log}\n1:needle\n", cwd=str(repo))
    assert observed == {os.path.realpath(str(log)): {1}}


# --- citation matching -------------------------------------------------------


def test_citation_observed_zero_overlap_semantics():
    observed = {os.path.realpath("/repo/a.py"): {10, 11, 12}}
    assert citation_observed(observed, _cite("/repo/a.py", 12, 20)) is True  # overlaps on 12
    assert citation_observed(observed, _cite("/repo/a.py", 40, 60)) is False  # no overlap
    assert citation_observed(observed, _cite("/repo/never.py", 1, 5)) is False  # unseen file


def test_citation_observed_normalizes_relative_paths(tmp_path):
    """Bug: a relative citation was resolved against the process cwd, never the agent work_dir."""
    repo = _repo(tmp_path)
    observed = {os.path.realpath(str(repo / "alpha.py")): {1, 2, 3}}
    assert citation_observed(observed, _cite("alpha.py", 2, 2), cwd=str(repo)) is True
    assert citation_observed(observed, _cite("./alpha.py", 2, 2), cwd=str(repo)) is True
    assert citation_observed(observed, _cite("alpha.py", 2, 2)) is False  # without cwd: no match


def test_unverified_citations_filters():
    observed = {os.path.realpath("/repo/a.py"): {1, 2, 3}}
    bad = unverified_citations(observed, [_cite("/repo/a.py", 1, 2), _cite("/repo/a.py", 90, 95)])
    assert len(bad) == 1 and bad[0]["line_range"] == "90-95"


def test_correction_message_lists_offenders():
    assert "/repo/a.py:90-95" in correction_message([_cite("/repo/a.py", 90, 95)])


# --- final-answer formatting -------------------------------------------------


def test_parse_citations_no_tag_returns_empty_list():
    # Regression: previously returned a dict, which made format_citations raise TypeError.
    assert parse_citations("just prose, no tag") == []
    assert get_final_answer("just prose, no tag", observed={}) == "<final_answer>\n\n</final_answer>"


def test_format_citations_drops_unverified(tmp_path):
    repo = _repo(tmp_path)
    f = repo / "alpha.py"
    text = f"<final_answer>\n{f}:1-3 (seen)\n{f}:90-95 (guessed)\n</final_answer>"
    observed = {os.path.realpath(str(f)): {1, 2, 3}}
    out = format_citations(parse_citations(text), observed=observed)
    assert f"{f}:1-3 (seen)" in out
    assert "90-95" not in out
    assert "unverified" not in out


def test_format_citations_without_observed_keeps_everything(tmp_path):
    repo = _repo(tmp_path)
    f = repo / "alpha.py"
    out = format_citations(parse_citations(f"<final_answer>\n{f}:90-95\n</final_answer>"))
    assert f"{f}:90-95" in out


def test_format_citations_validates_relative_path_against_cwd(tmp_path):
    # A relative citation must be validated against work_dir, not the process cwd. Before the fix
    # os.path.isfile("alpha.py") was checked from the wrong directory and the citation was dropped.
    repo = _repo(tmp_path)
    text = "<final_answer>\nalpha.py:1-3 (relative)\n</final_answer>"
    out = format_citations(parse_citations(text), cwd=str(repo))
    assert "alpha.py:1-3 (relative)" in out


# --- end-to-end correction loop ----------------------------------------------


class _FakeLLM:
    """Returns scripted assistant messages, ignoring the conversation."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.model = "fake"
        self.calls = 0

    async def acall(self, messages, tools=None, **kwargs):
        self.calls += 1
        return self.scripted.pop(0)


def _agent(tmp_path, scripted):
    repo = _repo(tmp_path)
    toolset = ToolSet([ReadTool(), GrepTool(), GlobTool()], work_dir=str(repo))
    agent = Agent(
        name="t",
        system_prompt="sys",
        llm=_FakeLLM(scripted),
        toolset=toolset,
        trajectory_file=str(tmp_path / "traj" / "t.jsonl"),
        work_dir=str(repo),
    )
    return agent, repo


def _tool_call(name, args):
    return Message(role="assistant", tool_calls=[FunctionCall(id="1", name=name, arguments=json.dumps(args))])


def _final(body):
    return Message(role="assistant", content=f"<final_answer>\n{body}\n</final_answer>")


def test_single_file_grep_citation_survives(tmp_path):
    repo = tmp_path / "repo"
    target = Path(repo) / "alpha.py"
    grep = _tool_call("Grep", {"pattern": "needle", "path": str(target), "output_mode": "content", "-C": 1})
    agent, repo = _agent(tmp_path, [grep, _final(f"{target}:3-5 (the target)")])
    result = asyncio.run(agent.run("q", citation=True))
    assert f"{target}:3-5 (the target)" in result
    assert agent.llm.calls == 2  # no correction round was needed


def test_hallucinated_citation_is_dropped_after_corrections(tmp_path):
    repo = tmp_path / "repo"
    target = Path(repo) / "alpha.py"
    read = _tool_call("Read", {"path": str(target), "limit": 2})
    scripted = [read] + [_final(f"{target}:1-2 (real)\n{target}:100-105 (invented)")] * (MAX_CITATION_CORRECTIONS + 1)
    agent, repo = _agent(tmp_path, scripted)
    result = asyncio.run(agent.run("q", max_turns=10, citation=True))
    assert f"{target}:1-2 (real)" in result
    assert "100-105" not in result
    assert "invented" not in result
    assert "unverified" not in result
    assert agent.llm.calls == MAX_CITATION_CORRECTIONS + 2


def test_correction_loop_lets_the_model_fix_itself(tmp_path):
    repo = tmp_path / "repo"
    target = Path(repo) / "alpha.py"
    scripted = [
        _final(f"{target}:3-4 (guessed without reading)"),
        _tool_call("Read", {"path": str(target)}),
        _final(f"{target}:3-4 (now actually read)"),
    ]
    agent, repo = _agent(tmp_path, scripted)
    result = asyncio.run(agent.run("q", max_turns=10, citation=True))
    assert f"{target}:3-4 (now actually read)" in result
    assert agent.llm.calls == 3


def test_citation_pointing_at_missing_file_is_still_dropped(tmp_path):
    repo = tmp_path / "repo"
    target = Path(repo) / "alpha.py"
    read = _tool_call("Read", {"path": str(target), "limit": 2})
    scripted = [read] + [_final(f"{target}:1-2 (real)\n/nonexistent/ghost.py:1-2 (gone)")] * (
        MAX_CITATION_CORRECTIONS + 1
    )
    agent, repo = _agent(tmp_path, scripted)
    result = asyncio.run(agent.run("q", max_turns=10, citation=True))
    assert f"{target}:1-2 (real)" in result
    assert "ghost.py" not in result


def test_citation_disabled_returns_raw_content(tmp_path):
    agent, repo = _agent(tmp_path, [Message(role="assistant", content="plain answer")])
    assert asyncio.run(agent.run("q")) == "plain answer"
