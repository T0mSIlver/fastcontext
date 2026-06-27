"""Tests for citation hallucination detection and the correction loop.

A citation is "verified" if the model actually opened at least one line of the
cited range, either through the Read tool or through Grep output (including
context lines). Unverified citations trigger up to two correction requests;
survivors are kept but marked ``(unverified)``.
"""

import asyncio
import json
import os

from fastcontext.agent.agent import MAX_CITATION_CORRECTIONS, Agent
from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.observed import (
    citation_observed,
    correction_message,
    record_grep,
    record_read,
    unverified_citations,
)
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.glob import GlobTool
from fastcontext.agent.tool.grep import GrepTool
from fastcontext.agent.tool.read import ReadTool
from fastcontext.agent.utils import format_citations, parse_citations


# --- observed-lines tracking -------------------------------------------------


def test_record_read_captures_returned_range():
    observed: dict = {}
    record_read(observed, "```/repo/a.py:10-13\n10|x\n11|y\n12|z\n13|w\n```")
    key = os.path.realpath("/repo/a.py")
    assert observed[key] == {10, 11, 12, 13}


def test_record_grep_captures_match_and_context_lines():
    observed: dict = {}
    output = (
        "/repo/a.py\n"
        "2-line2\n"
        "3:line3 foo\n"
        "4-line4\n"
        "--\n"
        "7:line7 foo\n"
        "\n"
        "/repo/b.py\n"
        "1:foo here\n"
    )
    record_grep(observed, output)
    assert observed[os.path.realpath("/repo/a.py")] == {2, 3, 4, 7}
    assert observed[os.path.realpath("/repo/b.py")] == {1}


def test_citation_observed_zero_overlap_semantics():
    observed = {os.path.realpath("/repo/a.py"): {10, 11, 12}}

    def cite(path, s, e):
        return {"path": path, "start_line": s, "end_line": e, "line_range": f"{s}-{e}"}

    assert citation_observed(observed, cite("/repo/a.py", 12, 20)) is True  # overlap on 12
    assert citation_observed(observed, cite("/repo/a.py", 40, 60)) is False  # no overlap
    assert citation_observed(observed, cite("/repo/never.py", 1, 5)) is False  # unseen file


def test_unverified_citations_filters():
    observed = {os.path.realpath("/repo/a.py"): {1, 2, 3}}
    cites = [
        {"path": "/repo/a.py", "start_line": 1, "end_line": 2, "line_range": "1-2"},
        {"path": "/repo/a.py", "start_line": 90, "end_line": 95, "line_range": "90-95"},
    ]
    bad = unverified_citations(observed, cites)
    assert len(bad) == 1 and bad[0]["line_range"] == "90-95"


def test_correction_message_lists_offenders():
    msg = correction_message([{"path": "/repo/a.py", "line_range": "90-95"}])
    assert "/repo/a.py:90-95" in msg


# --- final-answer formatting -------------------------------------------------


def test_parse_citations_no_tag_returns_empty_list():
    # Regression: previously returned a dict, breaking format_citations.
    assert parse_citations("just prose, no tag") == []


def test_format_citations_marks_unverified(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("\n".join(f"l{i}" for i in range(1, 30)), encoding="utf-8")
    text = f"<final_answer>\n{f}:1-3 (seen)\n{f}:90-95 (guessed)\n</final_answer>"
    observed = {os.path.realpath(str(f)): {1, 2, 3}}
    out = format_citations(parse_citations(text), observed=observed)
    # The seen range is clean; the guessed range is flagged.
    assert f"{f}:1-3 (seen)" in out
    assert "(unverified" in out
    assert out.count("unverified") == 1


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
    repo = tmp_path / "repo"
    (repo).mkdir()
    f = repo / "foo.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 6)), encoding="utf-8")  # 5 lines
    toolset = ToolSet([ReadTool(), GrepTool(), GlobTool()], work_dir=str(repo))
    agent = Agent(
        name="t",
        system_prompt="sys",
        llm=_FakeLLM(scripted),
        toolset=toolset,
        trajectory_file=str(tmp_path / "traj" / "t.jsonl"),
        work_dir=str(repo),
    )
    return agent, f


def _read_call(f):
    return Message(
        role="assistant",
        tool_calls=[FunctionCall(id="1", name="Read", arguments=json.dumps({"path": str(f)}))],
    )


def _final(f, rng):
    return Message(role="assistant", content=f"<final_answer>\n{f}:{rng}\n</final_answer>")


def test_loop_corrects_then_accepts_verified_citation(tmp_path):
    # Read the file, then cite an unread range (triggers correction), then cite
    # a range that was read (accepted).
    agent, f = _agent(tmp_path, [_read_call(f := tmp_path / "repo" / "foo.py"), _final(f, "100-105"), _final(f, "1-3")])
    result = asyncio.run(agent.run("q", citation=True))
    assert f"{f}:1-3" in result
    assert "unverified" not in result
    assert agent.llm.calls == 3  # one correction round happened


def test_loop_marks_unverified_after_exhausting_corrections(tmp_path):
    f = tmp_path / "repo" / "foo.py"
    scripted = [_read_call(f)] + [_final(f, "100-105")] * (MAX_CITATION_CORRECTIONS + 1)
    agent, f = _agent(tmp_path, scripted)
    result = asyncio.run(agent.run("q", citation=True))
    assert f"{f}:100-105" in result  # kept, not dropped
    assert "unverified" in result  # but flagged
    assert agent.llm.calls == MAX_CITATION_CORRECTIONS + 2
