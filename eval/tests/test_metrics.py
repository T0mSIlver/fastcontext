"""Tests for trajectory scoring, using synthetic and real example trajectories."""

import json
import os

import pytest

from fc_eval.metrics import compute_metrics, parse_citations, score_run

EXAMPLES = {
    "/home/dev/work/llama.cpp/.fastcontext/trajectory-tui2.jsonl": "/home/dev/work/llama.cpp",
    "/home/dev/work/codspeed/.fastcontext/trajectory_verbose.jsonl": "/home/dev/work/codspeed",
}


def _msg(**kw):
    return kw


def test_counts_turns_tools_and_duplicates():
    messages = [
        _msg(role="system", content="sys"),
        _msg(role="user", content="<query>find x</query>"),
        _msg(
            role="assistant",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            tool_calls=[
                {"id": "a", "function": {"name": "Glob", "arguments": '{"pattern": "**/*.py"}'}},
                {"id": "b", "function": {"name": "Glob", "arguments": '{"pattern": "**/*.py"}'}},  # duplicate
            ],
        ),
        _msg(role="tool", tool_call_id="a", content="No files found"),
        _msg(role="tool", tool_call_id="b", content="<system-reminder>Error: directory `/x` does not exist.</system-reminder>"),
        _msg(role="assistant", usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
             content="done\n<final_answer>\n</final_answer>"),
    ]
    m = compute_metrics(messages)
    assert m.turns == 2
    assert m.tool_calls_total == 2
    assert m.tool_usage == {"Glob": 2}
    assert m.duplicate_tool_calls == 1
    assert m.failed_tool_calls == 1   # only the system-reminder, not "No files found"
    assert m.total_tokens == 43


def test_correction_loop_counted():
    messages = [
        _msg(role="user", content="<query>q</query>"),
        _msg(role="assistant", content="<final_answer>\n/a/b.py:1-2\n</final_answer>"),
        _msg(role="user", content="Your final answer cites line ranges you never opened during exploration:\n- /a/b.py:1-2"),
        _msg(role="assistant", content="<final_answer>\n</final_answer>"),
    ]
    m = compute_metrics(messages)
    assert m.corrections == 1


def test_unverified_citation_detection(tmp_path):
    f = tmp_path / "real.py"
    f.write_text("a\nb\nc\nd\ne\n")
    read_output = f"```{f}:1-3\n1|a\n2|b\n3|c\n"
    messages = [
        _msg(role="assistant", tool_calls=[{"id": "r", "function": {"name": "Read", "arguments": f'{{"path": "{f}"}}'}}]),
        _msg(role="tool", tool_call_id="r", content=read_output),
        _msg(role="assistant", content=f"<final_answer>\n{f}:1-2 (seen)\n{f}:80-90 (never opened)\n</final_answer>"),
    ]
    m = compute_metrics(messages)
    assert m.citations_total == 2          # both point at an existing file
    assert m.unverified_citations == 1     # only the 80-90 range was never read


def test_missing_file_citation():
    messages = [
        _msg(role="assistant", content="<final_answer>\n/does/not/exist.py:1-2\n</final_answer>"),
    ]
    m = compute_metrics(messages)
    assert m.citations_missing_file == 1
    assert m.citations_total == 0


def test_parse_citations_handles_explanations():
    text = "<final_answer>\n/a/b.py:10-15 (core logic)\n/c/d.js:5\n</final_answer>"
    cites = parse_citations(text)
    assert cites[0]["start_line"] == 10 and cites[0]["end_line"] == 15
    assert cites[1]["start_line"] == cites[1]["end_line"] == 5


@pytest.mark.parametrize("path,cwd", list(EXAMPLES.items()))
def test_real_example_trajectories_score(path, cwd):
    if not os.path.isfile(path):
        pytest.skip(f"example trajectory not present: {path}")
    m = score_run(path, cwd=cwd)
    assert m.turns > 0
    assert m.tool_calls_total > 0
    # Trajectory is valid JSONL the whole way through.
    with open(path) as fh:
        for line in fh:
            if line.strip():
                json.loads(line)


def test_verbose_example_has_unverified_citations():
    path = "/home/dev/work/codspeed/.fastcontext/trajectory_verbose.jsonl"
    if not os.path.isfile(path):
        pytest.skip("example not present")
    m = score_run(path, cwd="/home/dev/work/codspeed")
    # This run cited five ranges but only opened one file -> four unverified.
    assert m.unverified_citations == 4
