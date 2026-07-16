"""The per-result cap (--max-tool-result-chars) and how it composes with the per-turn budget.

The turn budget is spent greedily in call order, so one oversized result can consume all of it and
leave the later calls of the same turn with nothing. The per-result cap bounds each result first so
every call in a turn survives.
"""

import asyncio

from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.tool import Tool


class _BigTool(Tool):
    """Returns a fixed-size blob so the caps are the only thing shaping the output."""

    name = "Big"
    description = "test"
    parameters = {"type": "object", "properties": {}}

    async def call(self, parameters: str, **kwargs) -> str:
        import json

        return "y" * json.loads(parameters)["n"]


def _msg(*sizes: int) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[FunctionCall(id=f"c{i}", name="Big", arguments=f'{{"n": {n}}}') for i, n in enumerate(sizes)],
    )


def _run(toolset, msg):
    return [m.content for m in asyncio.run(toolset.call(msg))]


def test_per_result_cap_truncates_each_result(tmp_path):
    toolset = ToolSet([_BigTool()], work_dir=str(tmp_path), max_tool_result_chars=100)
    (a, b) = _run(toolset, _msg(5000, 5000))
    assert a.startswith("y" * 100) and "Output truncated" in a
    assert b.startswith("y" * 100) and "Output truncated" in b


def test_per_result_cap_disabled_by_zero(tmp_path):
    toolset = ToolSet([_BigTool()], work_dir=str(tmp_path), max_tool_result_chars=0)
    (only,) = _run(toolset, _msg(5000))
    assert only == "y" * 5000


def test_per_result_cap_leaves_small_results_alone(tmp_path):
    toolset = ToolSet([_BigTool()], work_dir=str(tmp_path), max_tool_result_chars=1000)
    (only,) = _run(toolset, _msg(10))
    assert only == "y" * 10


def test_per_result_cap_stops_one_call_starving_the_rest(tmp_path):
    """The regression the flag exists for.

    Turn budget 3000 spent greedily: without a per-result cap the first 5000-char result eats all of
    it and the second call comes back empty. With a 1000-char per-result cap both survive.
    """
    starved = ToolSet([_BigTool()], work_dir=str(tmp_path), max_tool_output_chars=3000)
    first, second = _run(starved, _msg(5000, 5000))
    assert second.strip().startswith("<system-reminder>")  # nothing but the truncation notice

    fair = ToolSet(
        [_BigTool()], work_dir=str(tmp_path), max_tool_output_chars=3000, max_tool_result_chars=1000
    )
    first, second = _run(fair, _msg(5000, 5000))
    assert first.startswith("y" * 1000)
    assert second.startswith("y" * 1000)


def test_the_two_caps_say_which_one_truncated(tmp_path):
    """The notices must not both claim "per-result".

    The model can act on the difference -- one oversized result means narrow this call, a turn over
    budget means ask for less at once -- so a turn-capped result labelled "per-result" points it at
    the wrong fix.
    """
    per_result = ToolSet([_BigTool()], work_dir=str(tmp_path), max_tool_result_chars=50)
    (out,) = _run(per_result, _msg(5000))
    assert "per-result limit" in out

    per_turn = ToolSet([_BigTool()], work_dir=str(tmp_path), max_tool_output_chars=50)
    (out,) = _run(per_turn, _msg(5000))
    assert "per-turn total output limit" in out


def test_turn_budget_still_applies_under_a_per_result_cap(tmp_path):
    """The per-result cap does not let a turn exceed the turn budget: N results each under the
    per-result cap still add N x cap, which is what the turn budget exists to stop."""
    toolset = ToolSet(
        [_BigTool()], work_dir=str(tmp_path), max_tool_output_chars=1000, max_tool_result_chars=900
    )
    outputs = _run(toolset, _msg(900, 900, 900))
    payload = sum(len(o.split("<system-reminder>")[0]) for o in outputs)
    assert payload <= 1000
