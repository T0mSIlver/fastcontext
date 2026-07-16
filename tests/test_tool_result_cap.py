"""The per-result cap (--max-result-output-tokens) and how it composes with the per-turn budget.

The turn budget is spent greedily in call order, so one oversized result can consume all of it and
leave the later calls of the same turn with nothing. The per-result cap bounds each result first so
every call in a turn survives.
"""

import asyncio

from fastcontext.agent.budget import estimate_tokens
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
    toolset = ToolSet([_BigTool()], work_dir=str(tmp_path), max_result_output_tokens=100)
    (a, b) = _run(toolset, _msg(5000, 5000))
    for out in (a, b):
        assert "Output truncated" in out
        assert estimate_tokens(out.split("\n<system-reminder>")[0]) <= 100


def test_per_result_cap_disabled_by_zero(tmp_path):
    toolset = ToolSet([_BigTool()], work_dir=str(tmp_path), max_result_output_tokens=0)
    (only,) = _run(toolset, _msg(5000))
    assert only == "y" * 5000


def test_per_result_cap_leaves_small_results_alone(tmp_path):
    toolset = ToolSet([_BigTool()], work_dir=str(tmp_path), max_result_output_tokens=1000)
    (only,) = _run(toolset, _msg(10))
    assert only == "y" * 10


def test_per_result_cap_stops_one_call_starving_the_rest(tmp_path):
    """The regression the flag exists for.

    Turn budget 1000 tokens spent greedily: without a per-result cap the first result (~1667 tokens)
    eats all of it and the second call comes back empty. With a 400-token per-result cap both survive.
    """
    starved = ToolSet([_BigTool()], work_dir=str(tmp_path), max_turn_output_tokens=1000)
    first, second = _run(starved, _msg(5000, 5000))
    assert second.strip().startswith("<system-reminder>")  # nothing but the truncation notice

    fair = ToolSet(
        [_BigTool()], work_dir=str(tmp_path), max_turn_output_tokens=1000, max_result_output_tokens=400
    )
    first, second = _run(fair, _msg(5000, 5000))
    assert first.count("y") > 0 and second.count("y") > 0, "both calls must survive the turn"


def test_the_two_caps_say_which_one_truncated(tmp_path):
    """The notices must not both claim "per-result".

    The model can act on the difference -- one oversized result means narrow this call, a turn over
    budget means ask for less at once -- so a turn-capped result labelled "per-result" points it at
    the wrong fix.
    """
    per_result = ToolSet([_BigTool()], work_dir=str(tmp_path), max_result_output_tokens=50)
    (out,) = _run(per_result, _msg(5000))
    assert "per-result limit" in out

    per_turn = ToolSet([_BigTool()], work_dir=str(tmp_path), max_turn_output_tokens=50)
    (out,) = _run(per_turn, _msg(5000))
    assert "per-turn total output limit" in out


def test_turn_budget_still_applies_under_a_per_result_cap(tmp_path):
    """The per-result cap does not let a turn exceed the turn budget: N results each under the
    per-result cap still add N x cap, which is what the turn budget exists to stop."""
    toolset = ToolSet(
        [_BigTool()], work_dir=str(tmp_path), max_turn_output_tokens=1000, max_result_output_tokens=900
    )
    outputs = _run(toolset, _msg(2700, 2700, 2700))
    payload = sum(estimate_tokens(o.split("\n<system-reminder>")[0]) for o in outputs)
    assert payload <= 1000
