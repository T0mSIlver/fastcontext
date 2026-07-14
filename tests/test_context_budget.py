"""Tests for context-budget management.

Without a budget the agent grows its prompt until the provider rejects the request and the
run dies with no answer. The budget makes it stop exploring while a request still fits, and
the per-result cap stops one oversized Read from exhausting the window in a single call.
"""

import json
import tempfile
from pathlib import Path

from fastcontext.agent.agent import Agent
from fastcontext.agent.budget import (
    ContextBudget,
    cap_tool_output,
    estimate_messages_tokens,
    estimate_tokens,
)
from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.tool.read import ReadTool
from fastcontext.agent.tool.tool import ToolSet


# --------------------------------------------------------------------------- estimation


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("x" * 300) > estimate_tokens("x" * 30)


def test_estimate_counts_tool_call_arguments():
    # Tool-call arguments are part of the prompt on the next turn and must be counted.
    bare = [{"role": "assistant", "content": ""}]
    with_call = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "Read", "arguments": json.dumps({"path": "/x" * 200})}}],
        }
    ]
    assert estimate_messages_tokens(with_call) > estimate_messages_tokens(bare) + 50


# --------------------------------------------------------------------------- tool cap


def test_cap_tool_output_truncates_and_explains():
    out = cap_tool_output("y" * 5000, limit=1000)
    assert len(out) < 5000
    assert "Output truncated" in out
    assert "offset/limit" in out  # tells the model how to get the rest


def test_cap_tool_output_leaves_small_output_alone():
    assert cap_tool_output("small", limit=1000) == "small"


def test_cap_tool_output_disabled_with_zero():
    assert cap_tool_output("y" * 5000, limit=0) == "y" * 5000


async def test_toolset_caps_an_oversized_read():
    # The real failure: Read's own ceiling is 2000 lines x 500 chars, so one call can return
    # ~250k tokens and make the conversation unsendable.
    with tempfile.TemporaryDirectory() as tmp:
        big = Path(tmp) / "big.py"
        big.write_text("\n".join("x" * 200 for _ in range(1000)), encoding="utf-8")

        toolset = ToolSet([ReadTool()], work_dir=tmp, max_tool_output_chars=5_000)
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[FunctionCall(id="1", name="Read", arguments=json.dumps({"path": str(big)}))],
        )
        results = await toolset.call(msg)
        assert len(results[0].content) < 6_000
        assert "Output truncated" in results[0].content


# --------------------------------------------------------------------------- budget


def test_budget_disabled_never_finalizes():
    budget = ContextBudget(max_context=0)
    assert not budget.enabled
    huge = [{"role": "user", "content": "x" * 10_000_000}]
    assert not budget.must_finalize(huge)


def test_budget_trips_when_projection_crosses_limit():
    budget = ContextBudget(max_context=10_000, reserve=2_000)
    assert budget.limit == 8_000
    assert not budget.must_finalize([{"role": "user", "content": "x" * 300}])
    # ~10k tokens of content at 3 chars/token, well past the 8k limit
    assert budget.must_finalize([{"role": "user", "content": "x" * 30_000}])


def test_measured_usage_replaces_the_estimate():
    # The provider's exact prompt_tokens must win, so estimation error cannot accumulate.
    budget = ContextBudget(max_context=10_000, reserve=2_000)
    budget.record_usage({"prompt_tokens": 7_900})
    assert not budget.must_finalize([])
    budget.add_pending([{"role": "tool", "content": "x" * 900}])  # ~300 more tokens
    assert budget.must_finalize([])


def test_record_usage_clears_pending():
    budget = ContextBudget(max_context=100_000)
    budget.record_usage({"prompt_tokens": 500})
    budget.add_pending([{"role": "tool", "content": "x" * 3_000}])
    # exact prefix (500) + estimate for what was appended since (~1000)
    assert budget.projected_tokens([]) > 1_000
    # A fresh measurement supersedes the estimate entirely rather than stacking on it.
    budget.record_usage({"prompt_tokens": 1_600})
    assert budget.projected_tokens([]) == 1_600


def test_estimate_used_before_the_first_response():
    # No measurement yet, so the whole conversation is estimated from the messages themselves.
    budget = ContextBudget(max_context=10_000, reserve=1_000)
    assert budget.projected_tokens([{"role": "user", "content": "x" * 3_000}]) > 900


def test_record_usage_ignores_missing_usage():
    budget = ContextBudget(max_context=100_000)
    budget.record_usage(None)
    budget.record_usage({})
    assert budget.projected_tokens([{"role": "user", "content": "hi"}]) > 0


# --------------------------------------------------------------------------- end to end


class _ScriptedLLM:
    """Replays canned assistant messages and reports a prompt size that grows each turn."""

    model = "scripted"

    def __init__(self, replies: list[Message], prompt_tokens: list[int]):
        self._replies = replies
        self._prompt_tokens = prompt_tokens
        self.calls: list[dict] = []

    async def acall(self, messages, tools, event_sink=None, turn=0):
        self.calls.append({"tools": tools, "messages": list(messages)})
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        reply = reply.model_copy()
        reply.usage = {"prompt_tokens": self._prompt_tokens[min(len(self.calls) - 1, len(self._prompt_tokens) - 1)]}
        return reply


def _read_call(path: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[FunctionCall(id="c1", name="Read", arguments=json.dumps({"path": path}))],
    )


async def test_agent_finalizes_instead_of_overflowing():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "a.py"
        src.write_text("\n".join(f"line{i}" for i in range(1, 40)), encoding="utf-8")
        traj = str(Path(tmp) / "t" / "traj.jsonl")

        # Turn 1: the model reads a file. The provider then reports a prompt already near the
        # window. Turn 2 must therefore be the final one -- with no tools offered.
        answer = Message(role="assistant", content=f"<final_answer>\n{src}:1-5 (why)\n</final_answer>")
        llm = _ScriptedLLM(replies=[_read_call(str(src)), answer], prompt_tokens=[9_500, 9_800])

        toolset = ToolSet([ReadTool()], work_dir=tmp, max_tool_output_chars=30_000)
        agent = Agent(
            name="t",
            system_prompt="sys",
            llm=llm,
            toolset=toolset,
            trajectory_file=traj,
            work_dir=tmp,
            budget=ContextBudget(max_context=10_000, reserve=1_000),
        )

        result = await agent.run(prompt="q", max_turns=10, citation=True)

        # It answered rather than exploring until the provider rejected the prompt.
        assert "a.py:1-5" in result
        # Two turns: the tools were withheld on the second so it could not keep exploring.
        assert len(llm.calls) == 2
        assert llm.calls[0]["tools"], "tools must be offered while there is room"
        assert llm.calls[1]["tools"] is None, "tools must be withheld once the budget trips"
        # The model was told why.
        assert any(
            "approaching the context limit" in (m.get("content") or "") for m in llm.calls[1]["messages"]
        )


async def test_agent_explores_freely_when_budget_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "a.py"
        src.write_text("line1\nline2\n", encoding="utf-8")
        traj = str(Path(tmp) / "t" / "traj.jsonl")

        answer = Message(role="assistant", content=f"<final_answer>\n{src}:1-2 (why)\n</final_answer>")
        # A huge reported prompt must NOT trip anything when the budget is off.
        llm = _ScriptedLLM(replies=[_read_call(str(src)), answer], prompt_tokens=[999_999, 999_999])

        agent = Agent(
            name="t",
            system_prompt="sys",
            llm=llm,
            toolset=ToolSet([ReadTool()], work_dir=tmp),
            trajectory_file=traj,
            work_dir=tmp,
            budget=ContextBudget(max_context=0),
        )
        await agent.run(prompt="q", max_turns=10, citation=True)
        assert all(call["tools"] for call in llm.calls), "budget disabled must never withhold tools"
