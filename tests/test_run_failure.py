"""A run that cannot produce an answer must raise AgentRunError, not return the error as an answer.

This is what lets the CLI exit nonzero on failure: previously the agent returned the failure text
and exited 0, so a driving agent had to scan stdout for "LLM API call failed" to notice.

Two ways a run ends with no answer: the endpoint call fails, or the model never stops calling tools
-- including on the turn that explicitly asked it for the final answer.
"""

import asyncio

import pytest

from fastcontext.agent.agent import Agent, AgentRunError
from fastcontext.agent.llm import FunctionCall, Message, LLMAPIError
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool


class _FailingLLM:
    """Raises the same error the real LLM raises when the endpoint call fails."""

    model = "fake"

    async def acall(self, messages, tools=None, **kwargs):
        raise LLMAPIError("connection refused")


class _NeverAnswersLLM:
    """Always calls a tool and never emits a final answer, even when asked for one."""

    model = "fake"

    async def acall(self, messages, tools=None, **kwargs):
        return Message(
            role="assistant",
            content="",
            tool_calls=[FunctionCall(id="c1", name="Read", arguments='{"path": "a.py"}')],
        )


def _agent(tmp_path, llm=None):
    return Agent(
        name="t",
        system_prompt="sys",
        llm=llm or _FailingLLM(),
        toolset=ToolSet([ReadTool()], work_dir=str(tmp_path)),
        trajectory_file=str(tmp_path / "traj" / "t.jsonl"),
        work_dir=str(tmp_path),
    )


def test_llm_failure_raises_agent_run_error(tmp_path):
    agent = _agent(tmp_path)
    with pytest.raises(AgentRunError) as exc:
        asyncio.run(agent.run("q", max_turns=3))
    assert "LLM API call failed" in str(exc.value)


def test_failure_is_recorded_to_the_trajectory(tmp_path):
    """The failure is still written to context before raising, so the trajectory is not lost."""
    agent = _agent(tmp_path)
    with pytest.raises(AgentRunError):
        asyncio.run(agent.run("q", max_turns=3))
    contents = [m.get("content") or "" for m in agent.context.get_messages()]
    assert any("LLM API call failed" in c for c in contents)


def test_exhausting_turns_without_an_answer_raises(tmp_path):
    """A model that keeps calling tools past the final-answer request produced no answer.

    Returning "No final answer after N turns." would put that sentence on stdout with exit 0, where
    a caller parsing the answer reads the apology as the finding.
    """
    agent = _agent(tmp_path, llm=_NeverAnswersLLM())
    with pytest.raises(AgentRunError) as exc:
        asyncio.run(agent.run("q", max_turns=2))
    assert "No final answer after 2 turns" in str(exc.value)
