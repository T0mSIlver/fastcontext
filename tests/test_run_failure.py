"""A run that hits an LLM API failure must raise AgentRunError, not return the error as an answer.

This is what lets the CLI exit nonzero on failure: previously the agent returned the failure text
and exited 0, so a driving agent had to scan stdout for "LLM API call failed" to notice.
"""

import asyncio

import pytest

from fastcontext.agent.agent import Agent, AgentRunError
from fastcontext.agent.llm import RequestyAPIError
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool


class _FailingLLM:
    """Raises the same error the real LLM raises when the endpoint call fails."""

    model = "fake"

    async def acall(self, messages, tools=None, **kwargs):
        raise RequestyAPIError("connection refused")


def _agent(tmp_path):
    return Agent(
        name="t",
        system_prompt="sys",
        llm=_FailingLLM(),
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
