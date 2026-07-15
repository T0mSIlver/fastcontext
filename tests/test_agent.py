"""Live agent test. Skipped unless FC_MODEL and FC_BASE_URL are set (see conftest)."""

from fastcontext.agent.agent import Agent
from fastcontext.agent.llm import LLM
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool


async def test_agent(llm_endpoint, tmp_path):
    (tmp_path / "README.md").write_text("FastContext explores repositories with read-only tools.\n", encoding="utf-8")

    llm = LLM(**llm_endpoint, debug=True)
    toolset = ToolSet(tools=[ReadTool()], work_dir=str(tmp_path))
    agent = Agent(
        name="TestAgent",
        system_prompt="You are a helpful coding assistant.",
        llm=llm,
        toolset=toolset,
        trajectory_file=str(tmp_path / "trajectory.jsonl"),
        work_dir=str(tmp_path),
    )

    result = await agent.run(
        f"Summarize the content of '{tmp_path / 'README.md'}' in one sentence.",
        max_turns=5,
    )
    assert isinstance(result, str)
    assert result
