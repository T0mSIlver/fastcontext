import os

from swefc.agent.agent import Agent
from swefc.agent.llm import LLM
from swefc.agent.tool.tool import ToolSet

from swefc.agent.utils import load_system_prompt


def make_swefc_agent(
    trajectory_file: str,
    work_dir: str,
    **kwargs,
) -> Agent:
    name = "SWE-FastContext"
    system_prompt = kwargs.get("system_prompt", None)
    if system_prompt is None:
        system_prompt = load_system_prompt(work_dir)

    llm = LLM(
        model=os.getenv("MODEL"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    from swefc.agent.tool.glob import GlobTool
    from swefc.agent.tool.grep import GrepTool
    from swefc.agent.tool.read import ReadTool

    toolset = ToolSet([ReadTool(), GlobTool(), GrepTool()], work_dir=work_dir)
    return Agent(
        name=name,
        system_prompt=system_prompt,
        llm=llm,
        toolset=toolset,
        trajectory_file=trajectory_file,
        work_dir=work_dir,
    )
