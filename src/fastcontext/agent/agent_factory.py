import os

from fastcontext.agent.agent import Agent
from fastcontext.agent.llm import LLM
from fastcontext.agent.tool.tool import ToolSet
from fastcontext.agent.tool.utils import RG_PATH
from fastcontext.agent.utils import load_system_prompt


def _get_env(name: str, legacy_name: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    if legacy_name:
        return os.getenv(legacy_name)
    return None


def _require_env(name: str, legacy_name: str | None = None) -> str:
    value = _get_env(name, legacy_name)
    if value:
        return value
    legacy_hint = f" or {legacy_name}" if legacy_name else ""
    raise RuntimeError(f"Missing required environment variable {name}{legacy_hint}.")


def make_fastcontext_agent(
    trajectory_file: str,
    work_dir: str,
    **kwargs,
) -> Agent:
    name = "FastContext"
    system_prompt = kwargs.get("system_prompt", None)
    if system_prompt is None:
        system_prompt = load_system_prompt(work_dir)

    max_tokens = os.getenv("FC_MAX_TOKENS", "4096").strip()
    temperature = os.getenv("FC_TEMPERATURE", "0.7").strip()
    try:
        max_tokens = int(max_tokens)
    except ValueError:
        max_tokens = 4096
    try:
        temperature = float(temperature)
    except ValueError:
        temperature = 0.7

    llm = LLM(
        model=_require_env("FC_MODEL", "MODEL"),
        api_key=_get_env("FC_API_KEY", "API_KEY"),
        base_url=_require_env("FC_BASE_URL", "BASE_URL"),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )

    from fastcontext.agent.tool.glob import GlobTool
    from fastcontext.agent.tool.grep import GrepTool
    from fastcontext.agent.tool.read import ReadTool

    if not RG_PATH:
        raise RuntimeError(
            "Grep tool requires ripgrep (rg) to be installed, but it was not found in current environment.\n"
            "Install it from: https://github.com/BurntSushi/ripgrep"
        )

    toolset = ToolSet([ReadTool(), GlobTool(), GrepTool()], work_dir=work_dir)
    return Agent(
        name=name,
        system_prompt=system_prompt,
        llm=llm,
        toolset=toolset,
        trajectory_file=trajectory_file,
        work_dir=work_dir,
    )
