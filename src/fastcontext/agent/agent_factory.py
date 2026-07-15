import os
import sys

from fastcontext.agent.agent import Agent
from fastcontext.agent.budget import (
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    ContextBudget,
    required_reserve,
)
from fastcontext.agent.llm import LLM, resolve_max_tokens
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


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def make_fastcontext_agent(
    trajectory_file: str,
    work_dir: str,
    **kwargs,
) -> Agent:
    name = "FastContext"
    system_prompt = kwargs.get("system_prompt", None)
    if system_prompt is None:
        system_prompt = load_system_prompt(work_dir)

    model = _require_env("FC_MODEL", "MODEL")
    api_key = _get_env("FC_API_KEY", "API_KEY")
    base_url = _require_env("FC_BASE_URL", "BASE_URL")

    # Context window in tokens. 0 disables the budget (unbounded, the old behavior): we cannot
    # guess it safely, because a server's usable window is often far below its configured one --
    # llama.cpp with --parallel 2 halves it per slot.
    max_context = kwargs.get("max_context")
    if max_context is None:
        max_context = _int_env("FC_MAX_CONTEXT", 0)
    max_tool_output_chars = kwargs.get("max_tool_output_chars")
    if max_tool_output_chars is None:
        max_tool_output_chars = _int_env("FC_MAX_TOOL_OUTPUT_CHARS", DEFAULT_MAX_TOOL_OUTPUT_CHARS)

    # max_tokens (the per-response completion cap) precedence: explicit CLI arg > FC_MAX_TOKENS env
    # > provider auto-detection > built-in default. "auto" (or unset) triggers a lookup of the
    # model's context length from the provider's /models endpoint.
    max_tokens_source = kwargs.get("max_tokens")
    if max_tokens_source is None:
        max_tokens_source = os.getenv("FC_MAX_TOKENS")
    max_tokens = resolve_max_tokens(
        max_tokens_source,
        base_url=base_url,
        api_key=api_key,
        model=model,
        verbose=kwargs.get("verbose", False),
    )

    temperature = os.getenv("FC_TEMPERATURE", "0.7").strip()
    try:
        temperature = float(temperature)
    except ValueError:
        temperature = 0.7

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # The budget trips only after a turn's tool results have landed, so the reserve must absorb a
    # full turn of tool output plus the completion -- otherwise the agent can cross the limit and
    # find that even the final-answer request no longer fits.
    reserve = _int_env("FC_CONTEXT_RESERVE", 0) or required_reserve(max_tool_output_chars, int(max_tokens))

    from fastcontext.agent.tool.glob import GlobTool
    from fastcontext.agent.tool.grep import GrepTool
    from fastcontext.agent.tool.read import ReadTool

    if not RG_PATH:
        raise RuntimeError(
            "Grep tool requires ripgrep (rg) to be installed, but it was not found in current environment.\n"
            "Install it from: https://github.com/BurntSushi/ripgrep"
        )

    # Say so out loud when the run is unprotected. Silence here reads as "I am safe", and the
    # failure it hides is the run dying mid-exploration with no answer at all.
    if max_context <= 0:
        print(
            "warning: context budget disabled (FC_MAX_CONTEXT/--max-context is 0). A long "
            "exploration can grow the prompt until the provider rejects it and the run ends with "
            "no answer. Set it to the model's usable window -- note a server's usable window is "
            "often below its configured one (llama.cpp --parallel 2 halves it per slot).",
            file=sys.stderr,
        )
    elif max_tool_output_chars <= 0:
        print(
            "warning: tool-output cap disabled (FC_MAX_TOOL_OUTPUT_CHARS is 0) while a context "
            "budget is set. A single Read can return ~250k tokens and overshoot the window in one "
            "turn, which the budget cannot undo.",
            file=sys.stderr,
        )

    toolset = ToolSet(
        [ReadTool(), GlobTool(), GrepTool()],
        work_dir=work_dir,
        max_tool_output_chars=max_tool_output_chars,
    )
    return Agent(
        name=name,
        system_prompt=system_prompt,
        llm=llm,
        toolset=toolset,
        trajectory_file=trajectory_file,
        work_dir=work_dir,
        budget=ContextBudget(max_context=max_context, reserve=reserve),
    )
