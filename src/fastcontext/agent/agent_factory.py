import sys

from fastcontext.agent.agent import Agent
from fastcontext.agent.budget import (
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    ContextBudget,
    required_reserve,
)
from fastcontext.agent.config import load_settings
from fastcontext.agent.llm import LLM, resolve_max_tokens
from fastcontext.agent.tool.tool import ToolSet
from fastcontext.agent.tool.utils import RG_PATH
from fastcontext.agent.utils import load_system_prompt

# Config keys that may be passed as explicit overrides via kwargs (highest precedence).
_OVERRIDE_KEYS = (
    "model",
    "base_url",
    "api_key",
    "temperature",
    "max_tokens",
    "max_context",
    "max_tool_output_chars",
    "context_reserve",
    "reasoning_effort",
    "max_citations",
)

# A generous default cap: it bounds a runaway/hallucinated citation list without touching the
# handful a normal answer returns.
DEFAULT_MAX_CITATIONS = 25


def make_fastcontext_agent(
    trajectory_file: str,
    work_dir: str,
    **kwargs,
) -> Agent:
    name = "FastContext"
    system_prompt = kwargs.get("system_prompt") or load_system_prompt(work_dir)

    # Resolve every knob through the layered config: override > FC_* env > project/user config file
    # > default. This lets a config file supply the endpoint and tuning once, so callers (and the
    # coding agent driving this over bash) need not re-declare them on each invocation.
    settings = load_settings(
        work_dir,
        overrides={key: kwargs.get(key) for key in _OVERRIDE_KEYS},
        config_path=kwargs.get("config_path"),
    )

    model = settings.require("model", "FC_MODEL", "MODEL")
    api_key = settings.str_("api_key", "FC_API_KEY", "API_KEY")
    base_url = settings.require("base_url", "FC_BASE_URL", "BASE_URL")

    # Context window in tokens. 0 disables the budget (unbounded, the old behavior): we cannot
    # guess it safely, because a server's usable window is often far below its configured one --
    # llama.cpp with --parallel 2 halves it per slot.
    max_context = settings.int_("max_context", "FC_MAX_CONTEXT", 0)
    max_tool_output_chars = settings.int_(
        "max_tool_output_chars", "FC_MAX_TOOL_OUTPUT_CHARS", DEFAULT_MAX_TOOL_OUTPUT_CHARS
    )
    max_citations = settings.int_("max_citations", "FC_MAX_CITATIONS", DEFAULT_MAX_CITATIONS)

    # max_tokens (the per-response completion cap): the resolved source (override > FC_MAX_TOKENS env
    # > config file) feeds provider auto-detection. "auto" (or unset) triggers a lookup of the
    # model's context length from the provider's /models endpoint; otherwise an integer is used.
    max_tokens = resolve_max_tokens(
        settings.raw("max_tokens", "FC_MAX_TOKENS"),
        base_url=base_url,
        api_key=api_key,
        model=model,
        verbose=kwargs.get("verbose", False),
    )
    temperature = settings.float_("temperature", "FC_TEMPERATURE", 0.7)

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=settings.str_("reasoning_effort", "FC_REASONING_EFFORT"),
    )

    # The budget trips only after a turn's tool results have landed, so the reserve must absorb a
    # full turn of tool output plus the completion -- otherwise the agent can cross the limit and
    # find that even the final-answer request no longer fits.
    reserve = settings.int_("context_reserve", "FC_CONTEXT_RESERVE", 0) or required_reserve(
        max_tool_output_chars, max_tokens
    )

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
        max_citations=max_citations,
    )
