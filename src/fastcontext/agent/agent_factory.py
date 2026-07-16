import sys

from fastcontext.agent.agent import Agent
from fastcontext.agent.budget import (
    DEFAULT_MAX_TURN_OUTPUT_TOKENS,
    ContextBudget,
    required_reserve,
)
from fastcontext.agent.config import adopt_renamed_overrides, load_settings
from fastcontext.agent.llm import LLM, resolve_max_completion_tokens, resolve_max_context
from fastcontext.agent.tool.tool import ToolSet
from fastcontext.agent.tool.utils import RG_PATH
from fastcontext.agent.utils import load_system_prompt

# Config keys that may be passed as explicit overrides via kwargs (highest precedence).
_OVERRIDE_KEYS = (
    "model",
    "base_url",
    "api_key",
    "temperature",
    "max_completion_tokens",
    "max_context",
    "max_turn_output_tokens",
    "max_result_output_tokens",
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
        overrides=adopt_renamed_overrides({key: kwargs.get(key) for key in _OVERRIDE_KEYS}, kwargs),
        config_path=kwargs.get("config_path"),
    )

    model = settings.require("model", "FC_MODEL", "MODEL")
    api_key = settings.str_("api_key", "FC_API_KEY", "API_KEY")
    base_url = settings.require("base_url", "FC_BASE_URL", "BASE_URL")

    # The usable context window in tokens, which the provider can often tell us -- including the
    # llama.cpp case where the configured window is shared between --parallel slots. 0 disables the
    # budget; "auto" (the default) asks the provider and leaves it off if nothing is advertised,
    # because a guessed window is worse than none.
    max_context = resolve_max_context(
        settings.raw("max_context", "FC_MAX_CONTEXT"),
        base_url=base_url,
        api_key=api_key,
        model=model,
        verbose=kwargs.get("verbose", False),
    )
    max_turn_output_tokens = settings.int_(
        "max_turn_output_tokens", "FC_MAX_TURN_OUTPUT_TOKENS", DEFAULT_MAX_TURN_OUTPUT_TOKENS
    )
    # Off by default: the turn budget above is what protects the window, and this only changes how
    # that budget is shared between the calls of a single turn.
    max_result_output_tokens = settings.int_("max_result_output_tokens", "FC_MAX_RESULT_OUTPUT_TOKENS", 0)
    max_citations = settings.int_("max_citations", "FC_MAX_CITATIONS", DEFAULT_MAX_CITATIONS)

    # How long ONE response may be. Never detected from the provider: what a provider advertises is
    # the context WINDOW (see max_context above), and feeding that in here makes the reserve -- which
    # is 2 x this -- exceed the window, so the agent finalizes before its first turn.
    max_completion_tokens = resolve_max_completion_tokens(
        settings.raw("max_completion_tokens", "FC_MAX_COMPLETION_TOKENS"),
        verbose=kwargs.get("verbose", False),
    )
    temperature = settings.float_("temperature", "FC_TEMPERATURE", 0.7)

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        reasoning_effort=settings.str_("reasoning_effort", "FC_REASONING_EFFORT"),
    )

    # The budget trips only after a turn's tool results have landed, so the reserve must absorb a
    # full turn of tool output plus the completion -- otherwise the agent can cross the limit and
    # find that even the final-answer request no longer fits.
    reserve = settings.int_("context_reserve", "FC_CONTEXT_RESERVE", 0) or required_reserve(
        max_turn_output_tokens, max_completion_tokens
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
    elif max_turn_output_tokens <= 0:
        print(
            "warning: tool-output cap disabled (FC_MAX_TURN_OUTPUT_TOKENS is 0) while a context "
            "budget is set. A single Read can return ~250k tokens and overshoot the window in one "
            "turn, which the budget cannot undo.",
            file=sys.stderr,
        )

    toolset = ToolSet(
        [ReadTool(), GlobTool(), GrepTool()],
        work_dir=work_dir,
        max_turn_output_tokens=max_turn_output_tokens,
        max_result_output_tokens=max_result_output_tokens,
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
