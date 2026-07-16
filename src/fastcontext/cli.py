import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from fastcontext.agent.agent import AgentRunError
from fastcontext.agent.agent_factory import make_fastcontext_agent
from fastcontext.agent.config import warn_renamed_flag
from fastcontext.agent.llm import DEFAULT_MAX_COMPLETION_TOKENS

# Superseded spellings of --max-turn-output-tokens. Both measured CHARACTERS; the cap is now in
# tokens, so a value passed under these names is converted, not reused verbatim.
_DEPRECATED_TURN_FLAGS = ("--max-tool-output-chars", "--max-turn-output-chars")

# Deprecated alias of --max-completion-tokens. Same quantity, so argparse just accepts it.
_DEPRECATED_COMPLETION_FLAG = "--max-tokens"


def _run_init(args) -> int:
    """`fastcontext init`: scaffold a starter config file."""
    from fastcontext.agent.config import user_config_path, write_starter_config

    path = Path(args.path).expanduser() if args.path else user_config_path()
    try:
        written = write_starter_config(path, force=args.force)
    except FileExistsError:
        print(f"{path} already exists; pass --force to overwrite.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"could not write {path}: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote starter config to {written}\nEdit it, then run: fastcontext -q \"...\"")
    return 0


def main():
    """FastContext Command Line Interface"""
    parser = argparse.ArgumentParser(
        description="FastContext CLI",
    )

    # Optional subcommands. Without one, the arguments below run an exploration, so the existing
    # `fastcontext -q "..."` invocation is unchanged.
    subparsers = parser.add_subparsers(dest="command")
    init_parser = subparsers.add_parser("init", help="scaffold a starter config.toml (from current FC_* env vars)")
    init_parser.add_argument(
        "--path",
        type=str,
        default=None,
        metavar="PATH",
        help="where to write the config (default: $XDG_CONFIG_HOME/fastcontext/config.toml).",
    )
    init_parser.add_argument("--force", action="store_true", help="overwrite an existing config file.")

    parser.add_argument("--query", "-q", type=str, help="query to ask the agent")
    parser.add_argument(
        "--traj",
        "-t",
        type=str,
        help="agent trajectory file",
        default=f".fastcontext/trajectory_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.jsonl",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=12,
        metavar="N",
        help=(
            "maximum exploration turns before the agent is asked for its final answer. Default 12: "
            "measured across six repos, runs that converged on their own needed 5-14 turns (median "
            "~9), so the previous default of 4 cut most real explorations short. A cap above what a "
            "question needs costs nothing -- a simple lookup stops on its own well before it."
        ),
    )
    parser.add_argument(
        "--max-completion-tokens",
        # Deprecated alias. Same quantity, so it is accepted rather than rejected; it is renamed
        # because "max tokens" named neither which tokens nor whose, and sat one letter from
        # --max-turns.
        _DEPRECATED_COMPLETION_FLAG,
        dest="max_completion_tokens",
        type=str,
        default=None,
        metavar="N",
        help=(
            f"how long ONE response may be, in tokens (default {DEFAULT_MAX_COMPLETION_TOKENS}). This "
            "is not the model's window -- see --max-context. Overrides FC_MAX_COMPLETION_TOKENS."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="whether to run in verbose mode")
    parser.add_argument(
        "--tui",
        action="store_true",
        help="stream the run in a collapsible Textual TUI (every reasoning, tool call and result)",
    )
    parser.add_argument("--citation", action="store_true", help="Only return the citations in the final answer")
    parser.add_argument(
        "--max-context",
        type=str,
        default=None,
        metavar="N|auto",
        help=(
            "usable context window in tokens: an integer, 0 to disable the budget, or 'auto' (the "
            "default) to ask the provider. When the conversation approaches it, the agent stops "
            "exploring and produces its final answer instead of growing the prompt until the "
            "provider rejects it. Overrides FC_MAX_CONTEXT. Note a server's usable window can be "
            "well below its configured one (llama.cpp --parallel 2 halves it); auto accounts for that."
        ),
    )
    parser.add_argument(
        "--max-turn-output-tokens",
        dest="max_turn_output_tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "total tokens of tool output ONE TURN may add, across all of its tool calls "
            "(0 disables). Guards against a turn exhausting the whole window; the context reserve is "
            "sized against it. Overrides FC_MAX_TURN_OUTPUT_TOKENS."
        ),
    )
    parser.add_argument(
        "--max-result-output-tokens",
        dest="max_result_output_tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "truncate a SINGLE tool result above this many tokens (0 disables, the default). "
            "The turn budget is spent in call order, so one huge result can starve the later calls "
            "of that turn; this caps each result first. Overrides FC_MAX_RESULT_OUTPUT_TOKENS."
        ),
    )
    parser.add_argument(
        "--max-citations",
        type=int,
        default=None,
        help=(
            "cap the number of citations in the final answer (0 disables). A safety bound on a "
            "runaway/hallucinated list; the model's order is kept. Overrides FC_MAX_CITATIONS "
            "(default 25)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "path to a TOML config file (overrides FC_CONFIG and config-file discovery). "
            "Without it, settings are read from ./.fastcontext/config.toml then "
            "$XDG_CONFIG_HOME/fastcontext/config.toml. Env vars and CLI flags still win."
        ),
    )

    # Before parse_args, which would otherwise reject these with a bare "unrecognized arguments" and
    # exit before anything explained why. They are gone rather than aliased: they took a CHARACTER
    # count and the cap is now in tokens, so silently reusing the number would roughly triple it.
    for flag in _DEPRECATED_TURN_FLAGS:
        if any(a == flag or a.startswith(f"{flag}=") for a in sys.argv[1:]):
            warn_renamed_flag(flag, "--max-turn-output-tokens (a TOKEN count -- roughly chars / 3)")
    if any(a == _DEPRECATED_COMPLETION_FLAG or a.startswith(f"{_DEPRECATED_COMPLETION_FLAG}=") for a in sys.argv[1:]):
        warn_renamed_flag(_DEPRECATED_COMPLETION_FLAG, "--max-completion-tokens")

    args = parser.parse_args()

    if args.command == "init":
        raise SystemExit(_run_init(args))

    work_dir = os.getcwd()
    agent = make_fastcontext_agent(
        trajectory_file=args.traj,
        work_dir=work_dir,
        max_completion_tokens=args.max_completion_tokens,
        max_context=args.max_context,
        max_turn_output_tokens=args.max_turn_output_tokens,
        max_result_output_tokens=args.max_result_output_tokens,
        max_citations=args.max_citations,
        verbose=args.verbose,
        config_path=args.config,
    )

    prompt = args.query

    if args.tui:
        from fastcontext.tui import FastContextTUI

        app = FastContextTUI(agent=agent, prompt=prompt, max_turns=args.max_turns, citation=args.citation)
        app.run()
        if app.final_answer is not None:
            print(app.final_answer)
        elif app.error is not None:
            # The run failed inside the TUI worker; surface it the same way as the headless path.
            print(app.error, file=sys.stderr)
            raise SystemExit(1)
        return

    try:
        final_output = asyncio.run(
            agent.run(prompt=prompt, max_turns=args.max_turns, verbose=args.verbose, citation=args.citation)
        )
    except AgentRunError as exc:
        # A failed run exits nonzero with the error on stderr, so a driving agent can detect failure
        # from the exit code instead of scanning stdout. stdout stays clean (no partial answer).
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    print(final_output)


if __name__ == "__main__":
    main()
