import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from fastcontext.agent.agent import AgentRunError
from fastcontext.agent.agent_factory import make_fastcontext_agent


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
    parser.add_argument("--max-turns", type=int, help="maximum number of turns", default=4)
    parser.add_argument(
        "--max-tokens",
        type=str,
        default=None,
        metavar="N|auto",
        help=(
            "max completion tokens per response: an integer, or 'auto' to fetch the model's "
            "context length from the provider. Overrides FC_MAX_TOKENS. "
            "Default: auto-detect, falling back to 4096."
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
        type=int,
        default=None,
        help=(
            "usable context window in tokens. When the conversation approaches it, the agent stops "
            "exploring and produces its final answer instead of growing the prompt until the "
            "provider rejects it. 0 disables the budget. Overrides FC_MAX_CONTEXT. Note a server's "
            "usable window can be well below its configured one (llama.cpp --parallel 2 halves it)."
        ),
    )
    parser.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=None,
        help=(
            "total characters of tool output ONE TURN may add, across all of its tool calls "
            "(0 disables). Guards against a turn exhausting the whole window; the context reserve is "
            "sized against it. Overrides FC_MAX_TOOL_OUTPUT_CHARS."
        ),
    )
    parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=None,
        help=(
            "truncate a SINGLE tool result above this many characters (0 disables, the default). "
            "The turn budget is spent in call order, so one huge result can starve the later calls "
            "of that turn; this caps each result first. Overrides FC_MAX_TOOL_RESULT_CHARS."
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

    args = parser.parse_args()

    if args.command == "init":
        raise SystemExit(_run_init(args))

    work_dir = os.getcwd()
    agent = make_fastcontext_agent(
        trajectory_file=args.traj,
        work_dir=work_dir,
        max_tokens=args.max_tokens,
        max_context=args.max_context,
        max_tool_output_chars=args.max_tool_output_chars,
        max_tool_result_chars=args.max_tool_result_chars,
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
