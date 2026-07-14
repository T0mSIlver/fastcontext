"""CLI for the FastContext eval harness.

    uv run fc-eval run       # launch runs (branches x tasks), save trajectories
    uv run fc-eval analyze   # score trajectories -> results/summary.json
    uv run fc-eval dashboard # render results/dashboard.html (and optionally serve)
    uv run fc-eval all       # run + analyze + dashboard
"""

from __future__ import annotations

import argparse
import os
import webbrowser

from .analyze import analyze, write_summary
from .config import load_config
from .dashboard import write_dashboard
from .run import run_all

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_ROOT = os.path.dirname(_HERE)
DEFAULT_CONFIG = os.path.join(_EVAL_ROOT, "config.yaml")
DEFAULT_RESULTS = os.path.join(_EVAL_ROOT, "results")


def _cmd_run(args):
    cfg = load_config(args.config)
    if args.retries is not None:
        cfg.retries = max(0, args.retries)
    print(f"Launching {len(cfg.branches)} branch(es) x {len(cfg.tasks)} task(s)...")
    run_all(cfg, args.results, timeout=args.timeout, keep_worktrees=not args.clean_worktrees)


def _cmd_analyze(args):
    cfg = load_config(args.config)
    summary = analyze(args.results, config=cfg)
    out = os.path.join(args.results, "summary.json")
    write_summary(summary, out)
    print(f"Scored {len(summary['rows'])} run(s) -> {out}")
    for branch, agg in summary["aggregates"].items():
        print(
            f"  {branch}: {agg['scored_runs']}/{agg['runs']} scored, errors: "
            f"{agg['errored_runs']}/{agg['runs']} | mean failed={agg['failed_tool_calls']} "
            f"dup={agg['duplicate_tool_calls']} corr={agg['corrections']} "
            f"unverified={agg['unverified_citations']} tokens={agg['total_tokens']:.0f}"
        )
    for row in summary["rows"]:
        err = row.get("run_error") or row.get("error")
        if err:
            print(f"  ! ERRORED (excluded from means): [{row['branch']}] {row['task']}: {err}")


def _cmd_dashboard(args):
    cfg = load_config(args.config)
    summary = analyze(args.results, config=cfg)
    write_summary(summary, os.path.join(args.results, "summary.json"))
    out = os.path.join(args.results, "dashboard.html")
    write_dashboard(summary, out)
    print(f"Dashboard -> {out}")
    if args.open:
        webbrowser.open(f"file://{out}")
    if args.serve:
        _serve(args.results, args.port)


def _cmd_all(args):
    _cmd_run(args)
    _cmd_dashboard(args)


def _serve(results_dir: str, port: int):
    import functools
    import http.server
    import socketserver

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=results_dir)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving {results_dir} at http://localhost:{port}/dashboard.html (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def _common_parser() -> argparse.ArgumentParser:
    """-c/-r, accepted both before and after the subcommand.

    The defaults are SUPPRESS-ed and applied after parsing: a subparser inheriting
    these actions would otherwise overwrite a value given to the top-level parser
    with its own default.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", default=argparse.SUPPRESS, help="path to config.yaml")
    common.add_argument("--results", "-r", default=argparse.SUPPRESS, help="results directory")
    return common


def main(argv=None):
    common = _common_parser()
    p = argparse.ArgumentParser(prog="fc-eval", description="FastContext eval harness", parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="launch runs and save trajectories", parents=[common])
    pr.add_argument("--timeout", type=int, default=600, help="per-run timeout in seconds")
    pr.add_argument("--clean-worktrees", action="store_true", help="remove branch worktrees after running")
    pr.add_argument("--retries", type=int, default=None, help="retries per failed run (overrides config)")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="score trajectories into summary.json", parents=[common])
    pa.set_defaults(func=_cmd_analyze)

    pd = sub.add_parser("dashboard", help="build the HTML dashboard", parents=[common])
    pd.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    pd.add_argument("--serve", action="store_true", help="serve the results dir over HTTP")
    pd.add_argument("--port", type=int, default=8009)
    pd.set_defaults(func=_cmd_dashboard)

    pall = sub.add_parser("all", help="run + analyze + dashboard", parents=[common])
    pall.add_argument("--timeout", type=int, default=600)
    pall.add_argument("--clean-worktrees", action="store_true")
    pall.add_argument("--retries", type=int, default=None)
    pall.add_argument("--open", action="store_true")
    pall.add_argument("--serve", action="store_true")
    pall.add_argument("--port", type=int, default=8009)
    pall.set_defaults(func=_cmd_all)

    args = p.parse_args(argv)
    # The agent subprocess runs with cwd=<explored repo>: relative paths here would
    # write trajectories into that repo instead of the results dir.
    args.config = os.path.abspath(os.path.expanduser(getattr(args, "config", None) or DEFAULT_CONFIG))
    args.results = os.path.abspath(os.path.expanduser(getattr(args, "results", None) or DEFAULT_RESULTS))
    try:
        args.func(args)
    except RuntimeError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
