"""Aggregate trajectory metrics into a summary for the dashboard."""

from __future__ import annotations

import glob
import json
import os

from .config import Config
from .metrics import score_run


def _read_final_answer(meta: dict, run_dir: str) -> str | None:
    fa = os.path.join(run_dir, "final_answer.txt")
    if os.path.isfile(fa):
        text = open(fa, encoding="utf-8").read().strip()
        return text or None
    return None


def analyze(results_dir: str, config: Config | None = None) -> dict:
    """Score every run under ``results_dir/runs`` plus any reference runs."""
    rows: list[dict] = []

    for meta_path in sorted(glob.glob(os.path.join(results_dir, "runs", "*", "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        traj = meta.get("trajectory") or os.path.join(run_dir, "trajectory.jsonl")
        final_answer = _read_final_answer(meta, run_dir)
        row = {
            "branch": meta.get("branch", os.path.basename(os.path.dirname(run_dir))),
            "task": meta.get("task", os.path.basename(run_dir)),
            "repo": meta.get("repo", ""),
            "query": meta.get("query", ""),
            "wall_seconds": meta.get("wall_seconds"),
            "run_error": meta.get("error"),
            "kind": "run",
        }
        if os.path.isfile(traj):
            cwd = meta.get("repo") or None
            m = score_run(traj, cwd=cwd, final_answer=final_answer)
            row.update(m.to_dict())
        else:
            row["error"] = "no trajectory file"
        rows.append(row)

    # Reference (pre-recorded) trajectories declared in config.
    if config:
        for ref in config.reference_runs:
            row = {"branch": ref.branch, "task": ref.name, "repo": ref.repo or "", "query": "", "kind": "reference"}
            if os.path.isfile(ref.trajectory):
                m = score_run(ref.trajectory, cwd=ref.repo)
                row.update(m.to_dict())
            else:
                row["error"] = f"missing trajectory: {ref.trajectory}"
            rows.append(row)

    summary = {
        "rows": rows,
        "branches": sorted({r["branch"] for r in rows}),
        "tasks": sorted({r["task"] for r in rows}),
        "aggregates": _aggregate_by_branch(rows),
    }
    return summary


# Numeric metrics we average per branch.
_AGG_FIELDS = [
    "turns",
    "tool_calls_total",
    "failed_tool_calls",
    "duplicate_tool_calls",
    "corrections",
    "citations_total",
    "unverified_citations",
    "citations_missing_file",
    "total_tokens",
]


def row_error(row: dict) -> str | None:
    """The run's failure, if any: a harness/LLM error recorded at run time or at scoring."""
    err = row.get("run_error") or row.get("error")
    return str(err) if err else None


def _aggregate_by_branch(rows: list[dict]) -> dict:
    by_branch: dict[str, list[dict]] = {}
    for r in rows:
        by_branch.setdefault(r["branch"], []).append(r)

    agg = {}
    for branch, brows in by_branch.items():
        # An errored run is not a data point: averaging it in would report a
        # broken run as a legitimate zero.
        scored = [r for r in brows if "turns" in r and not row_error(r)]
        n = len(scored) or 1
        means = {f: round(sum(r.get(f, 0) or 0 for r in scored) / n, 1) for f in _AGG_FIELDS}
        means["runs"] = len(brows)
        means["scored_runs"] = len(scored)
        means["reached_final_answer"] = sum(1 for r in scored if r.get("reached_final_answer"))
        means["errored_runs"] = sum(1 for r in brows if row_error(r))
        agg[branch] = means
    return agg


def write_summary(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
