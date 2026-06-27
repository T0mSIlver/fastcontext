"""Launch FastContext runs across branches x tasks and save trajectories.

For each (branch, task) pair we:

1. materialize the branch as a git worktree of the FastContext source repo,
2. run ``fastcontext`` *inside the target repo* (so ``work_dir`` is the repo
   being explored) using that branch's code via ``uv run --project <worktree>``,
3. save the trajectory, the final-answer stdout, and run metadata under
   ``results/runs/<branch>/<task>/``.

Running each branch from its own worktree is what lets the eval compare two
versions of FastContext on identical queries.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass

from .config import Config, Task


@dataclass
class RunResult:
    branch: str
    task: str
    repo: str
    query: str
    trajectory: str
    final_answer_file: str
    exit_code: int
    wall_seconds: float
    started_at: float
    error: str | None = None


def _ensure_worktree(fastcontext_repo: str, branch: str, worktrees_dir: str) -> str:
    """Create (or reuse) a git worktree for ``branch`` and return its path."""
    wt = os.path.join(worktrees_dir, branch.replace("/", "__"))
    if os.path.isdir(os.path.join(wt, ".git")) or os.path.isfile(os.path.join(wt, ".git")):
        return wt
    os.makedirs(worktrees_dir, exist_ok=True)
    # Verify the branch exists before trying to add a worktree for it.
    rev = subprocess.run(
        ["git", "-C", fastcontext_repo, "rev-parse", "--verify", "--quiet", branch],
        capture_output=True,
        text=True,
    )
    if rev.returncode != 0:
        raise RuntimeError(f"branch '{branch}' not found in {fastcontext_repo}")
    add = subprocess.run(
        ["git", "-C", fastcontext_repo, "worktree", "add", "--force", wt, branch],
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        raise RuntimeError(f"git worktree add failed for {branch}:\n{add.stderr}")
    return wt


def _check_env() -> None:
    if not (os.getenv("FC_MODEL") or os.getenv("MODEL")):
        raise RuntimeError("FC_MODEL (or MODEL) is not set -- configure the FastContext endpoint first.")
    if not (os.getenv("FC_BASE_URL") or os.getenv("BASE_URL")):
        raise RuntimeError("FC_BASE_URL (or BASE_URL) is not set -- configure the FastContext endpoint first.")


def run_one(cfg: Config, branch: str, worktree: str, task: Task, results_dir: str, timeout: int) -> RunResult:
    out_dir = os.path.join(results_dir, "runs", branch.replace("/", "__"), task.name)
    os.makedirs(out_dir, exist_ok=True)
    traj = os.path.join(out_dir, "trajectory.jsonl")
    final_file = os.path.join(out_dir, "final_answer.txt")
    # Start fresh so a re-run never appends to a previous trajectory.
    for p in (traj, final_file):
        if os.path.exists(p):
            os.remove(p)

    cmd = [
        "uv", "run", "--project", worktree,
        "fastcontext", "-q", task.query,
        "--max-turns", str(cfg.max_turns),
        "-t", traj,
    ]
    if cfg.citation:
        cmd.append("--citation")

    started = time.time()
    error: str | None = None
    try:
        proc = subprocess.run(cmd, cwd=task.repo, capture_output=True, text=True, timeout=timeout)
        exit_code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        exit_code = -1
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\n[timed out after {timeout}s]"
        error = f"timeout after {timeout}s"
    wall = time.time() - started

    with open(final_file, "w", encoding="utf-8") as f:
        f.write(stdout)
    if exit_code != 0 and error is None:
        error = (stderr or "non-zero exit").strip().splitlines()[-1] if stderr.strip() else f"exit {exit_code}"

    result = RunResult(
        branch=branch,
        task=task.name,
        repo=task.repo,
        query=task.query,
        trajectory=traj,
        final_answer_file=final_file,
        exit_code=exit_code,
        wall_seconds=round(wall, 2),
        started_at=started,
        error=error,
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    with open(os.path.join(out_dir, "stderr.log"), "w", encoding="utf-8") as f:
        f.write(stderr)
    return result


def run_all(cfg: Config, results_dir: str, timeout: int = 600, keep_worktrees: bool = True) -> list[RunResult]:
    _check_env()
    worktrees_dir = os.path.join(results_dir, ".worktrees")
    results: list[RunResult] = []
    created_worktrees: list[str] = []
    for branch in cfg.branches:
        try:
            wt = _ensure_worktree(cfg.fastcontext_repo, branch, worktrees_dir)
            created_worktrees.append(wt)
        except RuntimeError as e:
            print(f"  ! skipping branch '{branch}': {e}")
            continue
        for task in cfg.tasks:
            print(f"  -> [{branch}] {task.name} ({task.repo})")
            res = run_one(cfg, branch, wt, task, results_dir, timeout)
            status = "ok" if res.error is None else f"ERROR: {res.error}"
            print(f"     done in {res.wall_seconds}s ({status})")
            results.append(res)

    if not keep_worktrees:
        for wt in created_worktrees:
            subprocess.run(
                ["git", "-C", cfg.fastcontext_repo, "worktree", "remove", "--force", wt],
                capture_output=True,
            )
            shutil.rmtree(wt, ignore_errors=True)
    return results
