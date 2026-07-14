"""Eval configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class Task:
    name: str
    repo: str
    query: str


@dataclass
class ReferenceRun:
    branch: str
    name: str
    trajectory: str
    repo: str | None = None


@dataclass
class Config:
    fastcontext_repo: str
    branches: list[str]
    tasks: list[Task]
    max_turns: int = 6
    citation: bool = True
    retries: int = 2  # extra attempts for a run that failed (flaky LLM endpoint)
    retry_backoff: float = 5.0  # seconds to wait between attempts
    reference_runs: list[ReferenceRun] = field(default_factory=list)


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _abs(p: str) -> str:
        return os.path.abspath(os.path.expanduser(p))

    tasks = [Task(name=t["name"], repo=_abs(t["repo"]), query=t["query"]) for t in raw.get("tasks", [])]
    refs = [
        ReferenceRun(
            branch=r.get("branch", "reference"),
            name=r["name"],
            trajectory=_abs(r["trajectory"]),
            repo=_abs(r["repo"]) if r.get("repo") else None,
        )
        for r in raw.get("reference_runs", [])
    ]
    return Config(
        fastcontext_repo=_abs(raw["fastcontext_repo"]),
        branches=list(raw.get("branches", [])),
        tasks=tasks,
        max_turns=int(raw.get("max_turns", 6)),
        citation=bool(raw.get("citation", True)),
        retries=max(0, int(raw.get("retries", 2))),
        retry_backoff=float(raw.get("retry_backoff", 5.0)),
        reference_runs=refs,
    )
