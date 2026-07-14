"""Tests for the runner: path handling, sweep resilience, retries, error reporting."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from fc_eval import cli
from fc_eval import run as run_mod
from fc_eval.analyze import _aggregate_by_branch
from fc_eval.config import Config, Task

OK_STDOUT = "<final_answer>\n/a/b.py:1-2 (x)\n</final_answer>"
LLM_FAIL_STDOUT = "LLM API call failed. So stopping the agent."


def _cfg(tmp_path, retries=0):
    task = Task(name="t1", repo=str(tmp_path / "repo"), query="where is x?")
    os.makedirs(task.repo, exist_ok=True)
    return Config(
        fastcontext_repo=str(tmp_path / "fc"),
        branches=["main"],
        tasks=[task],
        retries=retries,
        retry_backoff=0.0,
    )


def _fake_runner(stdouts, calls):
    """Stand in for the fastcontext subprocess: writes the trajectory it is handed.

    Resolves the -t path against ``cwd`` exactly like the real CLI would, so a
    relative results dir shows up as a file inside the explored repo.
    """

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        traj = cmd[cmd.index("-t") + 1]
        resolved = os.path.join(cwd or ".", traj)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "assistant", "content": "hi", "usage": {"total_tokens": 7}}) + "\n")
        calls.append({"cmd": cmd, "cwd": cwd, "traj": traj, "resolved": resolved})
        out = stdouts[min(len(calls) - 1, len(stdouts) - 1)]
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    return fake_run


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("FC_MODEL", "m")
    monkeypatch.setenv("FC_BASE_URL", "http://localhost:8080/v1")


def test_relative_results_dir_never_writes_into_task_repo(tmp_path, monkeypatch, env):
    cfg = _cfg(tmp_path)
    calls = []
    monkeypatch.setattr(run_mod.subprocess, "run", _fake_runner([OK_STDOUT], calls))
    monkeypatch.chdir(tmp_path)

    results = run_mod.run_all(cfg, "results-rel")  # relative on purpose

    traj = results[0].trajectory
    assert os.path.isabs(traj)
    assert traj.startswith(str(tmp_path / "results-rel"))
    assert os.path.isfile(traj)
    assert os.path.isabs(calls[0]["traj"])
    # Nothing at all leaked into the explored repo.
    assert os.listdir(cfg.tasks[0].repo) == []


def test_one_failing_run_does_not_abort_the_sweep(tmp_path, monkeypatch, env):
    cfg = _cfg(tmp_path)
    cfg.tasks.append(Task(name="t2", repo=cfg.tasks[0].repo, query="q2"))
    calls = []
    ok = _fake_runner([OK_STDOUT], calls)

    def boom(cmd, cwd=None, **kwargs):
        if cmd[0] != "git" and f"{os.sep}t1{os.sep}" in cmd[cmd.index("-t") + 1]:
            raise FileNotFoundError("uv")
        return ok(cmd, cwd=cwd, **kwargs)

    monkeypatch.setattr(run_mod.subprocess, "run", boom)
    results = run_mod.run_all(cfg, str(tmp_path / "results"))

    assert [r.task for r in results] == ["t1", "t2"]
    assert "FileNotFoundError" in results[0].error
    assert results[1].error is None
    # The failed run still gets a meta.json so the analyzer sees it as an error.
    meta = json.load(open(os.path.join(tmp_path, "results", "runs", "main", "t1", "meta.json")))
    assert meta["error"].startswith("FileNotFoundError")


def test_llm_failure_is_detected_and_retried(tmp_path, monkeypatch, env):
    cfg = _cfg(tmp_path, retries=2)
    calls = []
    monkeypatch.setattr(run_mod.subprocess, "run", _fake_runner([LLM_FAIL_STDOUT, LLM_FAIL_STDOUT, OK_STDOUT], calls))

    results = run_mod.run_all(cfg, str(tmp_path / "results"))

    assert results[0].attempts == 3
    assert results[0].error is None


def test_llm_failure_exhausting_retries_is_marked_errored(tmp_path, monkeypatch, env):
    cfg = _cfg(tmp_path, retries=1)
    calls = []
    monkeypatch.setattr(run_mod.subprocess, "run", _fake_runner([LLM_FAIL_STDOUT], calls))

    results = run_mod.run_all(cfg, str(tmp_path / "results"))

    assert results[0].attempts == 2
    assert results[0].error == "LLM API call failed"


def test_errored_runs_excluded_from_aggregates():
    rows = [
        {"branch": "a", "task": "t1", "turns": 4, "total_tokens": 100, "reached_final_answer": True},
        {"branch": "a", "task": "t2", "turns": 0, "total_tokens": 0, "run_error": "LLM API call failed"},
    ]
    agg = _aggregate_by_branch(rows)["a"]
    assert agg["runs"] == 2
    assert agg["scored_runs"] == 1
    assert agg["errored_runs"] == 1
    # Averaged over the one good run, not diluted to 50 by the failure.
    assert agg["total_tokens"] == 100
    assert agg["turns"] == 4


@pytest.mark.parametrize(
    "argv",
    [
        ["-c", "CFG", "-r", "RES", "analyze"],
        ["analyze", "-c", "CFG", "-r", "RES"],
        ["-c", "CFG", "analyze", "-r", "RES"],
    ],
)
def test_config_and_results_flags_accepted_on_both_sides(argv, tmp_path, monkeypatch):
    seen = {}

    def fake_analyze(results_dir, config=None):
        seen["results"] = results_dir
        return {"rows": [], "aggregates": {}}

    def fake_load_config(path):
        seen["config"] = path
        return None

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "analyze", fake_analyze)
    monkeypatch.setattr(cli, "write_summary", lambda summary, path: None)
    monkeypatch.chdir(tmp_path)

    cli.main([a.replace("CFG", "my.yaml").replace("RES", "out") for a in argv])

    assert seen["config"] == str(tmp_path / "my.yaml")
    assert seen["results"] == str(tmp_path / "out")


def test_timeout_is_reported_as_an_error(tmp_path, monkeypatch, env):
    cfg = _cfg(tmp_path)

    def timeout(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(run_mod.subprocess, "run", timeout)
    results = run_mod.run_all(cfg, str(tmp_path / "results"), timeout=1)
    assert results[0].error == "timeout after 1s"
