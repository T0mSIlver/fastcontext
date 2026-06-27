import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from fastcontext.agent.tool.grep import GrepTool


def test_grep_returns_reminder_on_timeout():
    """run_rg must bound the ripgrep subprocess and surface a system-reminder
    instead of hanging / raising when it times out. (The outer asyncio
    wait_for cannot interrupt the blocking subprocess.run, so the timeout
    must live on the subprocess itself.)"""
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "a.txt").write_text("MATCH\n", encoding="utf-8")

        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="rg", timeout=10)):
            out = asyncio.run(grep.call(json.dumps({"pattern": "MATCH", "output_mode": "content"}), cwd=cwd))

        assert "<system-reminder>" in out
        assert "timed out" in out


def test_grep_passes_timeout_to_subprocess():
    """The subprocess.run call must receive a positive timeout kwarg."""
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "a.txt").write_text("MATCH\n", encoding="utf-8")

        real_run = subprocess.run
        captured = {}

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_run(*args, **kwargs)

        with mock.patch("subprocess.run", side_effect=spy):
            asyncio.run(grep.call(json.dumps({"pattern": "MATCH", "output_mode": "content"}), cwd=cwd))

        assert captured.get("timeout", 0) and captured["timeout"] > 0
