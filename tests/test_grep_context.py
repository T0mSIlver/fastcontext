import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.grep import GrepTool


def _search(grep, cwd, params):
    return asyncio.run(grep.call(json.dumps(params), cwd=cwd))


def test_grep_no_context_unless_requested():
    """content mode must not inject surrounding context by default; -C only
    applies when the model explicitly provides it."""
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "a.txt").write_text("before\nMATCH here\nafter\n", encoding="utf-8")

        # No -C: only the matching line, no neighbors.
        out = _search(grep, cwd, {"pattern": "MATCH", "output_mode": "content"})
        assert "MATCH here" in out
        assert "before" not in out, f"unexpected default context: {out!r}"
        assert "after" not in out, f"unexpected default context: {out!r}"

        # Explicit -C: neighbors are included.
        out = _search(grep, cwd, {"pattern": "MATCH", "output_mode": "content", "-C": 1})
        assert "before" in out and "after" in out, f"expected context with -C=1: {out!r}"
