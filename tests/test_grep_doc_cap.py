import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.grep import GrepTool


def test_grep_doc_describes_real_cap():
    """The Grep description must describe the real 100-line cap and the
    actual truncation note, not the fictional 'several thousand'/'at least'
    behavior."""
    desc = GrepTool().description
    assert "several thousand" not in desc
    assert "at least" not in desc
    assert "100" in desc
    assert "head_limit" in desc
    # The documented truncation wording must match what the tool actually emits.
    assert "Results truncated to first" in desc


def test_grep_doc_matches_runtime_truncation():
    """Cross-check: the truncation note produced at runtime uses exactly the
    phrasing the doc promises."""
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "haystack.txt").write_text("\n".join("MATCH" for _ in range(150)), encoding="utf-8")
        out = asyncio.run(grep.call(json.dumps({"pattern": "MATCH", "output_mode": "content"}), cwd=cwd))
        assert "Results truncated to first 100 lines" in out
        assert "Results truncated to first" in GrepTool().description
