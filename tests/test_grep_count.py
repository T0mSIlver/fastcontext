import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.grep import GrepTool


def test_grep_count_mode():
    """output_mode 'count' (the advertised enum value) must emit per-file
    match counts via rg --count-matches, not fall through to content."""
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "haystack.txt").write_text("MATCH\nMATCH\nMATCH\nnope\n", encoding="utf-8")

        out = asyncio.run(grep.call(json.dumps({"pattern": "MATCH", "output_mode": "count"}), cwd=cwd))

        # rg --count-matches prints "<path>:<count>"; 3 matches in the file.
        assert out.strip().endswith(":3"), f"expected a per-file count of 3, got: {out!r}"
        # It must not be content mode (which would echo the matched line text).
        assert "MATCH\n" not in out and not out.strip().endswith("MATCH")
