import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.read import ReadTool


def test_read_truncates_long_lines_at_500():
    """read.md promises lines longer than 500 chars are truncated to 500 with
    '...'. Verify the runtime honors that length."""
    read = ReadTool()
    with tempfile.TemporaryDirectory() as cwd:
        p = Path(cwd) / "long.txt"
        p.write_text("a" * 600 + "\n" + "short\n", encoding="utf-8")

        out = asyncio.run(read.call(json.dumps({"path": str(p)}), cwd=cwd))

        # Exactly 500 chars of the long line survive, with a "..." marker.
        assert "a" * 500 in out
        assert "a" * 501 not in out
        assert "..." in out
        # Short lines are passed through untouched.
        assert "short" in out


def test_read_doc_matches_truncation_length():
    """The documented truncation length must match the implementation."""
    from fastcontext.agent.tool.read import MAX_LINE_LENGTH

    assert MAX_LINE_LENGTH == 500
    assert "500 characters" in ReadTool().description
