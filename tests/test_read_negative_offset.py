import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.read import ReadTool


def _read(read, cwd, params):
    return asyncio.run(read.call(json.dumps(params), cwd=cwd))


def test_read_negative_offset_counts_from_end():
    """A negative offset must count backwards from the end of the file
    (FastContext paper, Appendix E, p. 19: 'Negative values count backwards
    from the end')."""
    read = ReadTool()
    with tempfile.TemporaryDirectory() as cwd:
        p = Path(cwd) / "f.txt"
        p.write_text("".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8")  # 10 lines

        # -2 -> start at line 9, read to end.
        out = _read(read, cwd, {"path": str(p), "offset": -2})
        assert "9|line9" in out and "10|line10" in out
        assert "8|line8" not in out

        # -1 with limit 1 -> only the last line.
        out = _read(read, cwd, {"path": str(p), "offset": -1, "limit": 1})
        assert "10|line10" in out and "9|line9" not in out

        # Over-long negative offset clamps to the start of the file.
        out = _read(read, cwd, {"path": str(p), "offset": -999})
        assert "1|line1" in out

        # Zero is still rejected.
        out = _read(read, cwd, {"path": str(p), "offset": 0})
        assert "<system-reminder>Error" in out and "non-zero" in out
