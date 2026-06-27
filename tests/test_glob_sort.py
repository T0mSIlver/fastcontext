import asyncio
import json
import os
import tempfile
from pathlib import Path

from fastcontext.agent.tool.glob import GlobTool


def test_glob_sorts_by_modification_time():
    """glob.md promises results sorted by modification time. Create files out
    of mtime order on disk and assert the output is ordered oldest -> newest
    (rg --sort modified)."""
    glob = GlobTool()
    with tempfile.TemporaryDirectory() as cwd:
        # Create in a deliberately non-mtime order, then stamp mtimes.
        for name, mtime in [("new.txt", 3000), ("old.txt", 1000), ("mid.txt", 2000)]:
            p = Path(cwd) / name
            p.write_text("x", encoding="utf-8")
            os.utime(p, (mtime, mtime))

        out = asyncio.run(glob.call(json.dumps({"directory": cwd, "pattern": "*.txt"}), cwd=cwd))
        names = [Path(line).name for line in out.splitlines()]

        assert names == ["old.txt", "mid.txt", "new.txt"], f"expected mtime order, got {names}"
