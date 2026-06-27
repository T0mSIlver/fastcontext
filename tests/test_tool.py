import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.glob import GlobTool
from fastcontext.agent.tool.grep import GrepTool
from fastcontext.agent.tool.read import ReadTool


def test_grep_head_limit():
    """head_limit must be honored for values above the default 100-line cap,
    and fall back to 100 when unspecified."""
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        # 250 matching lines, single file, so output exceeds any tested limit.
        (Path(cwd) / "haystack.txt").write_text("\n".join("MATCH" for _ in range(250)), encoding="utf-8")

        def lines_for(params):
            out = asyncio.run(grep.call(json.dumps(params), cwd=cwd))
            return out.splitlines()

        # Above the default cap: previously clamped to 100, must now be honored.
        # Truncated output = first `limit` lines + one "Results truncated" note.
        out = lines_for({"pattern": "MATCH", "output_mode": "content", "head_limit": 150})
        assert len(out) == 151, f"expected 150 lines + note, got {len(out)}"
        assert "truncated to first 150" in out[-1]

        # Unspecified: 100-line default cap still applies.
        out = lines_for({"pattern": "MATCH", "output_mode": "content"})
        assert len(out) == 101, f"expected 100 lines + note, got {len(out)}"
        assert "truncated to first 100" in out[-1]


def test_grep_tool():
    grep = GrepTool()
    params = {
        "pattern": "grep.call",
        "path": ".",
        "glob": "*.py",
        "output_mode": "content",
        "head_limit": 100,
        "-C": 3,
    }

    output = asyncio.run(grep.call(json.dumps(params)))
    print(output)

    # /testbed/**: No such file or directory (os error 2)
    params = {"pattern": "arithmetic", "path": "/testbed/**", "output_mode": "files_with_matches", "head_limit": 200}
    output = asyncio.run(grep.call(json.dumps(params)))
    print(output)


def test_glob_tool():
    glob = GlobTool()
    params = {
        "directory": "./src",
        "pattern": "**/*.py",
    }
    output = asyncio.run(glob.call(json.dumps(params)))
    print(output)


def test_read_tool_path_traversal():
    read = ReadTool()
    cwd = "/tmp/"

    # Should be blocked: outside cwd
    output = asyncio.run(read.call(json.dumps({"path": "/etc/passwd"})))
    assert "<system-reminder>Permission error" in output, f"Expected permission error, got: {output}"

    # Should be blocked: not within cwd
    output = asyncio.run(read.call(json.dumps({"path": f"{cwd}/README.md"}), cwd=cwd))
    assert "<system-reminder>Error:" in output

    from pathlib import Path

    cwd = Path.cwd().as_posix()
    output = asyncio.run(read.call(json.dumps({"path": f"{cwd}/test_llm.py"}), cwd="./"))


if __name__ == "__main__":
    test_grep_tool()
    test_glob_tool()
    test_read_tool_path_traversal()
