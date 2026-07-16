import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.glob import GlobTool
from fastcontext.agent.tool.grep import DEFAULT_HEAD_LIMIT, GrepTool
from fastcontext.agent.tool.read import ReadTool


def test_grep_head_limit():
    """An explicit head_limit is honored, and the default applies when unspecified.

    Asserted against DEFAULT_HEAD_LIMIT rather than a literal: the default is a runaway guard whose
    value is expected to move, and the model is told it in grep.md.
    """
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        # More matches than the default cap, so both cases truncate.
        n = DEFAULT_HEAD_LIMIT + 50
        (Path(cwd) / "haystack.txt").write_text("\n".join("MATCH" for _ in range(n)), encoding="utf-8")

        def lines_for(params):
            out = asyncio.run(grep.call(json.dumps(params), cwd=cwd))
            return out.splitlines()

        # Explicit limit is honored. Truncated output = first `limit` lines + one "truncated" note.
        out = lines_for({"pattern": "MATCH", "output_mode": "content", "head_limit": 150})
        assert len(out) == 151, f"expected 150 lines + note, got {len(out)}"
        assert "truncated to first 150" in out[-1]

        # Unspecified: the default cap applies.
        out = lines_for({"pattern": "MATCH", "output_mode": "content"})
        assert len(out) == DEFAULT_HEAD_LIMIT + 1, f"expected {DEFAULT_HEAD_LIMIT} lines + note, got {len(out)}"
        assert f"truncated to first {DEFAULT_HEAD_LIMIT}" in out[-1]


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
