import asyncio
import json

from fastcontext.agent.tool.glob import GlobTool
from fastcontext.agent.tool.grep import GrepTool
from fastcontext.agent.tool.read import ReadTool


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
