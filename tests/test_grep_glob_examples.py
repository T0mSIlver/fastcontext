import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.glob import GlobTool
from fastcontext.agent.tool.grep import GrepTool


def test_grep_doc_glob_examples_are_valid():
    """The glob examples in the Grep description must be extension globs, not
    the malformed dotfile patterns '.js' / '**/.tsx'."""
    desc = GrepTool().description
    assert '"*.js"' in desc and '"**/*.tsx"' in desc
    # The old malformed examples must be gone.
    assert '(e.g., ".js"' not in desc
    assert '"**/.tsx"' not in desc


def test_documented_extension_glob_actually_matches():
    """Sanity-check that the documented '*.js' pattern matches by extension
    (the malformed '.js' would only match a literal dotfile)."""
    glob = GlobTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "app.js").write_text("x", encoding="utf-8")
        (Path(cwd) / "readme.md").write_text("x", encoding="utf-8")

        out = asyncio.run(glob.call(json.dumps({"directory": cwd, "pattern": "*.js"}), cwd=cwd))
        names = {Path(line).name for line in out.splitlines()}
        assert names == {"app.js"}
