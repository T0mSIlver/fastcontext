import asyncio
import json
import re
import tempfile
from pathlib import Path

from fastcontext.agent.tool.grep import GrepTool


def test_grep_defaults_to_content():
    """With output_mode omitted, Grep must behave as 'content' (matching lines
    with line numbers), not as files_with_matches and not as a flag-less rg
    call that drops -n."""
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "haystack.txt").write_text("alpha\nMATCH here\nbeta\n", encoding="utf-8")

        out = asyncio.run(grep.call(json.dumps({"pattern": "MATCH"}), cwd=cwd))
        lines = out.splitlines()

        # Content mode with -n emits "<lineno>:<text>" rows for matches.
        assert any(re.match(r"\d+:.*MATCH", ln) for ln in lines), f"expected line-numbered match, got: {out!r}"
        # files_with_matches would emit only the bare path with no match text.
        assert "MATCH here" in out


def test_grep_schema_documents_content_default():
    schema = GrepTool().schema()
    desc = schema["function"]["parameters"]["properties"]["output_mode"]["description"]
    assert 'Defaults to "content"' in desc
    assert "files_with_matches" not in desc.split("Defaults to")[1]
