"""Grep against a single file must still name the file it matched in.

rg omits the filename when the target is one explicit file, so a Grep scoped to a single path came
back as bare `12:match` with the path nowhere in the output -- leaving the model to remember which
file it had asked about before it could cite the line.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.grep import GrepTool


def test_grep_on_a_single_file_names_the_file():
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        target = Path(cwd) / "alpha.py"
        target.write_text("MODEL = 'x'\nother = 1\n", encoding="utf-8")

        out = asyncio.run(grep.call(json.dumps({"pattern": "MODEL", "path": str(target)}), cwd=cwd))

        assert "MODEL = 'x'" in out
        assert "alpha.py" in out, f"single-file grep lost the path: {out!r}"


def test_grep_on_a_directory_still_names_each_file():
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        (Path(cwd) / "alpha.py").write_text("MODEL = 'x'\n", encoding="utf-8")
        (Path(cwd) / "beta.py").write_text("MODEL = 'y'\n", encoding="utf-8")

        out = asyncio.run(grep.call(json.dumps({"pattern": "MODEL", "path": cwd}), cwd=cwd))

        assert "alpha.py" in out and "beta.py" in out


def test_single_file_count_mode_names_the_file():
    grep = GrepTool()
    with tempfile.TemporaryDirectory() as cwd:
        target = Path(cwd) / "alpha.py"
        target.write_text("MODEL\nMODEL\n", encoding="utf-8")

        out = asyncio.run(
            grep.call(
                json.dumps({"pattern": "MODEL", "path": str(target), "output_mode": "count"}), cwd=cwd
            )
        )

        assert "alpha.py" in out
