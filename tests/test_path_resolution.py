"""Tests for resolve_path and the absolute-path directory listing.

resolve_path recovers from the model's habit of emitting a mangled absolute
prefix (e.g. using the repo name as the filesystem root) and tells the model
when the lossy suffix-matching fallback was used. The system-prompt listing is
rendered with absolute paths so the model is primed to pass absolute paths.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from fastcontext.agent.tool.read import ReadTool
from fastcontext.agent.tool.utils import resolve_path
from fastcontext.agent.utils import load_system_prompt


def _make_repo(tmp: str) -> Path:
    repo = Path(tmp) / "myrepo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "llm.py").write_text("MODEL = 'x'\n", encoding="utf-8")
    return repo


def test_absolute_in_cwd_is_verbatim_no_note():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        full = str(repo / "src" / "pkg" / "llm.py")
        resolved, note = resolve_path(full, str(repo))
        assert resolved == full
        assert note is None


def test_absolute_in_cwd_nonexistent_is_verbatim():
    # Existence is the caller's job; resolve_path must not rewrite an in-cwd
    # path just because the file is missing.
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        full = str(repo / "src" / "missing.py")
        resolved, note = resolve_path(full, str(repo))
        assert resolved == full
        assert note is None


def test_relative_path_resolved_to_cwd_no_note():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        resolved, note = resolve_path("src/pkg/llm.py", str(repo))
        assert resolved == str((repo / "src" / "pkg" / "llm.py").resolve())
        assert note is None


def test_mangled_repo_name_prefix_suffix_matches_with_note():
    # cwd=/.../myrepo, model emits /myrepo/src/pkg/llm.py (repo name as root).
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        resolved, note = resolve_path("/myrepo/src/pkg/llm.py", str(repo))
        assert resolved == str((repo / "src" / "pkg" / "llm.py").resolve())
        assert note is not None
        assert "<system-reminder>" in note


def test_arbitrary_mangled_prefix_suffix_matches():
    # Any wrong absolute prefix recovers to the longest existing suffix.
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        resolved, note = resolve_path("/totally/wrong/src/pkg/llm.py", str(repo))
        assert resolved == str((repo / "src" / "pkg" / "llm.py").resolve())
        assert note is not None


def test_longest_suffix_wins():
    # Both /foo.py and /src/pkg/foo.py exist; the longest suffix must win.
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        (repo / "llm.py").write_text("top\n", encoding="utf-8")
        resolved, _ = resolve_path("/x/src/pkg/llm.py", str(repo))
        assert resolved == str((repo / "src" / "pkg" / "llm.py").resolve())


def test_unresolvable_absolute_returns_unchanged_no_note():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        resolved, note = resolve_path("/nowhere/nope.py", str(repo))
        assert resolved == "/nowhere/nope.py"
        assert note is None


def test_relative_escape_returns_unchanged():
    # A relative path climbing out of the workspace is left for the caller to
    # reject, never silently resolved outside cwd.
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        resolved, note = resolve_path("../secret.py", str(repo))
        assert resolved == "../secret.py"
        assert note is None


def test_suffix_match_never_escapes_cwd():
    # A traversal payload in the suffix must not resolve outside the workspace.
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        outside = Path(tmp) / "secret.py"
        outside.write_text("secret\n", encoding="utf-8")
        resolved, note = resolve_path("/myrepo/../secret.py", str(repo))
        assert Path(resolved).resolve() != outside.resolve()
        assert note is None  # nothing inside cwd matched


def test_directory_listing_uses_absolute_paths():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        prompt = load_system_prompt(str(repo))
        assert str(repo / "src") in prompt
        # The bare relative name should not appear on its own line.
        assert "\nsrc\n" not in prompt


def test_read_tool_resolves_mangled_path_and_notifies():
    read = ReadTool()
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        out = asyncio.run(
            read.call(json.dumps({"path": "/myrepo/src/pkg/llm.py"}), cwd=str(repo))
        )
        assert "MODEL = 'x'" in out  # file content was actually read
        assert "<system-reminder>" in out  # model was told about the rewrite
