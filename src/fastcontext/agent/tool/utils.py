import os
import platform
import shutil
from pathlib import Path


def _find_existing_rg() -> str | None:
    rg_name = "rg.exe" if platform.system() == "Windows" else "rg"
    rg = shutil.which(rg_name)
    if rg and os.path.exists(rg):
        return rg
    return None


RG_PATH = _find_existing_rg()


def resolve_path(file_path: str, cwd: str) -> tuple[str, str | None]:
    """Best-effort map a model-supplied path onto a real path inside ``cwd``.

    Returns ``(resolved_path, note)``. ``note`` is ``None`` when the path was
    used verbatim or resolved unambiguously (a relative path joined to the
    workspace root). When the lossy suffix-matching heuristic is used to
    recover from a mangled absolute prefix (e.g. the model uses the repo name
    as the filesystem root: ``/myrepo/src/foo.py``), ``note`` is a
    ``<system-reminder>`` describing the rewrite so the model can correct the
    path form on subsequent calls.

    The caller is still responsible for the ``is_relative_to(cwd)`` permission
    check; this helper only proposes a candidate and never widens access
    (suffix candidates that escape ``cwd`` are rejected here).
    """
    cwd_path = Path(cwd).resolve()

    # 1. Absolute path already inside the workspace: use it verbatim. Existence
    #    is left to the caller so its "does not exist" error is preserved.
    try:
        p = Path(file_path)
        if p.is_absolute() and p.resolve().is_relative_to(cwd_path):
            return file_path, None
    except (OSError, ValueError):
        pass

    # 2. Relative path: resolve against the workspace root. Unambiguous, so no
    #    note. A relative path escaping the workspace (e.g. ``../x``) is left
    #    unchanged for the caller to reject.
    if not Path(file_path).is_absolute():
        candidate = (cwd_path / file_path).resolve()
        if candidate.is_relative_to(cwd_path):
            return str(candidate), None
        return file_path, None

    # 3. Absolute path outside the workspace: the model likely mangled the
    #    prefix. Recover by matching the longest trailing segment that exists
    #    inside the workspace (drop the fewest leading components first).
    parts = Path(file_path).parts  # e.g. ("/", "myrepo", "src", "foo.py")
    for i in range(1, len(parts)):
        candidate = (cwd_path / Path(*parts[i:])).resolve()
        if candidate.is_relative_to(cwd_path) and candidate.exists():
            note = (
                f"<system-reminder>Note: '{file_path}' is not inside the workspace; "
                f"interpreted it as the closest match '{candidate}'. Use absolute paths "
                f"under '{cwd_path}' exactly as shown in the workspace directory listing "
                f"and tool output to avoid ambiguity.</system-reminder>"
            )
            return str(candidate), note

    # 4. No match: return unchanged so the caller emits its normal error.
    return file_path, None
