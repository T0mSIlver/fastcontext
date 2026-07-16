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


def resolve_path(file_path: str, cwd: str) -> str:
    """Best-effort map a model-supplied path onto a real path inside ``cwd``.

    The model routinely emits an absolute path whose root is the repository name
    rather than the filesystem root (``/myrepo/src/foo.py`` for
    ``<cwd>/src/foo.py``). The intent is unambiguous, so the rewrite happens
    silently: the tool output carries no note about it. Telling the model it had
    "the wrong form" spent tokens on every such call and did not change the
    behaviour -- it is how the model refers to workspace files.

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
            return file_path
    except (OSError, ValueError):
        pass

    # 2. Relative path: resolve against the workspace root. A relative path
    #    escaping the workspace (e.g. ``../x``) is left unchanged for the caller
    #    to reject.
    if not Path(file_path).is_absolute():
        candidate = (cwd_path / file_path).resolve()
        if candidate.is_relative_to(cwd_path):
            return str(candidate)
        return file_path

    # 3. Absolute path outside the workspace: the model likely mangled the
    #    prefix. Recover by matching the longest trailing segment that exists
    #    inside the workspace (drop the fewest leading components first).
    parts = Path(file_path).parts  # e.g. ("/", "myrepo", "src", "foo.py")
    for i in range(1, len(parts)):
        candidate = (cwd_path / Path(*parts[i:])).resolve()
        if candidate.is_relative_to(cwd_path) and candidate.exists():
            return str(candidate)

    # 4. A single-component absolute path ("/myrepo") is the model using the repo
    #    name as the filesystem root: it denotes the workspace itself. Deeper paths
    #    are NOT collapsed to the workspace — an unmatched suffix must stay an error
    #    rather than silently widening a Read/Grep to the whole repository.
    if len(parts) == 2:
        return str(cwd_path)

    # 5. No match: return unchanged so the caller emits its normal error.
    return file_path
