"""Layered configuration for FastContext.

Settings resolve with precedence (highest first):

    explicit override (CLI flag / kwarg)
      > FC_* environment variable
      > project config  (./.fastcontext/config.toml, searched upward from cwd)
      > user config     ($XDG_CONFIG_HOME/fastcontext/config.toml)
      > built-in default

The config file is TOML, read with the stdlib ``tomllib`` (no new dependency).
Keys mirror the ``FC_*`` variables without the prefix, e.g. ``FC_BASE_URL`` ->
``base_url``. Environment variables and CLI flags stay the highest-priority
overrides, so nothing that worked before breaks -- a config file just makes them
optional, which is the point: a coding agent driving the harness over ``bash``
no longer has to re-declare the endpoint on every call.

Example ``~/.config/fastcontext/config.toml``::

    base_url = "http://127.0.0.1:11434/v1"
    model    = "fastcontext"
    api_key  = "dummy"
    max_tokens = 4096
    max_context = 70000
    reasoning_effort = "none"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import tomllib

USER_CONFIG_DIR = "fastcontext"
CONFIG_BASENAME = "config.toml"
PROJECT_CONFIG_DIR = ".fastcontext"


def user_config_path() -> Path:
    """`$XDG_CONFIG_HOME/fastcontext/config.toml` (default `~/.config/...`)."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / USER_CONFIG_DIR / CONFIG_BASENAME


def project_config_path(work_dir: str) -> Path | None:
    """First `.fastcontext/config.toml` found walking up from `work_dir`."""
    start = Path(work_dir).resolve()
    for directory in (start, *start.parents):
        candidate = directory / PROJECT_CONFIG_DIR / CONFIG_BASENAME
        if candidate.is_file():
            return candidate
    return None


def _load_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        # A broken config file must not crash a run; warn on stderr and ignore it.
        print(f"warning: ignoring FastContext config {path}: {exc}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


class Settings:
    """Resolved settings with layered precedence (see module docstring)."""

    def __init__(self, file_values: dict[str, Any], overrides: dict[str, Any] | None = None) -> None:
        self._file = file_values
        # Drop passed-through ``None``s so an unset CLI flag doesn't mask lower tiers.
        self._overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    def raw(self, key: str, env: str, legacy_env: str | None = None) -> Any:
        """The winning value for `key` before type coercion, or ``None``."""
        if key in self._overrides:
            return self._overrides[key]
        value = os.getenv(env)
        if value is None and legacy_env:
            value = os.getenv(legacy_env)
        if value is not None:
            return value
        if key in self._file:
            return self._file[key]
        return None

    def str_(self, key: str, env: str, legacy_env: str | None = None, default: str | None = None) -> str | None:
        value = self.raw(key, env, legacy_env)
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    def require(self, key: str, env: str, legacy_env: str | None = None) -> str:
        value = self.str_(key, env, legacy_env)
        if not value:
            legacy_hint = f" / {legacy_env}" if legacy_env else ""
            raise RuntimeError(
                f"Missing required setting '{key}'. Set {env}{legacy_hint}, pass it explicitly, or add "
                f'`{key} = "..."` to {user_config_path()} or ./{PROJECT_CONFIG_DIR}/{CONFIG_BASENAME}.'
            )
        return value

    def int_(self, key: str, env: str, default: int) -> int:
        value = self.raw(key, env)
        if value is None:
            return default
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def float_(self, key: str, env: str, default: float) -> float:
        value = self.raw(key, env)
        if value is None:
            return default
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default


def load_settings(
    work_dir: str,
    overrides: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> Settings:
    """Build a :class:`Settings` from the config files plus explicit overrides.

    An explicit ``config_path`` (or the ``FC_CONFIG`` env var) replaces file
    discovery entirely. Otherwise the user config is loaded first and the project
    config layered on top (project keys win).
    """
    explicit = config_path or os.getenv("FC_CONFIG")
    if explicit:
        file_values = _load_toml(Path(explicit).expanduser())
    else:
        user = _load_toml(user_config_path())
        project = _load_toml(project_config_path(work_dir))
        file_values = {**user, **project}
    return Settings(file_values, overrides)
