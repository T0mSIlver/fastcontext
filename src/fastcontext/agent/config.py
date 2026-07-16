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
import stat
import sys
from collections.abc import Mapping
from math import ceil
from pathlib import Path
from typing import Any

import tomllib

USER_CONFIG_DIR = "fastcontext"
CONFIG_BASENAME = "config.toml"
PROJECT_CONFIG_DIR = ".fastcontext"

# Config-file key -> (primary env var, legacy env var) it mirrors. Drives both discovery docs and
# `init`, which seeds a starter file from whatever FC_* vars are currently set.
_ENDPOINT_KEYS = (
    ("base_url", "FC_BASE_URL", "BASE_URL"),
    ("model", "FC_MODEL", "MODEL"),
    ("api_key", "FC_API_KEY", "API_KEY"),
)
_TUNING_KEYS = (
    ("max_tokens", "FC_MAX_TOKENS", None),
    ("max_context", "FC_MAX_CONTEXT", None),
    ("max_turn_output_tokens", "FC_MAX_TURN_OUTPUT_TOKENS", None),
    ("max_result_output_tokens", "FC_MAX_RESULT_OUTPUT_TOKENS", None),
    ("max_citations", "FC_MAX_CITATIONS", None),
    ("context_reserve", "FC_CONTEXT_RESERVE", None),
    ("reasoning_effort", "FC_REASONING_EFFORT", None),
    ("temperature", "FC_TEMPERATURE", None),
)

# Superseded settings: old name -> (new name, converter). The output caps moved from characters to
# tokens, so these are not plain aliases -- reusing a char number as a token budget would silently
# triple it. The value is converted at the ASCII ratio the old char cap was chosen against, which
# preserves what the setter meant (16000 chars of source ~= 5300 tokens) rather than what they typed.
#
# They are converted rather than ignored because the turn cap is what the context reserve is sized
# against: dropping it silently would move that cap without telling anyone. The warning is what makes
# the change visible.
_ASCII_CHARS_PER_TOKEN = 3.0


def _chars_to_tokens(value: Any) -> int | None:
    try:
        chars = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if chars <= 0:
        # 0 means "no cap" for these settings, under both the old name and the new one. Scaling it
        # would turn the documented disable value into a 1-token cap -- every tool result replaced by
        # its own truncation notice, so the agent sees nothing at all. A sentinel is not a quantity.
        return chars
    # Round UP: rounding a small cap down toward 0 would land on the disable sentinel and mean the
    # opposite of what was asked.
    return max(1, ceil(chars / _ASCII_CHARS_PER_TOKEN))


_RENAMED_KEYS: dict[str, tuple[str, Any]] = {
    "max_tool_output_chars": ("max_turn_output_tokens", _chars_to_tokens),
    "max_turn_output_chars": ("max_turn_output_tokens", _chars_to_tokens),
    "max_tool_result_chars": ("max_result_output_tokens", _chars_to_tokens),
    "max_result_output_chars": ("max_result_output_tokens", _chars_to_tokens),
}
_RENAMED_ENV: dict[str, tuple[str, Any]] = {
    "FC_MAX_TOOL_OUTPUT_CHARS": ("FC_MAX_TURN_OUTPUT_TOKENS", _chars_to_tokens),
    "FC_MAX_TURN_OUTPUT_CHARS": ("FC_MAX_TURN_OUTPUT_TOKENS", _chars_to_tokens),
    "FC_MAX_TOOL_RESULT_CHARS": ("FC_MAX_RESULT_OUTPUT_TOKENS", _chars_to_tokens),
    "FC_MAX_RESULT_OUTPUT_CHARS": ("FC_MAX_RESULT_OUTPUT_TOKENS", _chars_to_tokens),
}
# A new name may supersede several old ones; the lookup only needs one direction each.
_RENAMED_KEY_BY_NEW: dict[str, list[str]] = {}
for _old, (_new, _conv) in _RENAMED_KEYS.items():
    _RENAMED_KEY_BY_NEW.setdefault(_new, []).append(_old)
_RENAMED_ENV_BY_NEW: dict[str, list[str]] = {}
for _old, (_new, _conv) in _RENAMED_ENV.items():
    _RENAMED_ENV_BY_NEW.setdefault(_new, []).append(_old)

# Warn once per name per process; a per-run diagnostic repeated on every lookup is just noise.
_WARNED: set[str] = set()


def _warn_renamed(old: str, new: str, converted: Any = None) -> None:
    if old in _WARNED:
        return
    _WARNED.add(old)
    detail = ""
    if converted is not None:
        detail = (
            f" The cap is now measured in TOKENS, not characters, so the value was converted to "
            f"{converted}; set '{new}' explicitly to choose your own."
        )
    print(f"warning: '{old}' is deprecated and will be removed; use '{new}'.{detail}", file=sys.stderr)


def warn_renamed_flag(old_flag: str, new_flag: str) -> None:
    """Announce a deprecated CLI spelling. argparse resolves an option alias silently, so without
    this the flag form would be the one old name that changes a cap without saying so."""
    _warn_renamed(old_flag, new_flag)


def adopt_renamed_overrides(overrides: dict[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Move a superseded setting passed under its old name onto its new key, converting the unit.

    A programmatic caller passing ``max_tool_output_chars=`` would otherwise have it swallowed by
    ``**kwargs`` and silently ignored -- landing on the default cap instead of the one they asked
    for. That is the same "moves the cap without telling anyone" failure the env/file shim exists to
    prevent, so it is closed the same way. An explicit new-name value always wins.
    """
    for old, (new, converter) in _RENAMED_KEYS.items():
        value = kwargs.get(old)
        if value is None or overrides.get(new) is not None:
            continue
        converted = converter(value) if converter else value
        if converted is None:
            continue
        _warn_renamed(old, new, converted if converter else None)
        overrides[new] = converted
    return overrides


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
        """The winning value for `key` before type coercion, or ``None``.

        A renamed setting's old name is consulted only after every tier of the new name has come up
        empty, so adopting the new name anywhere overrides a stale old one rather than fighting it.
        """
        if key in self._overrides:
            return self._overrides[key]
        value = os.getenv(env)
        if value is None and legacy_env:
            value = os.getenv(legacy_env)
        if value is not None:
            return value
        if key in self._file:
            return self._file[key]

        for old_env in _RENAMED_ENV_BY_NEW.get(env, ()):
            value = os.getenv(old_env)
            if value is None:
                continue
            converter = _RENAMED_ENV[old_env][1]
            converted = converter(value) if converter else value
            if converted is None:
                continue
            _warn_renamed(old_env, env, converted if converter else None)
            return converted
        for old_key in _RENAMED_KEY_BY_NEW.get(key, ()):
            if old_key not in self._file:
                continue
            converter = _RENAMED_KEYS[old_key][1]
            converted = converter(self._file[old_key]) if converter else self._file[old_key]
            if converted is None:
                continue
            _warn_renamed(old_key, key, converted if converter else None)
            return converted
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


def _toml_quote(value: str) -> str:
    """Quote a string as a TOML basic string (stdlib tomllib is read-only)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_starter_config(env: Mapping[str, str] | None = None) -> str:
    """Render a commented starter ``config.toml``.

    Any relevant ``FC_*`` (or legacy) variable already present in ``env`` is
    baked in as an active setting, so ``fastcontext init`` can freeze a working
    shell environment into a file; everything else is left as a commented hint.
    """
    env = os.environ if env is None else env

    def current(primary: str, legacy: str | None) -> str | None:
        value = env.get(primary)
        if not value and legacy:
            value = env.get(legacy)
        return value.strip() if value and value.strip() else None

    lines = [
        "# FastContext configuration -- keys mirror the FC_* env vars without the prefix.",
        "# Precedence: CLI flag > FC_* env var > project config > this file > built-in default.",
        "",
    ]
    for key, primary, legacy in _ENDPOINT_KEYS:
        value = current(primary, legacy)
        if value is not None:
            lines.append(f"{key} = {_toml_quote(value)}")
        elif key == "api_key":
            lines.append('# api_key = "..."   # only if your endpoint requires authentication')
        else:
            placeholder = "http://127.0.0.1:11434/v1" if key == "base_url" else "your-model-name"
            lines.append(f"{key} = {_toml_quote(placeholder)}")

    lines += ["", "# Optional tuning (uncomment to override defaults):"]
    for key, primary, _ in _TUNING_KEYS:
        value = env.get(primary)
        value = value.strip() if value and value.strip() else None
        if value is not None:
            # Quote non-numeric values (e.g. reasoning_effort, or max_tokens = "auto").
            rendered = value if value.lstrip("-").isdigit() else _toml_quote(value)
            lines.append(f"{key} = {rendered}")
        else:
            lines.append(f"# {key} =")
    return "\n".join(lines) + "\n"


def write_starter_config(path: Path, force: bool = False, env: Mapping[str, str] | None = None) -> Path:
    """Write a starter config to `path`. Raises FileExistsError unless `force`.

    The file is created with owner-only permissions (0600) since it may hold an
    API key.
    """
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_starter_config(env), encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # best-effort on platforms without POSIX permissions
    return path
