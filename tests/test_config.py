"""Hermetic tests for layered config resolution (no network, no real ~/.config)."""

import pytest

import tomllib

from fastcontext.agent import config
from fastcontext.agent.config import (
    load_settings,
    project_config_path,
    render_starter_config,
    user_config_path,
    write_starter_config,
)

# Every env var the settings look at, so a stray value in the test environment
# can't leak into a case.
_ENV_VARS = [
    "FC_MODEL",
    "MODEL",
    "FC_BASE_URL",
    "BASE_URL",
    "FC_API_KEY",
    "API_KEY",
    "FC_MAX_TOKENS",
    "FC_MAX_CONTEXT",
    "FC_MAX_TURN_OUTPUT_CHARS",
    "FC_MAX_RESULT_OUTPUT_CHARS",
    "FC_MAX_TOOL_OUTPUT_CHARS",  # renamed -> FC_MAX_TURN_OUTPUT_CHARS
    "FC_MAX_CITATIONS",
    "FC_CONTEXT_RESERVE",
    "FC_TEMPERATURE",
    "FC_REASONING_EFFORT",
    "FC_CONFIG",
    "XDG_CONFIG_HOME",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Point XDG at an empty dir so a real user config never bleeds in.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # The rename warning fires once per process; clear it so each case sees its own.
    config._WARNED.clear()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- discovery ---------------------------------------------------------------


def test_user_config_path_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert user_config_path() == tmp_path / "fastcontext" / "config.toml"


def test_project_config_walks_up(tmp_path):
    _write(tmp_path / ".fastcontext" / "config.toml", 'model = "m"\n')
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert project_config_path(str(nested)) == tmp_path / ".fastcontext" / "config.toml"


def test_project_config_absent(tmp_path):
    assert project_config_path(str(tmp_path)) is None


# --- precedence --------------------------------------------------------------


def test_override_beats_everything(monkeypatch, tmp_path):
    monkeypatch.setenv("FC_MODEL", "from-env")
    _write(tmp_path / ".fastcontext" / "config.toml", 'model = "from-file"\n')
    s = load_settings(str(tmp_path), overrides={"model": "from-override"})
    assert s.require("model", "FC_MODEL", "MODEL") == "from-override"


def test_env_beats_file(monkeypatch, tmp_path):
    monkeypatch.setenv("FC_MODEL", "from-env")
    _write(tmp_path / ".fastcontext" / "config.toml", 'model = "from-file"\n')
    s = load_settings(str(tmp_path))
    assert s.require("model", "FC_MODEL", "MODEL") == "from-env"


def test_project_beats_user(monkeypatch, tmp_path):
    _write(tmp_path / "xdg" / "fastcontext" / "config.toml", 'model = "user"\nbase_url = "u"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _write(tmp_path / ".fastcontext" / "config.toml", 'model = "project"\n')
    s = load_settings(str(tmp_path))
    assert s.require("model", "FC_MODEL") == "project"  # project wins
    assert s.require("base_url", "FC_BASE_URL") == "u"  # falls through to user


def test_none_override_does_not_mask_lower_tiers(tmp_path):
    _write(tmp_path / ".fastcontext" / "config.toml", 'model = "from-file"\n')
    s = load_settings(str(tmp_path), overrides={"model": None})
    assert s.require("model", "FC_MODEL") == "from-file"


def test_default_when_nothing_set(tmp_path):
    s = load_settings(str(tmp_path))
    assert s.str_("api_key", "FC_API_KEY", default=None) is None
    assert s.int_("max_context", "FC_MAX_CONTEXT", 0) == 0
    assert s.float_("temperature", "FC_TEMPERATURE", 0.7) == 0.7


# --- legacy aliases & explicit path ------------------------------------------


def test_legacy_env_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL", "legacy")
    s = load_settings(str(tmp_path))
    assert s.require("model", "FC_MODEL", "MODEL") == "legacy"


def test_explicit_config_path_replaces_discovery(tmp_path):
    _write(tmp_path / ".fastcontext" / "config.toml", 'model = "project"\n')
    explicit = _write(tmp_path / "custom.toml", 'model = "explicit"\n')
    s = load_settings(str(tmp_path), config_path=str(explicit))
    assert s.require("model", "FC_MODEL") == "explicit"


def test_fc_config_env_selects_file(monkeypatch, tmp_path):
    explicit = _write(tmp_path / "custom.toml", 'model = "via-env"\n')
    monkeypatch.setenv("FC_CONFIG", str(explicit))
    s = load_settings(str(tmp_path))
    assert s.require("model", "FC_MODEL") == "via-env"


# --- coercion & robustness ---------------------------------------------------


def test_int_and_float_coercion_from_file(tmp_path):
    _write(tmp_path / ".fastcontext" / "config.toml", "max_context = 70000\ntemperature = 0.2\n")
    s = load_settings(str(tmp_path))
    assert s.int_("max_context", "FC_MAX_CONTEXT", 0) == 70000
    assert s.float_("temperature", "FC_TEMPERATURE", 0.7) == 0.2


def test_bad_numeric_env_falls_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("FC_MAX_CONTEXT", "not-an-int")
    s = load_settings(str(tmp_path))
    assert s.int_("max_context", "FC_MAX_CONTEXT", 123) == 123


def test_malformed_toml_is_ignored(tmp_path, capsys):
    _write(tmp_path / ".fastcontext" / "config.toml", "this is = = not toml")
    s = load_settings(str(tmp_path))
    assert s.str_("model", "FC_MODEL", default="fallback") == "fallback"
    assert "ignoring FastContext config" in capsys.readouterr().err


def test_require_raises_with_helpful_message(tmp_path):
    s = load_settings(str(tmp_path))
    with pytest.raises(RuntimeError, match="Missing required setting 'model'"):
        s.require("model", "FC_MODEL", "MODEL")


def test_zero_override_is_respected(tmp_path):
    # Explicit 0 (disable budget) must survive the None-filtering.
    _write(tmp_path / ".fastcontext" / "config.toml", "max_context = 9999\n")
    s = load_settings(str(tmp_path), overrides={"max_context": 0})
    assert s.int_("max_context", "FC_MAX_CONTEXT", 0) == 0


# --- init / starter config ---------------------------------------------------


def test_starter_config_empty_env_is_valid_toml():
    text = render_starter_config(env={})
    data = tomllib.loads(text)  # must parse
    assert data["base_url"] == "http://127.0.0.1:11434/v1"
    assert data["model"] == "your-model-name"
    assert "api_key" not in data  # left commented when unset
    # tuning keys are commented out by default
    assert "max_context" not in data


def test_starter_config_bakes_in_current_env():
    env = {
        "FC_BASE_URL": "http://host:8080/v1",
        "FC_MODEL": "fastcontext",
        "FC_API_KEY": "secret",
        "FC_MAX_TOKENS": "auto",
        "FC_MAX_CONTEXT": "70000",
        "FC_REASONING_EFFORT": "none",
    }
    data = tomllib.loads(render_starter_config(env=env))
    assert data["base_url"] == "http://host:8080/v1"
    assert data["model"] == "fastcontext"
    assert data["api_key"] == "secret"
    assert data["max_tokens"] == "auto"  # non-numeric -> quoted string
    assert data["max_context"] == 70000  # numeric -> bare int
    assert data["reasoning_effort"] == "none"


def test_starter_config_uses_legacy_env_alias():
    data = tomllib.loads(render_starter_config(env={"MODEL": "legacy-model", "BASE_URL": "http://x/v1"}))
    assert data["model"] == "legacy-model"
    assert data["base_url"] == "http://x/v1"


def test_write_starter_config_creates_file(tmp_path):
    target = tmp_path / "sub" / "config.toml"
    written = write_starter_config(target, env={"FC_MODEL": "m", "FC_BASE_URL": "http://x/v1"})
    assert written == target
    assert tomllib.loads(target.read_text())["model"] == "m"


def test_write_starter_config_refuses_existing(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("model = \"keep\"\n")
    with pytest.raises(FileExistsError):
        write_starter_config(target, env={})
    assert "keep" in target.read_text()  # untouched


def test_write_starter_config_force_overwrites(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("model = \"old\"\n")
    write_starter_config(target, force=True, env={"FC_MODEL": "new", "FC_BASE_URL": "http://x/v1"})
    assert tomllib.loads(target.read_text())["model"] == "new"


def test_write_starter_config_is_owner_only(tmp_path):
    target = tmp_path / "config.toml"
    write_starter_config(target, env={})
    assert (target.stat().st_mode & 0o777) == 0o600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- renamed settings --------------------------------------------------------
#
# `max_tool_output_chars` said "tool" but bounded a whole TURN. It is honored under its old name
# rather than ignored: it is the cap the context reserve is sized against, so silently dropping it
# would move that cap without telling anyone.


def _turn_chars(settings):
    return settings.int_("max_turn_output_chars", "FC_MAX_TURN_OUTPUT_CHARS", 16000)


def test_renamed_config_key_is_honored_and_warns(tmp_path, capsys):
    path = _write(tmp_path / "c.toml", "max_tool_output_chars = 5000\n")
    settings = load_settings(str(tmp_path), config_path=str(path))
    assert _turn_chars(settings) == 5000
    err = capsys.readouterr().err
    assert "max_tool_output_chars" in err and "max_turn_output_chars" in err


def test_renamed_env_var_is_honored_and_warns(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FC_MAX_TOOL_OUTPUT_CHARS", "7000")
    settings = load_settings(str(tmp_path), config_path="/nonexistent")
    assert _turn_chars(settings) == 7000
    err = capsys.readouterr().err
    assert "FC_MAX_TOOL_OUTPUT_CHARS" in err and "FC_MAX_TURN_OUTPUT_CHARS" in err


def test_new_name_wins_over_the_old_one(monkeypatch, tmp_path):
    monkeypatch.setenv("FC_MAX_TOOL_OUTPUT_CHARS", "7000")
    monkeypatch.setenv("FC_MAX_TURN_OUTPUT_CHARS", "9000")
    settings = load_settings(str(tmp_path), config_path="/nonexistent")
    assert _turn_chars(settings) == 9000


def test_new_name_in_a_file_beats_the_old_env_var(monkeypatch, tmp_path):
    """Adopting the new name anywhere overrides a stale old one rather than fighting it."""
    monkeypatch.setenv("FC_MAX_TOOL_OUTPUT_CHARS", "7000")
    path = _write(tmp_path / "c.toml", "max_turn_output_chars = 9000\n")
    settings = load_settings(str(tmp_path), config_path=str(path))
    assert _turn_chars(settings) == 9000


def test_no_warning_when_only_the_new_name_is_used(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FC_MAX_TURN_OUTPUT_CHARS", "9000")
    settings = load_settings(str(tmp_path), config_path="/nonexistent")
    assert _turn_chars(settings) == 9000
    assert "deprecated" not in capsys.readouterr().err


def test_rename_warning_is_emitted_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FC_MAX_TOOL_OUTPUT_CHARS", "7000")
    settings = load_settings(str(tmp_path), config_path="/nonexistent")
    _turn_chars(settings)
    _turn_chars(settings)
    assert capsys.readouterr().err.count("deprecated") == 1


def test_starter_config_offers_the_new_names_only():
    rendered = render_starter_config(env={})
    assert "max_turn_output_chars" in rendered
    assert "max_result_output_chars" in rendered
    assert "max_tool_output_chars" not in rendered
