"""Hermetic tests for layered config resolution (no network, no real ~/.config)."""

import pytest

from fastcontext.agent.config import load_settings, project_config_path, user_config_path

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
