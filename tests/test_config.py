"""Config loading, role models, and environment isolation."""

from __future__ import annotations

import tomli_w

from lit.config import (
    DEFAULT_ROLE_EFFORT,
    DEFAULT_ROLE_MODELS,
    Config,
    config_path,
    load_config,
)


def test_defaults():
    cfg = Config()
    assert cfg.llm.model == "sonnet"
    assert cfg.llm.models == DEFAULT_ROLE_MODELS
    assert cfg.llm.max_parallel >= 1


def test_config_dir_honours_env_set_after_import(tmp_path, monkeypatch):
    """Regression: the path was frozen at import, so a late LIT_CONFIG_DIR was
    ignored and writes landed in the real user config directory."""
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path / "cfg"))
    assert config_path() == tmp_path / "cfg" / "config.toml"

    Config(root=str(tmp_path / "libs")).save()
    assert (tmp_path / "cfg" / "config.toml").exists()


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path))
    cfg = Config(root=str(tmp_path / "libs"), default_library="mylib")
    cfg.llm.models["reader"] = "opus"
    cfg.fetch.email = "me@example.org"
    cfg.save()

    loaded = load_config()
    assert loaded.default_library == "mylib"
    assert loaded.llm.models["reader"] == "opus"
    assert loaded.fetch.email == "me@example.org"


def test_partial_role_overrides_keep_the_other_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_bytes(
        tomli_w.dumps({"llm": {"models": {"reader": "opus"}}}).encode()
    )
    cfg = load_config()
    assert cfg.llm.models["reader"] == "opus"
    assert cfg.llm.models["scout"] == "haiku"  # not wiped by the partial table


def test_env_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LIT_ROOT", "/custom/root")
    monkeypatch.setenv("LIT_MODEL_READER", "opus")
    monkeypatch.setenv("LIT_LIBRARY", "envlib")
    cfg = load_config()
    assert cfg.root == "/custom/root"
    assert cfg.llm.models["reader"] == "opus"
    assert cfg.default_library == "envlib"


def test_missing_config_file_yields_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path / "absent"))
    assert load_config().llm.model == "sonnet"


# --------------------------------------------------------------------------
# Per-role reasoning effort
# --------------------------------------------------------------------------

def test_effort_defaults():
    cfg = Config()
    assert cfg.llm.effort == "medium"
    assert cfg.llm.efforts == DEFAULT_ROLE_EFFORT


def test_every_role_names_its_own_effort():
    """A role in one map and not the other silently inherits the fallback.

    That is precisely the un-chosen effort this setting exists to abolish, and
    it is invisible at runtime — the call still works, it just costs more. So
    the two maps are pinned to each other rather than left to be remembered
    whenever a new role is added.
    """
    assert set(DEFAULT_ROLE_EFFORT) == set(DEFAULT_ROLE_MODELS)


def test_partial_effort_overrides_keep_the_other_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_bytes(
        tomli_w.dumps({"llm": {"efforts": {"reader": "high"}}}).encode()
    )
    cfg = load_config()
    assert cfg.llm.efforts["reader"] == "high"
    assert cfg.llm.efforts["scout"] == "low"   # not wiped by the partial table


def test_effort_env_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LIT_EFFORT_READER", "xhigh")
    assert load_config().llm.efforts["reader"] == "xhigh"


def test_effort_survives_a_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path))
    cfg = Config(root=str(tmp_path / "libs"))
    cfg.llm.efforts["reader"] = "high"
    cfg.save()
    assert load_config().llm.efforts["reader"] == "high"


def test_effort_can_be_turned_off_entirely(tmp_path, monkeypatch):
    """Empty means: state nothing, and let the CLI's own setting decide."""
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_bytes(
        tomli_w.dumps({"llm": {"effort": ""}}).encode()
    )
    cfg = load_config()
    assert cfg.llm.effort_for("nonexistent") == ""
