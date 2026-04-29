"""Tests for the on-disk config (`~/.coralbricks/config.json`)."""

from __future__ import annotations

import os
import stat

import pytest

from coralbricks.cli import config as cfg_mod


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Redirect platformdirs to a per-test tmp dir so real user state is untouched."""
    monkeypatch.setattr(cfg_mod, "config_dir", lambda: tmp_path / "coralbricks")
    monkeypatch.setattr(cfg_mod, "config_path", lambda: tmp_path / "coralbricks" / "config.json")
    monkeypatch.delenv(cfg_mod.ENV_API_KEY, raising=False)
    monkeypatch.delenv(cfg_mod.ENV_SERVER_URL, raising=False)


def test_load_returns_defaults_when_absent():
    cfg = cfg_mod.load()
    assert cfg.api_key is None
    assert cfg.server_url == cfg_mod.DEFAULT_SERVER_URL
    assert cfg.user_id is None


def test_save_round_trip_and_permissions():
    cfg = cfg_mod.Config(
        api_key="ak_abc",
        server_url="http://localhost:3000",
        user_id=42,
        email="arpit@coralbricks.ai",
    )
    cfg_mod.save(cfg)

    loaded = cfg_mod.load()
    assert loaded.api_key == "ak_abc"
    assert loaded.user_id == 42
    assert loaded.email == "arpit@coralbricks.ai"
    assert loaded.server_url == "http://localhost:3000"

    mode = stat.S_IMODE(os.stat(cfg_mod.config_path()).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_clear_removes_file():
    cfg = cfg_mod.Config(api_key="ak_abc")
    cfg_mod.save(cfg)
    assert cfg_mod.config_path().exists()
    assert cfg_mod.clear() is True
    assert not cfg_mod.config_path().exists()
    # Clearing again is a no-op.
    assert cfg_mod.clear() is False


def test_env_overrides_take_precedence(monkeypatch):
    cfg = cfg_mod.Config(api_key="ak_stored", server_url="http://from-disk")
    monkeypatch.setenv(cfg_mod.ENV_API_KEY, "ak_env")
    monkeypatch.setenv(cfg_mod.ENV_SERVER_URL, "http://from-env")
    assert cfg.effective_api_key() == "ak_env"
    assert cfg.effective_server_url() == "http://from-env"
