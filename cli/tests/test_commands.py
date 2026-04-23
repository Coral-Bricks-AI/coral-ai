"""End-to-end-ish tests for the CLI commands using a mocked backend."""

from __future__ import annotations

import pytest
import responses
from click.testing import CliRunner

from coralbricks.cli import config as cfg_mod
from coralbricks.cli.app import cli


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg_mod, "config_dir", lambda: tmp_path / "coralbricks")
    monkeypatch.setattr(cfg_mod, "config_path", lambda: tmp_path / "coralbricks" / "config.json")
    monkeypatch.delenv(cfg_mod.ENV_API_KEY, raising=False)
    monkeypatch.setenv(cfg_mod.ENV_SERVER_URL, "http://backend.test")


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as r:
        yield r


def _login(runner: CliRunner) -> None:
    cfg = cfg_mod.Config(
        api_key="ak_test",
        server_url="http://backend.test",
        user_id=1,
        email="test@coralbricks.ai",
    )
    cfg_mod.save(cfg)


def test_whoami_unauthenticated():
    runner = CliRunner()
    result = runner.invoke(cli, ["whoami"])
    assert result.exit_code != 0
    assert "Not logged in" in result.output


def test_login_validates_and_persists(mocked_responses):
    mocked_responses.post(
        "http://backend.test/cli/v1/auth/validate",
        json={
            "success": True,
            "data": {"userId": 42, "email": "arpit@coralbricks.ai", "plan": "free"},
        },
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["login", "--api-key", "ak_test"])
    assert result.exit_code == 0, result.output
    assert "arpit@coralbricks.ai" in result.output
    assert "Welcome" in result.output

    cfg = cfg_mod.load()
    assert cfg.api_key == "ak_test"
    assert cfg.user_id == 42
    assert cfg.email == "arpit@coralbricks.ai"


def test_login_rejects_invalid_key(mocked_responses):
    mocked_responses.post(
        "http://backend.test/cli/v1/auth/validate",
        json={"success": False, "error": "Invalid API key"},
        status=401,
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["login", "--api-key", "ak_bad"])
    assert result.exit_code != 0
    assert "Invalid API key" in result.output


def test_whoami_calls_validate(mocked_responses):
    _login(CliRunner())
    mocked_responses.post(
        "http://backend.test/cli/v1/auth/validate",
        json={
            "success": True,
            "data": {"userId": 1, "email": "test@coralbricks.ai", "plan": "free"},
        },
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "test@coralbricks.ai" in result.output


def test_sources_renders_table(mocked_responses):
    _login(CliRunner())
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={
            "success": True,
            "data": {
                "connectors": [
                    {"sourceId": "notion", "displayName": "Notion", "authType": "oauth2"},
                    {"sourceId": "stripe", "displayName": "Stripe", "authType": "api_key"},
                ]
            },
        },
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["sources"])
    assert result.exit_code == 0, result.output
    assert "notion" in result.output
    assert "stripe" in result.output
    assert "oauth2" in result.output
    assert "api_key" in result.output


def test_connections_empty(mocked_responses):
    _login(CliRunner())
    mocked_responses.get(
        "http://backend.test/cli/v1/connections",
        json={"success": True, "data": {"connections": []}},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["connections"])
    assert result.exit_code == 0
    assert "No connections" in result.output


def test_connections_renders_table(mocked_responses):
    _login(CliRunner())
    mocked_responses.get(
        "http://backend.test/cli/v1/connections",
        json={
            "success": True,
            "data": {
                "connections": [
                    {
                        "id": 7,
                        "sourceId": "notion",
                        "status": "active",
                        "externalAccountLabel": "my-workspace",
                    }
                ]
            },
        },
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["connections"])
    assert result.exit_code == 0, result.output
    assert "notion" in result.output
    assert "my-workspace" in result.output
    assert "active" in result.output


def test_logout_removes_config():
    _login(CliRunner())
    assert cfg_mod.config_path().exists()
    runner = CliRunner()
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert "Logged out" in result.output
    assert not cfg_mod.config_path().exists()
