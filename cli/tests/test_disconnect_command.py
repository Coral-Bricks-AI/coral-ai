"""Tests for `coralbricks disconnect <source>`."""

from __future__ import annotations

from typing import Any

import pytest
import responses
from click.testing import CliRunner
from coralbricks.cli import config as cfg_mod
from coralbricks.cli.app import cli

BACKEND = "http://backend.test"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg_mod, "config_dir", lambda: tmp_path / "coralbricks")
    monkeypatch.setattr(cfg_mod, "config_path", lambda: tmp_path / "coralbricks" / "config.json")
    monkeypatch.delenv(cfg_mod.ENV_API_KEY, raising=False)
    monkeypatch.setenv(cfg_mod.ENV_SERVER_URL, BACKEND)
    cfg_mod.save(
        cfg_mod.Config(
            api_key="ak_test",
            server_url=BACKEND,
            user_id=1,
            email="test@coralbricks.ai",
        )
    )


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as r:
        yield r


def _connections_resp(*items: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": {"connections": list(items)}}


def test_disconnect_removes_existing_connection(mocked_responses):
    mocked_responses.add(
        responses.GET,
        f"{BACKEND}/cli/v1/connections",
        json=_connections_resp({"id": 13, "sourceId": "notion", "status": "active"}),
    )
    mocked_responses.add(
        responses.DELETE,
        f"{BACKEND}/cli/v1/connections/13",
        json={"success": True, "message": "Connection deleted"},
    )
    result = CliRunner().invoke(cli, ["disconnect", "notion", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Disconnected" in result.output
    assert "conn #13" in result.output


def test_disconnect_missing_source_errors(mocked_responses):
    mocked_responses.add(
        responses.GET,
        f"{BACKEND}/cli/v1/connections",
        json=_connections_resp({"id": 13, "sourceId": "notion"}),
    )
    result = CliRunner().invoke(cli, ["disconnect", "stripe", "--yes"])
    assert result.exit_code != 0
    assert "No connection found" in result.output


def test_disconnect_rejects_invalid_source_id(mocked_responses):
    # No HTTP calls expected — regex rejects before network.
    result = CliRunner().invoke(cli, ["disconnect", "../admin", "--yes"])
    assert result.exit_code != 0
    assert "Invalid source id" in result.output


def test_disconnect_declines_on_confirmation_no(mocked_responses):
    mocked_responses.add(
        responses.GET,
        f"{BACKEND}/cli/v1/connections",
        json=_connections_resp({"id": 13, "sourceId": "notion"}),
    )
    # No DELETE registered — if the CLI tried to hit it, responses would raise.
    result = CliRunner().invoke(cli, ["disconnect", "notion"], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output
