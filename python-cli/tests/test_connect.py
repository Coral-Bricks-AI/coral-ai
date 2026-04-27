"""Tests for `coralbricks connect` — both api_key and OAuth loopback paths."""

from __future__ import annotations

import threading
import time
from urllib.request import urlopen

import pytest
import responses
from click.testing import CliRunner

from coralbricks.cli import config as cfg_mod
from coralbricks.cli.app import cli
from coralbricks.cli.oauth import LoopbackServer


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


def _login() -> None:
    cfg_mod.save(
        cfg_mod.Config(
            api_key="ak_test",
            server_url="http://backend.test",
            user_id=1,
            email="test@coralbricks.ai",
        )
    )


# ---------- loopback server ----------

def test_loopback_captures_connection_id():
    with LoopbackServer(timeout=5.0) as lb:
        url = lb.url + "?connection_id=42&source_id=notion"

        def hit():
            # Tiny delay so the server is ready before we poke it.
            time.sleep(0.05)
            with urlopen(url) as r:
                assert r.status == 200

        t = threading.Thread(target=hit)
        t.start()
        result = lb.wait()
        t.join(timeout=2.0)

    assert result.connection_id == 42
    assert result.source_id == "notion"
    assert result.error is None


def test_loopback_captures_error():
    with LoopbackServer(timeout=5.0) as lb:
        url = lb.url + "?error=access_denied"

        def hit():
            time.sleep(0.05)
            with urlopen(url) as r:
                assert r.status == 200

        t = threading.Thread(target=hit)
        t.start()
        result = lb.wait()
        t.join(timeout=2.0)

    assert result.connection_id is None
    assert result.error == "access_denied"


def test_loopback_times_out():
    with LoopbackServer(timeout=0.2) as lb:
        result = lb.wait()
    assert result.connection_id is None
    assert result.error is not None
    assert "timeout" in result.error.lower()


# ---------- connect (api_key) ----------

def test_connect_api_key_prompts_and_creates(mocked_responses):
    _login()
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={
            "success": True,
            "data": {
                "connectors": [
                    {
                        "sourceId": "stripe",
                        "displayName": "Stripe",
                        "authType": "api_key",
                        "apiKeyFields": [
                            {"key": "api_key", "label": "Secret key", "required": True, "secret": True}
                        ],
                    }
                ]
            },
        },
    )
    mocked_responses.post(
        "http://backend.test/cli/v1/connect/stripe",
        json={
            "success": True,
            "data": {"connection": {"id": 101, "sourceId": "stripe", "status": "active"}},
        },
        status=201,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "stripe"], input="sk_test_123\n")
    assert result.exit_code == 0, result.output
    assert "Connected" in result.output
    assert "stripe" in result.output
    # Numeric connection ids are internal — they must not leak into the CLI UI.
    assert "#101" not in result.output
    assert "#" not in result.output.replace("# ", "")  # catch stray ids

    sent = [c for c in mocked_responses.calls if c.request.url.endswith("/connect/stripe")]
    assert len(sent) == 1
    body = sent[0].request.body
    assert b"sk_test_123" in body
    assert b"credentials" in body


def test_connect_warns_on_existing_and_aborts(mocked_responses):
    _login()
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={
            "success": True,
            "data": {
                "connectors": [
                    {
                        "sourceId": "stripe",
                        "displayName": "Stripe",
                        "authType": "api_key",
                        "apiKeyFields": [
                            {"key": "api_key", "label": "Secret key", "required": True, "secret": True}
                        ],
                    }
                ]
            },
        },
    )
    mocked_responses.get(
        "http://backend.test/cli/v1/connections",
        json={
            "success": True,
            "data": {
                "connections": [{"id": 9, "sourceId": "stripe", "status": "active"}]
            },
        },
    )

    runner = CliRunner()
    # Answer "n" at the "Continue?" prompt.
    result = runner.invoke(cli, ["connect", "stripe"], input="n\n")
    assert result.exit_code != 0
    assert "already connected" in result.output.lower()
    # The api-key prompt should never have fired because we aborted first.
    assert "Secret key" not in result.output


def test_connect_reconnect_with_yes_skips_prompt(mocked_responses):
    _login()
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={
            "success": True,
            "data": {
                "connectors": [
                    {
                        "sourceId": "stripe",
                        "displayName": "Stripe",
                        "authType": "api_key",
                        "apiKeyFields": [
                            {"key": "api_key", "label": "Secret key", "required": True, "secret": True}
                        ],
                    }
                ]
            },
        },
    )
    mocked_responses.get(
        "http://backend.test/cli/v1/connections",
        json={
            "success": True,
            "data": {
                "connections": [{"id": 9, "sourceId": "stripe", "status": "active"}]
            },
        },
    )
    mocked_responses.post(
        "http://backend.test/cli/v1/connect/stripe",
        json={
            "success": True,
            "data": {
                "connection": {"id": 9, "sourceId": "stripe", "status": "active"},
                "reconnected": True,
            },
        },
        status=200,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "stripe", "--yes"], input="sk_test_rot\n")
    assert result.exit_code == 0, result.output
    assert "Reconnected" in result.output
    assert "#9" not in result.output
    # The warning should NOT have printed because --yes skips the check prompt.
    assert "Continue?" not in result.output


def test_connect_different_source_no_warning(mocked_responses):
    """Warning is sourceId-scoped — connecting stripe should not warn about notion."""
    _login()
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={
            "success": True,
            "data": {
                "connectors": [
                    {
                        "sourceId": "stripe",
                        "displayName": "Stripe",
                        "authType": "api_key",
                        "apiKeyFields": [
                            {"key": "api_key", "label": "Secret key", "required": True, "secret": True}
                        ],
                    }
                ]
            },
        },
    )
    mocked_responses.get(
        "http://backend.test/cli/v1/connections",
        json={
            "success": True,
            "data": {"connections": [{"id": 9, "sourceId": "notion", "status": "active"}]},
        },
    )
    mocked_responses.post(
        "http://backend.test/cli/v1/connect/stripe",
        json={
            "success": True,
            "data": {
                "connection": {"id": 11, "sourceId": "stripe", "status": "active"},
                "reconnected": False,
            },
        },
        status=201,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "stripe"], input="sk_test_new\n")
    assert result.exit_code == 0, result.output
    assert "already connected" not in result.output.lower()
    assert "Connected" in result.output


def test_connect_unknown_source_errors(mocked_responses):
    _login()
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={"success": True, "data": {"connectors": [{"sourceId": "stripe", "authType": "api_key"}]}},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "doesnotexist"])
    assert result.exit_code != 0
    assert "Unknown source" in result.output


# ---------- connect (oauth2) ----------

def test_connect_oauth_flow(mocked_responses, monkeypatch):
    """Simulate the OAuth happy path without actually opening a browser.

    We intercept `webbrowser.open` and, as soon as the CLI kicks it off,
    fire a request at the loopback URL with ?connection_id=…, mirroring
    what the backend would do after a real provider redirect.
    """
    _login()
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={
            "success": True,
            "data": {
                "connectors": [
                    {
                        "sourceId": "notion",
                        "displayName": "Notion",
                        "authType": "oauth2",
                    }
                ]
            },
        },
    )
    # Capture the loopback URL the CLI sends so we know where to redirect.
    captured: dict[str, str] = {}

    def record_connect_start(request):
        import json as _json

        body = _json.loads(request.body)
        captured["loopback"] = body["cliLoopback"]
        return (200, {}, _json.dumps({"success": True, "data": {"authUrl": "https://notion.example/authorize"}}))

    mocked_responses.add_callback(
        responses.POST,
        "http://backend.test/cli/v1/connect/notion",
        callback=record_connect_start,
        content_type="application/json",
    )

    def fake_open(url: str, *args, **kwargs) -> bool:
        # Tap the loopback in a thread so the CLI's blocking .wait() returns.
        def hit():
            time.sleep(0.05)
            with urlopen(captured["loopback"] + "?connection_id=77&source_id=notion") as r:
                assert r.status == 200

        threading.Thread(target=hit, daemon=True).start()
        return True

    monkeypatch.setattr("coralbricks.cli.commands.connect.webbrowser.open", fake_open)

    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "notion"])
    assert result.exit_code == 0, result.output
    assert "Connected" in result.output
    assert "#77" not in result.output
    assert "loopback" in captured
    assert captured["loopback"].startswith("http://127.0.0.1:")


def test_connect_oauth_error_from_loopback(mocked_responses, monkeypatch):
    _login()
    mocked_responses.get(
        "http://backend.test/cli/v1/sources",
        json={
            "success": True,
            "data": {
                "connectors": [
                    {"sourceId": "notion", "displayName": "Notion", "authType": "oauth2"}
                ]
            },
        },
    )
    captured: dict[str, str] = {}

    def record_start(request):
        import json as _json

        captured["loopback"] = _json.loads(request.body)["cliLoopback"]
        return (200, {}, _json.dumps({"success": True, "data": {"authUrl": "https://x.example/authorize"}}))

    mocked_responses.add_callback(
        responses.POST,
        "http://backend.test/cli/v1/connect/notion",
        callback=record_start,
        content_type="application/json",
    )

    def fake_open(url: str, *args, **kwargs) -> bool:
        def hit():
            time.sleep(0.05)
            with urlopen(captured["loopback"] + "?error=access_denied") as r:
                assert r.status == 200

        threading.Thread(target=hit, daemon=True).start()
        return True

    monkeypatch.setattr("coralbricks.cli.commands.connect.webbrowser.open", fake_open)

    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "notion"])
    assert result.exit_code != 0
    assert "access_denied" in result.output
