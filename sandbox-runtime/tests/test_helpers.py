"""Tests for the user-facing helper modules (tools, llm, cancel_event)."""

from __future__ import annotations

import os

import pytest

from coralbricks.sandbox import cancel_event, llm, tools

from conftest import MockGateway


def test_tools_ping_routes_to_tools_namespace(mock_gateway: MockGateway) -> None:
    mock_gateway.set_handler(lambda method, params: {"pong": True, "method": method})

    out = tools.ping(socket_path=mock_gateway.socket_path)

    assert out == {"pong": True, "method": "tools.ping"}
    assert mock_gateway.received[0][0] == "tools.ping"


def test_llm_ping_routes_to_llm_namespace(mock_gateway: MockGateway) -> None:
    mock_gateway.set_handler(lambda method, params: {"pong": True, "method": method})

    out = llm.ping(socket_path=mock_gateway.socket_path)

    assert out == {"pong": True, "method": "llm.ping"}
    assert mock_gateway.received[0][0] == "llm.ping"


def test_cancel_event_returns_false_when_env_var_unset(monkeypatch) -> None:
    monkeypatch.delenv(cancel_event.CANCEL_FILE_ENV_VAR, raising=False)

    assert cancel_event.is_set() is False


def test_cancel_event_returns_false_when_file_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(cancel_event.CANCEL_FILE_ENV_VAR, str(tmp_path / "nope"))

    assert cancel_event.is_set() is False


def test_cancel_event_returns_true_once_file_appears(monkeypatch, tmp_path) -> None:
    sentinel = tmp_path / "cancel"
    monkeypatch.setenv(cancel_event.CANCEL_FILE_ENV_VAR, str(sentinel))

    assert cancel_event.is_set() is False
    sentinel.write_bytes(b"")
    assert cancel_event.is_set() is True
