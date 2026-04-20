"""Tests for the runtime's length-prefixed JSON RPC client."""

from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
from typing import Any

import pytest

from coralbricks.sandbox import (
    RpcCallError,
    RpcConnectError,
    RpcProtocolError,
    SOCKET_ENV_VAR,
)
from coralbricks.sandbox import _rpc

from conftest import MockGateway


def test_round_trip_returns_handler_result(mock_gateway: MockGateway) -> None:
    mock_gateway.set_handler(lambda method, params: {"echoed": params, "method": method})

    out = _rpc.call("tools.ping", {"hello": "world"}, socket_path=mock_gateway.socket_path)

    assert out == {"echoed": {"hello": "world"}, "method": "tools.ping"}
    assert mock_gateway.received == [("tools.ping", {"hello": "world"})]


def test_call_with_no_params_sends_empty_dict(mock_gateway: MockGateway) -> None:
    mock_gateway.set_handler(lambda method, params: list(params.keys()))

    out = _rpc.call("tools.ping", socket_path=mock_gateway.socket_path)

    assert out == []
    assert mock_gateway.received[0][1] == {}


def test_handler_exception_surfaces_as_rpc_call_error(mock_gateway: MockGateway) -> None:
    def boom(method: str, params: dict[str, Any]) -> Any:
        raise PermissionError("nope")
    mock_gateway.set_handler(boom)

    with pytest.raises(RpcCallError) as exc_info:
        _rpc.call("tools.ping", socket_path=mock_gateway.socket_path)

    assert exc_info.value.remote_type == "PermissionError"
    assert "nope" in exc_info.value.message
    # repr should be informative for stack traces.
    assert "PermissionError" in str(exc_info.value)


def test_unknown_socket_path_raises_connect_error(short_tmp_path) -> None:
    # AF_UNIX paths max ~104 chars on macOS; pytest's tmp_path under
    # /var/folders/... blows past that. Use a short /tmp path instead.
    nonexistent = os.path.join(short_tmp_path, "no-such.sock")

    with pytest.raises(RpcConnectError):
        _rpc.call("tools.ping", socket_path=nonexistent)


def test_socket_env_var_is_picked_up(monkeypatch, mock_gateway: MockGateway) -> None:
    mock_gateway.set_handler(lambda *_: "ok")
    monkeypatch.setenv(SOCKET_ENV_VAR, mock_gateway.socket_path)

    assert _rpc.call("tools.ping") == "ok"


def test_missing_env_var_and_no_socket_path_raises(monkeypatch) -> None:
    monkeypatch.delenv(SOCKET_ENV_VAR, raising=False)

    with pytest.raises(RpcConnectError, match="not set"):
        _rpc.call("tools.ping")


def test_oversized_outgoing_frame_is_refused(mock_gateway: MockGateway, monkeypatch) -> None:
    # Shrink the cap so we don't actually allocate 64 MiB in a test.
    monkeypatch.setattr(_rpc, "_MAX_FRAME_BYTES", 1024)

    big = "x" * 4096
    with pytest.raises(RpcProtocolError, match="refusing to send"):
        _rpc.call("tools.ping", {"big": big}, socket_path=mock_gateway.socket_path)


def test_malformed_response_raises_protocol_error(short_tmp_path) -> None:
    """A 'gateway' that sends garbage instead of a JSON envelope."""
    sock_path = os.path.join(short_tmp_path, "bad.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def serve() -> None:
        conn, _ = srv.accept()
        # Read+discard one frame, reply with a deliberately broken
        # frame: length prefix + bytes that aren't valid JSON.
        header = _recv_all(conn, 4)
        (n,) = struct.Struct(">I").unpack(header)
        _recv_all(conn, n)
        bad = b"\xff\xfe not json"
        conn.sendall(struct.Struct(">I").pack(len(bad)) + bad)
        conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        with pytest.raises(RpcProtocolError):
            _rpc.call("tools.ping", socket_path=sock_path)
    finally:
        t.join(timeout=2.0)
        srv.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass


def test_premature_close_during_response_raises_protocol_error(short_tmp_path) -> None:
    sock_path = os.path.join(short_tmp_path, "halfclose.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def serve() -> None:
        conn, _ = srv.accept()
        header = _recv_all(conn, 4)
        (n,) = struct.Struct(">I").unpack(header)
        _recv_all(conn, n)
        # Promise 64 bytes, send only 4 then close.
        conn.sendall(struct.Struct(">I").pack(64) + b"oops")
        conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        with pytest.raises(RpcProtocolError, match="closed socket"):
            _rpc.call("tools.ping", socket_path=sock_path)
    finally:
        t.join(timeout=2.0)
        srv.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass


def test_response_envelope_with_unknown_ok_value_raises_call_error(
    short_tmp_path,
) -> None:
    sock_path = os.path.join(short_tmp_path, "weird.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def serve() -> None:
        conn, _ = srv.accept()
        header = _recv_all(conn, 4)
        (n,) = struct.Struct(">I").unpack(header)
        _recv_all(conn, n)
        body = json.dumps({"id": "x", "ok": False,
                           "error": {"type": "Whatever", "message": "boom"}}).encode()
        conn.sendall(struct.Struct(">I").pack(len(body)) + body)
        conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        with pytest.raises(RpcCallError) as exc:
            _rpc.call("tools.ping", socket_path=sock_path)
    finally:
        t.join(timeout=2.0)
        srv.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass

    assert exc.value.remote_type == "Whatever"
    assert exc.value.message == "boom"


def _recv_all(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("short")
        buf.extend(chunk)
    return bytes(buf)
