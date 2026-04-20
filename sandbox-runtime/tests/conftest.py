"""Shared fixtures for the sandbox-runtime tests.

We deliberately do not import the gateway-side RPC server here --
the runtime package has to be unit-testable from a clean stdlib
checkout, with no platform dep. The fixtures below spin up a
threaded mock UDS server that speaks the same length-prefixed JSON
wire format as the real gateway.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import pytest

_LEN = struct.Struct(">I")
HandlerFn = Callable[[str, dict[str, Any]], Any]


@dataclass
class MockGateway:
    """A tiny threaded UDS server that mimics the gateway's RPC framing.

    Tests register a ``handler(method, params) -> result`` callable;
    raising from the handler is reported back as an error envelope so
    the runtime client's :class:`RpcCallError` path is exercised.

    The server runs one connection at a time -- the runtime client
    opens a fresh socket per call anyway, and tests stay deterministic.
    """

    socket_path: str
    _thread: Optional[threading.Thread] = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _server: Optional[socket.socket] = None
    _handler: Optional[HandlerFn] = None
    received: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def set_handler(self, handler: HandlerFn) -> None:
        self._handler = handler

    def start(self) -> None:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        srv.listen(8)
        # 100ms accept poll so .stop() doesn't block long.
        srv.settimeout(0.1)
        self._server = srv
        t = threading.Thread(target=self._serve_forever, name="mock-gateway", daemon=True)
        t.start()
        self._thread = t

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

    def _serve_forever(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                self._handle_one(conn)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle_one(self, conn: socket.socket) -> None:
        conn.settimeout(2.0)
        try:
            header = _recv_exact(conn, _LEN.size)
            (n,) = _LEN.unpack(header)
            body = _recv_exact(conn, n)
        except (ConnectionError, socket.timeout):
            # Client opened the connection then bailed (e.g. the
            # oversized-frame test refuses to send). Not our problem;
            # don't surface as an unhandled thread exception.
            return
        try:
            req = json.loads(body.decode("utf-8"))
            method = str(req["method"])
            params = dict(req.get("params") or {})
            req_id = str(req.get("id") or "")
        except Exception as exc:
            self._send(conn, {"id": "", "ok": False,
                              "error": {"type": "ProtocolError", "message": str(exc)}})
            return

        self.received.append((method, params))

        handler = self._handler
        if handler is None:
            self._send(conn, {"id": req_id, "ok": False,
                              "error": {"type": "NoHandler", "message": method}})
            return

        try:
            result = handler(method, params)
        except Exception as exc:  # noqa: BLE001 -- mirror gateway behavior
            self._send(conn, {"id": req_id, "ok": False,
                              "error": {"type": type(exc).__name__, "message": str(exc)}})
            return

        self._send(conn, {"id": req_id, "ok": True, "result": result})

    @staticmethod
    def _send(conn: socket.socket, envelope: dict[str, Any]) -> None:
        payload = json.dumps(envelope).encode("utf-8")
        conn.sendall(_LEN.pack(len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError(f"peer closed after {len(out)}/{n}")
        out.extend(chunk)
    return bytes(out)


@pytest.fixture
def short_tmp_path() -> Iterator[str]:
    """Like ``tmp_path`` but guaranteed short enough for AF_UNIX.

    macOS caps Unix-domain socket paths at ~104 bytes, and pytest's
    ``tmp_path`` lives under ``/var/folders/...`` which already eats
    most of that budget. Tests that bind sockets directly need a
    short root.
    """
    tmpdir = tempfile.mkdtemp(prefix="cb-rt-")
    try:
        yield tmpdir
    finally:
        # Best-effort cleanup; tests are responsible for unlinking
        # any sockets they created.
        try:
            for name in os.listdir(tmpdir):
                try:
                    os.unlink(os.path.join(tmpdir, name))
                except OSError:
                    pass
            os.rmdir(tmpdir)
        except OSError:
            pass


@pytest.fixture
def mock_gateway() -> Iterator[MockGateway]:
    # tempfile.mkdtemp because the socket path itself must not exist
    # before bind(); using NamedTemporaryFile would race that.
    tmpdir = tempfile.mkdtemp(prefix="cb-rt-test-")
    sock_path = os.path.join(tmpdir, "gateway.sock")
    gw = MockGateway(socket_path=sock_path)
    gw.start()
    try:
        yield gw
    finally:
        gw.stop()
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
