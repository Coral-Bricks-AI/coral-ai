"""Length-prefixed JSON RPC over a Unix-domain socket.

This is the only IO surface ``coralbricks.sandbox`` opens. The gateway
mounts a per-run UDS into the sandbox at ``$CORAL_GATEWAY_SOCKET``;
every public helper in this package (``tools.search``, ``llm.chat``,
...) is a thin wrapper that builds a method name + params dict and
calls :func:`call`.

Wire format
-----------

Each request and response is a single frame::

    [4 bytes big-endian length N][N bytes UTF-8 JSON]

Request body::

    {"id": "<uuid>", "method": "tools.search", "params": {...}}

Response body, on success::

    {"id": "<uuid>", "ok": true, "result": <any json>}

Response body, on failure::

    {"id": "<uuid>", "ok": false,
     "error": {"type": "<class>", "message": "<str>"}}

The protocol is intentionally minimal:

* No JSON-RPC 2.0 envelope -- every byte added here lands in every
  hot path inside every pipeline.
* No notifications, no batches, no streaming. Pipelines are short.
* No persistent connection across calls -- each :func:`call` opens
  the socket, writes one frame, reads one frame, closes. Connection
  setup over an abstract / filesystem UDS is sub-millisecond, and a
  per-call socket means a slow tool call cannot wedge an unrelated
  later call from the same pipeline.

Why stdlib only
---------------

This module is part of the package the gateway pre-installs into
every per-pipeline venv. Adding a third-party RPC library here
would either pin all customer pipelines to that library's version
or balloon the cold-start install time. ``socket`` + ``struct`` +
``json`` is enough.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import uuid
from typing import Any, Mapping, Optional

# 4-byte big-endian length prefix. Frames larger than ~4 GiB are
# rejected on the gateway side; the practical cap is far smaller.
_LEN_STRUCT = struct.Struct(">I")
# Hard guard against a malicious / buggy gateway sending a giant
# length and starving the sandbox of memory. 64 MiB is plenty for
# any reasonable tool/LLM payload; raise this only if a real workload
# justifies it.
_MAX_FRAME_BYTES = 64 * 1024 * 1024

# Default socket timeout for a single RPC. The sandbox already has a
# wall-clock timeout enforced by the gateway (see
# ``platform/sandbox/subprocess_executor.py``); this is a per-call
# safety net so a hung gateway doesn't burn the entire wall budget
# on one tool call.
DEFAULT_RPC_TIMEOUT_S = 45.0

# Env var name the sandbox runner sets before exec'ing the pipeline.
SOCKET_ENV_VAR = "CORAL_GATEWAY_SOCKET"


class RpcError(RuntimeError):
    """Base class for all errors raised by the sandbox RPC client."""


class RpcConnectError(RpcError):
    """Could not open / write to the gateway socket at all."""


class RpcProtocolError(RpcError):
    """The gateway sent a frame we couldn't parse."""


class RpcCallError(RpcError):
    """The gateway accepted the call but returned an error envelope.

    ``remote_type`` is the class name the gateway reported (e.g.
    ``"ToolNotAllowed"``, ``"ModelNotAllowed"``); ``message`` is the
    human-readable string. Pipeline code that wants to handle a
    specific failure mode should match on ``remote_type`` rather than
    parsing the message.
    """

    def __init__(self, remote_type: str, message: str) -> None:
        super().__init__(f"{remote_type}: {message}")
        self.remote_type = remote_type
        self.message = message


def _resolve_socket_path(socket_path: Optional[str]) -> str:
    if socket_path is not None:
        return socket_path
    env_path = os.environ.get(SOCKET_ENV_VAR)
    if not env_path:
        raise RpcConnectError(
            f"no socket path passed and ${SOCKET_ENV_VAR} is not set; "
            "this code is meant to run inside a Coral sandbox"
        )
    return env_path


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` or raise.

    ``socket.recv`` is allowed to short-read at any kernel buffer
    boundary, so anything reading length-prefixed frames must loop.
    """
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RpcProtocolError(
                f"gateway closed socket after {n - remaining} of {n} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    if len(payload) > _MAX_FRAME_BYTES:
        raise RpcProtocolError(
            f"refusing to send {len(payload)} byte frame "
            f"(max {_MAX_FRAME_BYTES})"
        )
    sock.sendall(_LEN_STRUCT.pack(len(payload)) + payload)


def _recv_frame(sock: socket.socket) -> bytes:
    header = _recv_exactly(sock, _LEN_STRUCT.size)
    (n,) = _LEN_STRUCT.unpack(header)
    if n > _MAX_FRAME_BYTES:
        raise RpcProtocolError(
            f"gateway announced {n} byte frame (max {_MAX_FRAME_BYTES})"
        )
    return _recv_exactly(sock, n)


def call(
    method: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    socket_path: Optional[str] = None,
    timeout_s: float = DEFAULT_RPC_TIMEOUT_S,
) -> Any:
    """Make one synchronous RPC call to the gateway.

    Pipeline-facing wrappers (``tools.search``, ``llm.chat``, ...)
    should layer on top of this; pipelines themselves rarely call
    :func:`call` directly.

    Parameters
    ----------
    method:
        Dotted RPC method name. The gateway dispatches on this exact
        string -- there is no client-side routing.
    params:
        JSON-serializable mapping. ``None`` is treated as ``{}``.
    socket_path:
        UDS path to connect to. Tests pass an explicit path; in
        production the path comes from ``$CORAL_GATEWAY_SOCKET``.
    timeout_s:
        Per-call socket timeout. The gateway enforces its own per-run
        wall budget on top of this.

    Returns
    -------
    The ``result`` field of the gateway's response envelope, decoded
    from JSON. Type is whatever the called method returns.

    Raises
    ------
    RpcConnectError:
        Could not open the socket (no path, kernel said no, etc.).
    RpcProtocolError:
        Frame was malformed or the gateway closed the connection
        mid-frame. Indicates a gateway bug or a kernel-level kill.
    RpcCallError:
        Gateway accepted the call and returned a structured error.
        The pipeline's own try/except should catch this.
    """
    path = _resolve_socket_path(socket_path)
    request_id = uuid.uuid4().hex
    body = json.dumps(
        {"id": request_id, "method": method, "params": dict(params or {})}
    ).encode("utf-8")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        try:
            sock.connect(path)
        except OSError as exc:
            raise RpcConnectError(
                f"cannot connect to gateway socket {path!r}: {exc}"
            ) from exc

        try:
            _send_frame(sock, body)
            raw = _recv_frame(sock)
        except socket.timeout as exc:
            raise RpcConnectError(
                f"gateway RPC {method} timed out after {timeout_s}s"
            ) from exc
        except OSError as exc:
            raise RpcConnectError(
                f"gateway socket I/O failed for {method}: {exc}"
            ) from exc
    finally:
        try:
            sock.close()
        except OSError:
            pass

    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RpcProtocolError(
            f"gateway returned non-JSON response for {method}: {exc}"
        ) from exc

    if not isinstance(envelope, dict):
        raise RpcProtocolError(
            f"gateway returned non-object response for {method}: {type(envelope).__name__}"
        )

    # We don't fail on a missing/mismatched request id -- the protocol
    # is one-shot per connection, so id correlation is purely
    # diagnostic. The gateway logs use it; we don't need to.
    if envelope.get("ok") is True:
        return envelope.get("result")

    err = envelope.get("error") or {}
    raise RpcCallError(
        remote_type=str(err.get("type") or "RpcCallError"),
        message=str(err.get("message") or "unknown gateway error"),
    )


__all__ = [
    "DEFAULT_RPC_TIMEOUT_S",
    "RpcCallError",
    "RpcConnectError",
    "RpcError",
    "RpcProtocolError",
    "SOCKET_ENV_VAR",
    "call",
]
