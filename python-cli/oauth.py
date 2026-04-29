"""Ephemeral localhost HTTP server that captures the OAuth callback.

The pattern is the same one `gh`, `stripe`, `vercel`, `supabase` use:

    1. CLI binds an OS-assigned port on 127.0.0.1
    2. CLI tells the backend its loopback URL via the connect-start request
    3. Backend redirects the browser to us after a successful OAuth
       handshake with the provider — the URL carries either
       ?connection_id=<id> on success or ?error=<msg> on failure
    4. We return a simple HTML "you can close this tab" page, capture
       the query params, and shut down

The server is single-shot — one request and it stops. A timeout wraps
the whole thing so we don't hang forever if the user abandons the flow.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class OAuthResult:
    connection_id: int | None = None
    error: str | None = None
    source_id: str | None = None
    reconnected: bool = False


_SUCCESS_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Coral Bricks</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;background:#0b0f14;color:#e6edf3;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{text-align:center;padding:2.5rem 3rem;border-radius:14px;background:#111823;
box-shadow:0 10px 40px rgba(0,0,0,.35);max-width:420px}
h1{margin:.2rem 0 .5rem;font-size:1.4rem;color:#7ee787}
p{margin:.25rem 0;color:#9da7b3}
code{background:#1c2531;padding:.1rem .35rem;border-radius:4px;color:#d2a8ff}
</style></head>
<body><div class="card">
<h1>&#10003; Connected</h1>
<p>Coral Bricks captured the credentials.</p>
<p>You can close this tab and return to <code>coralbricks</code>.</p>
</div></body></html>
"""

_ERROR_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Coral Bricks</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;background:#0b0f14;color:#e6edf3;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{text-align:center;padding:2.5rem 3rem;border-radius:14px;background:#111823;
box-shadow:0 10px 40px rgba(0,0,0,.35);max-width:480px}
h1{margin:.2rem 0 .5rem;font-size:1.4rem;color:#ff7b72}
p{margin:.25rem 0;color:#9da7b3}
</style></head>
<body><div class="card">
<h1>&#10007; Connect failed</h1>
<p>Return to <code>coralbricks</code> for the full error.</p>
</div></body></html>
"""


class LoopbackServer:
    """Starts on __enter__, captures one callback, stops on __exit__."""

    def __init__(self, *, path: str = "/callback", timeout: float = 300.0):
        self._path = path
        self._timeout = timeout
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._result = OAuthResult()

    def __enter__(self) -> "LoopbackServer":
        # Port 0 → kernel picks a free port. Bind to 127.0.0.1 only so the
        # server is unreachable from anywhere else on the network.
        self._httpd = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def port(self) -> int:
        assert self._httpd is not None
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self._path}"

    def wait(self) -> OAuthResult:
        """Block until the callback arrives or we time out."""
        if not self._done.wait(timeout=self._timeout):
            return OAuthResult(error="timeout waiting for browser callback")
        return self._result

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            # Silence the default access-log spam so the CLI output stays clean.
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != server._path:
                    self.send_response(404)
                    self.end_headers()
                    return
                params = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
                cid = params.get("connection_id")
                try:
                    connection_id = int(cid) if cid else None
                except ValueError:
                    connection_id = None
                server._result = OAuthResult(
                    connection_id=connection_id,
                    error=params.get("error"),
                    source_id=params.get("source_id"),
                    reconnected=params.get("reconnected") == "1",
                )
                body = _SUCCESS_HTML if connection_id and not params.get("error") else _ERROR_HTML
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                server._done.set()

        return Handler


def find_free_port() -> int:
    """Ask the OS for a free loopback port (used only if we need one up-front)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
