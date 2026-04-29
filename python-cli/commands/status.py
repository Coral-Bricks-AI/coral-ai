"""`coralbricks status` — one-screen overview.

Combines whoami + connections + recent runs in a single render so a
buyer demoing the CLI sees the full state of their account in one
command. Falls back gracefully when each backend call fails — partial
state is more useful than a hard crash.
"""

from __future__ import annotations

from typing import Any

import click
from rich.console import Group
from rich.text import Text

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client
from .runs import _fmt_bytes, _fmt_when


@click.command("status")
def status_cmd() -> None:
    """Show identity, connections, and recent runs in one screen."""
    cfg = cfg_mod.load()
    if not cfg.effective_api_key():
        raise click.ClickException("Not logged in. Run `coralbricks login` first.")

    client = Client(cfg)

    identity = _fetch_identity(client, cfg)
    connections = _fetch_connections(client)
    recent_runs = _fetch_recent_runs(client, connections)

    tui.banner()
    _render_identity_panel(identity, cfg)
    _render_connections_panel(connections)
    _render_recent_runs_panel(recent_runs)
    tui.blank()


# ---------- fetch ----------


def _fetch_identity(client: Client, cfg: cfg_mod.Config) -> dict[str, Any]:
    try:
        resp = client.post("/cli/v1/auth/validate", json={})
    except (ApiError, AuthError):
        # Best-effort — fall back to cached config if the validate call
        # is unreachable (offline demo, transient backend hiccup).
        return {"email": cfg.email, "plan": "free", "isVerified": False, "_offline": True}
    if isinstance(resp, dict):
        return resp
    return {"email": cfg.email, "plan": "free", "isVerified": False}


def _fetch_connections(client: Client) -> list[dict[str, Any]]:
    try:
        resp = client.get("/cli/v1/connections")
    except (ApiError, AuthError):
        return []
    if isinstance(resp, dict):
        return resp.get("connections") or []
    if isinstance(resp, list):
        return resp
    return []


def _fetch_recent_runs(
    client: Client, connections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per connected source — most recent run only."""
    runs: list[dict[str, Any]] = []
    for c in connections:
        source_id = c.get("sourceId")
        if not source_id:
            continue
        try:
            resp = client.get("/cli/v1/runs", params={"sourceId": source_id, "limit": 1})
        except (ApiError, AuthError):
            continue
        items: list[dict[str, Any]] = []
        if isinstance(resp, dict):
            items = resp.get("runs") or []
        elif isinstance(resp, list):
            items = resp
        if items:
            row = dict(items[0])
            row.setdefault("sourceId", source_id)
            runs.append(row)
    return runs


# ---------- render ----------


def _render_identity_panel(identity: dict[str, Any], cfg: cfg_mod.Config) -> None:
    email = identity.get("email") or cfg.email or "-"
    plan = identity.get("plan") or "free"
    verified = bool(identity.get("isVerified"))
    server = cfg.effective_server_url()
    custom_server = server.rstrip("/") != cfg_mod.DEFAULT_SERVER_URL.rstrip("/")
    offline = bool(identity.get("_offline"))

    body = tui.kv_renderable(
        [
            ("email", Text(email, style=f"bold {tui.CORAL}")),
            ("plan", Text(plan, style="magenta" if plan == "paid" else "white")),
            (
                "status",
                Text("offline (cached)", style="yellow")
                if offline
                else (Text("verified", style="green") if verified else Text("unverified", style="yellow")),
            ),
            ("server", Text(server, style="dim") if custom_server else None),
        ]
    )
    pill = tui.pill("OFFLINE", "warn") if offline else tui.pill("ACTIVE", "success")
    tui.panel(body, title="Identity", title_extra=pill)


def _render_connections_panel(connections: list[dict[str, Any]]) -> None:
    if not connections:
        body: Any = Group(
            Text("No connections yet.", style="dim"),
            Text(""),
            Text("  → coralbricks connect <source>", style=f"bold {tui.CORAL}"),
        )
        tui.panel(body, title="Connections", title_extra=tui.pill("EMPTY", "neutral"))
        return

    rows = []
    for c in connections:
        rows.append(
            [
                Text(c.get("sourceId", "?"), style=f"bold {tui.CORAL}"),
                tui.status_label(c.get("status")),
                Text(str(c.get("lastSyncAt") or "never"), style="dim" if not c.get("lastSyncAt") else "white"),
            ]
        )
    tui.panel(
        tui.table_renderable(rows, headers=["source", "status", "last sync"]),
        title=f"Connections ({len(connections)})",
        title_extra=tui.pill(f"{len(connections)} ACTIVE", "success"),
    )


def _render_recent_runs_panel(runs: list[dict[str, Any]]) -> None:
    if not runs:
        body = Group(
            Text("No syncs yet.", style="dim"),
            Text(""),
            Text("  → coralbricks sync <source>", style=f"bold {tui.CORAL}"),
        )
        tui.panel(body, title="Recent activity", title_extra=tui.pill("EMPTY", "neutral"))
        return

    rows = []
    for r in runs:
        rows.append(
            (
                Text(str(r.get("sourceId", "?")), style=f"bold {tui.CORAL}"),
                Text(f"#{r.get('id')}", style="bold"),
                tui.status_label(r.get("status")),
                Text(f"{int(r.get('recordsWritten') or 0):,}", style="white"),
                Text(_fmt_bytes(r.get("bytesWritten")), style="white"),
                Text(_fmt_when(r.get("finishedAt") or r.get("startedAt") or r.get("createdAt"))),
            )
        )
    tui.panel(
        tui.table_renderable(
            rows,
            headers=["source", "run", "status", "records", "bytes", "when"],
        ),
        title="Recent activity",
        footer="more:  coralbricks runs <source>",
    )
