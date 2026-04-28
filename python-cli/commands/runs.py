"""`coralbricks runs <source>` — show recent sync runs for a source."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import click
from rich.text import Text

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


@click.command("runs")
@click.argument("source_id")
@click.option("--limit", default=25, show_default=True, help="Max runs to show.")
def runs_cmd(source_id: str, limit: int) -> None:
    """List recent sync runs for a source."""
    if not _SOURCE_ID_RE.match(source_id):
        raise click.ClickException(
            f"Invalid source id: {source_id!r}. Use lowercase letters, digits, '-' or '_'."
        )

    cfg = cfg_mod.load()
    client = Client(cfg)

    try:
        resp = client.get("/cli/v1/runs", params={"sourceId": source_id, "limit": limit})
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(f"Failed to list runs ({e.status}): {e.message}") from e

    runs: list[dict[str, Any]] = []
    if isinstance(resp, dict):
        runs = resp.get("runs") or []
    elif isinstance(resp, list):
        runs = resp

    if not runs:
        tui.blank()
        tui.console.print(Text(f"No runs yet for {source_id}.", style="dim"))
        tui.hint(f"Start one:  coralbricks sync {source_id}")
        tui.blank()
        return

    rows = [
        (
            Text(f"#{r.get('id')}", style="bold"),
            tui.status_label(r.get("status")),
            Text(r.get("syncMode") or "-", style="dim"),
            Text(f"{int(r.get('recordsWritten') or 0):,}", style="white"),
            Text(_fmt_bytes(r.get("bytesWritten")), style="white"),
            Text(_fmt_when(r.get("finishedAt") or r.get("startedAt") or r.get("createdAt"))),
        )
        for r in runs
    ]

    tui.blank()
    tui.panel(
        tui.table_renderable(rows, headers=("run", "status", "mode", "records", "bytes", "when")),
        title=f"Runs for {source_id}",
        footer=f"showing {len(runs)} most recent",
    )
    tui.blank()


def _fmt_bytes(n: Any) -> str:
    try:
        v = float(n) if n is not None else 0.0
    except (TypeError, ValueError):
        return "?"
    if v < 1024:
        return f"{int(v)} B"
    for unit in ("KB", "MB", "GB"):
        v /= 1024
        if v < 1024:
            return f"{v:.1f} {unit}"
    return f"{v:.1f} TB"


def _fmt_when(ts: Any) -> str:
    """Pretty relative time ('3m ago') with absolute fallback."""
    if not ts:
        return "—"
    raw = str(ts)[:19].replace("T", " ")
    parsed = _parse_iso(str(ts))
    if parsed is None:
        return raw
    delta = datetime.now(timezone.utc) - parsed
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return raw
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 86400 * 7:
        return f"{seconds // 86400}d ago"
    return raw


def _parse_iso(s: str) -> datetime | None:
    try:
        # Tolerate both Z-suffixed and offset-form ISO 8601 timestamps.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
