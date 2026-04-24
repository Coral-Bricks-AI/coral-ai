"""`coralbricks runs <source>` — show recent sync runs for a source."""

from __future__ import annotations

import re
from typing import Any

import click

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

    click.echo()
    click.secho(f"Runs for {source_id}", bold=True)
    click.echo()

    if not runs:
        tui.hint("No runs yet. Start one:  coralbricks sync " + source_id)
        click.echo()
        return

    rows = [
        (
            f"#{r.get('id')}",
            f"{tui.status_dot(r.get('status'))} {r.get('status') or '-'}",
            r.get("syncMode") or "-",
            str(r.get("recordsWritten") or 0),
            _fmt_bytes(r.get("bytesWritten")),
            _fmt_time(r.get("finishedAt") or r.get("startedAt") or r.get("createdAt")),
        )
        for r in runs
    ]
    tui.table(rows, headers=("run", "status", "mode", "records", "bytes", "when"))
    click.echo()


def _fmt_bytes(n: Any) -> str:
    try:
        v = int(n) if n is not None else 0
    except (TypeError, ValueError):
        return "?"
    if v < 1024:
        return f"{v} B"
    for unit in ("KB", "MB", "GB"):
        v /= 1024
        if v < 1024:
            return f"{v:.1f} {unit}"
    return f"{v:.1f} TB"


def _fmt_time(ts: Any) -> str:
    if not ts:
        return click.style("—", dim=True)
    return str(ts)[:19].replace("T", " ")
