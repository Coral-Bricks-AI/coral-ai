"""`coralbricks connections` — list the user's configured connections."""

from __future__ import annotations

import json as _json

import click

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client


@click.command("connections")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON instead of a table.")
def connections_cmd(as_json: bool) -> None:
    """List your active data source connections."""
    cfg = cfg_mod.load()
    client = Client(cfg)
    try:
        resp = client.get("/cli/v1/connections")
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(
            f"Failed to fetch connections ({e.status}): {e.message}"
        ) from e

    if isinstance(resp, dict):
        items = resp.get("connections") or []
    else:
        items = resp or []

    if as_json:
        click.echo(_json.dumps(items, indent=2, sort_keys=True))
        return

    click.echo()
    if not items:
        click.secho("No connections yet", bold=True)
        click.echo()
        tui.hint("Add one:  coralbricks connect <source>")
        click.echo()
        return

    click.secho(f"Your connections ({len(items)})", bold=True)
    rows = []
    for c in items:
        source = click.style(c.get("sourceId", "?"), fg="cyan")
        status_val = c.get("status") or "unknown"
        status = f"{tui.status_dot(status_val)} {status_val}"
        last_sync = c.get("lastSyncAt") or click.style("never", dim=True)
        rows.append([source, status, last_sync])

    tui.table(rows, headers=["source", "status", "last sync"])
    click.echo()
    tui.hint("Sync now:     coralbricks sync <source>")
    click.echo()
