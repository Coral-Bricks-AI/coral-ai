"""`coralbricks connections` — list the user's configured connections."""

from __future__ import annotations

import json as _json

import click

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client


@click.command("connections")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON instead of a table.")
@click.option("--ids", is_flag=True, help="Include numeric connection IDs in the output.")
def connections_cmd(as_json: bool, ids: bool) -> None:
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
    headers = ["source", "label", "status"]
    if ids:
        headers = ["id", *headers]

    rows = []
    for c in items:
        source = click.style(c.get("sourceId", "?"), fg="cyan")
        label = c.get("externalAccountLabel") or click.style("—", dim=True)
        status_val = c.get("status") or "unknown"
        status = f"{tui.status_dot(status_val)} {status_val}"
        row = [source, label, status]
        if ids:
            row = [click.style(str(c.get("id", "")), dim=True), *row]
        rows.append(row)

    tui.table(rows, headers=headers)
    click.echo()
    tui.hint("Sync now:     coralbricks sync <source>")
    click.echo()
