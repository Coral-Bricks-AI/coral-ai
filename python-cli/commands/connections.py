"""`coralbricks connections` — list the user's configured connections."""

from __future__ import annotations

import json as _json

import click
from rich.text import Text

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

    if not items:
        tui.blank()
        tui.console.print(Text("No connections yet.", style="dim"))
        tui.hint("Add one:  coralbricks connect <source>")
        tui.blank()
        return

    rows = []
    for c in items:
        source = Text(c.get("sourceId", "?"), style=f"bold {tui.CORAL}")
        last_sync = c.get("lastSyncAt") or "never"
        last_sync_renderable = (
            Text(str(last_sync)) if last_sync != "never" else Text("never", style="dim")
        )
        rows.append([source, tui.status_label(c.get("status")), last_sync_renderable])

    tui.blank()
    tui.panel(
        tui.table_renderable(rows, headers=["source", "status", "last sync"]),
        title=f"Your connections ({len(items)})",
        footer="sync now:  coralbricks sync <source>",
    )
    tui.blank()
