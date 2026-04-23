"""`coralbricks sources` — list available connectors from the registry."""

from __future__ import annotations

import json as _json

import click

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client


@click.command("sources")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON instead of a table.")
def sources_cmd(as_json: bool) -> None:
    """List available data sources (Notion, Slack, Stripe, …)."""
    cfg = cfg_mod.load()
    client = Client(cfg)
    try:
        resp = client.get("/cli/v1/sources")
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(f"Failed to fetch sources ({e.status}): {e.message}") from e

    if isinstance(resp, dict):
        items = resp.get("connectors") or resp.get("sources") or []
    else:
        items = resp or []

    if as_json:
        click.echo(_json.dumps(items, indent=2, sort_keys=True))
        return

    if not items:
        click.echo(click.style("No connectors available.", dim=True))
        return

    click.echo()
    click.secho(f"Available sources ({len(items)})", bold=True)
    rows = [
        [
            click.style(s.get("sourceId", "?"), fg="white", bold=True),
            s.get("displayName", "") or "",
            tui.auth_pill(s.get("authType")),
        ]
        for s in items
    ]
    tui.table(rows, headers=["id", "name", "auth"])
    click.echo()
    tui.hint("Connect one:  coralbricks connect <id>")
    click.echo()
