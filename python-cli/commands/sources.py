"""`coralbricks sources` — list available connectors from the registry."""

from __future__ import annotations

import json as _json

import click
from rich.text import Text

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client


@click.command("sources")
@click.option(
    "--filter",
    "-f",
    "filter_text",
    default=None,
    help="Substring filter on source id or display name.",
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON instead of a table.")
def sources_cmd(filter_text: str | None, as_json: bool) -> None:
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

    if filter_text:
        needle = filter_text.lower().strip()
        items = [
            s
            for s in items
            if needle in (s.get("sourceId", "").lower())
            or needle in (s.get("displayName", "").lower())
        ]

    if as_json:
        click.echo(_json.dumps(items, indent=2, sort_keys=True))
        return

    if not items:
        tui.blank()
        if filter_text:
            tui.console.print(Text(f"No sources match {filter_text!r}.", style="dim"))
        else:
            tui.console.print(Text("No connectors available.", style="dim"))
        tui.blank()
        return

    rows = [
        [
            Text(s.get("sourceId", "?"), style=f"bold {tui.CORAL}"),
            Text(s.get("displayName", "") or "", style="white"),
            tui.auth_pill(s.get("authType")),
        ]
        for s in items
    ]

    tui.blank()
    title = f"Available sources ({len(items)})"
    if filter_text:
        title += f" · matching {filter_text!r}"
    tui.panel(
        tui.table_renderable(rows, headers=["id", "name", "auth"]),
        title=title,
        footer="connect:  coralbricks connect <id>",
    )
    tui.blank()
