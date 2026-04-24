"""`coralbricks disconnect <source>` — remove a previously-configured connection.

Resolves the user's connection for the named source via
`GET /cli/v1/connections`, then `DELETE /cli/v1/connections/{id}`.
Back-end enforces ownership (connection.userId === auth'd user).
"""

from __future__ import annotations

import re
from typing import Any

import click

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


@click.command("disconnect")
@click.argument("source_id")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def disconnect_cmd(source_id: str, yes: bool) -> None:
    """Remove the connection for SOURCE_ID."""
    if not _SOURCE_ID_RE.match(source_id):
        raise click.ClickException(
            f"Invalid source id: {source_id!r}. Use lowercase letters, digits, '-' or '_'."
        )

    cfg = cfg_mod.load()
    client = Client(cfg)

    conn = _find_connection(client, source_id)
    if conn is None:
        raise click.ClickException(f"No connection found for {source_id!r}.")

    conn_id = conn.get("id")
    if not isinstance(conn_id, int):
        raise click.ClickException("Malformed connection record (missing id).")

    if not yes and not click.confirm(
        f"Remove connection for {click.style(source_id, fg='cyan', bold=True)} (conn #{conn_id})?",
        default=False,
    ):
        tui.hint("Cancelled.")
        return

    try:
        client.delete(f"/cli/v1/connections/{conn_id}")
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(f"Failed to disconnect ({e.status}): {e.message}") from e

    click.echo()
    tui.ok(f"Disconnected {click.style(source_id, fg='cyan', bold=True)} (conn #{conn_id})")
    click.echo()


def _find_connection(client: Client, source_id: str) -> dict[str, Any] | None:
    try:
        resp = client.get("/cli/v1/connections")
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(f"Failed to list connections ({e.status}): {e.message}") from e

    items: list[dict[str, Any]] = []
    if isinstance(resp, dict):
        items = resp.get("connections") or []
    elif isinstance(resp, list):
        items = resp

    for item in items:
        if isinstance(item, dict) and item.get("sourceId") == source_id:
            return item
    return None
