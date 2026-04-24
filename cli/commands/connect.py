"""`coralbricks connect <source>` — create a new data-source connection.

Two flows depending on the connector's authType:

- `oauth2`: spin a one-shot loopback HTTP server, POST the loopback URL to
  the backend, open the returned authorize URL in the user's browser,
  wait for the provider → backend → our loopback redirect chain to land.
- `api_key`: prompt for each required field (masked for secrets), POST
  credentials directly. The connection is created synchronously.

The backend (`/cli/v1/connect/:sourceId`) dispatches on authType so the
CLI doesn't have to know which route to hit — one endpoint, two branches.

Dedupe policy: one connection per (user, source). Re-running `connect`
for an already-connected source refreshes credentials in place — no
duplicate is ever created.
"""

from __future__ import annotations

import re
import webbrowser
from typing import Any
from urllib.parse import quote, urlparse

import click

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client
from ..oauth import LoopbackServer

# sourceIds are registry keys controlled by our backend — lowercase letters,
# digits, hyphens, underscores. We enforce it client-side so a typo like
# `../admin` is rejected locally before it ever hits the wire.
_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


@click.command("connect")
@click.argument("source_id")
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't auto-open the browser — print the authorize URL instead.",
)
@click.option(
    "--yes",
    "-y",
    "assume_yes",
    is_flag=True,
    help="Skip the re-connect confirmation prompt.",
)
def connect_cmd(source_id: str, no_browser: bool, assume_yes: bool) -> None:
    """Connect a data source (OAuth or API key)."""
    if not _SOURCE_ID_RE.match(source_id):
        raise click.ClickException(
            f"Invalid source id: {source_id!r}. Use lowercase letters, digits, '-' or '_'."
        )

    cfg = cfg_mod.load()
    client = Client(cfg)

    connector = _find_connector(client, source_id)
    auth_type = connector.get("authType")

    click.echo()
    click.secho(f"Connecting {connector.get('displayName') or source_id}", bold=True)
    tui.kv(
        [
            ("source", click.style(source_id, fg="cyan")),
            ("auth", tui.auth_pill(auth_type)),
        ]
    )
    click.echo()

    # Warn when the user already has this source connected so they know
    # they're about to refresh credentials rather than adding something
    # new. The server-side upsert is the authoritative dedupe.
    existing = _find_existing(client, source_id)
    if existing and not assume_yes:
        _prompt_reconnect_confirmation(existing)

    if auth_type == "oauth2":
        _connect_oauth(client, source_id, connector, no_browser)
    elif auth_type == "api_key":
        _connect_api_key(client, source_id, connector)
    else:
        raise click.ClickException(
            f"Connector '{source_id}' uses unsupported auth type: {auth_type!r}"
        )


def _find_existing(client: Client, source_id: str) -> dict[str, Any] | None:
    """Return the user's current connection for this source, if any.

    Called purely for the UX warning — the authoritative dedupe happens
    server-side. Any error fetching here is swallowed so a transient list
    failure never blocks the primary connect flow.
    """
    try:
        resp = client.get("/cli/v1/connections")
    except (ApiError, AuthError):
        return None

    items: list[dict[str, Any]] = []
    if isinstance(resp, dict):
        items = resp.get("connections") or []
    elif isinstance(resp, list):
        items = resp

    for c in items:
        if c.get("sourceId") == source_id:
            return c
    return None


def _prompt_reconnect_confirmation(existing: dict[str, Any]) -> None:
    status = existing.get("status") or "unknown"
    tui.warn(
        click.style("You're already connected to this source.", fg="yellow", bold=True)
    )
    tui.kv(
        [
            ("status", f"{tui.status_dot(status)} {status}"),
            (
                "last sync",
                str(existing.get("lastSyncAt")) if existing.get("lastSyncAt") else click.style("never", dim=True),
            ),
        ]
    )
    click.echo()
    click.secho("  Continuing will refresh its credentials in place.", dim=True)
    click.echo()
    if not click.confirm(click.style("  Continue?", fg="white"), default=False):
        raise click.Abort()
    click.echo()


def _find_connector(client: Client, source_id: str) -> dict[str, Any]:
    try:
        resp = client.get("/cli/v1/sources")
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(f"Failed to fetch sources ({e.status}): {e.message}") from e

    items: list[dict[str, Any]] = []
    if isinstance(resp, dict):
        items = resp.get("connectors") or resp.get("sources") or []
    elif isinstance(resp, list):
        items = resp

    for c in items:
        if c.get("sourceId") == source_id:
            return c
    known = ", ".join(c.get("sourceId", "?") for c in items[:6])
    suffix = f" (known: {known}…)" if items else ""
    raise click.ClickException(f"Unknown source: {source_id}{suffix}")


def _connect_oauth(
    client: Client,
    source_id: str,
    connector: dict[str, Any],
    no_browser: bool,
) -> None:
    with LoopbackServer() as loopback:
        body: dict[str, Any] = {"cliLoopback": loopback.url}

        try:
            resp = client.post(f"/cli/v1/connect/{quote(source_id, safe='')}", json=body)
        except AuthError as e:
            raise click.ClickException(e.message) from e
        except ApiError as e:
            raise click.ClickException(f"Failed to start OAuth ({e.status}): {e.message}") from e

        auth_url = resp.get("authUrl") if isinstance(resp, dict) else None
        if not auth_url:
            raise click.ClickException("Backend did not return an authUrl.")
        # Defensive: never hand webbrowser.open a non-http(s) URL. Guards
        # against a compromised/MITM'd backend smuggling `file://`,
        # `javascript:`, or other schemes through the authUrl.
        scheme = urlparse(auth_url).scheme.lower()
        if scheme not in ("http", "https"):
            raise click.ClickException(f"Backend returned an unsafe authUrl scheme: {scheme!r}")

        tui.hint(f"Loopback listening on {loopback.url}")
        if not no_browser:
            tui.hint("Opening browser…")
            webbrowser.open(auth_url, new=2)

        # Always surface the URL — users may want to paste it into a
        # different browser (different profile, incognito, another
        # device) or the auto-open may have silently failed.
        click.echo()
        click.echo(
            "  " + click.style("Or open this URL:", bold=True) + " " + click.style(auth_url, fg="cyan", underline=True)
        )
        click.echo()
        click.secho("  Waiting for the callback…", dim=True)

        result = loopback.wait()

    if result.error:
        raise click.ClickException(f"OAuth failed: {result.error}")
    if not result.connection_id:
        raise click.ClickException("OAuth completed but no connection id was returned.")

    _print_connected(source_id, reconnected=result.reconnected)


def _connect_api_key(
    client: Client,
    source_id: str,
    connector: dict[str, Any],
) -> None:
    fields = connector.get("apiKeyFields") or []
    if not fields:
        raise click.ClickException(
            f"Connector '{source_id}' is api_key but exposes no fields — backend misconfigured."
        )

    credentials: dict[str, Any] = {}
    for field in fields:
        credentials[field["key"]] = _prompt_field(field)

    config_fields = connector.get("configFields") or []
    config: dict[str, Any] = {}
    for field in config_fields:
        config[field["key"]] = _prompt_field(field)

    body: dict[str, Any] = {"credentials": credentials}
    if config:
        body["config"] = config

    try:
        resp = client.post(f"/cli/v1/connect/{quote(source_id, safe='')}", json=body)
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(f"Connect failed ({e.status}): {e.message}") from e

    reconnected = bool(resp.get("reconnected")) if isinstance(resp, dict) else False
    _print_connected(source_id, reconnected=reconnected)


def _print_connected(source_id: str, *, reconnected: bool) -> None:
    verb = "Reconnected" if reconnected else "Connected"
    tui.banner(tagline=f"{verb.lower()} {source_id}")
    tui.ok(f"{verb} " + click.style(source_id, fg="cyan", bold=True))
    if reconnected:
        click.secho("  credentials refreshed in place", dim=True)
    click.echo()
    tui.hint("List yours:   coralbricks connections")
    tui.hint(f"Sync now:     coralbricks sync {source_id}")
    click.echo()


def _prompt_field(field: dict[str, Any]) -> Any:
    label = field.get("label") or field.get("key") or "value"
    required = bool(field.get("required", True))
    is_secret = bool(field.get("secret")) or field.get("type") in ("password", "secret")
    split_lines = bool(field.get("splitLines"))

    prompt_label = click.style(label, fg="white")
    if not required:
        prompt_label += click.style(" (optional)", dim=True)
    if split_lines:
        prompt_label += click.style(" (comma- or newline-separated)", dim=True)

    default = "" if not required else None
    value = click.prompt(prompt_label, hide_input=is_secret, default=default, show_default=False)
    if isinstance(value, str):
        value = value.strip()
    if split_lines and isinstance(value, str):
        parts = [p.strip() for p in value.replace("\n", ",").split(",") if p.strip()]
        return parts
    return value
