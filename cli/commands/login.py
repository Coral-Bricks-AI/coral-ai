"""login / logout / whoami — API-key auth against the Coral Bricks backend."""

from __future__ import annotations

import click

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client

SIGNUP_URL = "https://coralbricks.ai/settings/api-keys"


@click.command("login")
@click.option(
    "--api-key",
    "api_key",
    help="API key (otherwise prompted interactively).",
    default=None,
)
@click.option(
    "--server-url",
    "server_url",
    help="Override backend URL (default: https://backend.coralbricks.ai).",
    default=None,
)
def login_cmd(api_key: str | None, server_url: str | None) -> None:
    """Authenticate with a Coral Bricks API key."""
    cfg = cfg_mod.load()
    if server_url:
        cfg.server_url = server_url.rstrip("/")

    if not api_key:
        click.echo(click.style("Coral Bricks", bold=True, fg="cyan"))
        click.echo(
            click.style(f"  {tui.ARROW} ", fg="blue")
            + click.style("Create an API key at ", dim=True)
            + click.style(SIGNUP_URL, fg="cyan")
        )
        api_key = click.prompt(
            click.style("  Paste your API key", fg="white"),
            hide_input=True,
        )

    cfg.api_key = api_key.strip()
    client = Client(cfg)
    try:
        resp = client.post("/cli/v1/auth/validate", json={})
    except AuthError as e:
        raise click.ClickException(f"Login failed: {e.message}") from e
    except ApiError as e:
        raise click.ClickException(f"Login failed ({e.status}): {e.message}") from e

    cfg.user_id = resp.get("userId") if isinstance(resp, dict) else None
    cfg.email = resp.get("email") if isinstance(resp, dict) else None
    cfg_mod.save(cfg)

    plan = (resp.get("plan") if isinstance(resp, dict) else None) or "free"
    verified = resp.get("isVerified") if isinstance(resp, dict) else None
    server = cfg.effective_server_url()
    custom_server = server.rstrip("/") != cfg_mod.DEFAULT_SERVER_URL.rstrip("/")

    click.echo()
    tui.ok(
        "Welcome to Coral Bricks, "
        + click.style(cfg.email or "friend", fg="cyan", bold=True)
        + "."
    )
    click.echo()
    tui.kv(
        [
            ("plan", click.style(plan, fg="magenta" if plan == "paid" else "white")),
            ("status", click.style("verified", fg="green") if verified else None),
            ("server", click.style(server, dim=True) if custom_server else None),
        ]
    )
    click.echo()
    tui.hint("Browse connectors:    coralbricks sources")
    tui.hint("Connect a source:     coralbricks connect <name>")
    click.echo()


@click.command("logout")
def logout_cmd() -> None:
    """Clear the stored API key."""
    removed = cfg_mod.clear()
    if removed:
        tui.ok("Logged out.")
    else:
        click.echo(click.style("Already logged out.", dim=True))


@click.command("whoami")
def whoami_cmd() -> None:
    """Show the currently logged-in user."""
    cfg = cfg_mod.load()
    if not cfg.effective_api_key():
        raise click.ClickException("Not logged in. Run `coralbricks login` first.")
    client = Client(cfg)
    try:
        resp = client.post("/cli/v1/auth/validate", json={})
    except AuthError as e:
        raise click.ClickException(f"Session invalid: {e.message}") from e
    except ApiError as e:
        raise click.ClickException(f"API error ({e.status}): {e.message}") from e

    email = cfg.email
    plan = "free"
    verified = False
    if isinstance(resp, dict):
        email = resp.get("email") or email
        plan = resp.get("plan") or plan
        verified = bool(resp.get("isVerified"))

    server = cfg.effective_server_url()
    custom_server = server.rstrip("/") != cfg_mod.DEFAULT_SERVER_URL.rstrip("/")

    click.echo()
    tui.kv(
        [
            ("email", click.style(email or "-", fg="cyan", bold=True)),
            ("plan", click.style(plan, fg="magenta" if plan == "paid" else "white")),
            ("status", click.style("verified", fg="green") if verified else None),
            ("server", click.style(server, dim=True) if custom_server else None),
        ]
    )
    click.echo()
