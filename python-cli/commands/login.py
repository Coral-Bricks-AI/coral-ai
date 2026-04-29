"""login / logout / whoami — API-key auth against the Coral Bricks backend."""

from __future__ import annotations

import click
from rich.console import Group
from rich.text import Text

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

    # Already-logged-in short-circuit — surface current identity and
    # offer an inline switch so users don't have to chain logout+login.
    # Skipped when --api-key is passed explicitly (power-user override).
    if not api_key and cfg.effective_api_key():
        tui.blank()
        tui.ok(
            Text("Already logged in as ", style="white").append(
                cfg.email or "unknown", style=f"bold {tui.CORAL}"
            ).append(".")
        )
        tui.blank()
        if not click.confirm(
            click.style("  Switch account?", fg="white"), default=False
        ):
            return
        cfg_mod.clear()
        cfg = cfg_mod.load()
        if server_url:
            cfg.server_url = server_url.rstrip("/")
        tui.blank()

    if not api_key:
        tui.banner()
        tui.console.print(
            Text("  ").append(tui.ARROW + " ", style="blue").append(
                "Create an API key at ", style="dim"
            ).append(SIGNUP_URL, style=f"{tui.CORAL} underline")
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
    verified = bool(resp.get("isVerified")) if isinstance(resp, dict) else False
    server = cfg.effective_server_url()
    custom_server = server.rstrip("/") != cfg_mod.DEFAULT_SERVER_URL.rstrip("/")

    _render_welcome(cfg.email, plan, verified, server if custom_server else None)


def _render_welcome(
    email: str | None, plan: str, verified: bool, server: str | None
) -> None:
    headline = Text()
    headline.append("Welcome to Coral Bricks, ", style="white")
    headline.append(email or "friend", style=f"bold {tui.CORAL}")
    headline.append(".")

    identity = tui.kv_renderable(
        [
            ("plan", Text(plan, style="magenta" if plan == "paid" else "white")),
            ("status", Text("verified", style="green") if verified else None),
            ("server", Text(server, style="dim") if server else None),
        ]
    )

    next_steps = Text("WHAT'S NEXT", style="bold")
    steps_lines = []
    for cmd, desc in (
        ("coralbricks sources", "browse 600+ connectors"),
        ("coralbricks connect <id>", "add a data source"),
        ("coralbricks sync <id>", "run a sync locally"),
    ):
        line = Text("  ")
        line.append(cmd, style=f"bold {tui.CORAL}")
        line.append("  ")
        line.append(desc, style="dim")
        steps_lines.append(line)

    body = Group(
        Text(tui.CHECK + " ", style="ok").append_text(headline),
        Text(""),
        identity,
        Text(""),
        next_steps,
        *steps_lines,
    )
    tui.banner()
    tui.panel(body, title_extra=tui.pill("LOGGED IN", "success"), accent="green")
    tui.blank()


@click.command("logout")
def logout_cmd() -> None:
    """Clear the stored API key."""
    removed = cfg_mod.clear()
    if removed:
        tui.ok("Logged out.")
    else:
        tui.console.print(Text("Already logged out.", style="dim"))


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

    body = tui.kv_renderable(
        [
            ("email", Text(email or "-", style=f"bold {tui.CORAL}")),
            ("plan", Text(plan, style="magenta" if plan == "paid" else "white")),
            ("status", Text("verified", style="green") if verified else Text("unverified", style="yellow")),
            ("server", Text(server, style="dim") if custom_server else None),
        ]
    )
    tui.blank()
    tui.panel(body, title="Identity", title_extra=tui.pill("ACTIVE", "success"))
    tui.blank()
