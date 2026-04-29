"""Design system for the Coral Bricks CLI.

Built on rich for layout primitives + click for argv. Every command's
output flows through the shared `console` so we own one global output
contract: theme, truecolor, terminal width, emoji-off. In non-TTY
contexts (CI, pipes, tests) rich downgrades cleanly to plain text.

Hero moments use Panels with rounded borders and the brand coral
accent. Inline messages (ok / warn / err / hint) keep the click-era
look so panel-flow and inline-flow chain visually.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .logo import LOGO_ICON_ANSI

# Brand --------------------------------------------------------

CORAL = "#d64027"
CORAL_DIM = "#a13320"

# Glyphs exposed for command code that composes Text directly.
CHECK = "✓"
CROSS = "✗"
BULLET = "●"
ARROW = "→"
RULE = "─"

# Theme + console ----------------------------------------------

_theme = Theme(
    {
        "coral": CORAL,
        "coral.bold": f"bold {CORAL}",
        "coral.dim": CORAL_DIM,
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "hint": "dim",
        "label": "dim",
        "value": "white",
        "muted": "grey50",
        "kbd": "bold",
        "status.active": "green",
        "status.success": "green",
        "status.succeeded": "green",
        "status.running": "cyan",
        "status.connecting": "cyan",
        "status.queued": "cyan",
        "status.pending": "yellow",
        "status.error": "red",
        "status.failed": "red",
        "status.inactive": "yellow",
        "status.disconnected": "yellow",
        "auth.oauth2": "cyan",
        "auth.api_key": "magenta",
    }
)

# emoji=False so a literal `:` in user output (e.g. an HTTP error)
# isn't accidentally interpreted as a rich emoji shortcode.
console = Console(theme=_theme, soft_wrap=False, highlight=False, emoji=False)

_PILL_BG = {
    "success": "green",
    "ok": "green",
    "running": "cyan",
    "connecting": "cyan",
    "info": "blue",
    "warn": "yellow",
    "error": "red",
    "fail": "red",
    "neutral": "white",
    "coral": CORAL,
}


# Inline messages ---------------------------------------------


def ok(msg: str | Text) -> None:
    console.print(Text(f"{CHECK} ", style="ok") + _to_text(msg))


def warn(msg: str | Text) -> None:
    console.print(Text("! ", style="warn") + _to_text(msg))


def err(msg: str | Text) -> None:
    console.print(Text(f"{CROSS} ", style="err") + _to_text(msg))


def hint(msg: str) -> None:
    console.print(Text(f"  {ARROW} ", style="blue") + Text(msg, style="dim"))


def heading(msg: str) -> None:
    console.print()
    console.print(Text(msg, style="bold"))
    console.print(Text(RULE * max(len(msg), 8), style="dim"))


def rule(width: int = 40) -> None:
    console.print(Text(RULE * width, style="dim"))


def blank() -> None:
    console.print()


# Status pills + dots -----------------------------------------


def pill(text: str, kind: str = "info") -> Text:
    """Block-style status badge: `█ SUCCESS █` with bg color."""
    bg = _PILL_BG.get(kind.lower(), "blue")
    return Text(f" {text.upper()} ", style=f"black on {bg} bold")


def status_dot(status: str | None) -> Text:
    s = (status or "").lower()
    style = f"status.{s}" if f"status.{s}" in _theme.styles else "white"
    return Text(BULLET, style=style)


def status_label(status: str | None) -> Text:
    """Dot + colored word — used in tables where a pill is too loud."""
    s = (status or "unknown").lower()
    out = status_dot(status)
    out.append(" ")
    style = f"status.{s}" if f"status.{s}" in _theme.styles else "white"
    out.append(s, style=style)
    return out


def auth_pill(auth_type: str | None) -> Text:
    a = (auth_type or "-").lower()
    style = f"auth.{a}" if f"auth.{a}" in _theme.styles else "white"
    return Text(auth_type or "-", style=style)


# Phase markers ------------------------------------------------


def phase(n: int, total: int, label: str, *, elapsed: float | None = None) -> None:
    line = Text()
    line.append(f"[{n}/{total}] ", style="bold cyan")
    line.append(label)
    if elapsed is not None:
        line.append(f"  {elapsed:.1f}s", style="dim")
    console.print(line)


# Panels -------------------------------------------------------


def panel(
    content: RenderableType,
    *,
    title: str | None = None,
    title_extra: RenderableType | None = None,
    footer: str | None = None,
    accent: str = "grey42",
    padding: tuple[int, int] = (1, 2),
) -> None:
    """Render a rounded-border panel with the brand accent.

    `title_extra` lets a caller compose a richer title (e.g. a status
    pill + the run id) without us reaching for `Text.from_markup`.
    """
    title_renderable: RenderableType | None = None
    if title and title_extra is not None:
        title_renderable = Text.assemble(
            (title, "bold"), "  ", title_extra
        )
    elif title:
        title_renderable = Text(title, style="bold")
    elif title_extra is not None:
        title_renderable = title_extra

    p = Panel(
        content,
        title=title_renderable,
        title_align="left",
        subtitle=Text(footer, style="dim") if footer else None,
        subtitle_align="left",
        border_style=accent,
        padding=padding,
        box=ROUNDED,
    )
    console.print(p)


# KV pairs -----------------------------------------------------


def kv(pairs: Iterable[tuple[str, Any]], *, indent: int = 2) -> None:
    rows = [(k, v) for k, v in pairs if v not in (None, "")]
    if not rows:
        return
    width = max(len(k) for k, _ in rows)
    pad = " " * indent
    for k, v in rows:
        line = Text(pad)
        line.append(k.ljust(width), style="dim")
        line.append("  ")
        line.append_text(_to_text(v))
        console.print(line)


def kv_renderable(pairs: Iterable[tuple[str, Any]]) -> RenderableType:
    """Same as `kv` but returns a renderable for embedding in panels."""
    items: list[Text] = []
    rows = [(k, v) for k, v in pairs if v not in (None, "")]
    if not rows:
        return Text("")
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        line = Text()
        line.append(k.ljust(width), style="dim")
        line.append("  ")
        line.append_text(_to_text(v))
        items.append(line)
    return Group(*items)


# Tables -------------------------------------------------------


def table(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[str],
    indent: int = 2,
) -> None:
    if not rows:
        return
    t = _build_table(rows, headers=headers)
    console.print(Padding(t, (0, 0, 0, indent)) if indent else t)


def table_renderable(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[str],
) -> RenderableType:
    return _build_table(rows, headers=headers)


def _build_table(rows: Sequence[Sequence[Any]], *, headers: Sequence[str]) -> Table:
    t = Table(box=None, show_edge=False, pad_edge=False, padding=(0, 2), show_header=True)
    for h in headers:
        t.add_column(Text(h.upper(), style="bold"))
    for row in rows:
        t.add_row(*[_to_text(c) for c in row])
    return t


# Branded banner ----------------------------------------------


def banner(tagline: str | None = "connect 600+ data sources") -> None:
    """Print the chafa logomark + wordmark side-by-side, no panel."""
    indent = "  "
    gap = "   "
    logo_lines = LOGO_ICON_ANSI.rstrip("\n").split("\n")
    console.print()
    for i, logo_line in enumerate(logo_lines):
        rendered = Text.from_ansi(indent + logo_line)
        if i == 3:
            rendered.append(gap)
            rendered.append("CoralBricks", style=f"bold {CORAL}")
        elif i == 4 and tagline:
            rendered.append(gap)
            rendered.append(tagline, style="dim")
        console.print(rendered)
    console.print()


def welcome_panel() -> None:
    """Hero shown on `coralbricks` (no subcommand) + as login fallback."""
    panel(
        _welcome_body(),
        title="CoralBricks",
        footer="docs · github.com/Coral-Bricks-AI/coral-ai",
        accent="coral",
    )


def _welcome_body() -> Group:
    items: list[RenderableType] = []
    items.append(Text("connect 600+ data sources, run syncs locally.", style="dim"))
    items.append(Text(""))
    items.append(Text("GETTING STARTED", style="bold"))
    for cmd, desc in (
        ("coralbricks login", "authenticate with an api key"),
        ("coralbricks connect stripe", "connect your first source"),
        ("coralbricks sync stripe", "run your first sync"),
    ):
        line = Text("  ")
        line.append(cmd, style=f"bold {CORAL}")
        line.append("  ")
        line.append(desc, style="dim")
        items.append(line)
    items.append(Text(""))
    items.append(Text("COMMANDS", style="bold"))
    for name, desc in (
        ("login", "authenticate (or switch account)"),
        ("status", "one-screen overview"),
        ("sources", "browse available connectors"),
        ("connect", "add a new data source"),
        ("connections", "list your connected sources"),
        ("sync", "run a sync locally"),
        ("runs", "show recent sync history"),
        ("disconnect", "remove a connection"),
        ("logout", "clear stored credentials"),
    ):
        line = Text("  ")
        line.append(name.ljust(13), style="bold")
        line.append(desc, style="dim")
        items.append(line)
    return Group(*items)


# Helpers ------------------------------------------------------


def _to_text(v: Any) -> Text:
    if isinstance(v, Text):
        return v
    return Text(str(v))
