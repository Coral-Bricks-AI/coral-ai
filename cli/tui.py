"""Shared styling helpers so every command looks the same.

Nothing in here is fancy — just click.secho wrappers for checkmarks,
colored status dots, and aligned key/value + table printers. Keeping it
small means any new subcommand inherits the look for free.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import click

CHECK = "✓"
CROSS = "✗"
BULLET = "●"
ARROW = "→"
RULE = "─"

# Source-auth and run/connection status colors. Keeping a single source
# of truth lets `sources`, `connections`, and `runs` render consistently.
_AUTH_COLORS = {
    "oauth2": "cyan",
    "api_key": "magenta",
}

_STATUS_COLORS = {
    "active": "green",
    "success": "green",
    "succeeded": "green",
    "running": "cyan",
    "connecting": "cyan",
    "queued": "cyan",
    "pending": "yellow",
    "error": "red",
    "failed": "red",
    "inactive": "yellow",
    "disconnected": "yellow",
}


def ok(msg: str) -> None:
    click.echo(click.style(f"{CHECK} ", fg="green", bold=True) + msg)


def warn(msg: str) -> None:
    click.echo(click.style(f"! ", fg="yellow", bold=True) + msg)


def err(msg: str) -> None:
    click.echo(click.style(f"{CROSS} ", fg="red", bold=True) + msg)


def hint(msg: str) -> None:
    click.echo("  " + click.style(f"{ARROW} ", fg="blue") + click.style(msg, dim=True))


def rule(width: int = 40) -> None:
    click.echo(click.style(RULE * width, dim=True))


def heading(msg: str) -> None:
    click.echo()
    click.secho(msg, bold=True, fg="white")
    click.echo(click.style(RULE * max(len(msg), 8), dim=True))


def kv(pairs: Iterable[tuple[str, object]], *, indent: int = 2) -> None:
    """Print aligned key/value pairs. `value=None` rows are skipped."""
    rows = [(k, v) for k, v in pairs if v not in (None, "")]
    if not rows:
        return
    width = max(len(k) for k, _ in rows)
    pad = " " * indent
    for k, v in rows:
        click.echo(f"{pad}{click.style(k.ljust(width), dim=True)}  {v}")


def status_dot(status: str | None) -> str:
    color = _STATUS_COLORS.get((status or "").lower(), "white")
    return click.style(BULLET, fg=color)


def auth_pill(auth_type: str | None) -> str:
    color = _AUTH_COLORS.get((auth_type or "").lower(), "white")
    return click.style(auth_type or "-", fg=color)


def table(
    rows: Sequence[Sequence[object]],
    *,
    headers: Sequence[str],
    indent: int = 2,
) -> None:
    """Render a simple aligned table with a dim rule beneath bold headers.

    Cells pass through str() after their ANSI escape codes are stripped
    for width calculation so colored cells line up correctly.
    """
    if not rows:
        return
    stringified = [[_stringify(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in stringified:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(cell))

    pad = " " * indent
    header_line = "  ".join(
        click.style(h.upper().ljust(widths[i]), bold=True) for i, h in enumerate(headers)
    )
    click.echo(pad + header_line)
    click.echo(pad + click.style("  ".join(RULE * w for w in widths), dim=True))
    for row in stringified:
        cells = []
        for i, cell in enumerate(row):
            padding = widths[i] - _visible_len(cell)
            cells.append(cell + (" " * padding))
        click.echo(pad + "  ".join(cells))


def _stringify(value: object) -> str:
    return "" if value is None else str(value)


def _visible_len(styled: str) -> int:
    """Length after stripping ANSI escape sequences."""
    out = []
    i = 0
    while i < len(styled):
        ch = styled[i]
        if ch == "\x1b" and i + 1 < len(styled) and styled[i + 1] == "[":
            j = i + 2
            while j < len(styled) and styled[j] != "m":
                j += 1
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return len("".join(out))
