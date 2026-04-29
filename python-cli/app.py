"""Top-level Click group — the `coralbricks` command.

`coralbricks` (no subcommand) renders a branded welcome panel — feels
like a real product instead of dumping click's bare help. `--help`
still works as the dev escape hatch.
"""

from __future__ import annotations

import sys

import click

from . import __version__, tui
from .commands.connect import connect_cmd
from .commands.connections import connections_cmd
from .commands.disconnect import disconnect_cmd
from .commands.login import login_cmd, logout_cmd, whoami_cmd
from .commands.runs import runs_cmd
from .commands.sources import sources_cmd
from .commands.status import status_cmd
from .commands.sync import sync_cmd


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(__version__, prog_name="coralbricks")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Coral Bricks — connect data sources and run syncs locally into managed storage."""
    if ctx.invoked_subcommand is None:
        tui.banner()
        tui.welcome_panel()
        tui.blank()
        sys.exit(0)


cli.add_command(login_cmd)
cli.add_command(logout_cmd)
cli.add_command(whoami_cmd)
cli.add_command(status_cmd)
cli.add_command(sources_cmd)
cli.add_command(connect_cmd)
cli.add_command(connections_cmd)
cli.add_command(disconnect_cmd)
cli.add_command(sync_cmd)
cli.add_command(runs_cmd)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
