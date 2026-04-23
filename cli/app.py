"""Top-level Click group — the `coralbricks` command."""

from __future__ import annotations

import click

from . import __version__
from .commands.connections import connections_cmd
from .commands.login import login_cmd, logout_cmd, whoami_cmd
from .commands.sources import sources_cmd


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="coralbricks")
def cli() -> None:
    """Coral Bricks — connect data sources and run Airbyte sync locally."""


cli.add_command(login_cmd)
cli.add_command(logout_cmd)
cli.add_command(whoami_cmd)
cli.add_command(sources_cmd)
cli.add_command(connections_cmd)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
