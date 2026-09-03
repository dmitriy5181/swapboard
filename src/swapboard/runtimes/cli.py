from pathlib import Path

import click

from swapboard.common.paths import Layout
from swapboard.runtimes import installer
from swapboard.runtimes.manifest import RUNTIMES, UnsupportedPlatformError, resolve

prefix_option = click.option(
    "--prefix",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Installation prefix (default: the parent of the active virtualenv).",
)


def _runtimes_dir(prefix: Path | None) -> Path:
    layout = Layout(prefix.resolve()) if prefix else Layout.default()
    return layout.runtimes


class RuntimeGroup(click.Group):
    """Reports an unsupported platform as an error rather than a traceback.

    Pinned builds only exist for macOS. Running these commands anywhere else
    is a normal thing for a user to try, so it should read as a plain message
    explaining the alternative.
    """

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except UnsupportedPlatformError as error:
            raise click.ClickException(str(error)) from error


@click.group(cls=RuntimeGroup)
def main() -> None:
    """Manage the private llama.cpp and llama-swap builds swapboard runs."""


@main.command()
@prefix_option
@click.option(
    "--only",
    type=click.Choice(RUNTIMES),
    default=None,
    help="Install a single runtime instead of all of them.",
)
@click.option("--force", is_flag=True, help="Reinstall even if already current.")
def install(prefix: Path | None, only: str | None, force: bool) -> None:
    """Download and verify the pinned runtimes."""
    runtimes_dir = _runtimes_dir(prefix)
    targets = (only,) if only else RUNTIMES
    for runtime in targets:
        path = installer.install(runtime, runtimes_dir, force=force)
        click.echo(f"{runtime} {resolve(runtime).version} -> {path}")


@main.command()
@prefix_option
def status(prefix: Path | None) -> None:
    """Show which runtimes are installed and whether they are current."""
    runtimes_dir = _runtimes_dir(prefix)
    for runtime in RUNTIMES:
        pinned = resolve(runtime).version
        found = installer.installed_version(runtime, runtimes_dir)
        if found is None:
            click.echo(f"{runtime}: not installed (pinned {pinned})")
        elif installer.is_current(runtime, runtimes_dir):
            click.echo(f"{runtime}: {found} (current)")
        else:
            click.echo(f"{runtime}: {found} (pinned {pinned})")


@main.command()
@prefix_option
@click.argument("binary", type=click.Choice(["llama-server", "llama-swap"]))
def path(prefix: Path | None, binary: str) -> None:
    """Print the absolute path of a runtime binary."""
    layout = Layout(prefix.resolve()) if prefix else Layout.default()
    resolved = (
        layout.llama_server_bin if binary == "llama-server" else layout.llama_swap_bin
    )
    if not resolved.is_file():
        raise click.ClickException(f"{binary} is not installed at {resolved}")
    click.echo(str(resolved))


@main.command()
@prefix_option
@click.option(
    "--only",
    type=click.Choice(RUNTIMES),
    default=None,
    help="Remove a single runtime instead of all of them.",
)
def remove(prefix: Path | None, only: str | None) -> None:
    """Delete installed runtimes."""
    runtimes_dir = _runtimes_dir(prefix)
    for runtime in (only,) if only else RUNTIMES:
        if installer.remove(runtime, runtimes_dir):
            click.echo(f"Removed {runtime}")
        else:
            click.echo(f"{runtime} was not installed")


if __name__ == "__main__":
    main()
