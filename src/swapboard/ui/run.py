import click

from swapboard.ui.factory import create_app
from swapboard.ui.settings import UISettings


@click.command()
@click.option("--host", default=None, help="Bind address (default: SWAPBOARD_UI_HOST).")
@click.option(
    "--port", type=int, default=None, help="Bind port (default: SWAPBOARD_UI_PORT)."
)
@click.option("--debug", is_flag=True, help="Run with the Flask debugger enabled.")
def main(host: str | None, port: int | None, debug: bool) -> None:
    """Run the swapboard dashboard."""
    settings = UISettings()
    create_app().run(
        host=host or settings.host,
        port=port or settings.port,
        debug=debug,
    )


if __name__ == "__main__":
    main()
