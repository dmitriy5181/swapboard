"""Console entry point for the swapboard API.

Uses argparse rather than click so that a bare `pip install swapboard` needs no
dependency beyond the ASGI stack the service already requires.
"""

import argparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8771


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "swapboard.api.main:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swapboard-api", description="Run the swapboard API."
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Reload on source changes."
    )
    return parser


if __name__ == "__main__":
    main()
