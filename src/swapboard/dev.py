"""Runs llama-swap, the API and the dashboard together as local processes.

Shipped with the package rather than kept as a repository script so that anyone
who installs swapboard can start the whole stack without deploying it.
"""

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import click

from swapboard.common.network import (
    DEFAULT_API_PORT,
    DEFAULT_LLAMA_SWAP_PORT,
    DEFAULT_UI_PORT,
)
from swapboard.common.paths import LLAMA_CPP_RUNTIME, LLAMA_SWAP_RUNTIME, Layout
from swapboard.runtimes import installer
from swapboard.runtimes.manifest import is_supported

DEFAULT_PREFIX = Path("data")
DEFAULT_CONFIG = Path("llama-swap.example.yml")
POLL_INTERVAL = 0.5


class ProcessGroup:
    """Runs child processes together and takes them all down as one."""

    def __init__(self) -> None:
        self._processes: list[tuple[str, subprocess.Popen[bytes]]] = []

    def start(self, label: str, command: list[str], env: dict[str, str]) -> None:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._processes.append((label, process))
        threading.Thread(
            target=_prefix_output, args=(process.stdout, f"[{label}]"), daemon=True
        ).start()

    def terminate(self) -> None:
        for _, process in self._processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        for _, process in self._processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def first_exit(self) -> tuple[str, int] | None:
        for label, process in self._processes:
            code = process.poll()
            if code is not None:
                return label, code
        return None

    def wait(self) -> int:
        """Blocks until any child exits, then stops the rest."""
        while True:
            exited = self.first_exit()
            if exited is not None:
                label, code = exited
                click.echo(f"[{label}] exited with code {code}; stopping.", err=True)
                self.terminate()
                return code
            time.sleep(POLL_INTERVAL)


def _prefix_output(stream, prefix: str) -> None:
    try:
        for line in iter(stream.readline, b""):
            sys.stdout.write(f"{prefix} {line.decode(errors='replace')}")
            sys.stdout.flush()
    except OSError, ValueError:
        pass


def read_env_file(path: Path) -> dict[str, str]:
    """Reads a minimal `KEY=value` env file, ignoring comments and blanks."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_runtime(
    runtime: str, binary: str, layout: Layout, override: Path | None
) -> str:
    """Finds a runtime binary, preferring the private one over the host's.

    Managed runtimes only exist for platforms in the manifest; elsewhere this
    falls back to whatever is on PATH so development still works.
    """
    if override is not None:
        return str(override)
    if is_supported():
        return str(installer.install(runtime, layout.runtimes))
    found = shutil.which(binary)
    if found is None:
        raise click.ClickException(
            f"{binary} was not found on PATH and no pinned build exists for this "
            f"platform. Install {binary} or pass an explicit path."
        )
    return found


def build_llama_swap_command(binary: str, config: Path, port: int) -> list[str]:
    return [binary, "--config", str(config), "--listen", f"127.0.0.1:{port}"]


def build_api_command(port: int, reload: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "swapboard.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")
    return command


def build_ui_command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "swapboard.ui.run",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def build_environment(
    base: dict[str, str],
    *,
    layout: Layout,
    config: Path,
    models_dir: Path,
    llama_server: str,
    api_port: int,
    llama_swap_port: int,
    hf_token: str | None,
) -> dict[str, str]:
    env = dict(base)
    env["MODELS_DIR"] = str(models_dir)
    env["LLAMA_SERVER_BIN"] = llama_server
    env["SWAPBOARD_MODELS_PATH"] = str(models_dir)
    env["SWAPBOARD_LLAMA_SWAP_CONFIG_PATH"] = str(config)
    env["SWAPBOARD_LLAMA_SWAP_PORT"] = str(llama_swap_port)
    env["SWAPBOARD_UI_API_URL"] = f"http://127.0.0.1:{api_port}"
    if hf_token:
        env["SWAPBOARD_HF_TOKEN"] = hf_token
    return env


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="llama-swap configuration to run.",
)
@click.option(
    "--prefix",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_PREFIX,
    show_default=True,
    help="Directory holding the runtimes and models.",
)
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Model storage directory (default: <prefix>/models).",
)
@click.option("--api-port", default=DEFAULT_API_PORT, show_default=True)
@click.option("--ui-port", default=DEFAULT_UI_PORT, show_default=True)
@click.option("--llama-swap-port", default=DEFAULT_LLAMA_SWAP_PORT, show_default=True)
@click.option(
    "--llama-swap-bin",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Use a specific llama-swap binary instead of the pinned one.",
)
@click.option(
    "--llama-server-bin",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Use a specific llama-server binary instead of the pinned one.",
)
@click.option("--hf-token", default=None, help="Hugging Face token for downloads.")
@click.option(
    "--env-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path(".env"),
    show_default=True,
)
@click.option(
    "--no-llama-swap", is_flag=True, help="Use an already-running llama-swap."
)
@click.option("--no-ui", is_flag=True, help="Do not start the dashboard.")
@click.option("--reload/--no-reload", default=True, show_default=True)
def main(
    config: Path,
    prefix: Path,
    models_dir: Path | None,
    api_port: int,
    ui_port: int,
    llama_swap_port: int,
    llama_swap_bin: Path | None,
    llama_server_bin: Path | None,
    hf_token: str | None,
    env_file: Path,
    no_llama_swap: bool,
    no_ui: bool,
    reload: bool,
) -> None:
    """Run llama-swap, the swapboard API and the dashboard locally."""
    layout = Layout(prefix.resolve())
    config = config.resolve()
    if not config.is_file():
        raise click.ClickException(f"llama-swap config not found: {config}")

    models = (models_dir or layout.models).resolve()
    models.mkdir(parents=True, exist_ok=True)

    llama_server = resolve_runtime(
        LLAMA_CPP_RUNTIME, "llama-server", layout, llama_server_bin
    )
    env = build_environment(
        {**os.environ, **read_env_file(env_file)},
        layout=layout,
        config=config,
        models_dir=models,
        llama_server=llama_server,
        api_port=api_port,
        llama_swap_port=llama_swap_port,
        hf_token=hf_token,
    )

    group = ProcessGroup()

    def shutdown(_signum, _frame) -> None:
        click.echo("\nShutting down...")
        group.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        if not no_llama_swap:
            binary = resolve_runtime(
                LLAMA_SWAP_RUNTIME, "llama-swap", layout, llama_swap_bin
            )
            group.start(
                LLAMA_SWAP_RUNTIME,
                build_llama_swap_command(binary, config, llama_swap_port),
                env,
            )
        group.start("api", build_api_command(api_port, reload), env)
        if not no_ui:
            group.start("ui", build_ui_command(ui_port), env)
    except OSError as exc:
        group.terminate()
        raise click.ClickException(f"Failed to start a service: {exc}") from exc

    click.echo(f"API       http://127.0.0.1:{api_port}")
    if not no_ui:
        click.echo(f"Dashboard http://127.0.0.1:{ui_port}")
    sys.exit(group.wait())


if __name__ == "__main__":
    main()
