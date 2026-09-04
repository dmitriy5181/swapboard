"""Native macOS deployment.

Everything a deployment owns lives under one prefix, and the llama.cpp and
llama-swap builds are private to it: they are never linked onto the host PATH,
so whatever the machine has installed itself is left untouched.

All three services run as user launchd agents this module writes directly.
Owning the llama-swap job — rather than delegating it to a package manager — is
what allows its shutdown grace period to cover a multi-gigabyte model load;
without that, launchd SIGKILLs llama-swap and orphans the llama-server processes
it spawned, which keeps their ports assigned and stops those models restarting.
"""

import fcntl
import os
import platform
import plistlib
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import click

from swapboard.common.network import (
    DEFAULT_API_PORT,
    DEFAULT_HOST,
    DEFAULT_LLAMA_SWAP_PORT,
    DEFAULT_UI_PORT,
)
from swapboard.common.paths import LLAMA_CPP_RUNTIME, LLAMA_SWAP_RUNTIME, Layout
from swapboard.runtimes import installer
from swapboard.runtimes.manifest import UnsupportedPlatformError, is_supported

LLAMA_SWAP_LABEL = "com.swapboard.llama-swap"
API_LABEL = "com.swapboard.api"
UI_LABEL = "com.swapboard.ui"
# Ordered so dependants stop before what they depend on.
ALL_LABELS = (UI_LABEL, API_LABEL, LLAMA_SWAP_LABEL)

# Deliberately excludes Homebrew: the services must resolve their binaries from
# the private runtimes directory, never from whatever the host happens to have.
SERVICE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# llama-swap stops every llama-server it spawned before exiting, and a large
# model can still be loading when that starts. launchd's 20s default would
# SIGKILL it mid-shutdown.
LLAMA_SWAP_EXIT_TIMEOUT = 120
MODEL_SERVER_STOP_TIMEOUT = 30
AGENT_STOP_TIMEOUT = 30
HEALTH_TIMEOUT = 120


@dataclass(frozen=True)
class DeployOptions:
    config: Path
    api_port: int = DEFAULT_API_PORT
    ui_port: int = DEFAULT_UI_PORT
    llama_swap_port: int = DEFAULT_LLAMA_SWAP_PORT
    api_host: str = DEFAULT_HOST
    llama_swap_host: str = DEFAULT_HOST
    with_ui: bool = True
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def hf_token(self) -> str | None:
        token = self.environment.get("SWAPBOARD_HF_TOKEN", "").strip()
        return token or None


@dataclass(frozen=True)
class LaunchAgent:
    """A user level launchd job owned by a swapboard deployment."""

    label: str
    program_arguments: list[str]
    environment_variables: dict[str, str]
    log_path: Path
    working_directory: Path
    exit_timeout: int | None = None

    @property
    def plist_path(self) -> Path:
        return Path.home() / "Library/LaunchAgents" / f"{self.label}.plist"

    def definition(self) -> dict[str, object]:
        definition: dict[str, object] = {
            "Label": self.label,
            "ProgramArguments": self.program_arguments,
            "EnvironmentVariables": self.environment_variables,
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(self.log_path),
            "StandardErrorPath": str(self.log_path),
            "WorkingDirectory": str(self.working_directory),
        }
        if self.exit_timeout is not None:
            definition["ExitTimeOut"] = self.exit_timeout
        return definition

    def write(self) -> None:
        self.plist_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.plist_path.open("wb") as plist_file:
            plistlib.dump(self.definition(), plist_file)
        # The plist can carry a Hugging Face token, so keep it owner readable.
        self.plist_path.chmod(0o600)


class MacOSHost:
    """Shared launchd and process handling for deploy and uninstall."""

    def __init__(self, layout: Layout) -> None:
        self._layout = layout

    @staticmethod
    def validate_host() -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("Native deployment requires macOS")
        if not is_supported():
            raise UnsupportedPlatformError(
                f"No pinned runtimes for {platform.machine()}"
            )

    @contextmanager
    def deployment_lock(self):
        lock_path = Path.home() / ".swapboard-deploy.lock"
        with lock_path.open("w") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another swapboard deployment is running") from exc
            yield

    def stop_agents(self) -> None:
        for label in ALL_LABELS:
            self.bootout(label)
        self.stop_model_servers()

    def bootout(self, label: str) -> None:
        self._launchctl("bootout", f"gui/{os.getuid()}/{label}", check=False)
        deadline = time.monotonic() + AGENT_STOP_TIMEOUT
        while time.monotonic() < deadline:
            if not self._is_loaded(label):
                return
            time.sleep(0.5)
        raise RuntimeError(f"launchd agent did not stop: {label}")

    def bootstrap(self, agent: LaunchAgent) -> None:
        self.bootout(agent.label)
        agent.write()
        self._launchctl(
            "bootstrap", f"gui/{os.getuid()}", str(agent.plist_path), check=True
        )

    def stop_model_servers(self) -> None:
        """Terminates llama-server processes that outlived llama-swap.

        llama-swap assigns each model a fixed port, so a single orphan stops
        that model from ever starting again. Callers must stop llama-swap first,
        otherwise it respawns them. Matching on the private binary path means
        this can never touch a llama-server the host runs for its own purposes.
        """
        pattern = str(self._layout.llama_server_bin)
        for signal_option in ("-TERM", "-KILL"):
            if not self._model_servers_running(pattern):
                return
            subprocess.run(
                ["/usr/bin/pkill", signal_option, "-f", pattern], check=False
            )
            deadline = time.monotonic() + MODEL_SERVER_STOP_TIMEOUT
            while time.monotonic() < deadline and self._model_servers_running(pattern):
                time.sleep(1)
        if self._model_servers_running(pattern):
            raise RuntimeError(f"llama-server did not stop: {pattern}")

    def _is_loaded(self, label: str) -> bool:
        result = self._launchctl("print", f"gui/{os.getuid()}/{label}", check=False)
        return result.returncode == 0

    @staticmethod
    def _model_servers_running(pattern: str) -> bool:
        return (
            subprocess.run(
                ["/usr/bin/pgrep", "-f", pattern], capture_output=True
            ).returncode
            == 0
        )

    @staticmethod
    def _launchctl(*arguments: str, check: bool):
        return subprocess.run(
            ["/bin/launchctl", *arguments],
            text=True,
            capture_output=True,
            check=check,
        )


class MacOSDeployment(MacOSHost):
    def __init__(self, layout: Layout, options: DeployOptions) -> None:
        super().__init__(layout)
        self._options = options

    def deploy(self) -> None:
        self.validate_host()
        self._validate_options()
        with self.deployment_lock():
            # Stopping first releases the runtime directory before it is
            # replaced, and clears any llama-server orphaned by a previous run.
            self.stop_agents()
            installer.install_all(self._layout.runtimes)
            self._prepare_directories()
            self._install_config()
            try:
                for agent in self.agents():
                    self.bootstrap(agent)
                self._wait_for_health()
            except OSError, RuntimeError, subprocess.CalledProcessError:
                self._print_diagnostics()
                raise

    def agents(self) -> list[LaunchAgent]:
        agents = [self._llama_swap_agent(), self._api_agent()]
        if self._options.with_ui:
            agents.append(self._ui_agent())
        return agents

    def _validate_options(self) -> None:
        if not self._options.config.is_file():
            raise RuntimeError(f"llama-swap config not found: {self._options.config}")
        ports = [
            self._options.api_port,
            self._options.llama_swap_port,
        ] + ([self._options.ui_port] if self._options.with_ui else [])
        if len(set(ports)) != len(ports):
            raise RuntimeError(f"Service ports must differ, got {ports}")

    def _prepare_directories(self) -> None:
        for directory in (
            self._layout.models,
            self._layout.config,
            self._layout.log,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _install_config(self) -> None:
        try:
            shutil.copyfile(self._options.config, self._layout.llama_swap_config)
        except shutil.SameFileError:
            return

    def _llama_swap_agent(self) -> LaunchAgent:
        return LaunchAgent(
            label=LLAMA_SWAP_LABEL,
            program_arguments=[
                str(self._layout.llama_swap_bin),
                "--config",
                str(self._layout.llama_swap_config),
                "--listen",
                f"{self._options.llama_swap_host}:{self._options.llama_swap_port}",
            ],
            environment_variables={
                "MODELS_DIR": str(self._layout.models),
                "LLAMA_SERVER_BIN": str(self._layout.llama_server_bin),
                "PATH": SERVICE_PATH,
            },
            log_path=self._layout.log_file(LLAMA_SWAP_RUNTIME),
            working_directory=self._layout.prefix,
            exit_timeout=LLAMA_SWAP_EXIT_TIMEOUT,
        )

    def _api_agent(self) -> LaunchAgent:
        environment = {
            "SWAPBOARD_LLAMA_SWAP_CONFIG_PATH": str(self._layout.llama_swap_config),
            "SWAPBOARD_LLAMA_SWAP_PORT": str(self._options.llama_swap_port),
            "SWAPBOARD_MODELS_PATH": str(self._layout.models),
            "PATH": SERVICE_PATH,
        }
        token = self._options.hf_token
        if token:
            environment["SWAPBOARD_HF_TOKEN"] = token
        return LaunchAgent(
            label=API_LABEL,
            program_arguments=[
                str(self._layout.venv / "bin/uvicorn"),
                "swapboard.api.main:app",
                "--host",
                self._options.api_host,
                "--port",
                str(self._options.api_port),
            ],
            environment_variables=environment,
            log_path=self._layout.log_file("api"),
            working_directory=self._layout.prefix,
        )

    def _ui_agent(self) -> LaunchAgent:
        return LaunchAgent(
            label=UI_LABEL,
            program_arguments=[
                str(self._layout.venv / "bin/gunicorn"),
                "swapboard.ui.factory:create_app()",
                "--bind",
                f"{DEFAULT_HOST}:{self._options.ui_port}",
                "--workers",
                "2",
                "--timeout",
                "60",
            ],
            environment_variables={
                "SWAPBOARD_UI_API_URL": f"http://{DEFAULT_HOST}:{self._options.api_port}",
                "PATH": SERVICE_PATH,
            },
            log_path=self._layout.log_file("ui"),
            working_directory=self._layout.prefix,
        )

    def _wait_for_health(self) -> None:
        for port in self._health_ports():
            _wait_for_url(f"http://{DEFAULT_HOST}:{port}/health", HEALTH_TIMEOUT)

    def _health_ports(self) -> list[int]:
        ports = [self._options.llama_swap_port, self._options.api_port]
        if self._options.with_ui:
            ports.append(self._options.ui_port)
        return ports

    def _print_diagnostics(self) -> None:
        for name in (LLAMA_SWAP_RUNTIME, "api", "ui"):
            _print_log(self._layout.log_file(name))


class MacOSUninstaller(MacOSHost):
    def uninstall(self) -> None:
        """Removes the services and runtimes, keeping models and config."""
        self.validate_host()
        with self.deployment_lock():
            self.stop_agents()
            for label in ALL_LABELS:
                (Path.home() / "Library/LaunchAgents" / f"{label}.plist").unlink(
                    missing_ok=True
                )
            for runtime in (LLAMA_SWAP_RUNTIME, LLAMA_CPP_RUNTIME):
                installer.remove(runtime, self._layout.runtimes)


def _wait_for_url(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"Service did not become healthy: {url}")


def _print_log(log_path: Path) -> None:
    if not log_path.is_file():
        return
    click.echo(f"==> {log_path}", err=True)
    recent_lines = log_path.read_text(errors="replace").splitlines(True)[-100:]
    click.echo("".join(recent_lines), err=True)


def _layout_for(prefix: Path | None) -> Layout:
    return Layout(prefix.resolve()) if prefix else Layout.default()


prefix_option = click.option(
    "--prefix",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Installation prefix (default: the parent of the active virtualenv).",
)


@click.group()
def main() -> None:
    """Deploy swapboard as launchd services on macOS."""


@main.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    required=True,
    help="llama-swap configuration to install.",
)
@prefix_option
@click.option("--api-port", default=DEFAULT_API_PORT, show_default=True)
@click.option("--ui-port", default=DEFAULT_UI_PORT, show_default=True)
@click.option("--llama-swap-port", default=DEFAULT_LLAMA_SWAP_PORT, show_default=True)
@click.option("--api-host", default=DEFAULT_HOST, show_default=True)
@click.option("--llama-swap-host", default=DEFAULT_HOST, show_default=True)
@click.option(
    "--with-ui/--no-ui", default=True, show_default=True, help="Deploy the dashboard."
)
def deploy(
    config: Path,
    prefix: Path | None,
    api_port: int,
    ui_port: int,
    llama_swap_port: int,
    api_host: str,
    llama_swap_host: str,
    with_ui: bool,
) -> None:
    """Install the runtimes and start the launchd services."""
    layout = _layout_for(prefix)
    options = DeployOptions(
        config=config.resolve(),
        api_port=api_port,
        ui_port=ui_port,
        llama_swap_port=llama_swap_port,
        api_host=api_host,
        llama_swap_host=llama_swap_host,
        with_ui=with_ui,
        environment=dict(os.environ),
    )
    MacOSDeployment(layout, options).deploy()
    click.echo(f"swapboard deployed to {layout.prefix}")


@main.command()
@prefix_option
def stop(prefix: Path | None) -> None:
    """Stop the services, leaving the installation in place.

    Upgrading replaces the virtualenv the API and dashboard run from, so their
    agents have to be stopped before that happens rather than afterwards.
    """
    layout = _layout_for(prefix)
    host = MacOSHost(layout)
    host.validate_host()
    with host.deployment_lock():
        host.stop_agents()
    click.echo(f"swapboard services stopped ({layout.prefix})")


@main.command()
@prefix_option
def uninstall(prefix: Path | None) -> None:
    """Stop the services and remove the runtimes."""
    layout = _layout_for(prefix)
    MacOSUninstaller(layout).uninstall()
    click.echo(f"swapboard removed from {layout.prefix} (models and config kept)")


if __name__ == "__main__":
    main()
