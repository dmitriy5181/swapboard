import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

from swapboard.common.paths import Layout
from swapboard.deployment import macos
from swapboard.deployment.macos import (
    ALL_LABELS,
    API_LABEL,
    LLAMA_SWAP_LABEL,
    DeployOptions,
    MacOSDeployment,
    MacOSUninstaller,
)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "llama-swap.yml"
    path.write_text("models: {}\n", encoding="utf-8")
    return path


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return Layout(tmp_path / "prefix")


@pytest.fixture
def quiet_host(monkeypatch):
    """Neutralises everything that would touch launchd or the network."""
    monkeypatch.setattr(macos.MacOSHost, "validate_host", staticmethod(lambda: None))
    monkeypatch.setattr(macos.MacOSHost, "deployment_lock", lambda self: _nullcontext())
    monkeypatch.setattr(macos, "_wait_for_url", lambda url, timeout: None)


@contextmanager
def _nullcontext():
    yield


def test_deploy_runs_the_steps_in_a_safe_order(
    layout, config, quiet_host, monkeypatch
) -> None:
    """Services must stop before the runtimes they execute are replaced."""
    events: list[str] = []

    monkeypatch.setattr(
        macos.MacOSHost, "stop_agents", lambda self: events.append("stop")
    )
    monkeypatch.setattr(
        macos.installer, "install_all", lambda directory: events.append("runtimes")
    )
    monkeypatch.setattr(
        macos.MacOSHost,
        "bootstrap",
        lambda self, agent: events.append(f"start:{agent.label}"),
    )

    MacOSDeployment(layout, DeployOptions(config=config)).deploy()

    assert events[0] == "stop"
    assert events[1] == "runtimes"
    assert events[2].startswith("start:")
    assert events.index("stop") < events.index("runtimes")


def test_deploy_starts_llama_swap_before_the_api(
    layout, config, quiet_host, monkeypatch
) -> None:
    started: list[str] = []
    monkeypatch.setattr(macos.MacOSHost, "stop_agents", lambda self: None)
    monkeypatch.setattr(macos.installer, "install_all", lambda directory: None)
    monkeypatch.setattr(
        macos.MacOSHost,
        "bootstrap",
        lambda self, agent: started.append(agent.label),
    )

    MacOSDeployment(layout, DeployOptions(config=config)).deploy()

    assert started.index(LLAMA_SWAP_LABEL) < started.index(API_LABEL)


def test_deploy_installs_the_config_and_directories(
    layout, config, quiet_host, monkeypatch
) -> None:
    monkeypatch.setattr(macos.MacOSHost, "stop_agents", lambda self: None)
    monkeypatch.setattr(macos.installer, "install_all", lambda directory: None)
    monkeypatch.setattr(macos.MacOSHost, "bootstrap", lambda self, agent: None)

    MacOSDeployment(layout, DeployOptions(config=config)).deploy()

    assert layout.llama_swap_config.read_text() == "models: {}\n"
    assert layout.models.is_dir()
    assert layout.log.is_dir()


def test_deploy_prints_logs_when_a_service_fails(
    layout, config, quiet_host, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(macos.MacOSHost, "stop_agents", lambda self: None)
    monkeypatch.setattr(macos.installer, "install_all", lambda directory: None)
    monkeypatch.setattr(macos.MacOSHost, "bootstrap", lambda self, agent: None)

    def explode(self) -> None:
        layout.log_file("api").write_text("traceback: boom\n")
        raise RuntimeError("did not become healthy")

    monkeypatch.setattr(macos.MacOSDeployment, "_wait_for_health", explode)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        MacOSDeployment(layout, DeployOptions(config=config)).deploy()

    assert "traceback: boom" in capsys.readouterr().err


def test_stop_agents_covers_every_label(layout, monkeypatch) -> None:
    booted: list[str] = []
    monkeypatch.setattr(
        macos.MacOSHost, "bootout", lambda self, label: booted.append(label)
    )
    monkeypatch.setattr(macos.MacOSHost, "stop_model_servers", lambda self: None)

    macos.MacOSHost(layout).stop_agents()

    assert booted == list(ALL_LABELS)
    assert booted[-1] == LLAMA_SWAP_LABEL


def test_bootout_fails_when_an_agent_will_not_stop(layout, monkeypatch) -> None:
    monkeypatch.setattr(macos, "AGENT_STOP_TIMEOUT", 0.01)
    monkeypatch.setattr(
        macos.MacOSHost, "_launchctl", staticmethod(lambda *a, check: _ok())
    )

    with pytest.raises(RuntimeError, match="did not stop"):
        macos.MacOSHost(layout).bootout(API_LABEL)


def _ok():
    return subprocess.CompletedProcess(args=[], returncode=0)


def _absent():
    return subprocess.CompletedProcess(args=[], returncode=1)


def test_bootout_returns_once_the_agent_is_gone(layout, monkeypatch) -> None:
    monkeypatch.setattr(
        macos.MacOSHost, "_launchctl", staticmethod(lambda *a, check: _absent())
    )

    macos.MacOSHost(layout).bootout(API_LABEL)


def test_stop_model_servers_escalates_to_kill(layout, monkeypatch) -> None:
    monkeypatch.setattr(macos, "MODEL_SERVER_STOP_TIMEOUT", 0.01)
    signals: list[str] = []
    monkeypatch.setattr(
        macos.MacOSHost, "_model_servers_running", staticmethod(lambda pattern: True)
    )
    monkeypatch.setattr(
        macos.subprocess, "run", lambda cmd, **kw: signals.append(cmd[1])
    )

    with pytest.raises(RuntimeError, match="did not stop"):
        macos.MacOSHost(layout).stop_model_servers()

    assert signals == ["-TERM", "-KILL"]


def test_stop_model_servers_is_a_no_op_when_none_are_running(
    layout, monkeypatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        macos.MacOSHost, "_model_servers_running", staticmethod(lambda pattern: False)
    )
    monkeypatch.setattr(macos.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    macos.MacOSHost(layout).stop_model_servers()

    assert calls == []


def test_uninstall_removes_agents_and_runtimes(layout, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(macos.MacOSHost, "validate_host", staticmethod(lambda: None))
    monkeypatch.setattr(macos.MacOSHost, "deployment_lock", lambda self: _nullcontext())
    monkeypatch.setattr(macos.MacOSHost, "stop_agents", lambda self: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    agents_dir = tmp_path / "home/Library/LaunchAgents"
    agents_dir.mkdir(parents=True)
    for label in ALL_LABELS:
        (agents_dir / f"{label}.plist").write_bytes(b"")
    removed: list[str] = []
    monkeypatch.setattr(
        macos.installer, "remove", lambda runtime, directory: removed.append(runtime)
    )

    MacOSUninstaller(layout).uninstall()

    assert list(agents_dir.iterdir()) == []
    assert removed == ["llama-swap", "llama-cpp"]


def test_uninstall_keeps_models_and_config(layout, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(macos.MacOSHost, "validate_host", staticmethod(lambda: None))
    monkeypatch.setattr(macos.MacOSHost, "deployment_lock", lambda self: _nullcontext())
    monkeypatch.setattr(macos.MacOSHost, "stop_agents", lambda self: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(macos.installer, "remove", lambda runtime, directory: True)
    layout.models.mkdir(parents=True)
    layout.config.mkdir(parents=True)
    layout.llama_swap_config.write_text("models: {}\n")

    MacOSUninstaller(layout).uninstall()

    assert layout.models.is_dir()
    assert layout.llama_swap_config.is_file()


def test_validate_host_rejects_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(macos.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="requires macOS"):
        macos.MacOSHost.validate_host()


def test_validate_host_rejects_an_unsupported_architecture(monkeypatch) -> None:
    monkeypatch.setattr(macos.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(macos, "is_supported", lambda: False)
    monkeypatch.setattr(macos.platform, "machine", lambda: "ppc64")

    with pytest.raises(RuntimeError, match="ppc64"):
        macos.MacOSHost.validate_host()


def test_deploy_cli_forwards_every_option(tmp_path, config, monkeypatch) -> None:
    captured: list[tuple[Layout, DeployOptions]] = []
    deployed: list[bool] = []

    class Recorder:
        def __init__(self, layout: Layout, options: DeployOptions) -> None:
            captured.append((layout, options))

        def deploy(self) -> None:
            deployed.append(True)

    monkeypatch.setattr(macos, "MacOSDeployment", Recorder)

    result = CliRunner().invoke(
        macos.main,
        [
            "deploy",
            "--config",
            str(config),
            "--prefix",
            str(tmp_path / "prefix"),
            "--api-port",
            "9001",
            "--ui-port",
            "9000",
            "--llama-swap-port",
            "9002",
            "--api-host",
            "0.0.0.0",
            "--llama-swap-host",
            "0.0.0.0",
            "--no-ui",
        ],
    )

    assert result.exit_code == 0, result.output
    layout, options = captured[0]
    assert deployed == [True]
    assert layout.prefix == tmp_path / "prefix"
    assert options.api_port == 9001
    assert options.ui_port == 9000
    assert options.llama_swap_port == 9002
    assert options.api_host == "0.0.0.0"
    assert options.llama_swap_host == "0.0.0.0"
    assert options.with_ui is False


def test_deploy_cli_rejects_a_missing_config(tmp_path, monkeypatch) -> None:
    result = CliRunner().invoke(
        macos.main, ["deploy", "--config", str(tmp_path / "absent.yml")]
    )

    assert result.exit_code != 0


def test_stop_cli_stops_the_agents_without_removing_anything(
    tmp_path, monkeypatch
) -> None:
    """Upgrading replaces the venv the agents run from, so stop must come first."""
    stopped: list[Path] = []
    monkeypatch.setattr(macos.MacOSHost, "validate_host", staticmethod(lambda: None))
    monkeypatch.setattr(macos.MacOSHost, "deployment_lock", lambda self: _nullcontext())
    monkeypatch.setattr(
        macos.MacOSHost,
        "stop_agents",
        lambda self: stopped.append(self._layout.prefix),
    )
    removed: list[str] = []
    monkeypatch.setattr(
        macos.installer, "remove", lambda runtime, directory: removed.append(runtime)
    )

    result = CliRunner().invoke(
        macos.main, ["stop", "--prefix", str(tmp_path / "prefix")]
    )

    assert result.exit_code == 0, result.output
    assert stopped == [tmp_path / "prefix"]
    assert removed == []


def test_uninstall_cli_uses_the_resolved_prefix(tmp_path, monkeypatch) -> None:
    captured: list[Layout] = []
    uninstalled: list[bool] = []

    class Recorder:
        def __init__(self, layout: Layout) -> None:
            captured.append(layout)

        def uninstall(self) -> None:
            uninstalled.append(True)

    monkeypatch.setattr(macos, "MacOSUninstaller", Recorder)

    result = CliRunner().invoke(
        macos.main, ["uninstall", "--prefix", str(tmp_path / "prefix")]
    )

    assert result.exit_code == 0
    assert uninstalled == [True]
    assert captured[0].prefix == tmp_path / "prefix"


def test_hf_token_is_read_from_the_process_environment(config, monkeypatch) -> None:
    monkeypatch.setenv("SWAPBOARD_HF_TOKEN", "from-env")
    captured: list[DeployOptions] = []

    class Recorder:
        def __init__(self, layout: Layout, options: DeployOptions) -> None:
            captured.append(options)

        def deploy(self) -> None:
            pass

    monkeypatch.setattr(macos, "MacOSDeployment", Recorder)

    CliRunner().invoke(macos.main, ["deploy", "--config", str(config)])

    assert captured[0].hf_token == "from-env"


def test_wait_for_url_gives_up_after_the_timeout(monkeypatch) -> None:
    def refuse(url, timeout):
        raise OSError("refused")

    monkeypatch.setattr(macos, "urlopen", refuse)
    monkeypatch.setattr(macos.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        macos._wait_for_url("http://127.0.0.1:1/health", timeout=0.01)
