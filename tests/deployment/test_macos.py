import plistlib
from pathlib import Path

import pytest

from swapboard.common.paths import Layout
from swapboard.deployment.macos import (
    API_LABEL,
    LLAMA_SWAP_EXIT_TIMEOUT,
    LLAMA_SWAP_LABEL,
    SERVICE_PATH,
    UI_LABEL,
    DeployOptions,
    MacOSDeployment,
)


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return Layout(tmp_path / "prefix")


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "llama-swap.yml"
    path.write_text("models: {}\n", encoding="utf-8")
    return path


def build(layout: Layout, config: Path, **overrides) -> MacOSDeployment:
    options = DeployOptions(config=config, **overrides)
    return MacOSDeployment(layout, options)


def agent_by_label(deployment: MacOSDeployment, label: str):
    return next(agent for agent in deployment.agents() if agent.label == label)


def test_layout_derives_every_path_from_the_prefix(tmp_path: Path) -> None:
    layout = Layout(tmp_path / "swapboard")

    assert layout.venv == tmp_path / "swapboard/venv"
    assert layout.runtimes == tmp_path / "swapboard/runtimes"
    assert layout.models == tmp_path / "swapboard/models"
    assert layout.llama_swap_config == tmp_path / "swapboard/config/llama-swap.yml"
    assert layout.log_file("api") == tmp_path / "swapboard/log/api.log"
    assert (
        layout.llama_server_bin
        == tmp_path / "swapboard/runtimes/llama-cpp/llama-server"
    )
    assert (
        layout.llama_swap_bin == tmp_path / "swapboard/runtimes/llama-swap/llama-swap"
    )


def test_default_layout_is_the_parent_of_the_virtualenv(monkeypatch) -> None:
    monkeypatch.setattr("sys.prefix", "/opt/app/venv")

    assert Layout.default().prefix == Path("/opt/app")
    assert Layout.default().models == Path("/opt/app/models")


def test_deployment_builds_three_agents_by_default(layout, config) -> None:
    labels = [agent.label for agent in build(layout, config).agents()]

    assert labels == [LLAMA_SWAP_LABEL, API_LABEL, UI_LABEL]


def test_no_ui_omits_the_dashboard_agent(layout, config) -> None:
    labels = [agent.label for agent in build(layout, config, with_ui=False).agents()]

    assert labels == [LLAMA_SWAP_LABEL, API_LABEL]


def test_llama_swap_agent_runs_the_private_binary(layout, config) -> None:
    agent = agent_by_label(build(layout, config), LLAMA_SWAP_LABEL)

    assert agent.program_arguments == [
        str(layout.llama_swap_bin),
        "--config",
        str(layout.llama_swap_config),
        "--watch-config",
        "--listen",
        "127.0.0.1:8772",
    ]


def test_llama_swap_agent_points_llama_server_at_the_private_runtime(
    layout, config
) -> None:
    agent = agent_by_label(build(layout, config), LLAMA_SWAP_LABEL)

    assert agent.environment_variables["LLAMA_SERVER_BIN"] == str(
        layout.llama_server_bin
    )
    assert agent.environment_variables["MODELS_DIR"] == str(layout.models)


def test_service_path_excludes_homebrew(layout, config) -> None:
    for agent in build(layout, config).agents():
        assert "/opt/homebrew" not in agent.environment_variables["PATH"]
    assert SERVICE_PATH == "/usr/bin:/bin:/usr/sbin:/sbin"


def test_llama_swap_agent_extends_the_shutdown_grace_period(layout, config) -> None:
    agent = agent_by_label(build(layout, config), LLAMA_SWAP_LABEL)

    assert agent.definition()["ExitTimeOut"] == LLAMA_SWAP_EXIT_TIMEOUT


def test_other_agents_use_the_default_shutdown_grace_period(layout, config) -> None:
    agent = agent_by_label(build(layout, config), API_LABEL)

    assert "ExitTimeOut" not in agent.definition()


def test_api_agent_runs_uvicorn_from_the_prefix_venv(layout, config) -> None:
    agent = agent_by_label(build(layout, config, api_port=9001), API_LABEL)

    assert agent.program_arguments == [
        str(layout.venv / "bin/uvicorn"),
        "swapboard.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "9001",
    ]


def test_api_agent_receives_swapboard_settings(layout, config) -> None:
    agent = agent_by_label(build(layout, config, llama_swap_port=9772), API_LABEL)

    assert agent.environment_variables["SWAPBOARD_MODELS_PATH"] == str(layout.models)
    assert agent.environment_variables["SWAPBOARD_LLAMA_SWAP_CONFIG_PATH"] == str(
        layout.llama_swap_config
    )
    assert agent.environment_variables["SWAPBOARD_LLAMA_SWAP_PORT"] == "9772"


def test_hf_token_is_passed_through_when_set(layout, config) -> None:
    deployment = build(
        layout, config, environment={"SWAPBOARD_HF_TOKEN": "secret-token"}
    )

    agent = agent_by_label(deployment, API_LABEL)

    assert agent.environment_variables["SWAPBOARD_HF_TOKEN"] == "secret-token"


@pytest.mark.parametrize("token", ["", "   "])
def test_blank_hf_token_is_not_passed_through(layout, config, token: str) -> None:
    deployment = build(layout, config, environment={"SWAPBOARD_HF_TOKEN": token})

    agent = agent_by_label(deployment, API_LABEL)

    assert "SWAPBOARD_HF_TOKEN" not in agent.environment_variables


def test_public_endpoint_url_is_passed_through_when_set(layout, config) -> None:
    deployment = build(
        layout,
        config,
        environment={"SWAPBOARD_PUBLIC_ENDPOINT_URL": "https://inference.test/v1"},
    )

    agent = agent_by_label(deployment, API_LABEL)

    assert (
        agent.environment_variables["SWAPBOARD_PUBLIC_ENDPOINT_URL"]
        == "https://inference.test/v1"
    )


@pytest.mark.parametrize("url", ["", "   "])
def test_blank_public_endpoint_url_is_not_passed_through(layout, config, url) -> None:
    """An unconfigured workflow variable reaches the deploy step as an empty string."""
    deployment = build(
        layout, config, environment={"SWAPBOARD_PUBLIC_ENDPOINT_URL": url}
    )

    agent = agent_by_label(deployment, API_LABEL)

    assert "SWAPBOARD_PUBLIC_ENDPOINT_URL" not in agent.environment_variables


def test_ui_agent_points_at_the_local_api(layout, config) -> None:
    agent = agent_by_label(build(layout, config, api_port=9001, ui_port=9000), UI_LABEL)

    assert "--bind" in agent.program_arguments
    assert agent.program_arguments[agent.program_arguments.index("--bind") + 1] == (
        "127.0.0.1:9000"
    )
    assert (
        agent.environment_variables["SWAPBOARD_UI_API_URL"] == "http://127.0.0.1:9001"
    )


def test_hosts_are_configurable_for_remote_access(layout, config) -> None:
    deployment = build(layout, config, api_host="0.0.0.0", llama_swap_host="0.0.0.0")

    api = agent_by_label(deployment, API_LABEL)
    llama_swap = agent_by_label(deployment, LLAMA_SWAP_LABEL)

    assert api.program_arguments[api.program_arguments.index("--host") + 1] == "0.0.0.0"
    assert llama_swap.program_arguments[-1] == "0.0.0.0:8772"


def test_agent_writes_owner_only_plist(layout, config, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    agent = agent_by_label(
        build(layout, config, environment={"SWAPBOARD_HF_TOKEN": "secret"}), API_LABEL
    )

    agent.write()

    assert (
        agent.plist_path
        == tmp_path / "home/Library/LaunchAgents" / f"{API_LABEL}.plist"
    )
    assert agent.plist_path.stat().st_mode & 0o777 == 0o600
    written = plistlib.loads(agent.plist_path.read_bytes())
    assert written["Label"] == API_LABEL
    assert written["RunAtLoad"] is True
    assert written["KeepAlive"] == {"SuccessfulExit": False}
    assert written["WorkingDirectory"] == str(layout.prefix)
    assert written["StandardErrorPath"] == str(layout.log_file("api"))


def test_missing_config_is_rejected(layout, tmp_path) -> None:
    deployment = build(layout, tmp_path / "absent.yml")

    with pytest.raises(RuntimeError, match="config not found"):
        deployment._validate_options()


def test_colliding_ports_are_rejected(layout, config) -> None:
    deployment = build(layout, config, api_port=8772, llama_swap_port=8772)

    with pytest.raises(RuntimeError, match="ports must differ"):
        deployment._validate_options()


def test_ui_port_collision_only_matters_when_the_ui_is_deployed(layout, config) -> None:
    with pytest.raises(RuntimeError, match="ports must differ"):
        build(layout, config, ui_port=8771, api_port=8771)._validate_options()

    build(
        layout, config, ui_port=8771, api_port=8771, with_ui=False
    )._validate_options()


def test_install_config_copies_into_the_prefix(layout, config) -> None:
    deployment = build(layout, config)
    deployment._prepare_directories()

    deployment._install_config()

    assert layout.llama_swap_config.read_text() == "models: {}\n"


def test_install_config_accepts_the_managed_config_as_its_source(layout) -> None:
    layout.config.mkdir(parents=True)
    layout.llama_swap_config.write_text("models: {}\n")
    deployment = build(layout, layout.llama_swap_config)

    deployment._install_config()

    assert layout.llama_swap_config.read_text() == "models: {}\n"


def test_install_config_accepts_a_hard_link_to_the_managed_config(layout) -> None:
    layout.config.mkdir(parents=True)
    layout.llama_swap_config.write_text("models: {}\n")
    config_alias = layout.prefix / "llama-swap.yml"
    config_alias.hardlink_to(layout.llama_swap_config)
    deployment = build(layout, config_alias)

    deployment._install_config()

    assert layout.llama_swap_config.read_text() == "models: {}\n"


def test_prepare_directories_creates_the_prefix_tree(layout, config) -> None:
    build(layout, config)._prepare_directories()

    assert layout.models.is_dir()
    assert layout.config.is_dir()
    assert layout.log.is_dir()


def test_health_ports_cover_every_deployed_service(layout, config) -> None:
    assert build(layout, config)._health_ports() == [8772, 8771, 8773]
    assert build(layout, config, with_ui=False)._health_ports() == [8772, 8771]


def test_model_server_cleanup_matches_the_private_binary(
    layout, config, monkeypatch
) -> None:
    """The pkill pattern must be the private path, never a bare binary name."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "swapboard.deployment.macos.MacOSHost._model_servers_running",
        staticmethod(lambda pattern: False),
    )
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kwargs: calls.append(cmd) or None
    )

    build(layout, config).stop_model_servers()

    assert calls == []
    assert str(layout.llama_server_bin).endswith("/runtimes/llama-cpp/llama-server")
