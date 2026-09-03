import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from swapboard import dev
from swapboard.common.paths import Layout


def test_llama_swap_command_binds_locally(tmp_path: Path) -> None:
    config = tmp_path / "llama-swap.yml"

    command = dev.build_llama_swap_command("/runtimes/llama-swap", config, 8772)

    assert command == [
        "/runtimes/llama-swap",
        "--config",
        str(config),
        "--listen",
        "127.0.0.1:8772",
    ]


def test_api_command_runs_in_the_current_interpreter() -> None:
    command = dev.build_api_command(8771, reload=False)

    assert command[:4] == [sys.executable, "-m", "uvicorn", "swapboard.api.main:app"]
    assert "--reload" not in command


def test_api_command_can_enable_reload() -> None:
    assert "--reload" in dev.build_api_command(8771, reload=True)


def test_ui_command_targets_the_ui_module() -> None:
    command = dev.build_ui_command(8770)

    assert command[:3] == [sys.executable, "-m", "swapboard.ui.run"]
    assert command[-2:] == ["--port", "8770"]


def test_environment_wires_llama_swap_to_the_resolved_binary(tmp_path: Path) -> None:
    layout = Layout(tmp_path)
    config = tmp_path / "llama-swap.yml"

    env = dev.build_environment(
        {"HOME": "/home/dev"},
        layout=layout,
        config=config,
        models_dir=tmp_path / "models",
        llama_server="/runtimes/llama-cpp/llama-server",
        api_port=8771,
        llama_swap_port=8772,
        hf_token=None,
    )

    assert env["HOME"] == "/home/dev"
    assert env["LLAMA_SERVER_BIN"] == "/runtimes/llama-cpp/llama-server"
    assert env["MODELS_DIR"] == str(tmp_path / "models")
    assert env["SWAPBOARD_MODELS_PATH"] == str(tmp_path / "models")
    assert env["SWAPBOARD_LLAMA_SWAP_CONFIG_PATH"] == str(config)
    assert env["SWAPBOARD_LLAMA_SWAP_PORT"] == "8772"
    assert env["SWAPBOARD_UI_API_URL"] == "http://127.0.0.1:8771"
    assert "SWAPBOARD_HF_TOKEN" not in env


def test_environment_includes_hf_token_when_given(tmp_path: Path) -> None:
    env = dev.build_environment(
        {},
        layout=Layout(tmp_path),
        config=tmp_path / "c.yml",
        models_dir=tmp_path,
        llama_server="llama-server",
        api_port=8771,
        llama_swap_port=8772,
        hf_token="token",
    )

    assert env["SWAPBOARD_HF_TOKEN"] == "token"


def test_read_env_file_parses_pairs_and_skips_noise(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "SWAPBOARD_HF_TOKEN=token\n"
        'SWAPBOARD_MODELS_PATH="/models"\n'
        "SWAPBOARD_UI_PORT = 8770 \n"
        "not-a-pair\n",
        encoding="utf-8",
    )

    values = dev.read_env_file(env_file)

    assert values == {
        "SWAPBOARD_HF_TOKEN": "token",
        "SWAPBOARD_MODELS_PATH": "/models",
        "SWAPBOARD_UI_PORT": "8770",
    }


def test_read_env_file_tolerates_a_missing_file(tmp_path: Path) -> None:
    assert dev.read_env_file(tmp_path / "absent") == {}


def test_resolve_runtime_prefers_an_explicit_override(tmp_path: Path) -> None:
    resolved = dev.resolve_runtime(
        "llama-swap", "llama-swap", Layout(tmp_path), tmp_path / "custom"
    )

    assert resolved == str(tmp_path / "custom")


def test_resolve_runtime_installs_the_pinned_build(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dev, "is_supported", lambda: True)
    monkeypatch.setattr(
        dev.installer, "install", lambda runtime, directory: directory / runtime / "bin"
    )
    layout = Layout(tmp_path)

    resolved = dev.resolve_runtime("llama-swap", "llama-swap", layout, None)

    assert resolved == str(layout.runtimes / "llama-swap" / "bin")


def test_resolve_runtime_falls_back_to_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dev, "is_supported", lambda: False)
    monkeypatch.setattr(dev.shutil, "which", lambda binary: f"/usr/local/bin/{binary}")

    resolved = dev.resolve_runtime("llama-swap", "llama-swap", Layout(tmp_path), None)

    assert resolved == "/usr/local/bin/llama-swap"


def test_resolve_runtime_reports_a_missing_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dev, "is_supported", lambda: False)
    monkeypatch.setattr(dev.shutil, "which", lambda binary: None)

    with pytest.raises(Exception, match="was not found on PATH"):
        dev.resolve_runtime("llama-swap", "llama-swap", Layout(tmp_path), None)


def test_cli_rejects_a_missing_config(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        dev.main, ["--config", str(tmp_path / "absent.yml"), "--prefix", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "config not found" in result.output


def test_cli_exposes_the_documented_options() -> None:
    output = CliRunner().invoke(dev.main, ["--help"]).output

    for option in (
        "--config",
        "--prefix",
        "--models-dir",
        "--api-port",
        "--ui-port",
        "--llama-swap-port",
        "--llama-swap-bin",
        "--hf-token",
        "--env-file",
        "--no-llama-swap",
        "--no-ui",
        "--reload",
    ):
        assert option in output


def test_process_group_reports_the_first_exit() -> None:
    group = dev.ProcessGroup()
    group.start("quick", [sys.executable, "-c", "raise SystemExit(3)"], {})

    assert group.wait() == 3
