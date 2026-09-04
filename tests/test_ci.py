import os
import subprocess
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"


def test_validate_deployment_masks_and_exports_default_paths(tmp_path: Path) -> None:
    environment = build_environment(tmp_path)
    prefix = Path(environment["HOME"]) / ".swapboard"
    prefix.mkdir(parents=True)
    (prefix / "llama-swap.yml").write_text("models: {}\n")

    result = run_step("Validate the deployment configuration", environment)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"::add-mask::{prefix}/llama-swap.yml",
        f"::add-mask::{prefix}/venv",
        f"::add-mask::{prefix}",
        f"::add-mask::{environment['HOME']}",
    ]
    assert read_environment(Path(environment["GITHUB_ENV"])) == {
        "INSTALL_PREFIX": str(prefix),
        "APP_VENV": str(prefix / "venv"),
        "SWAPBOARD_CONFIG_PATH": str(prefix / "llama-swap.yml"),
    }


def test_validate_deployment_uses_config_override(tmp_path: Path) -> None:
    environment = build_environment(tmp_path)
    config = tmp_path / "custom.yml"
    config.write_text("models: {}\n")
    environment["SWAPBOARD_CONFIG_PATH"] = str(config)

    result = run_step("Validate the deployment configuration", environment)

    assert result.returncode == 0, result.stderr
    exported = read_environment(Path(environment["GITHUB_ENV"]))
    assert exported["SWAPBOARD_CONFIG_PATH"] == str(config)


def test_validate_deployment_rejects_missing_config(tmp_path: Path) -> None:
    environment = build_environment(tmp_path)

    result = run_step("Validate the deployment configuration", environment)

    assert result.returncode == 1
    assert "No llama-swap config on this host" in result.stderr


def test_stop_step_passes_the_installation_prefix(tmp_path: Path) -> None:
    environment = build_environment(tmp_path)
    executable = tmp_path / "venv/bin/swapboard-deploy"
    calls = tmp_path / "calls"
    write_recorder(executable, calls)
    environment |= {
        "APP_VENV": str(tmp_path / "venv"),
        "INSTALL_PREFIX": str(tmp_path / "prefix"),
        "CALLS": str(calls),
    }

    result = run_step("Stop the running services", environment)

    assert result.returncode == 0, result.stderr
    assert calls.read_text() == f"stop --prefix {tmp_path}/prefix\n"


def test_install_step_installs_and_verifies_release(tmp_path: Path) -> None:
    environment = build_install_environment(tmp_path, version="2026.9.1")

    result = run_step("Install the published release", environment)

    assert result.returncode == 0, result.stderr
    calls = Path(environment["UV_CALLS"]).read_text()
    assert "pip install" in calls
    assert "swapboard[all]==2026.9.1" in calls


def test_install_step_fails_after_retry_exhaustion(tmp_path: Path) -> None:
    environment = build_install_environment(tmp_path, version="2026.9.1")
    write_executable(
        Path(environment["FAKE_BIN"]) / "uv",
        '#!/bin/sh\necho "$*" >>"$UV_CALLS"\nexit 1\n',
    )
    write_executable(Path(environment["FAKE_BIN"]) / "sleep", "#!/bin/sh\nexit 0\n")

    result = run_step("Install the published release", environment)

    assert result.returncode == 1
    assert Path(environment["UV_CALLS"]).read_text().count("pip install") == 6
    assert "is not installable from PyPI" in result.stderr


def test_install_step_rejects_the_wrong_version(tmp_path: Path) -> None:
    environment = build_install_environment(tmp_path, version="2026.9.0")
    environment["RELEASE_VERSION"] = "2026.9.1"

    result = run_step("Install the published release", environment)

    assert result.returncode == 1
    assert "Installed 2026.9.0, expected 2026.9.1" in result.stderr


def test_deploy_step_passes_prefix_and_config(tmp_path: Path) -> None:
    environment = build_environment(tmp_path)
    executable = tmp_path / "venv/bin/swapboard-deploy"
    calls = tmp_path / "calls"
    write_recorder(executable, calls)
    environment |= {
        "APP_VENV": str(tmp_path / "venv"),
        "INSTALL_PREFIX": str(tmp_path / "prefix"),
        "SWAPBOARD_CONFIG_PATH": str(tmp_path / "llama-swap.yml"),
        "CALLS": str(calls),
    }

    result = run_step("Deploy the launchd services", environment)

    assert result.returncode == 0, result.stderr
    assert calls.read_text() == (
        f"deploy --prefix {tmp_path}/prefix --config {tmp_path}/llama-swap.yml\n"
    )


def test_deploy_step_forwards_the_public_endpoint_variable() -> None:
    """Dropping this pass-through silently reverts the dashboard to localhost."""
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    deploy = next(
        step
        for step in workflow["jobs"]["deploy-macos"]["steps"]
        if step["name"] == "Deploy the launchd services"
    )

    assert (
        deploy["env"]["SWAPBOARD_PUBLIC_ENDPOINT_URL"]
        == "${{ vars.SWAPBOARD_PUBLIC_ENDPOINT_URL }}"
    )


def test_publish_uses_the_node_24_artifact_action() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    download = next(
        step
        for step in workflow["jobs"]["publish-pypi"]["steps"]
        if step["name"] == "Download distributions"
    )

    assert download["uses"] == "actions/download-artifact@v8"


def run_step(
    name: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-e"],
        input=step_script(name),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def step_script(name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    return next(
        step["run"]
        for step in workflow["jobs"]["deploy-macos"]["steps"]
        if step["name"] == name
    )


def build_environment(tmp_path: Path) -> dict[str, str]:
    github_environment = tmp_path / "github-env"
    github_environment.touch()
    return os.environ | {
        "HOME": str(tmp_path / "home"),
        "GITHUB_ENV": str(github_environment),
        "SWAPBOARD_CONFIG_PATH": "",
    }


def build_install_environment(tmp_path: Path, version: str) -> dict[str, str]:
    environment = build_environment(tmp_path)
    fake_bin = tmp_path / "bin"
    app_venv = tmp_path / "venv"
    calls = tmp_path / "uv-calls"
    write_python_stub(app_venv / "bin/python")
    write_uv_stub(fake_bin / "uv")
    environment |= {
        "APP_VENV": str(app_venv),
        "FAKE_BIN": str(fake_bin),
        "FAKE_VERSION": version,
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "RELEASE_VERSION": version,
        "UV_CALLS": str(calls),
    }
    return environment


def write_python_stub(path: Path) -> None:
    write_executable(
        path,
        """#!/bin/sh
case "$2" in
  *version_info*) exit 0 ;;
  *importlib.metadata*) echo "$FAKE_VERSION" ;;
esac
""",
    )


def write_uv_stub(path: Path) -> None:
    write_executable(path, '#!/bin/sh\necho "$*" >>"$UV_CALLS"\n')


def write_recorder(path: Path, calls: Path) -> None:
    write_executable(path, '#!/bin/sh\necho "$*" >>"$CALLS"\n')
    calls.touch()


def write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)


def read_environment(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())
