from pathlib import Path

from click.testing import CliRunner

from swapboard.runtimes import cli, installer, manifest


def invoke(*arguments: str):
    return CliRunner().invoke(cli.main, list(arguments))


def test_install_explains_an_unsupported_platform(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manifest, "current_platform", lambda: ("linux", "x86_64"))

    result = invoke("install", "--prefix", str(tmp_path))

    assert result.exit_code != 0
    assert "Managed runtimes are available on macOS" in result.output
    # A ClickException, not an unhandled error surfacing as a traceback.
    assert isinstance(result.exception, SystemExit)


def test_status_explains_an_unsupported_platform(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manifest, "current_platform", lambda: ("linux", "x86_64"))

    result = invoke("status", "--prefix", str(tmp_path))

    assert result.exit_code != 0
    assert "Managed runtimes are available on macOS" in result.output
    assert isinstance(result.exception, SystemExit)


def test_install_reports_each_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        installer,
        "install",
        lambda runtime, directory, force=False: directory / runtime / "bin",
    )

    result = invoke("install", "--prefix", str(tmp_path))

    assert result.exit_code == 0
    assert "llama-cpp" in result.output
    assert "llama-swap" in result.output


def test_install_only_targets_one_runtime(tmp_path: Path, monkeypatch) -> None:
    requested: list[str] = []
    monkeypatch.setattr(
        installer,
        "install",
        lambda runtime, directory, force=False: (
            requested.append(runtime) or directory / runtime
        ),
    )

    result = invoke("install", "--prefix", str(tmp_path), "--only", "llama-swap")

    assert result.exit_code == 0
    assert requested == ["llama-swap"]


def test_install_rejects_an_unknown_runtime(tmp_path: Path) -> None:
    result = invoke("install", "--prefix", str(tmp_path), "--only", "nope")

    assert result.exit_code != 0


def test_install_forwards_the_force_flag(tmp_path: Path, monkeypatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(
        installer,
        "install",
        lambda runtime, directory, force=False: (
            seen.append(force) or directory / runtime
        ),
    )

    invoke("install", "--prefix", str(tmp_path), "--force")

    assert seen == [True, True]


def test_status_reports_missing_runtimes(tmp_path: Path) -> None:
    result = invoke("status", "--prefix", str(tmp_path))

    assert result.exit_code == 0
    assert result.output.count("not installed") == 2


def test_status_reports_current_runtimes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(installer, "installed_version", lambda runtime, directory: "9")
    monkeypatch.setattr(installer, "is_current", lambda runtime, directory: True)

    result = invoke("status", "--prefix", str(tmp_path))

    assert result.output.count("(current)") == 2


def test_status_reports_an_outdated_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(installer, "installed_version", lambda runtime, directory: "1")
    monkeypatch.setattr(installer, "is_current", lambda runtime, directory: False)

    result = invoke("status", "--prefix", str(tmp_path))

    assert "pinned" in result.output
    assert "(current)" not in result.output


def test_path_prints_an_installed_binary(tmp_path: Path) -> None:
    binary = tmp_path / "runtimes/llama-cpp/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("")

    result = invoke("path", "--prefix", str(tmp_path), "llama-server")

    assert result.exit_code == 0
    assert result.output.strip() == str(binary)


def test_path_fails_when_not_installed(tmp_path: Path) -> None:
    result = invoke("path", "--prefix", str(tmp_path), "llama-swap")

    assert result.exit_code != 0
    assert "is not installed" in result.output


def test_remove_reports_what_was_deleted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        installer, "remove", lambda runtime, directory: runtime == "llama-swap"
    )

    result = invoke("remove", "--prefix", str(tmp_path))

    assert "Removed llama-swap" in result.output
    assert "llama-cpp was not installed" in result.output


def test_prefix_defaults_to_the_virtualenv_parent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("sys.prefix", str(tmp_path / "venv"))
    seen: list[Path] = []
    monkeypatch.setattr(
        installer,
        "install",
        lambda runtime, directory, force=False: (
            seen.append(directory) or directory / runtime
        ),
    )

    invoke("install")

    assert seen[0] == tmp_path / "runtimes"
