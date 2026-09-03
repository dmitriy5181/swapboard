import pytest

from swapboard.api import run


def test_defaults_bind_locally() -> None:
    arguments = run.build_parser().parse_args([])

    assert arguments.host == "127.0.0.1"
    assert arguments.port == 8771
    assert arguments.reload is False


def test_host_and_port_are_configurable() -> None:
    arguments = run.build_parser().parse_args(
        ["--host", "0.0.0.0", "--port", "9001", "--reload"]
    )

    assert arguments.host == "0.0.0.0"
    assert arguments.port == 9001
    assert arguments.reload is True


def test_non_numeric_port_is_rejected() -> None:
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["--port", "http"])


def test_main_runs_the_asgi_app(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeUvicorn:
        @staticmethod
        def run(target, **kwargs):
            captured["target"] = target
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", FakeUvicorn)

    run.main(["--port", "9002"])

    assert captured["target"] == "swapboard.api.main:app"
    assert captured["port"] == 9002
    assert captured["host"] == "127.0.0.1"
