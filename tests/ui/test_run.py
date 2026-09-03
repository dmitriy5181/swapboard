from click.testing import CliRunner

from swapboard.ui import run


class FakeApp:
    def __init__(self) -> None:
        self.called_with: dict[str, object] = {}

    def run(self, **kwargs) -> None:
        self.called_with = kwargs


def test_defaults_come_from_settings(monkeypatch) -> None:
    monkeypatch.delenv("SWAPBOARD_UI_HOST", raising=False)
    monkeypatch.delenv("SWAPBOARD_UI_PORT", raising=False)
    app = FakeApp()
    monkeypatch.setattr(run, "create_app", lambda: app)

    result = CliRunner().invoke(run.main, [])

    assert result.exit_code == 0
    assert app.called_with == {"host": "127.0.0.1", "port": 8770, "debug": False}


def test_command_line_overrides_settings(monkeypatch) -> None:
    app = FakeApp()
    monkeypatch.setattr(run, "create_app", lambda: app)

    result = CliRunner().invoke(
        run.main, ["--host", "0.0.0.0", "--port", "9000", "--debug"]
    )

    assert result.exit_code == 0
    assert app.called_with == {"host": "0.0.0.0", "port": 9000, "debug": True}


def test_settings_are_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("SWAPBOARD_UI_HOST", "0.0.0.0")
    monkeypatch.setenv("SWAPBOARD_UI_PORT", "9100")
    app = FakeApp()
    monkeypatch.setattr(run, "create_app", lambda: app)

    CliRunner().invoke(run.main, [])

    assert app.called_with["host"] == "0.0.0.0"
    assert app.called_with["port"] == 9100
