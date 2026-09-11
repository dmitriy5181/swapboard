from pathlib import Path

import pytest

from swapboard.api.configfile import ConfigFile

VALID = """
models:
  llama-3:
    cmd: llama-server -m /models/acme/llama-3-GGUF/llama-3-Q4_K_M.gguf
"""


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "llama-swap.yml"
    path.write_text(VALID, encoding="utf-8")
    return path


def backup_of(config: Path) -> Path:
    return config.with_name(f"{config.name}.bak")


def test_shipped_example_config_is_accepted() -> None:
    """The config a fresh install starts from has to survive its own editor."""
    example = Path(__file__).resolve().parents[2] / "llama-swap.example.yml"

    validation = ConfigFile(example).validate(example.read_text(encoding="utf-8"))

    assert validation.valid is True
    assert validation.errors == []


def test_read_returns_the_file_verbatim(config: Path) -> None:
    assert ConfigFile(config).read() == VALID


def test_valid_config_is_written_and_backed_up(config: Path) -> None:
    replacement = VALID.replace("llama-3", "llama-4")

    validation = ConfigFile(config).write(replacement)

    assert validation.valid is True
    assert config.read_text(encoding="utf-8") == replacement
    assert backup_of(config).read_text(encoding="utf-8") == VALID


def test_invalid_yaml_is_rejected_and_changes_nothing(config: Path) -> None:
    validation = ConfigFile(config).write("models: [unclosed")

    assert validation.valid is False
    assert "Invalid YAML" in validation.errors[0]
    assert config.read_text(encoding="utf-8") == VALID
    assert not backup_of(config).exists()


def test_empty_config_is_rejected_in_its_own_words(config: Path) -> None:
    validation = ConfigFile(config).write("   \n")

    assert validation.errors == ["The configuration is empty."]
    assert config.read_text(encoding="utf-8") == VALID


def test_model_without_a_command_is_rejected_with_its_path(config: Path) -> None:
    validation = ConfigFile(config).write("models:\n  llama-3:\n    ttl: 30\n")

    assert validation.valid is False
    assert validation.errors == ["models/llama-3: 'cmd' is a required property"]
    assert config.read_text(encoding="utf-8") == VALID


def test_config_without_models_is_rejected(config: Path) -> None:
    validation = ConfigFile(config).write("healthCheckTimeout: 90\n")

    assert validation.valid is False
    assert "'models' is a required property" in validation.errors[0]


def test_port_macro_in_a_proxy_url_is_accepted(config: Path) -> None:
    """`format: uri` would reject the macro llama-swap's own docs prescribe."""
    text = (
        "models:\n"
        "  llama-3:\n"
        "    cmd: llama-server -m /models/acme/repo/llama.gguf\n"
        '    proxy: "http://localhost:${PORT}"\n'
    )

    assert ConfigFile(config).write(text).valid is True


def test_unknown_model_option_is_accepted(config: Path) -> None:
    """A newer llama-swap must not be locked out by an older vendored schema."""
    text = (
        "models:\n"
        "  llama-3:\n"
        "    cmd: llama-server -m /models/acme/repo/llama.gguf\n"
        "    optionFromALaterRelease: true\n"
    )

    assert ConfigFile(config).write(text).valid is True


def test_unmanageable_model_saves_with_a_warning(config: Path) -> None:
    """A path swapboard cannot trace would otherwise vanish without explanation."""
    text = "models:\n  llama-3:\n    cmd: llama-server -m /opt/llama.gguf\n"

    validation = ConfigFile(config).write(text)

    assert validation.valid is True
    assert config.read_text(encoding="utf-8") == text
    assert len(validation.warnings) == 1
    assert "llama-3" in validation.warnings[0]


def test_manageable_model_saves_without_warnings(config: Path) -> None:
    assert ConfigFile(config).write(VALID).warnings == []
