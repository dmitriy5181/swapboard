from swapboard.api.settings import Settings
from swapboard.common.network import DEFAULT_LLAMA_SWAP_PORT


def test_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("SWAPBOARD_LLAMA_SWAP_CONFIG_PATH", "/tmp/config.yml")
    monkeypatch.setenv("SWAPBOARD_LLAMA_SWAP_PORT", "9772")
    monkeypatch.setenv("SWAPBOARD_MODELS_PATH", "/tmp/models")
    monkeypatch.setenv("SWAPBOARD_HF_TOKEN", "token")
    monkeypatch.setenv("SWAPBOARD_PUBLIC_ENDPOINT_URL", "  https://inference.test/v1  ")

    settings = Settings()

    assert settings.llama_swap_config_path == "/tmp/config.yml"
    assert settings.llama_swap_port == 9772
    assert settings.models_path == "/tmp/models"
    assert settings.hf_token == "token"
    assert settings.public_endpoint_url == "https://inference.test/v1"


def _clear_settings_env(monkeypatch) -> None:
    for name in (
        "SWAPBOARD_LLAMA_SWAP_CONFIG_PATH",
        "SWAPBOARD_LLAMA_SWAP_PORT",
        "SWAPBOARD_MODELS_PATH",
        "SWAPBOARD_HF_TOKEN",
        "SWAPBOARD_PUBLIC_ENDPOINT_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_defaults_to_the_layout_of_its_own_installation(
    monkeypatch, tmp_path
) -> None:
    """An installed API must find its own config and models unconfigured."""
    _clear_settings_env(monkeypatch)
    monkeypatch.setattr("sys.prefix", str(tmp_path / "venv"))

    settings = Settings(_env_file=None)

    assert settings.llama_swap_config_path == str(tmp_path / "config/llama-swap.yml")
    assert settings.models_path == str(tmp_path / "models")


def test_settings_falls_back_to_defaults(monkeypatch) -> None:
    _clear_settings_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.llama_swap_port == DEFAULT_LLAMA_SWAP_PORT
    assert settings.hf_token is None


def test_settings_normalizes_blank_hf_token(monkeypatch) -> None:
    monkeypatch.setenv("SWAPBOARD_HF_TOKEN", "  ")

    settings = Settings()

    assert settings.hf_token is None


def test_settings_normalizes_blank_public_endpoint_url(monkeypatch) -> None:
    """An unset workflow variable arrives as an empty string, not as absent."""
    monkeypatch.setenv("SWAPBOARD_PUBLIC_ENDPOINT_URL", "")

    settings = Settings()

    assert settings.public_endpoint_url is None


def test_settings_accepts_keyword_arguments() -> None:
    settings = Settings(
        llama_swap_config_path="/tmp/config.yml",
        llama_swap_port=1234,
        models_path="/tmp/models",
        hf_token=None,
    )

    assert settings.llama_swap_port == 1234
