from swapboard.api.settings import Settings


def test_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("SWAPBOARD_LLAMA_SWAP_CONFIG_PATH", "/tmp/config.yml")
    monkeypatch.setenv("SWAPBOARD_LLAMA_SWAP_PORT", "9772")
    monkeypatch.setenv("SWAPBOARD_MODELS_PATH", "/tmp/models")
    monkeypatch.setenv("SWAPBOARD_HF_TOKEN", "token")

    settings = Settings()

    assert settings.llama_swap_config_path == "/tmp/config.yml"
    assert settings.llama_swap_port == 9772
    assert settings.models_path == "/tmp/models"
    assert settings.hf_token == "token"


def test_settings_falls_back_to_defaults(monkeypatch) -> None:
    for name in (
        "SWAPBOARD_LLAMA_SWAP_CONFIG_PATH",
        "SWAPBOARD_LLAMA_SWAP_PORT",
        "SWAPBOARD_MODELS_PATH",
        "SWAPBOARD_HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llama_swap_config_path == "/etc/llama-swap/config/config.yaml"
    assert settings.llama_swap_port == 8080
    assert settings.models_path == "/models"
    assert settings.hf_token is None


def test_settings_normalizes_blank_hf_token(monkeypatch) -> None:
    monkeypatch.setenv("SWAPBOARD_HF_TOKEN", "  ")

    settings = Settings()

    assert settings.hf_token is None


def test_settings_accepts_keyword_arguments() -> None:
    settings = Settings(
        llama_swap_config_path="/tmp/config.yml",
        llama_swap_port=1234,
        models_path="/tmp/models",
        hf_token=None,
    )

    assert settings.llama_swap_port == 1234
