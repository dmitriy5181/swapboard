import threading
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from swapboard.api.service import Downloads, ModelsService
from swapboard.api.settings import Settings
from swapboard.common.models import DownloadState

DEFAULT_CONFIG = """\
models:
  embeddinggemma-300M:
    cmd: |
      llama-server --port ${PORT}
      -m /models/ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf
      --embeddings
"""

MULTIMODAL_CONFIG = """\
models:
  qwen3.5-4b-q4_k_m:
    cmd: |
      llama-server --port ${PORT}
      -m /models/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf
      --mmproj /models/unsloth/Qwen3.5-4B-GGUF/mmproj-F16.gguf
"""


def build_settings(
    tmp_models: Path,
    *,
    llama_swap_port: int = 8080,
    hf_token: str | None = None,
    config: Path | None = None,
    public_endpoint_url: str | None = None,
) -> Settings:
    config = config or write_default_config(tmp_models)
    return Settings(
        llama_swap_config_path=str(config),
        llama_swap_port=llama_swap_port,
        models_path=str(tmp_models),
        hf_token=hf_token,
        public_endpoint_url=public_endpoint_url,
    )


def build_client(
    tmp_models: Path,
    llama_swap_port: int = 8080,
    *,
    config: Path | None = None,
    public_endpoint_url: str | None = None,
) -> TestClient:
    settings = build_settings(
        tmp_models,
        llama_swap_port=llama_swap_port,
        config=config,
        public_endpoint_url=public_endpoint_url,
    )

    import swapboard.api.main as main

    main.settings = settings
    main.service = ModelsService(settings)
    return TestClient(main.app)


def write_default_config(directory: Path) -> Path:
    config = directory / "default.yml"
    config.write_text(DEFAULT_CONFIG, encoding="utf-8")
    return config


def write_multimodal_config(directory: Path) -> Path:
    config = directory / "multimodal.yml"
    config.write_text(MULTIMODAL_CONFIG, encoding="utf-8")
    return config


def test_health_returns_ok(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_reports_missing(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    names = {model["name"] for model in body}
    assert "embeddinggemma-300M" in names
    assert all(model["present"] is False for model in body)
    assert all(model["download_state"] == "idle" for model in body)


def test_info_returns_configured_port(tmp_path: Path) -> None:
    client = build_client(tmp_path, llama_swap_port=9090)

    response = client.get("/info")

    assert response.status_code == 200
    assert response.json() == {"port": 9090, "endpoint_url": None}


def test_info_reports_the_public_endpoint_when_configured(tmp_path: Path) -> None:
    """Behind a reverse proxy the address users need is not the local one."""
    client = build_client(tmp_path, public_endpoint_url="https://inference.test/v1")

    response = client.get("/info")

    assert response.json()["endpoint_url"] == "https://inference.test/v1"


def test_get_unknown_model_returns_404(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/models/does-not-exist")

    assert response.status_code == 404


def test_downloads_try_begin_allows_only_one_winner() -> None:
    downloads = Downloads()

    assert downloads.try_begin("model") is True
    assert downloads.try_begin("model") is False


def test_download_unknown_model_returns_404(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/models/does-not-exist/download")

    assert response.status_code == 404


def test_download_present_model_does_not_start(tmp_path: Path) -> None:
    model_dir = tmp_path / "ggml-org" / "embeddinggemma-300M-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "embeddinggemma-300M-Q8_0.gguf").write_bytes(b"fake-gguf")
    client = build_client(tmp_path)

    response = client.post("/models/embeddinggemma-300M/download")

    assert response.status_code == 200
    assert response.json() == {"started": False, "message": "Model already present"}


def test_present_when_file_exists(tmp_path: Path) -> None:
    model_dir = tmp_path / "ggml-org" / "embeddinggemma-300M-GGUF"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "embeddinggemma-300M-Q8_0.gguf"
    model_path.write_bytes(b"fake-gguf")
    client = build_client(tmp_path)

    response = client.get("/models/embeddinggemma-300M")

    assert response.status_code == 200
    body = response.json()
    assert body["repo_id"] == "ggml-org/embeddinggemma-300M-GGUF"
    assert body["present"] is True
    assert body["path"] == str(model_path)


def test_empty_file_does_not_count_as_present(tmp_path: Path) -> None:
    model_dir = tmp_path / "ggml-org" / "embeddinggemma-300M-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "embeddinggemma-300M-Q8_0.gguf").write_bytes(b"")
    client = build_client(tmp_path)

    assert client.get("/models/embeddinggemma-300M").json()["present"] is False


def test_multimodal_model_requires_projector(tmp_path: Path) -> None:
    model_dir = tmp_path / "unsloth" / "Qwen3.5-4B-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"fake-model")
    client = build_client(tmp_path, config=write_multimodal_config(tmp_path))

    missing_projector = client.get("/models/qwen3.5-4b-q4_k_m")
    (model_dir / "mmproj-F16.gguf").write_bytes(b"fake-projector")
    complete = client.get("/models/qwen3.5-4b-q4_k_m")

    assert missing_projector.json()["present"] is False
    assert complete.json()["present"] is True


def test_download_fetches_only_missing_projector(tmp_path: Path) -> None:
    model_dir = tmp_path / "unsloth" / "Qwen3.5-4B-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"fake-model")
    service = ModelsService(
        build_settings(
            tmp_path,
            hf_token="token",
            config=write_multimodal_config(tmp_path),
        )
    )

    def download_file(**kwargs: str | None) -> str:
        target = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        target.write_bytes(b"fake-projector")
        return str(target)

    with (
        patch(
            "swapboard.api.service.hf_hub_download",
            side_effect=download_file,
        ) as download,
        patch.object(threading.Thread, "start", lambda thread: thread.run()),
    ):
        outcome = service.start_download("qwen3.5-4b-q4_k_m")

    status = service.get_status("qwen3.5-4b-q4_k_m")
    assert outcome.started is True
    download.assert_called_once_with(
        repo_id="unsloth/Qwen3.5-4B-GGUF",
        filename="mmproj-F16.gguf",
        local_dir=str(model_dir),
        token="token",
    )
    assert status is not None
    assert status.present is True
    assert status.download_state == DownloadState.COMPLETED


def test_download_retries_failed_projector_without_fetching_primary_again(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "unsloth" / "Qwen3.5-4B-GGUF"
    model_dir.mkdir(parents=True)
    service = ModelsService(
        build_settings(tmp_path, config=write_multimodal_config(tmp_path))
    )
    requested_filenames: list[str] = []

    def download_file(**kwargs: str | None) -> str:
        filename = str(kwargs["filename"])
        requested_filenames.append(filename)
        if filename == "mmproj-F16.gguf" and requested_filenames.count(filename) == 1:
            raise RuntimeError("projector download failed")
        target = Path(str(kwargs["local_dir"])) / filename
        target.write_bytes(b"fake-gguf")
        return str(target)

    with (
        patch(
            "swapboard.api.service.hf_hub_download",
            side_effect=download_file,
        ),
        patch.object(threading.Thread, "start", lambda thread: thread.run()),
    ):
        first_outcome = service.start_download("qwen3.5-4b-q4_k_m")
        failed_status = service.get_status("qwen3.5-4b-q4_k_m")
        retry_outcome = service.start_download("qwen3.5-4b-q4_k_m")

    completed_status = service.get_status("qwen3.5-4b-q4_k_m")
    assert first_outcome.started is True
    assert failed_status is not None
    assert failed_status.download_state == DownloadState.FAILED
    assert failed_status.download_error == "projector download failed"
    assert retry_outcome.started is True
    assert requested_filenames == [
        "Qwen3.5-4B-Q4_K_M.gguf",
        "mmproj-F16.gguf",
        "mmproj-F16.gguf",
    ]
    assert completed_status is not None
    assert completed_status.present is True
    assert completed_status.download_state == DownloadState.COMPLETED


def test_download_in_progress_is_not_started_twice(tmp_path: Path) -> None:
    service = ModelsService(build_settings(tmp_path))

    with patch.object(threading.Thread, "start", lambda thread: None):
        first = service.start_download("embeddinggemma-300M")
        second = service.start_download("embeddinggemma-300M")

    assert first.started is True
    assert second.started is False
    assert second.message == "Download already in progress"
