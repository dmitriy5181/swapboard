import threading
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from swapboard.api.configfile import ConfigFile
from swapboard.api.llamaswap import LlamaSwapCatalog
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

MTP_CONFIG = """\
models:
  qwen3.8-27b:
    cmd: |
      llama-server --port ${PORT}
      -m ${models_dir}/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf
      --mmproj ${models_dir}/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf
      --spec-draft-model ${models_dir}/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Q4_0.gguf
      --spec-type draft-mtp
"""

NAMESPACED_CONFIG = """\
models:
  "local/embeddinggemma-300M":
    cmd: |
      llama-server --port ${PORT}
      -m /models/ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf
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


def build_catalog(payload: object) -> LlamaSwapCatalog:
    """A catalog that answers from a fixture rather than a live llama-swap."""
    catalog = LlamaSwapCatalog("http://llama-swap.test")
    catalog._client = httpx.Client(
        base_url="http://llama-swap.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    )
    return catalog


def build_service(settings: Settings, catalog_payload: object = None) -> ModelsService:
    return ModelsService(settings, build_catalog(catalog_payload or {"data": []}))


def build_client(
    tmp_models: Path,
    llama_swap_port: int = 8080,
    *,
    config: Path | None = None,
    public_endpoint_url: str | None = None,
    catalog_payload: object = None,
) -> TestClient:
    settings = build_settings(
        tmp_models,
        llama_swap_port=llama_swap_port,
        config=config,
        public_endpoint_url=public_endpoint_url,
    )

    import swapboard.api.main as main

    main.settings = settings
    main.service = build_service(settings, catalog_payload)
    main.config_file = ConfigFile(settings.llama_swap_config_path)
    return TestClient(main.app)


def write_default_config(directory: Path) -> Path:
    config = directory / "default.yml"
    config.write_text(DEFAULT_CONFIG, encoding="utf-8")
    return config


def write_multimodal_config(directory: Path) -> Path:
    config = directory / "multimodal.yml"
    config.write_text(MULTIMODAL_CONFIG, encoding="utf-8")
    return config


def write_mtp_config(directory: Path) -> Path:
    config = directory / "mtp.yml"
    config.write_text(MTP_CONFIG, encoding="utf-8")
    return config


def write_namespaced_config(directory: Path) -> Path:
    config = directory / "namespaced.yml"
    config.write_text(NAMESPACED_CONFIG, encoding="utf-8")
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


def test_downloads_try_claim_allows_only_one_winner() -> None:
    downloads = Downloads()

    assert downloads.try_claim("model") is True
    assert downloads.try_claim("model") is False


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


def test_get_model_accepts_a_name_containing_a_slash(tmp_path: Path) -> None:
    """llama-swap namespaces models, so a name can span several path segments."""
    client = build_client(tmp_path, config=write_namespaced_config(tmp_path))

    response = client.get("/models/local/embeddinggemma-300M")

    assert response.status_code == 200
    assert response.json()["name"] == "local/embeddinggemma-300M"


def test_download_accepts_a_name_containing_a_slash(tmp_path: Path) -> None:
    model_dir = tmp_path / "ggml-org" / "embeddinggemma-300M-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "embeddinggemma-300M-Q8_0.gguf").write_bytes(b"fake-gguf")
    client = build_client(tmp_path, config=write_namespaced_config(tmp_path))

    response = client.post("/models/local/embeddinggemma-300M/download")

    assert response.status_code == 200
    assert response.json() == {"started": False, "message": "Model already present"}


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
    service = build_service(
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


def test_download_fetches_nested_mtp_draft_into_the_repository(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "unsloth/Qwen3.8-27B-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3.8-27B-Q6_K.gguf").write_bytes(b"fake-model")
    (model_dir / "mmproj-F16.gguf").write_bytes(b"fake-projector")
    service = build_service(build_settings(tmp_path, config=write_mtp_config(tmp_path)))

    def download_file(**kwargs: str | None) -> str:
        target = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-draft")
        return str(target)

    with (
        patch(
            "swapboard.api.service.hf_hub_download",
            side_effect=download_file,
        ) as download,
        patch.object(threading.Thread, "start", lambda thread: thread.run()),
    ):
        outcome = service.start_download("qwen3.8-27b")

    status = service.get_status("qwen3.8-27b")
    assert outcome.started is True
    download.assert_called_once_with(
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        filename="MTP/mtp-Q4_0.gguf",
        local_dir=str(model_dir),
        token=None,
    )
    assert (model_dir / "MTP/mtp-Q4_0.gguf").is_file()
    assert status is not None
    assert status.present is True
    assert status.download_state == DownloadState.COMPLETED


def test_download_retries_failed_projector_without_fetching_primary_again(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "unsloth" / "Qwen3.5-4B-GGUF"
    model_dir.mkdir(parents=True)
    service = build_service(
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
    service = build_service(build_settings(tmp_path))

    with patch.object(threading.Thread, "start", lambda thread: None):
        first = service.start_download("embeddinggemma-300M")
        second = service.start_download("embeddinggemma-300M")

    assert first.started is True
    assert second.started is False
    assert second.message == "Download already in progress"


def test_list_models_reports_size_on_disk(tmp_path: Path) -> None:
    model_dir = tmp_path / "ggml-org" / "embeddinggemma-300M-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "embeddinggemma-300M-Q8_0.gguf").write_bytes(b"x" * 2048)
    client = build_client(tmp_path)

    payload = client.get("/models").json()

    assert payload[0]["size_bytes"] == 2048


def test_list_models_carries_the_metadata_llama_swap_reports(tmp_path: Path) -> None:
    client = build_client(
        tmp_path,
        catalog_payload={
            "data": [
                {
                    "id": "embeddinggemma-300M",
                    "context_length": 2048,
                    "meta": {
                        "llamaswap": {
                            "family": "EmbeddingGemma",
                            "quantization": "Q8_0",
                            "task": "embeddings",
                        }
                    },
                }
            ]
        },
    )

    meta = client.get("/models").json()[0]["meta"]

    assert meta["family"] == "EmbeddingGemma"
    assert meta["quantization"] == "Q8_0"
    assert meta["context_length"] == 2048


def test_list_models_leaves_metadata_absent_when_llama_swap_knows_nothing(
    tmp_path: Path,
) -> None:
    client = build_client(tmp_path)

    assert client.get("/models").json()[0]["meta"] is None


def test_list_stray_models_reports_only_unreferenced_files(tmp_path: Path) -> None:
    configured = tmp_path / "ggml-org" / "embeddinggemma-300M-GGUF"
    configured.mkdir(parents=True)
    (configured / "embeddinggemma-300M-Q8_0.gguf").write_bytes(b"x")
    forgotten = tmp_path / "acme" / "retired-GGUF"
    forgotten.mkdir(parents=True)
    (forgotten / "retired.gguf").write_bytes(b"x" * 64)
    client = build_client(tmp_path)

    payload = client.get("/stray-models").json()

    assert payload == [
        {
            "relative_path": "acme/retired-GGUF",
            "files": ["retired.gguf"],
            "size_bytes": 64,
        }
    ]


def test_remove_model_deletes_its_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "ggml-org" / "embeddinggemma-300M-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "embeddinggemma-300M-Q8_0.gguf").write_bytes(b"x")
    client = build_client(tmp_path)

    response = client.delete("/models/embeddinggemma-300M")

    assert response.status_code == 200
    assert response.json()["removed"] is True
    assert client.get("/models").json()[0]["present"] is False


def test_remove_model_reports_a_model_that_was_never_downloaded(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.delete("/models/embeddinggemma-300M")

    assert response.status_code == 200
    assert response.json() == {"removed": False, "message": "Model was not downloaded"}


def test_remove_unknown_model_is_not_found(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    assert client.delete("/models/absent").status_code == 404


def test_remove_stray_model_deletes_the_directory(tmp_path: Path) -> None:
    forgotten = tmp_path / "acme" / "retired-GGUF"
    forgotten.mkdir(parents=True)
    (forgotten / "retired.gguf").write_bytes(b"x")
    client = build_client(tmp_path)

    response = client.delete("/stray-models/acme/retired-GGUF")

    assert response.status_code == 200
    assert response.json()["removed"] is True
    assert not forgotten.exists()


def test_remove_unknown_stray_model_is_not_found(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    assert client.delete("/stray-models/acme/absent").status_code == 404


def test_remove_stray_model_refuses_to_escape_the_models_directory(
    tmp_path: Path,
) -> None:
    """An encoded `..` survives URL normalisation and reaches the handler intact.

    It names no file this store would report as stray, which is the whole of
    the answer: the target is unknown rather than forbidden.
    """
    outside = tmp_path.parent / "outside-the-store"
    outside.mkdir(exist_ok=True)
    client = build_client(tmp_path)

    response = client.delete("/stray-models/%2e%2e/outside-the-store")

    assert response.status_code == 404
    assert outside.exists()


def test_get_config_returns_the_current_file(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    payload = client.get("/config").json()

    assert payload["text"] == DEFAULT_CONFIG
    assert payload["warnings"] == []


def test_save_config_replaces_the_file(tmp_path: Path) -> None:
    config = write_default_config(tmp_path)
    client = build_client(tmp_path, config=config)
    replacement = DEFAULT_CONFIG.replace("--embeddings", "--embeddings --parallel 2")

    response = client.put("/config", json={"text": replacement})

    assert response.status_code == 200
    assert response.json()["saved"] is True
    assert config.read_text(encoding="utf-8") == replacement


def test_save_config_rejects_a_schema_violation_without_writing(tmp_path: Path) -> None:
    config = write_default_config(tmp_path)
    client = build_client(tmp_path, config=config)

    response = client.put("/config", json={"text": "models:\n  broken:\n    ttl: 5\n"})

    assert response.status_code == 422
    assert response.json()["errors"] == ["models/broken: 'cmd' is a required property"]
    assert config.read_text(encoding="utf-8") == DEFAULT_CONFIG


def test_saved_config_is_served_to_the_models_endpoint(tmp_path: Path) -> None:
    """`--watch-config` reloads llama-swap, and swapboard rereads on every call."""
    client = build_client(tmp_path, config=write_default_config(tmp_path))
    renamed = DEFAULT_CONFIG.replace("embeddinggemma-300M:", "renamed-model:")

    client.put("/config", json={"text": renamed})

    names = [model["name"] for model in client.get("/models").json()]

    assert names == ["renamed-model"]


SHARED_WEIGHTS_CONFIG = """\
models:
  fast:
    cmd: llama-server -m /models/acme/demo-GGUF/demo.gguf -c 4096
  long:
    cmd: llama-server -m /models/acme/demo-GGUF/demo.gguf -c 32768
"""

UNDERIVABLE_CONFIG = """\
models:
  solo:
    cmd: llama-server -m {path}
"""


def write_config(directory: Path, text: str) -> Path:
    config = directory / "config.yml"
    config.write_text(text, encoding="utf-8")
    return config


def test_removing_one_model_keeps_weights_a_second_model_shares(
    tmp_path: Path,
) -> None:
    """Both entries run the same file; deleting it would take the other down."""
    weights = tmp_path / "acme/demo-GGUF/demo.gguf"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x" * 32)
    client = build_client(
        tmp_path, config=write_config(tmp_path, SHARED_WEIGHTS_CONFIG)
    )

    response = client.delete("/models/fast")

    assert response.json() == {
        "removed": False,
        "message": "Files kept: another configured model uses them",
    }
    assert weights.exists()


def test_stray_removal_spares_a_configured_file_in_the_same_directory(
    tmp_path: Path,
) -> None:
    """The stray entry names the directory; only the unclaimed file is stray."""
    configured = (
        tmp_path / "ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
    )
    configured.parent.mkdir(parents=True)
    configured.write_bytes(b"x" * 16)
    extra = configured.parent / "extra.gguf"
    extra.write_bytes(b"y" * 16)
    client = build_client(tmp_path)

    response = client.delete("/stray-models/ggml-org/embeddinggemma-300M-GGUF")

    assert response.json() == {"removed": True, "message": "Stray model removed"}
    assert configured.exists()
    assert not extra.exists()


def test_a_model_only_the_config_can_name_is_never_listed_as_stray(
    tmp_path: Path,
) -> None:
    """swapboard cannot download it, so it must not offer to delete it either."""
    weights = tmp_path / "solo.gguf"
    weights.write_bytes(b"x" * 16)
    config = write_config(tmp_path, UNDERIVABLE_CONFIG.format(path=weights))
    client = build_client(tmp_path, config=config)

    assert client.get("/stray-models").json() == []
    assert client.delete("/stray-models/solo.gguf").status_code == 404
    assert weights.exists()


def test_removal_releases_the_claim_it_took_for_a_later_download(
    tmp_path: Path,
) -> None:
    """Holding the slot past the delete would lock the model out of downloading."""
    client = build_client(tmp_path)

    client.delete("/models/embeddinggemma-300M")

    with patch("swapboard.api.service.hf_hub_download"):
        response = client.post("/models/embeddinggemma-300M/download")

    assert response.json()["started"] is True


SORTING_CONFIG = """\
models:
  zephyr:
    cmd: llama-server -m /models/acme/zephyr-GGUF/zephyr.gguf
  Alpha:
    cmd: llama-server -m /models/acme/alpha-GGUF/alpha.gguf
  mistral:
    cmd: llama-server -m /models/acme/mistral-GGUF/mistral.gguf
"""


def test_list_models_is_sorted_by_name(tmp_path: Path) -> None:
    """Config order is written for llama-swap, not for anyone reading it."""
    config = tmp_path / "sorting.yml"
    config.write_text(SORTING_CONFIG, encoding="utf-8")
    client = build_client(tmp_path, config=config)

    names = [model["name"] for model in client.get("/models").json()]

    assert names == ["Alpha", "mistral", "zephyr"]
