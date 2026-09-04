import re

import pytest

from swapboard.common.models import (
    DownloadResponse,
    DownloadState,
    GatewayStatus,
    InferenceInfo,
    ModelStatus,
    ServiceStatus,
)
from swapboard.ui.factory import create_app


def model(
    name: str = "embeddinggemma-300M",
    *,
    present: bool = False,
    download_state: DownloadState = DownloadState.IDLE,
    download_error: str | None = None,
) -> ModelStatus:
    return ModelStatus(
        name=name,
        repo_id="ggml-org/embeddinggemma-300M-GGUF",
        filename="embeddinggemma-300M-Q8_0.gguf",
        path=f"/models/ggml-org/embeddinggemma-300M-GGUF/{name}.gguf",
        present=present,
        download_state=download_state,
        download_error=download_error,
    )


def online(*models: ModelStatus) -> GatewayStatus:
    return GatewayStatus(
        status=ServiceStatus.ok,
        info=InferenceInfo(port=8772),
        endpoint_url="http://127.0.0.1:8772/v1",
        health={"status": "ok"},
        models=list(models),
    )


class StubClient:
    def __init__(self, status: GatewayStatus, response=None, error=None) -> None:
        self.status = status
        self.response = response or DownloadResponse(
            started=True, message="Download started"
        )
        self.error = error
        self.downloaded: list[str] = []

    def get_status(self) -> GatewayStatus:
        return self.status

    def download_model(self, name: str) -> DownloadResponse:
        self.downloaded.append(name)
        if self.error is not None:
            raise self.error
        return self.response


def build_client(stub: StubClient):
    app = create_app({"SWAPBOARD_CLIENT": stub, "TESTING": True})
    return app.test_client()


@pytest.fixture
def stub() -> StubClient:
    return StubClient(online(model()))


def test_health_reports_ok(stub) -> None:
    response = build_client(stub).get("/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_index_renders_the_models_section(stub) -> None:
    response = build_client(stub).get("/")

    assert response.status_code == 200
    assert b'id="models-section"' in response.data
    assert b"embeddinggemma-300M" in response.data
    assert b"ggml-org/embeddinggemma-300M-GGUF" in response.data


def test_online_status_shows_the_endpoint_url(stub) -> None:
    response = build_client(stub).get("/")

    assert b"Online" in response.data
    assert b"http://127.0.0.1:8772/v1" in response.data


def test_unavailable_status_hides_the_table() -> None:
    stub = StubClient(GatewayStatus(status=ServiceStatus.unavailable))

    response = build_client(stub).get("/")

    assert b"Unavailable" in response.data
    assert b"could not be reached" in response.data
    assert b"/v1" not in response.data


def test_missing_model_offers_a_download_button(stub) -> None:
    response = build_client(stub).get("/partials/models")

    assert b"Not downloaded" in response.data
    assert b"/models/embeddinggemma-300M/download" in response.data


def test_present_model_is_marked_ready() -> None:
    stub = StubClient(online(model(present=True)))

    response = build_client(stub).get("/partials/models")

    assert b"Available" in response.data
    assert b"ready" in response.data
    assert b"/download" not in response.data


def test_failed_model_surfaces_its_error() -> None:
    stub = StubClient(
        online(model(download_state=DownloadState.FAILED, download_error="disk full"))
    )

    response = build_client(stub).get("/partials/models")

    assert b"Failed" in response.data
    assert b"disk full" in response.data


def test_in_progress_download_polls_and_disables_the_button() -> None:
    stub = StubClient(online(model(download_state=DownloadState.DOWNLOADING)))

    response = build_client(stub).get("/partials/models")

    assert b'hx-trigger="every 3s"' in response.data
    assert b"In progress" in response.data
    assert b"disabled" in response.data


def test_idle_section_does_not_poll(stub) -> None:
    response = build_client(stub).get("/partials/models")

    assert b"every 3s" not in response.data


def test_no_models_reports_an_empty_configuration() -> None:
    stub = StubClient(online())

    response = build_client(stub).get("/partials/models")

    assert b"No models are configured" in response.data


def test_download_starts_and_reports_success(stub) -> None:
    response = build_client(stub).post("/models/embeddinggemma-300M/download")

    assert response.status_code == 200
    assert stub.downloaded == ["embeddinggemma-300M"]
    assert b"Download started" in response.data
    assert b"alert-success" in response.data


def test_download_button_of_a_namespaced_model_is_reachable() -> None:
    """A slashed name has to survive url_for and still match the download route."""
    stub = StubClient(online(model(name="local/embeddinggemma-300M")))
    client = build_client(stub)
    partial = client.get("/partials/models")
    match = re.search(rb'hx-post="([^"]+)"', partial.data)
    assert match is not None

    response = client.post(match.group(1).decode())

    assert response.status_code == 200
    assert stub.downloaded == ["local/embeddinggemma-300M"]


def test_download_already_present_reports_information() -> None:
    stub = StubClient(
        online(model(present=True)),
        response=DownloadResponse(started=False, message="Model already present"),
    )

    response = build_client(stub).post("/models/embeddinggemma-300M/download")

    assert b"Model already present" in response.data
    assert b"alert-info" in response.data


def test_download_failure_is_reported_without_breaking_the_page() -> None:
    stub = StubClient(online(model()), error=RuntimeError("connection refused"))

    response = build_client(stub).post("/models/embeddinggemma-300M/download")

    assert response.status_code == 200
    assert b"Could not reach the swapboard API." in response.data
    assert b"alert-danger" in response.data
    assert b'id="models-section"' in response.data


def test_download_response_is_a_swappable_fragment(stub) -> None:
    response = build_client(stub).post("/models/embeddinggemma-300M/download")

    assert not response.data.lstrip().startswith(b"<!DOCTYPE")
    assert b'id="models-section"' in response.data
