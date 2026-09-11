import re

import httpx
import pytest

from swapboard.common.models import (
    ConfigDocument,
    ConfigSaveResponse,
    DownloadResponse,
    DownloadState,
    GatewayStatus,
    InferenceInfo,
    ModelMeta,
    ModelParameters,
    ModelStatus,
    RemovalResponse,
    ServiceStatus,
    StrayModel,
)
from swapboard.ui.factory import create_app

CONFIG_TEXT = "models:\n  embeddinggemma-300M:\n    cmd: llama-server\n"


def model(
    name: str = "embeddinggemma-300M",
    *,
    present: bool = False,
    download_state: DownloadState = DownloadState.IDLE,
    download_error: str | None = None,
    size_bytes: int = 0,
    meta: ModelMeta | None = None,
) -> ModelStatus:
    return ModelStatus(
        name=name,
        repo_id="ggml-org/embeddinggemma-300M-GGUF",
        filename="embeddinggemma-300M-Q8_0.gguf",
        path=f"/models/ggml-org/embeddinggemma-300M-GGUF/{name}.gguf",
        present=present,
        download_state=download_state,
        download_error=download_error,
        size_bytes=size_bytes,
        meta=meta,
    )


def stray(relative_path: str = "acme/retired-GGUF") -> StrayModel:
    return StrayModel(
        relative_path=relative_path, files=("retired.gguf",), size_bytes=2048
    )


def online(
    *models: ModelStatus, stray_models: list[StrayModel] | None = None
) -> GatewayStatus:
    return GatewayStatus(
        status=ServiceStatus.ok,
        info=InferenceInfo(port=8772),
        endpoint_url="http://127.0.0.1:8772/v1",
        health={"status": "ok"},
        models=list(models),
        stray=stray_models or [],
    )


class StubClient:
    def __init__(
        self,
        status: GatewayStatus,
        response=None,
        error=None,
        *,
        removal=None,
        config=None,
        save=None,
    ) -> None:
        self.status = status
        self.response = response or DownloadResponse(
            started=True, message="Download started"
        )
        self.removal = removal or RemovalResponse(removed=True, message="Model removed")
        self.config = config or ConfigDocument(text=CONFIG_TEXT)
        self.save = save or ConfigSaveResponse(
            saved=True, message="Configuration saved."
        )
        self.error = error
        self.downloaded: list[str] = []
        self.removed: list[str] = []
        self.removed_stray: list[str] = []
        self.saved: list[str] = []

    def get_status(self) -> GatewayStatus:
        return self.status

    def download_model(self, name: str) -> DownloadResponse:
        self.downloaded.append(name)
        self._fail_if_asked()
        return self.response

    def remove_model(self, name: str) -> RemovalResponse:
        self.removed.append(name)
        self._fail_if_asked()
        return self.removal

    def remove_stray(self, relative_path: str) -> RemovalResponse:
        self.removed_stray.append(relative_path)
        self._fail_if_asked()
        return self.removal

    def get_config(self) -> ConfigDocument:
        self._fail_if_asked()
        return self.config

    def save_config(self, text: str) -> ConfigSaveResponse:
        self.saved.append(text)
        self._fail_if_asked()
        return self.save

    def _fail_if_asked(self) -> None:
        if self.error is not None:
            raise self.error


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


def test_present_model_offers_removal_instead_of_download() -> None:
    stub = StubClient(online(model(present=True)))

    response = build_client(stub).get("/partials/models")

    assert b"Available" in response.data
    assert b'hx-delete="/models/embeddinggemma-300M"' in response.data
    assert b"hx-post" not in response.data


def test_failed_model_surfaces_its_error() -> None:
    stub = StubClient(
        online(model(download_state=DownloadState.FAILED, download_error="disk full"))
    )

    response = build_client(stub).get("/partials/models")

    assert b"Failed" in response.data
    assert b"disk full" in response.data


def test_in_progress_download_polls_and_disables_both_actions() -> None:
    stub = StubClient(online(model(download_state=DownloadState.DOWNLOADING)))

    response = build_client(stub).get("/partials/models")

    assert b", every 3s" in response.data
    assert b"hx-post" not in response.data
    assert b"hx-delete" not in response.data


def test_idle_section_does_not_poll(stub) -> None:
    response = build_client(stub).get("/partials/models")

    assert b"every 3s" not in response.data
    assert b'hx-trigger="models-updated from:body"' in response.data


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


def test_model_metadata_is_rendered_as_badges() -> None:
    stub = StubClient(
        online(
            model(
                meta=ModelMeta(
                    family="Gemma 4",
                    parameters=ModelParameters(total="8B", effective="4.5B"),
                    quantization="QAT-Q4_0",
                    task="chat",
                    context_length=131072,
                    capabilities=("function_calling", "vision"),
                )
            )
        )
    )

    response = build_client(stub).get("/partials/models")

    assert b"Gemma 4" in response.data
    assert b"8B (4.5B active)" in response.data
    assert b"QAT-Q4_0" in response.data
    assert b"128K ctx" in response.data
    assert b"bi-eye" in response.data
    assert b"bi-tools" in response.data


def test_loaded_model_is_marked_as_held_in_memory() -> None:
    stub = StubClient(online(model(meta=ModelMeta(state="loaded"))))

    assert b"Loaded" in build_client(stub).get("/partials/models").data


def test_unloaded_model_carries_no_state_badge() -> None:
    stub = StubClient(online(model(meta=ModelMeta(state="unloaded"))))

    assert b"Unloaded" not in build_client(stub).get("/partials/models").data


def test_model_without_metadata_still_shows_its_repository() -> None:
    """llama-swap being down must not blank the column it decorates."""
    stub = StubClient(online(model()))

    response = build_client(stub).get("/partials/models")

    assert b"ggml-org/embeddinggemma-300M-GGUF" in response.data


def test_size_on_disk_is_rendered_in_binary_units() -> None:
    stub = StubClient(online(model(present=True, size_bytes=1536 * 1024 * 1024)))

    assert b"1.5 GiB" in build_client(stub).get("/partials/models").data


def test_stray_models_are_listed_with_a_removal_button() -> None:
    stub = StubClient(online(model(), stray_models=[stray()]))

    response = build_client(stub).get("/partials/models")

    assert b"Stray models" in response.data
    assert b"acme/retired-GGUF" in response.data
    assert b'hx-delete="/stray-models/acme/retired-GGUF"' in response.data


def test_no_stray_models_hides_the_section(stub) -> None:
    assert b"Stray models" not in build_client(stub).get("/partials/models").data


def test_remove_model_reports_the_outcome(stub) -> None:
    response = build_client(stub).delete("/models/embeddinggemma-300M")

    assert response.status_code == 200
    assert stub.removed == ["embeddinggemma-300M"]
    assert b"Model removed" in response.data
    assert b'id="models-section"' in response.data


def test_remove_stray_model_reports_the_outcome(stub) -> None:
    response = build_client(stub).delete("/stray-models/acme/retired-GGUF")

    assert response.status_code == 200
    assert stub.removed_stray == ["acme/retired-GGUF"]
    assert b"alert-success" in response.data


def test_remove_failure_is_reported_without_breaking_the_page() -> None:
    stub = StubClient(online(model(present=True)), error=RuntimeError("refused"))

    response = build_client(stub).delete("/models/embeddinggemma-300M")

    assert response.status_code == 200
    assert b"Could not reach the swapboard API." in response.data
    assert b"alert-danger" in response.data


def test_config_editor_loads_the_current_configuration(stub) -> None:
    response = build_client(stub).get("/partials/config-editor")

    assert response.status_code == 200
    assert b"llama-swap configuration" in response.data
    assert b"cmd: llama-server" in response.data


def test_config_editor_shows_the_warnings_the_api_reports() -> None:
    stub = StubClient(
        online(model()),
        config=ConfigDocument(text=CONFIG_TEXT, warnings=["model 'x' is unmanageable"]),
    )

    response = build_client(stub).get("/partials/config-editor")

    assert b"model &#39;x&#39; is unmanageable" in response.data


def test_saving_the_config_reports_success_and_refreshes_the_table(stub) -> None:
    response = build_client(stub).put("/config", data={"text": CONFIG_TEXT})

    assert response.status_code == 200
    assert stub.saved == [CONFIG_TEXT]
    assert b"Configuration saved." in response.data
    assert response.headers["HX-Trigger"] == "models-updated"


def test_rejected_config_keeps_the_submitted_text_on_screen() -> None:
    """Losing the edit would force the user to retype it to fix one line."""
    stub = StubClient(
        online(model()),
        save=ConfigSaveResponse(
            saved=False,
            message="The configuration was not saved.",
            errors=["models/broken: 'cmd' is a required property"],
        ),
    )
    submitted = "models:\n  broken:\n    ttl: 5\n"

    response = build_client(stub).put("/config", data={"text": submitted})

    assert b"&#39;cmd&#39; is a required property" in response.data
    assert b"ttl: 5" in response.data
    assert "HX-Trigger" not in response.headers


def test_config_editor_survives_an_unreachable_api() -> None:
    stub = StubClient(online(model()), error=RuntimeError("refused"))

    response = build_client(stub).get("/partials/config-editor")

    assert response.status_code == 200
    assert b"Could not reach the swapboard API." in response.data


def refusal(status_code: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("DELETE", "http://api.test/models/x")
    response = httpx.Response(status_code, json={"detail": detail}, request=request)
    return httpx.HTTPStatusError("refused", request=request, response=response)


def test_refused_request_reports_the_api_explanation_not_a_lost_connection() -> None:
    """Blaming the connection would send the user looking for the wrong fault."""
    stub = StubClient(online(model()), error=refusal(404, "Unknown model 'ghost'"))

    response = build_client(stub).delete("/models/ghost")

    assert b"Unknown model &#39;ghost&#39;" in response.data
    assert b"Could not reach" not in response.data


def test_refusal_without_a_usable_body_still_names_the_action() -> None:
    request = httpx.Request("DELETE", "http://api.test/models/x")
    error = httpx.HTTPStatusError(
        "refused",
        request=request,
        response=httpx.Response(500, text="boom", request=request),
    )
    stub = StubClient(online(model()), error=error)

    response = build_client(stub).delete("/models/embeddinggemma-300M")

    assert b"refused to remove &#39;embeddinggemma-300M&#39;" in response.data


def test_transport_failure_still_reports_an_unreachable_api() -> None:
    stub = StubClient(online(model()), error=httpx.ConnectError("refused"))

    response = build_client(stub).delete("/models/embeddinggemma-300M")

    assert b"Could not reach the swapboard API." in response.data


def test_config_modal_lives_outside_the_swapped_section(stub) -> None:
    """Inside it, every refresh and download poll would tear an open editor out."""
    client = build_client(stub)

    page = client.get("/")
    partial = client.get("/partials/models")

    assert b'id="config-modal"' in page.data
    assert b'id="config-modal"' not in partial.data


def test_config_editor_is_the_modal_content_itself(stub) -> None:
    """A wrapper around the header and footer collapses Bootstrap's flex sizing."""
    response = build_client(stub).get("/partials/config-editor")

    assert response.data.lstrip().startswith(b'<form class="modal-content')


def test_api_failure_with_a_non_object_body_still_reports_the_refusal(stub) -> None:
    """`.get` on a JSON array would raise a second failure inside the first."""
    request = httpx.Request("DELETE", "http://api.test/models/demo")
    response = httpx.Response(502, json=["gateway", "error"], request=request)
    stub.error = httpx.HTTPStatusError("502", request=request, response=response)

    page = build_client(stub).delete("/models/demo")

    assert b"The swapboard API refused to remove &#39;demo&#39;." in page.data


def test_model_in_error_is_not_painted_as_held_in_memory(stub) -> None:
    """A failed load styled as success would read as a healthy model."""
    stub = StubClient(online(model(meta=ModelMeta(state="error"))))

    response = build_client(stub).get("/partials/models")

    assert b"text-bg-danger" in response.data
    assert b"Held in memory" not in response.data


def test_model_still_loading_is_marked_as_in_flight(stub) -> None:
    stub = StubClient(online(model(meta=ModelMeta(state="starting"))))

    response = build_client(stub).get("/partials/models")

    assert b"text-bg-warning" in response.data
    assert b"llama-swap reports this model as starting" in response.data


def test_config_modal_is_named_for_a_screen_reader(stub) -> None:
    page = build_client(stub).get("/")
    editor = build_client(stub).get("/partials/config-editor")

    assert b'aria-labelledby="config-modal-title"' in page.data
    assert b'id="config-modal-title"' in editor.data
