import httpx
import pytest

from swapboard.common.client import SwapboardClient
from swapboard.common.models import DownloadState, ServiceStatus

MODEL_PAYLOAD = {
    "name": "llama-3",
    "repo_id": "acme/llama-3-GGUF",
    "filename": "llama-3-Q4_K_M.gguf",
    "path": "/models/acme/llama-3-GGUF/llama-3-Q4_K_M.gguf",
    "present": False,
    "download_state": "idle",
    "download_error": None,
}


def build_client(handler, base_url: str = "http://gateway.test") -> SwapboardClient:
    client = SwapboardClient(base_url)
    client._client = httpx.Client(
        base_url=base_url, transport=httpx.MockTransport(handler)
    )
    return client


def test_list_models_parses_payload() -> None:
    client = build_client(lambda request: httpx.Response(200, json=[MODEL_PAYLOAD]))

    models = client.list_models()

    assert len(models) == 1
    assert models[0].name == "llama-3"
    assert models[0].download_state == DownloadState.IDLE


def test_get_model_requests_named_path() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json=MODEL_PAYLOAD)

    client = build_client(handler)

    model = client.get_model("llama-3")

    assert seen == ["/models/llama-3"]
    assert model.repo_id == "acme/llama-3-GGUF"


def test_download_model_posts_and_parses() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(
            200, json={"started": True, "message": "Download started"}
        )

    client = build_client(handler)

    result = client.download_model("llama-3")

    assert seen == [("POST", "/models/llama-3/download")]
    assert result.started is True
    assert result.message == "Download started"


def test_get_status_aggregates_every_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return httpx.Response(200, json={"port": 9000})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=[MODEL_PAYLOAD])

    client = build_client(handler, "http://gateway.test:8400")

    status = client.get_status()

    assert status.status == ServiceStatus.ok
    assert status.info is not None
    assert status.info.port == 9000
    assert status.health == {"status": "ok"}
    assert len(status.models) == 1


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://gateway.test:8400", "http://gateway.test:9000/v1"),
        ("https://gateway.test:8400", "https://gateway.test:9000/v1"),
        ("http://gateway.test", "http://gateway.test:9000/v1"),
    ],
)
def test_endpoint_url_swaps_port_and_keeps_scheme(base_url: str, expected: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return httpx.Response(200, json={"port": 9000})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=[])

    client = build_client(handler, base_url)

    assert client.get_status().endpoint_url == expected


def test_get_status_reports_unavailable_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = build_client(handler)

    status = client.get_status()

    assert status.status == ServiceStatus.unavailable
    assert status.info is None
    assert status.models == []


def test_get_status_reports_unavailable_on_http_error() -> None:
    client = build_client(lambda request: httpx.Response(503))

    assert client.get_status().status == ServiceStatus.unavailable


def test_get_status_reports_unavailable_on_malformed_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return httpx.Response(200, json={"port": "not-a-number"})
        return httpx.Response(200, json={})

    client = build_client(handler)

    assert client.get_status().status == ServiceStatus.unavailable


def test_list_models_raises_on_error_status() -> None:
    client = build_client(lambda request: httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        client.list_models()
