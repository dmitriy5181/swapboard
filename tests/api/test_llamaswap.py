from collections.abc import Callable
from unittest.mock import MagicMock, patch

import httpx

from swapboard.api.llamaswap import LlamaSwapCatalog

CHAT_ENTRY = {
    "id": "gemma-4-e4b",
    "context_length": 131072,
    "capabilities": {
        "vision": True,
        "function_calling": True,
        "audio_transcriptions": False,
    },
    "meta": {
        "llamaswap": {
            "family": "Gemma 4",
            "parameters": {"effective": "4.5B", "total": "8B"},
            "quantization": "QAT-Q4_0",
            "task": "chat",
            "type": "model",
        }
    },
    "status": {"value": "unloaded"},
}

PEER_ENTRY = {
    "id": "opencode-go/glm-5",
    "meta": {"llamaswap": {"peerID": "opencode-go", "type": "peer"}},
    "status": {"value": "unloaded"},
}


Handler = Callable[[httpx.Request], httpx.Response]


def build_catalog(handler: Handler) -> LlamaSwapCatalog:
    catalog = LlamaSwapCatalog("http://llama-swap.test")
    catalog._client = httpx.Client(
        base_url="http://llama-swap.test", transport=httpx.MockTransport(handler)
    )
    return catalog


def catalog_returning(payload: object) -> LlamaSwapCatalog:
    return build_catalog(lambda request: httpx.Response(200, json=payload))


def test_fetch_meta_requests_the_openai_model_list() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    build_catalog(handler).fetch_meta()

    assert seen == ["/v1/models"]


def test_fetch_meta_maps_every_declared_field() -> None:
    meta = catalog_returning({"data": [CHAT_ENTRY]}).fetch_meta()["gemma-4-e4b"]

    assert meta.family == "Gemma 4"
    assert meta.quantization == "QAT-Q4_0"
    assert meta.task == "chat"
    assert meta.context_length == 131072
    assert meta.state == "unloaded"


def test_fetch_meta_keeps_only_enabled_capabilities() -> None:
    meta = catalog_returning({"data": [CHAT_ENTRY]}).fetch_meta()["gemma-4-e4b"]

    assert meta.capabilities == ("function_calling", "vision")


def test_fetch_meta_reads_the_parameter_counts() -> None:
    meta = catalog_returning({"data": [CHAT_ENTRY]}).fetch_meta()["gemma-4-e4b"]

    assert meta.parameters is not None
    assert meta.parameters.total == "8B"
    assert meta.parameters.effective == "4.5B"
    assert meta.parameters.active is None


def test_fetch_meta_skips_models_served_by_a_peer() -> None:
    """A peer's files live on another host, so swapboard must not offer them."""
    catalog = catalog_returning({"data": [CHAT_ENTRY, PEER_ENTRY]})

    assert set(catalog.fetch_meta()) == {"gemma-4-e4b"}


def test_fetch_meta_tolerates_an_entry_without_metadata() -> None:
    meta = catalog_returning({"data": [{"id": "bare"}]}).fetch_meta()["bare"]

    assert meta.family is None
    assert meta.capabilities == ()


def test_fetch_meta_returns_nothing_when_llama_swap_is_unreachable() -> None:
    """Metadata only decorates the table; losing it must not empty the page."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert build_catalog(handler).fetch_meta() == {}


def test_fetch_meta_returns_nothing_on_an_error_status() -> None:
    assert build_catalog(lambda request: httpx.Response(503)).fetch_meta() == {}


def test_fetch_meta_returns_nothing_for_an_unexpected_payload() -> None:
    assert catalog_returning({"models": []}).fetch_meta() == {}


def test_fetch_meta_returns_nothing_for_malformed_capabilities() -> None:
    response = MagicMock()
    response.json.return_value = {
        "data": [{"id": "broken", "capabilities": {"chat": True, 1: True}}]
    }
    catalog = catalog_returning({"data": []})

    with patch.object(catalog._client, "get", return_value=response):
        assert catalog.fetch_meta() == {}


def test_fetch_meta_skips_an_entry_whose_id_is_not_a_name() -> None:
    """An unhashable id would otherwise raise past the fallback and break /models."""
    catalog = catalog_returning({"data": [{"id": []}, {"id": ""}, CHAT_ENTRY]})

    assert list(catalog.fetch_meta()) == ["gemma-4-e4b"]
