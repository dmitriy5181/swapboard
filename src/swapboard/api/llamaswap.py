"""Reads what llama-swap knows about the models it serves."""

import logging

import httpx

from swapboard.common.models import ModelMeta, ModelParameters

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0
PEER_TYPE = "peer"


class LlamaSwapCatalog:
    """Fetches per-model metadata from llama-swap's OpenAI-compatible list."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def fetch_meta(self) -> dict[str, ModelMeta]:
        """Maps model name to metadata, or nothing when llama-swap is silent.

        The metadata only decorates the dashboard, so a llama-swap that is down
        or answering something unexpected degrades to an undecorated table
        rather than an error.
        """
        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
            entries = response.json()["data"]
            return _index(entries)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.warning("llama-swap model metadata unavailable: %s", exc)
            return {}


def _index(entries: object) -> dict[str, ModelMeta]:
    if not isinstance(entries, list):
        return {}
    return {
        entry["id"]: _to_meta(entry)
        for entry in entries
        if isinstance(entry, dict) and _text(entry.get("id")) and not _is_peer(entry)
    }


def _is_peer(entry: dict) -> bool:
    """Reports models proxied from another llama-swap instance.

    Peers have no files on this host, so swapboard can neither download nor
    remove them and must not offer to.
    """
    return _llamaswap_meta(entry).get("type") == PEER_TYPE


def _to_meta(entry: dict) -> ModelMeta:
    meta = _llamaswap_meta(entry)
    return ModelMeta(
        family=_text(meta.get("family")),
        parameters=_to_parameters(meta.get("parameters")),
        quantization=_text(meta.get("quantization")),
        task=_text(meta.get("task")),
        context_length=_count(entry.get("context_length")),
        capabilities=_capabilities(entry.get("capabilities")),
        state=_text(_mapping(entry.get("status")).get("value")),
    )


def _to_parameters(value: object) -> ModelParameters | None:
    parameters = _mapping(value)
    if not parameters:
        return None
    return ModelParameters(
        total=_text(parameters.get("total")),
        active=_text(parameters.get("active")),
        effective=_text(parameters.get("effective")),
    )


def _capabilities(value: object) -> tuple[str, ...]:
    return tuple(sorted(name for name, on in _mapping(value).items() if on is True))


def _llamaswap_meta(entry: dict) -> dict:
    return _mapping(_mapping(entry.get("meta")).get("llamaswap"))


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
