"""Synchronous HTTP client for a swapboard API instance."""

import logging
from urllib.parse import quote

import httpx

from swapboard.common.models import (
    ConfigDocument,
    ConfigSaveResponse,
    DownloadResponse,
    GatewayStatus,
    InferenceInfo,
    ModelStatus,
    RemovalResponse,
    ServiceStatus,
    StrayModel,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
UNPROCESSABLE = 422


class SwapboardClient:
    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_health(self) -> dict[str, str]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def get_info(self) -> InferenceInfo:
        response = self._client.get("/info")
        response.raise_for_status()
        return InferenceInfo.model_validate(response.json())

    def list_models(self) -> list[ModelStatus]:
        response = self._client.get("/models")
        response.raise_for_status()
        return [ModelStatus.model_validate(item) for item in response.json()]

    def get_model(self, name: str) -> ModelStatus:
        response = self._client.get(_model_path(name))
        response.raise_for_status()
        return ModelStatus.model_validate(response.json())

    def download_model(self, name: str) -> DownloadResponse:
        response = self._client.post(f"{_model_path(name)}/download")
        response.raise_for_status()
        return DownloadResponse.model_validate(response.json())

    def remove_model(self, name: str) -> RemovalResponse:
        response = self._client.delete(_model_path(name))
        response.raise_for_status()
        return RemovalResponse.model_validate(response.json())

    def list_stray(self) -> list[StrayModel]:
        response = self._client.get("/stray-models")
        response.raise_for_status()
        return [StrayModel.model_validate(item) for item in response.json()]

    def remove_stray(self, relative_path: str) -> RemovalResponse:
        response = self._client.delete(_stray_path(relative_path))
        response.raise_for_status()
        return RemovalResponse.model_validate(response.json())

    def get_config(self) -> ConfigDocument:
        response = self._client.get("/config")
        response.raise_for_status()
        return ConfigDocument.model_validate(response.json())

    def save_config(self, text: str) -> ConfigSaveResponse:
        """Saves the config, returning a rejection rather than raising on one.

        A refused configuration is the expected answer to a typo, and the body
        carries the reasons the editor has to show, so 422 is read rather than
        turned into an exception.
        """
        response = self._client.put("/config", json={"text": text})
        if response.status_code != UNPROCESSABLE:
            response.raise_for_status()
        return ConfigSaveResponse.model_validate(response.json())

    def get_status(self) -> GatewayStatus:
        """Collects the full dashboard view, reporting unavailable on failure.

        Callers render this directly, so an unreachable or malformed service has
        to degrade into a status rather than propagate an exception.
        """
        try:
            info = self.get_info()
            return GatewayStatus(
                status=ServiceStatus.ok,
                info=info,
                endpoint_url=info.endpoint_url or self._endpoint_url(info.port),
                health=self.get_health(),
                models=self.list_models(),
                stray=self.list_stray(),
            )
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("swapboard API unavailable: %s", exc)
            return GatewayStatus(status=ServiceStatus.unavailable)

    def _endpoint_url(self, port: int) -> str:
        """Builds llama-swap's OpenAI-compatible URL from the API's own host.

        llama-swap listens on a different port of the same host, and only the
        API knows which one, so the base URL is reused with the port swapped.
        This is the fallback for installations that publish no public endpoint;
        behind a reverse proxy the API reports the address to use instead.
        """
        return str(httpx.URL(self._base_url).copy_with(port=port, path="/v1"))


def _model_path(name: str) -> str:
    """Builds the URL for a single model, escaping everything but slashes.

    llama-swap names may contain any character. Slashes stay literal because
    the API matches them as part of the path and proxies routinely reject or
    rewrite an encoded one; the rest is escaped so a name containing '?' or
    '#' cannot silently truncate the request.
    """
    return f"/models/{quote(name, safe='/')}"


def _stray_path(relative_path: str) -> str:
    return f"/stray-models/{quote(relative_path, safe='/')}"
