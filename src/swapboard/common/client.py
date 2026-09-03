"""Synchronous HTTP client for a swapboard API instance."""

import logging

import httpx

from swapboard.common.models import (
    DownloadResponse,
    GatewayStatus,
    InferenceInfo,
    ModelStatus,
    ServiceStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


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
        response = self._client.get(f"/models/{name}")
        response.raise_for_status()
        return ModelStatus.model_validate(response.json())

    def download_model(self, name: str) -> DownloadResponse:
        response = self._client.post(f"/models/{name}/download")
        response.raise_for_status()
        return DownloadResponse.model_validate(response.json())

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
                endpoint_url=self._endpoint_url(info.port),
                health=self.get_health(),
                models=self.list_models(),
            )
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("swapboard API unavailable: %s", exc)
            return GatewayStatus(status=ServiceStatus.unavailable)

    def _endpoint_url(self, port: int) -> str:
        """Builds llama-swap's OpenAI-compatible URL from the API's own host.

        llama-swap listens on a different port of the same host, and only the
        API knows which one, so the base URL is reused with the port swapped.
        """
        return str(httpx.URL(self._base_url).copy_with(port=port, path="/v1"))
