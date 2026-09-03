"""Wire models shared by the API, the client, and every consumer of either.

These definitions are the single source of truth: the FastAPI service returns
them, `SwapboardClient` parses them, and downstream projects import them rather
than restating their own copies.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DownloadState(StrEnum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


class ServiceStatus(StrEnum):
    ok = "ok"
    unavailable = "unavailable"


class ModelFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    relative_path: str
    repo_id: str
    filename: str


class ModelSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    files: tuple[ModelFile, ...] = Field(min_length=1)

    @property
    def primary_file(self) -> ModelFile:
        return self.files[0]


class DownloadProgress(BaseModel):
    state: DownloadState = DownloadState.IDLE
    error: str | None = None


class ModelStatus(BaseModel):
    name: str
    repo_id: str
    filename: str
    path: str
    present: bool
    download_state: DownloadState
    download_error: str | None = None


class DownloadOutcome(BaseModel):
    found: bool
    started: bool
    message: str


class DownloadResponse(BaseModel):
    started: bool
    message: str


class InferenceInfo(BaseModel):
    port: int


class GatewayStatus(BaseModel):
    """Everything a dashboard needs about a swapboard instance in one call.

    Aggregated client side rather than served as a single endpoint, so that an
    unreachable service degrades to `unavailable` instead of raising.
    """

    status: ServiceStatus
    info: InferenceInfo | None = None
    endpoint_url: str | None = None
    health: dict[str, str] = Field(default_factory=dict)
    models: list[ModelStatus] = Field(default_factory=list)
