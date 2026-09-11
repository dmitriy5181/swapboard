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


class ModelParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: str | None = None
    active: str | None = None
    effective: str | None = None


class ModelMeta(BaseModel):
    """What llama-swap reports about a model it serves, beyond its files.

    Every field is optional because llama-swap fills them from the GGUF it
    loaded: a model it has never seen, or a config that omits the metadata,
    yields a sparse record rather than an absent one.
    """

    model_config = ConfigDict(frozen=True)

    family: str | None = None
    parameters: ModelParameters | None = None
    quantization: str | None = None
    task: str | None = None
    context_length: int | None = None
    capabilities: tuple[str, ...] = ()
    state: str | None = None


class ModelStatus(BaseModel):
    name: str
    repo_id: str
    filename: str
    path: str
    present: bool
    download_state: DownloadState
    download_error: str | None = None
    size_bytes: int = 0
    meta: ModelMeta | None = None


class StrayModel(BaseModel):
    """GGUF files on disk that no configured model refers to."""

    model_config = ConfigDict(frozen=True)

    relative_path: str
    files: tuple[str, ...] = Field(min_length=1)
    size_bytes: int = 0


class DownloadOutcome(BaseModel):
    found: bool
    started: bool
    message: str


class DownloadResponse(BaseModel):
    started: bool
    message: str


class RemovalOutcome(BaseModel):
    found: bool
    removed: bool
    message: str


class RemovalResponse(BaseModel):
    removed: bool
    message: str


class ConfigValidation(BaseModel):
    """Why a configuration was rejected, and what it will silently cost.

    Errors block a save; warnings do not. A warning marks a model llama-swap
    accepts but swapboard cannot manage, which would otherwise just be missing
    from the dashboard with no explanation.
    """

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConfigDocument(BaseModel):
    text: str
    warnings: list[str] = Field(default_factory=list)


class ConfigUpdate(BaseModel):
    text: str


class ConfigSaveResponse(BaseModel):
    saved: bool
    message: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class InferenceInfo(BaseModel):
    port: int
    endpoint_url: str | None = None


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
    stray: list[StrayModel] = Field(default_factory=list)
