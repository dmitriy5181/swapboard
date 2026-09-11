from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Response, status

from swapboard.api.configfile import ConfigFile
from swapboard.api.service import ModelsService
from swapboard.api.settings import Settings
from swapboard.api.store import OutsideStoreError
from swapboard.common.models import (
    ConfigDocument,
    ConfigSaveResponse,
    ConfigUpdate,
    DownloadResponse,
    InferenceInfo,
    ModelStatus,
    RemovalOutcome,
    RemovalResponse,
    StrayModel,
)

settings = Settings()
service = ModelsService(settings)
config_file = ConfigFile(settings.llama_swap_config_path)

app = FastAPI(title="swapboard API", version="1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info", response_model=InferenceInfo)
def get_info() -> InferenceInfo:
    return InferenceInfo(
        port=settings.llama_swap_port,
        endpoint_url=settings.public_endpoint_url,
    )


@app.get("/config", response_model=ConfigDocument)
def get_config() -> ConfigDocument:
    text = config_file.read()
    return ConfigDocument(text=text, warnings=config_file.validate(text).warnings)


# The config's `cmd` lines are shell commands llama-swap executes, so anyone who
# can call this can run code as the llama-swap user. That is the same authority
# the file itself already grants, and the API is expected to stay on loopback;
# exposing it beyond that means exposing this.
@app.put("/config", response_model=ConfigSaveResponse)
def save_config(update: ConfigUpdate, response: Response) -> ConfigSaveResponse:
    validation = config_file.write(update.text)
    if not validation.valid:
        response.status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        return ConfigSaveResponse(
            saved=False,
            message="The configuration was not saved.",
            errors=validation.errors,
        )
    return ConfigSaveResponse(
        saved=True,
        message="Configuration saved.",
        warnings=validation.warnings,
    )


@app.get("/stray-models", response_model=list[StrayModel])
def list_stray_models() -> list[StrayModel]:
    return service.list_stray()


@app.delete("/stray-models/{relative_path:path}", response_model=RemovalResponse)
def remove_stray_model(relative_path: str) -> RemovalResponse:
    outcome = service.remove_stray(relative_path)
    if not outcome.found:
        raise HTTPException(status_code=404, detail=outcome.message)
    return RemovalResponse(removed=outcome.removed, message=outcome.message)


# llama-swap model names may contain slashes, so the routes below match the rest
# of the path instead of a single segment. That makes them greedy: any further
# route under /models has to be declared above them to stay reachable.
@app.get("/models", response_model=list[ModelStatus])
def list_models() -> list[ModelStatus]:
    return service.list_status()


@app.get("/models/{name:path}", response_model=ModelStatus)
def get_model(name: str) -> ModelStatus:
    status_for_model = service.get_status(name)
    if status_for_model is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
    return status_for_model


@app.delete("/models/{name:path}", response_model=RemovalResponse)
def remove_model(name: str) -> RemovalResponse:
    outcome = _guarded(lambda: service.remove(name))
    if not outcome.found:
        raise HTTPException(status_code=404, detail=outcome.message)
    return RemovalResponse(removed=outcome.removed, message=outcome.message)


@app.post("/models/{name:path}/download", response_model=DownloadResponse)
def download_model(name: str) -> DownloadResponse:
    outcome = service.start_download(name)
    if not outcome.found:
        raise HTTPException(status_code=404, detail=outcome.message)
    return DownloadResponse(started=outcome.started, message=outcome.message)


def _guarded(removal: Callable[[], RemovalOutcome]) -> RemovalOutcome:
    """Turns an escape from the models directory into a refusal, not a 500."""
    try:
        return removal()
    except OutsideStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
