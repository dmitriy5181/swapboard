from fastapi import FastAPI, HTTPException

from swapboard.api.service import ModelsService
from swapboard.api.settings import Settings
from swapboard.common.models import DownloadResponse, InferenceInfo, ModelStatus

settings = Settings()
service = ModelsService(settings)

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


@app.get("/models", response_model=list[ModelStatus])
def list_models() -> list[ModelStatus]:
    return service.list_status()


@app.get("/models/{name}", response_model=ModelStatus)
def get_model(name: str) -> ModelStatus:
    status = service.get_status(name)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
    return status


@app.post("/models/{name}/download", response_model=DownloadResponse)
def download_model(name: str) -> DownloadResponse:
    outcome = service.start_download(name)
    if not outcome.found:
        raise HTTPException(status_code=404, detail=outcome.message)
    return DownloadResponse(started=outcome.started, message=outcome.message)
