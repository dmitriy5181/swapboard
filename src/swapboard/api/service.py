import logging
import threading
from pathlib import Path

from huggingface_hub import hf_hub_download

from swapboard.api.config import parse_model_sources
from swapboard.api.settings import Settings
from swapboard.common.models import (
    DownloadOutcome,
    DownloadProgress,
    DownloadState,
    ModelFile,
    ModelSource,
    ModelStatus,
)

logger = logging.getLogger(__name__)


class Downloads:
    """Tracks in-flight downloads across the request threads that start them."""

    def __init__(self) -> None:
        self._by_model: dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> DownloadProgress:
        with self._lock:
            return self._by_model.get(name, DownloadProgress())

    def set(self, name: str, progress: DownloadProgress) -> None:
        with self._lock:
            self._by_model[name] = progress

    def try_begin(self, name: str) -> bool:
        """Claims the download slot for a model, or reports it already taken.

        Checking and claiming under one lock is what stops two concurrent
        requests from both starting a download of the same model.
        """
        with self._lock:
            current = self._by_model.get(name, DownloadProgress())
            if current.state == DownloadState.DOWNLOADING:
                return False
            self._by_model[name] = DownloadProgress(state=DownloadState.DOWNLOADING)
            return True


class ModelsService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._downloads = Downloads()

    def list_status(self) -> list[ModelStatus]:
        return [self._status_for(source) for source in self._sources()]

    def get_status(self, name: str) -> ModelStatus | None:
        source = self._source(name)
        if source is None:
            return None
        return self._status_for(source)

    def start_download(self, name: str) -> DownloadOutcome:
        source = self._source(name)
        if source is None:
            return DownloadOutcome(
                found=False, started=False, message=f"Unknown model '{name}'"
            )
        if self._is_present(source):
            return DownloadOutcome(
                found=True, started=False, message="Model already present"
            )
        if not self._downloads.try_begin(name):
            return DownloadOutcome(
                found=True, started=False, message="Download already in progress"
            )

        thread = threading.Thread(
            target=self._run_download, args=(source,), daemon=True
        )
        thread.start()
        return DownloadOutcome(found=True, started=True, message="Download started")

    def _sources(self) -> list[ModelSource]:
        """Re-reads the config on every call so edits are picked up live."""
        return parse_model_sources(self._settings.llama_swap_config_path)

    def _source(self, name: str) -> ModelSource | None:
        return next((source for source in self._sources() if source.name == name), None)

    def _status_for(self, source: ModelSource) -> ModelStatus:
        progress = self._downloads.get(source.name)
        primary_file = source.primary_file
        return ModelStatus(
            name=source.name,
            repo_id=primary_file.repo_id,
            filename=primary_file.filename,
            path=str(self._resolve(primary_file)),
            present=self._is_present(source),
            download_state=progress.state,
            download_error=progress.error,
        )

    def _resolve(self, model_file: ModelFile) -> Path:
        return Path(self._settings.models_path) / model_file.relative_path

    def _is_present(self, source: ModelSource) -> bool:
        return all(self._is_file_present(model_file) for model_file in source.files)

    def _is_file_present(self, model_file: ModelFile) -> bool:
        path = self._resolve(model_file)
        return path.exists() and path.stat().st_size > 0

    def _run_download(self, source: ModelSource) -> None:
        """Fetches whichever of a model's files are still missing.

        Skipping files that already arrived lets a failed multi-file download be
        retried without refetching the multi-gigabyte parts that succeeded.
        """
        try:
            for model_file in source.files:
                if self._is_file_present(model_file):
                    continue
                local_dir = self._resolve(model_file).parent
                hf_hub_download(
                    repo_id=model_file.repo_id,
                    filename=model_file.filename,
                    local_dir=str(local_dir),
                    token=self._settings.hf_token,
                )
            self._downloads.set(
                source.name, DownloadProgress(state=DownloadState.COMPLETED)
            )
        except Exception as exc:
            logger.exception("Failed to download model '%s'", source.name)
            self._downloads.set(
                source.name,
                DownloadProgress(state=DownloadState.FAILED, error=str(exc)),
            )
