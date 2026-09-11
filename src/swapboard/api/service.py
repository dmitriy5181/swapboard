import logging
import threading

from huggingface_hub import hf_hub_download

from swapboard.api.config import parse_model_sources
from swapboard.api.llamaswap import LlamaSwapCatalog
from swapboard.api.settings import Settings
from swapboard.api.store import ModelStore
from swapboard.common.models import (
    DownloadOutcome,
    DownloadProgress,
    DownloadState,
    ModelMeta,
    ModelSource,
    ModelStatus,
    RemovalOutcome,
    StrayModel,
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

    def clear(self, name: str) -> None:
        with self._lock:
            self._by_model.pop(name, None)

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
    def __init__(
        self, settings: Settings, catalog: LlamaSwapCatalog | None = None
    ) -> None:
        self._settings = settings
        self._store = ModelStore(settings.models_path)
        self._catalog = catalog or LlamaSwapCatalog(settings.llama_swap_url)
        self._downloads = Downloads()

    def list_status(self) -> list[ModelStatus]:
        meta_by_name = self._catalog.fetch_meta()
        return [
            self._status_for(source, meta_by_name.get(source.name))
            for source in self._sources()
        ]

    def get_status(self, name: str) -> ModelStatus | None:
        source = self._source(name)
        if source is None:
            return None
        return self._status_for(source, self._catalog.fetch_meta().get(name))

    def list_stray(self) -> list[StrayModel]:
        return self._store.stray(self._sources())

    def start_download(self, name: str) -> DownloadOutcome:
        source = self._source(name)
        if source is None:
            return DownloadOutcome(
                found=False, started=False, message=f"Unknown model '{name}'"
            )
        if self._store.is_present(source):
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

    def remove(self, name: str) -> RemovalOutcome:
        """Deletes a configured model's files, leaving it in the config.

        The model stays listed and downloadable; forgetting it entirely is an
        edit to the llama-swap config, not a removal.
        """
        source = self._source(name)
        if source is None:
            return RemovalOutcome(
                found=False, removed=False, message=f"Unknown model '{name}'"
            )
        if self._downloads.get(name).state == DownloadState.DOWNLOADING:
            return RemovalOutcome(
                found=True, removed=False, message="Download in progress"
            )

        removed = self._store.remove(source)
        self._downloads.clear(name)
        message = "Model removed" if removed else "Model was not downloaded"
        return RemovalOutcome(found=True, removed=removed, message=message)

    def remove_stray(self, relative_path: str) -> RemovalOutcome:
        if not self._store.remove_path(relative_path):
            return RemovalOutcome(
                found=False,
                removed=False,
                message=f"Unknown stray model '{relative_path}'",
            )
        return RemovalOutcome(found=True, removed=True, message="Stray model removed")

    def _sources(self) -> list[ModelSource]:
        """Re-reads the config on every call so edits are picked up live."""
        return parse_model_sources(self._settings.llama_swap_config_path)

    def _source(self, name: str) -> ModelSource | None:
        return next((source for source in self._sources() if source.name == name), None)

    def _status_for(self, source: ModelSource, meta: ModelMeta | None) -> ModelStatus:
        progress = self._downloads.get(source.name)
        primary_file = source.primary_file
        return ModelStatus(
            name=source.name,
            repo_id=primary_file.repo_id,
            filename=primary_file.filename,
            path=str(self._store.resolve(primary_file)),
            present=self._store.is_present(source),
            download_state=progress.state,
            download_error=progress.error,
            size_bytes=self._store.size_of(source),
            meta=meta,
        )

    def _run_download(self, source: ModelSource) -> None:
        """Fetches whichever of a model's files are still missing.

        Skipping files that already arrived lets a failed multi-file download be
        retried without refetching the multi-gigabyte parts that succeeded.
        """
        try:
            for model_file in source.files:
                if self._store.has_content(model_file):
                    continue
                local_dir = self._store.resolve(model_file).parent
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
