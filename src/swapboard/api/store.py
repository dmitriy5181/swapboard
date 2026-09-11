"""The models directory: what a configured model owns, and what nothing owns."""

import os
import shutil
from collections import defaultdict
from pathlib import Path

from swapboard.common.models import ModelFile, ModelSource, StrayModel

GGUF_SUFFIX = ".gguf"


class OutsideStoreError(ValueError):
    """Raised for a path that would reach outside the models directory."""


class ModelStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def resolve(self, model_file: ModelFile) -> Path:
        return self._within(model_file.relative_path)

    def is_present(self, source: ModelSource) -> bool:
        return all(self.has_content(file) for file in source.files)

    def has_content(self, model_file: ModelFile) -> bool:
        """A zero-byte file is a failed download, not an arrived one."""
        return self._size_of_file(model_file) > 0

    def size_of(self, source: ModelSource) -> int:
        return sum(self._size_of_file(file) for file in source.files)

    def remove(self, source: ModelSource) -> bool:
        """Deletes a model's files, reporting whether anything was there.

        Directories left holding no GGUF are removed outright, which also
        clears the `.cache` metadata `hf_hub_download` leaves behind. A
        directory shared with another model keeps that model's files and so
        survives.
        """
        directories = set()
        removed = False
        for model_file in source.files:
            path = self.resolve(model_file)
            directories.add(path.parent)
            if path.exists():
                path.unlink()
                removed = True
        for directory in directories:
            self._prune(directory)
        return removed

    def remove_path(self, relative_path: str) -> bool:
        target = self._within(relative_path)
        if not target.exists():
            return False
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return True

    def stray(self, sources: list[ModelSource]) -> list[StrayModel]:
        """Groups GGUF files no configured model claims, by their directory."""
        configured = {file.relative_path for source in sources for file in source.files}
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in self._gguf_files():
            relative = path.relative_to(self._root)
            if str(relative) in configured:
                continue
            grouped[_group_of(relative)].append(path)

        return [
            StrayModel(
                relative_path=relative_path,
                files=tuple(sorted(path.name for path in paths)),
                size_bytes=sum(_size(path) for path in paths),
            )
            for relative_path, paths in sorted(grouped.items())
        ]

    def _within(self, relative_path: str) -> Path:
        """Keeps a caller-supplied path inside the store.

        Stray paths arrive from HTTP requests, and a model path is only ever as
        trustworthy as the config it was read from.
        """
        root = self._root.resolve()
        candidate = (root / relative_path).resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise OutsideStoreError(
                f"'{relative_path}' is outside the models directory"
            )
        return candidate

    def _gguf_files(self) -> list[Path]:
        if not self._root.is_dir():
            return []
        return [
            path
            for path in self._root.rglob(f"*{GGUF_SUFFIX}")
            if path.is_file() and not _is_hidden(path.relative_to(self._root))
        ]

    def _prune(self, directory: Path) -> None:
        if directory == self._root.resolve() or not directory.is_dir():
            return
        if any(directory.rglob(f"*{GGUF_SUFFIX}")):
            return
        shutil.rmtree(directory)

    def _size_of_file(self, model_file: ModelFile) -> int:
        path = self.resolve(model_file)
        return path.stat().st_size if path.is_file() else 0


def _group_of(relative: Path) -> str:
    """Names the entry a stray file belongs to.

    Files sit at `<org>/<repo>/<file>` and are grouped by that directory. One
    dropped straight into the models directory has no directory to group by and
    stands alone, so that removing it cannot mean removing the whole store.
    """
    parent = relative.parent
    return str(relative) if parent == Path(".") else str(parent)


def _is_hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def _size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0
