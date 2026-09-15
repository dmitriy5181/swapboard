"""The models directory: what a configured model owns, and what nothing owns."""

import os
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from swapboard.common.models import ModelFile, ModelSource, StrayModel

GGUF_SUFFIX = ".gguf"


class OutsideStoreError(ValueError):
    """Raised for a path that would reach outside the models directory."""


class ModelStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).resolve()

    def resolve(self, model_file: ModelFile) -> Path:
        return self._within(model_file.relative_path)

    def repository_directory(self, model_file: ModelFile) -> Path:
        """Returns the local root corresponding to a Hugging Face repository."""
        return self._within(model_file.repo_id)

    def is_present(self, source: ModelSource) -> bool:
        return all(self.has_content(file) for file in source.files)

    def has_content(self, model_file: ModelFile) -> bool:
        """A zero-byte file is a failed download, not an arrived one."""
        return self._size_of_file(model_file) > 0

    def size_of(self, source: ModelSource) -> int:
        return sum(self._size_of_file(file) for file in source.files)

    def remove(self, source: ModelSource, claimed_elsewhere: Iterable[str]) -> bool:
        """Deletes a model's files, sparing any another model still claims.

        Two configured entries may name the same GGUF -- the same weights
        served with different settings -- and removing one of them must not
        take the file the other one runs on.

        The `.cache` metadata `hf_hub_download` leaves behind is removed once
        no model files remain, while unrelated files are preserved.
        """
        spared = self._occupied_by(claimed_elsewhere)
        directories = set()
        repositories = set()
        removed = False
        for model_file in source.files:
            path = self.resolve(model_file)
            if path in spared:
                continue
            directories.add(path.parent)
            repositories.add(self.repository_directory(model_file))
            if path.exists():
                path.unlink()
                removed = True
        self._prune_all(directories, repositories)
        return removed

    def remove_stray(self, relative_path: str, claimed: Iterable[str]) -> bool:
        """Deletes the unclaimed files of one stray entry, and nothing else.

        The inventory is recomputed here rather than taken on trust, so a path
        naming files a configured model still claims deletes none of them: a
        request can only ever remove what this store would report as stray.
        """
        targets = [
            path
            for path in self._unclaimed(claimed)
            if self._group_of(path) == relative_path
        ]
        if not targets:
            return False

        directories = {path.parent for path in targets}
        repositories = {self._repository_directory_for(path) for path in targets}
        for path in targets:
            path.unlink()
        self._prune_all(directories, repositories)
        return True

    def stray(self, claimed: Iterable[str]) -> list[StrayModel]:
        """Groups GGUF files no configured model claims, by their directory."""
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in self._unclaimed(claimed):
            grouped[self._group_of(path)].append(path)

        return [
            StrayModel(
                relative_path=relative_path,
                files=tuple(sorted(path.name for path in paths)),
                size_bytes=sum(_size(path) for path in paths),
            )
            for relative_path, paths in sorted(grouped.items())
        ]

    def _unclaimed(self, claimed: Iterable[str]) -> list[Path]:
        occupied = self._occupied_by(claimed)
        return [path for path in self._gguf_files() if path not in occupied]

    def _occupied_by(self, claimed: Iterable[str]) -> set[Path]:
        """Reads configured model paths as the files they occupy here.

        A configuration spells its models directory however it likes, and one
        whose path cannot be traced back to Hugging Face still names a file in
        use; resolving both against the store is what keeps a live model from
        being offered up as stray.
        """
        occupied = set()
        for claim in claimed:
            path = Path(claim)
            if not path.is_absolute():
                path = self._root / path
            occupied.add(path.resolve())
        return occupied

    def _group_of(self, path: Path) -> str:
        """Names the entry a stray file belongs to.

        Files sit at `<org>/<repo>/<file>` and are grouped by that directory.
        One dropped straight into the models directory has no directory to
        group by and stands alone.
        """
        relative = path.relative_to(self._root)
        return str(relative) if relative.parent == Path(".") else str(relative.parent)

    def _within(self, relative_path: str) -> Path:
        """Keeps a caller-supplied path inside the store.

        A model path is only ever as trustworthy as the config it was read
        from.
        """
        candidate = (self._root / relative_path).resolve()
        if candidate == self._root or not candidate.is_relative_to(self._root):
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

    def _prune_all(
        self, directories: Iterable[Path], repositories: Iterable[Path]
    ) -> None:
        repository_set = set(repositories)
        deepest_first = sorted(
            set(directories) | repository_set,
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in deepest_first:
            self._prune(directory, should_remove_cache=directory in repository_set)

    def _prune(self, directory: Path, *, should_remove_cache: bool) -> None:
        if directory == self._root or not directory.is_dir():
            return
        if self._contains_model_file(directory):
            return
        if should_remove_cache:
            cache = directory / ".cache"
            if cache.is_dir():
                shutil.rmtree(cache)
        if not any(directory.iterdir()):
            directory.rmdir()

    def _repository_directory_for(self, path: Path) -> Path:
        relative = path.relative_to(self._root)
        if len(relative.parts) < 3:
            return path.parent
        return self._root.joinpath(*relative.parts[:2])

    def _contains_model_file(self, directory: Path) -> bool:
        return any(
            path.is_file() and not _is_hidden(path.relative_to(self._root))
            for path in directory.rglob(f"*{GGUF_SUFFIX}")
        )

    def _size_of_file(self, model_file: ModelFile) -> int:
        path = self.resolve(model_file)
        return path.stat().st_size if path.is_file() else 0


def _is_hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def _size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0
