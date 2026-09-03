"""Downloads pinned runtimes into a private directory, off the host PATH."""

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from swapboard.runtimes.manifest import RuntimeArtifact, resolve

logger = logging.getLogger(__name__)

MARKER_NAME = ".swapboard-runtime.json"
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 1024 * 1024


class RuntimeVerificationError(RuntimeError):
    """Raised when a download does not match its pinned digest."""


def entrypoint_path(runtime: str, runtimes_dir: Path) -> Path:
    return runtimes_dir / runtime / resolve(runtime).entrypoint


def installed_version(runtime: str, runtimes_dir: Path) -> str | None:
    """Reports which version is installed, or None if nothing usable is."""
    marker = runtimes_dir / runtime / MARKER_NAME
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    version = recorded.get("version")
    return version if isinstance(version, str) else None


def is_current(runtime: str, runtimes_dir: Path) -> bool:
    if installed_version(runtime, runtimes_dir) != resolve(runtime).version:
        return False
    entrypoint = entrypoint_path(runtime, runtimes_dir)
    return entrypoint.is_file() and _is_executable(entrypoint)


def install(runtime: str, runtimes_dir: Path, *, force: bool = False) -> Path:
    """Ensures the pinned build of *runtime* is present, returning its binary."""
    if not force and is_current(runtime, runtimes_dir):
        logger.debug("%s is already at the pinned version", runtime)
        return entrypoint_path(runtime, runtimes_dir)

    artifact = resolve(runtime)
    runtimes_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Installing %s %s", artifact.runtime, artifact.version)

    with tempfile.TemporaryDirectory(dir=runtimes_dir) as work_dir:
        archive = Path(work_dir) / artifact.filename
        _download(artifact, archive)
        staged = Path(work_dir) / "staged"
        _extract(archive, staged, artifact.strip_components)
        entrypoint = staged / artifact.entrypoint
        if not entrypoint.is_file():
            raise RuntimeVerificationError(
                f"{artifact.runtime} archive does not contain {artifact.entrypoint}"
            )
        entrypoint.chmod(entrypoint.stat().st_mode | 0o111)
        _write_marker(staged, artifact)
        _swap_into_place(staged, runtimes_dir / runtime)

    return entrypoint_path(runtime, runtimes_dir)


def install_all(runtimes_dir: Path, *, force: bool = False) -> dict[str, Path]:
    from swapboard.runtimes.manifest import RUNTIMES

    return {
        runtime: install(runtime, runtimes_dir, force=force) for runtime in RUNTIMES
    }


def remove(runtime: str, runtimes_dir: Path) -> bool:
    target = runtimes_dir / runtime
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _download(artifact: RuntimeArtifact, destination: Path) -> None:
    """Streams the artifact to disk, verifying its digest before it is used."""
    digest = hashlib.sha256()
    with urlopen(artifact.url, timeout=DOWNLOAD_TIMEOUT) as response:
        with destination.open("wb") as handle:
            while chunk := response.read(CHUNK_SIZE):
                digest.update(chunk)
                handle.write(chunk)
    actual = digest.hexdigest()
    if actual != artifact.sha256:
        raise RuntimeVerificationError(
            f"{artifact.filename} digest mismatch: "
            f"expected {artifact.sha256}, got {actual}"
        )


def _extract(archive: Path, destination: Path, strip_components: int) -> None:
    """Unpacks *archive* to *destination*, dropping wrapper directories.

    `filter="data"` rejects absolute paths, parent traversal, links and device
    nodes, so a tampered archive cannot write outside the destination.
    """
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        root = Path(staging)
        for _ in range(strip_components):
            children = list(root.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise RuntimeVerificationError(
                    f"{archive.name} does not have a single top-level directory"
                )
            root = children[0]
        root.replace(destination)


def _write_marker(target: Path, artifact: RuntimeArtifact) -> None:
    marker = target / MARKER_NAME
    marker.write_text(
        json.dumps(
            {
                "runtime": artifact.runtime,
                "version": artifact.version,
                "sha256": artifact.sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _swap_into_place(staged: Path, target: Path) -> None:
    """Replaces *target* with *staged* through renames.

    Callers stop the services that use the runtime first, so the brief window
    where the directory is absent is not observable.
    """
    previous = target.with_name(f".{target.name}.previous")
    shutil.rmtree(previous, ignore_errors=True)
    if target.exists():
        target.replace(previous)
    try:
        staged.replace(target)
    except OSError:
        if previous.exists():
            previous.replace(target)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def _is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)
