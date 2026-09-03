import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from swapboard.runtimes import installer
from swapboard.runtimes.installer import RuntimeVerificationError
from swapboard.runtimes.manifest import RuntimeArtifact

RUNTIME = "llama-swap"


def build_archive(path: Path, entries: dict[str, bytes]) -> str:
    """Writes a gzip tarball and returns its digest."""
    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_artifact(
    archive: Path,
    digest: str,
    *,
    version: str = "1",
    strip_components: int = 0,
    entrypoint: str = "llama-swap",
) -> RuntimeArtifact:
    return RuntimeArtifact(
        runtime=RUNTIME,
        version=version,
        url=archive.as_uri(),
        sha256=digest,
        strip_components=strip_components,
        entrypoint=entrypoint,
    )


@pytest.fixture
def pin(monkeypatch):
    """Points the installer at a locally built archive instead of the network."""

    def apply(artifact: RuntimeArtifact) -> list[str]:
        requested: list[str] = []
        monkeypatch.setattr(installer, "resolve", lambda runtime: artifact)
        original = installer.urlopen

        def recording_urlopen(url, *args, **kwargs):
            requested.append(url)
            return original(url, *args, **kwargs)

        monkeypatch.setattr(installer, "urlopen", recording_urlopen)
        return requested

    return apply


def test_install_unpacks_flat_archive(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    digest = build_archive(archive, {"llama-swap": b"binary", "LICENSE.md": b"mit"})
    pin(build_artifact(archive, digest))
    runtimes_dir = tmp_path / "runtimes"

    entrypoint = installer.install(RUNTIME, runtimes_dir)

    assert entrypoint == runtimes_dir / RUNTIME / "llama-swap"
    assert entrypoint.read_bytes() == b"binary"
    assert (runtimes_dir / RUNTIME / "LICENSE.md").is_file()
    assert entrypoint.stat().st_mode & 0o111


def test_install_strips_wrapper_directory(tmp_path: Path, pin) -> None:
    archive = tmp_path / "wrapped.tar.gz"
    digest = build_archive(
        archive,
        {"llama-b1/llama-swap": b"binary", "llama-b1/libggml.dylib": b"lib"},
    )
    pin(build_artifact(archive, digest, strip_components=1))
    runtimes_dir = tmp_path / "runtimes"

    installer.install(RUNTIME, runtimes_dir)

    assert (runtimes_dir / RUNTIME / "llama-swap").read_bytes() == b"binary"
    assert (runtimes_dir / RUNTIME / "libggml.dylib").is_file()
    assert not (runtimes_dir / RUNTIME / "llama-b1").exists()


def test_install_records_version_and_is_idempotent(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    digest = build_archive(archive, {"llama-swap": b"binary"})
    requested = pin(build_artifact(archive, digest, version="250"))
    runtimes_dir = tmp_path / "runtimes"

    installer.install(RUNTIME, runtimes_dir)
    installer.install(RUNTIME, runtimes_dir)

    assert len(requested) == 1
    assert installer.installed_version(RUNTIME, runtimes_dir) == "250"
    assert installer.is_current(RUNTIME, runtimes_dir) is True


def test_force_reinstalls_current_runtime(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    digest = build_archive(archive, {"llama-swap": b"binary"})
    requested = pin(build_artifact(archive, digest))
    runtimes_dir = tmp_path / "runtimes"

    installer.install(RUNTIME, runtimes_dir)
    installer.install(RUNTIME, runtimes_dir, force=True)

    assert len(requested) == 2


def test_upgrade_replaces_previous_contents(tmp_path: Path, pin) -> None:
    runtimes_dir = tmp_path / "runtimes"
    old_archive = tmp_path / "old.tar.gz"
    old_digest = build_archive(
        old_archive, {"llama-swap": b"old", "stale.txt": b"remove me"}
    )
    pin(build_artifact(old_archive, old_digest, version="249"))
    installer.install(RUNTIME, runtimes_dir)

    new_archive = tmp_path / "new.tar.gz"
    new_digest = build_archive(new_archive, {"llama-swap": b"new"})
    pin(build_artifact(new_archive, new_digest, version="250"))
    installer.install(RUNTIME, runtimes_dir)

    assert (runtimes_dir / RUNTIME / "llama-swap").read_bytes() == b"new"
    assert not (runtimes_dir / RUNTIME / "stale.txt").exists()
    assert installer.installed_version(RUNTIME, runtimes_dir) == "250"
    assert not list(runtimes_dir.glob(".*.previous"))


def test_digest_mismatch_rejects_archive(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    build_archive(archive, {"llama-swap": b"binary"})
    pin(build_artifact(archive, "0" * 64))
    runtimes_dir = tmp_path / "runtimes"

    with pytest.raises(RuntimeVerificationError, match="digest mismatch"):
        installer.install(RUNTIME, runtimes_dir)

    assert not (runtimes_dir / RUNTIME).exists()


def test_digest_mismatch_leaves_existing_install_intact(tmp_path: Path, pin) -> None:
    runtimes_dir = tmp_path / "runtimes"
    good = tmp_path / "good.tar.gz"
    good_digest = build_archive(good, {"llama-swap": b"good"})
    pin(build_artifact(good, good_digest, version="249"))
    installer.install(RUNTIME, runtimes_dir)

    bad = tmp_path / "bad.tar.gz"
    build_archive(bad, {"llama-swap": b"bad"})
    pin(build_artifact(bad, "1" * 64, version="250"))

    with pytest.raises(RuntimeVerificationError):
        installer.install(RUNTIME, runtimes_dir)

    assert (runtimes_dir / RUNTIME / "llama-swap").read_bytes() == b"good"
    assert installer.installed_version(RUNTIME, runtimes_dir) == "249"


def test_missing_entrypoint_is_rejected(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    digest = build_archive(archive, {"README.md": b"docs"})
    pin(build_artifact(archive, digest))
    runtimes_dir = tmp_path / "runtimes"

    with pytest.raises(RuntimeVerificationError, match="does not contain llama-swap"):
        installer.install(RUNTIME, runtimes_dir)


def test_archive_without_single_root_rejects_strip(tmp_path: Path, pin) -> None:
    archive = tmp_path / "two-roots.tar.gz"
    digest = build_archive(archive, {"a/llama-swap": b"one", "b/llama-swap": b"two"})
    pin(build_artifact(archive, digest, strip_components=1))

    with pytest.raises(RuntimeVerificationError, match="single top-level directory"):
        installer.install(RUNTIME, tmp_path / "runtimes")


def test_path_traversal_is_refused(tmp_path: Path, pin) -> None:
    archive = tmp_path / "evil.tar.gz"
    digest = build_archive(archive, {"../escaped": b"pwned", "llama-swap": b"binary"})
    pin(build_artifact(archive, digest))
    runtimes_dir = tmp_path / "runtimes"

    with pytest.raises(tarfile.TarError):
        installer.install(RUNTIME, runtimes_dir)

    assert not (tmp_path / "escaped").exists()
    assert not (runtimes_dir / RUNTIME).exists()


def test_is_current_is_false_when_version_differs(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    digest = build_archive(archive, {"llama-swap": b"binary"})
    runtimes_dir = tmp_path / "runtimes"
    pin(build_artifact(archive, digest, version="249"))
    installer.install(RUNTIME, runtimes_dir)

    pin(build_artifact(archive, digest, version="250"))

    assert installer.is_current(RUNTIME, runtimes_dir) is False


def test_status_helpers_on_empty_directory(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    digest = build_archive(archive, {"llama-swap": b"binary"})
    pin(build_artifact(archive, digest))
    runtimes_dir = tmp_path / "runtimes"

    assert installer.installed_version(RUNTIME, runtimes_dir) is None
    assert installer.is_current(RUNTIME, runtimes_dir) is False
    assert installer.remove(RUNTIME, runtimes_dir) is False


def test_remove_deletes_installed_runtime(tmp_path: Path, pin) -> None:
    archive = tmp_path / "flat.tar.gz"
    digest = build_archive(archive, {"llama-swap": b"binary"})
    pin(build_artifact(archive, digest))
    runtimes_dir = tmp_path / "runtimes"
    installer.install(RUNTIME, runtimes_dir)

    assert installer.remove(RUNTIME, runtimes_dir) is True
    assert not (runtimes_dir / RUNTIME).exists()
