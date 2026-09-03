import re

import pytest

from swapboard.runtimes.manifest import (
    ARTIFACTS,
    RUNTIMES,
    UnsupportedPlatformError,
    is_supported,
    resolve,
)

SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
SUPPORTED_MACHINES = ("arm64", "x86_64")


def test_every_supported_platform_has_both_runtimes() -> None:
    for machine in SUPPORTED_MACHINES:
        for runtime in RUNTIMES:
            assert (runtime, "darwin", machine) in ARTIFACTS


def test_artifacts_are_pinned_and_well_formed() -> None:
    for (runtime, _, _), artifact in ARTIFACTS.items():
        assert artifact.runtime == runtime
        assert SHA256.match(artifact.sha256), artifact.url
        assert artifact.url.startswith("https://")
        assert artifact.version in artifact.url
        assert artifact.entrypoint
        assert artifact.strip_components >= 0


def test_digests_are_unique_per_artifact() -> None:
    digests = [artifact.sha256 for artifact in ARTIFACTS.values()]

    assert len(digests) == len(set(digests))


def test_filename_is_derived_from_url() -> None:
    artifact = resolve("llama-swap", "darwin", "arm64")

    assert artifact.filename == "llama-swap_250_darwin_arm64.tar.gz"


@pytest.mark.parametrize("machine", SUPPORTED_MACHINES)
def test_resolve_returns_artifact_for_supported_platform(machine: str) -> None:
    artifact = resolve("llama-cpp", "darwin", machine)

    assert artifact.entrypoint == "llama-server"
    assert artifact.strip_components == 1


def test_resolve_rejects_unsupported_platform() -> None:
    with pytest.raises(UnsupportedPlatformError, match="linux/x86_64"):
        resolve("llama-cpp", "linux", "x86_64")


def test_is_supported_reports_platform_coverage() -> None:
    assert is_supported("darwin", "arm64") is True
    assert is_supported("darwin", "x86_64") is True
    assert is_supported("linux", "x86_64") is False
