import hashlib
import re
from importlib.resources import files

import pytest
from jsonschema import Draft7Validator

from swapboard.runtimes.manifest import (
    ARTIFACTS,
    LLAMA_SWAP_CONFIG_SCHEMA,
    LLAMA_SWAP_VERSION,
    RUNTIMES,
    UnsupportedPlatformError,
    is_supported,
    load_config_schema,
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


def test_vendored_schema_matches_its_pinned_digest() -> None:
    """A refresh that skips the digest would validate against the wrong release."""
    resource = files("swapboard.runtimes").joinpath(LLAMA_SWAP_CONFIG_SCHEMA.resource)

    digest = hashlib.sha256(resource.read_bytes()).hexdigest()

    assert digest == LLAMA_SWAP_CONFIG_SCHEMA.sha256


def test_schema_is_pinned_to_the_installed_llama_swap() -> None:
    assert LLAMA_SWAP_CONFIG_SCHEMA.version == LLAMA_SWAP_VERSION
    assert f"/v{LLAMA_SWAP_VERSION}/" in LLAMA_SWAP_CONFIG_SCHEMA.url


def test_loaded_schema_is_usable_as_draft_seven() -> None:
    Draft7Validator.check_schema(load_config_schema())
