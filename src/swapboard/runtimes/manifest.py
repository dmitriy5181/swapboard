"""Pinned upstream builds of llama.cpp and llama-swap.

Every artifact is a released tarball with a recorded digest, so an installation
is reproducible and can be verified before anything is extracted. Bumping a
runtime means changing the version, URL and digest here and nowhere else.
"""

import platform
from dataclasses import dataclass

from swapboard.common.paths import LLAMA_CPP_RUNTIME, LLAMA_SWAP_RUNTIME

LLAMA_CPP_VERSION = "b10360"
LLAMA_SWAP_VERSION = "250"

LLAMA_CPP_RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"
LLAMA_SWAP_RELEASES = "https://github.com/mostlygeek/llama-swap/releases/download"


class UnsupportedPlatformError(RuntimeError):
    """Raised when no pinned build exists for the running machine."""


@dataclass(frozen=True)
class RuntimeArtifact:
    runtime: str
    version: str
    url: str
    sha256: str
    # llama.cpp wraps its build in a `llama-<version>/` directory; llama-swap
    # does not. Stripping normalises both onto a flat runtime directory.
    strip_components: int
    entrypoint: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


ARTIFACTS: dict[tuple[str, str, str], RuntimeArtifact] = {
    (LLAMA_CPP_RUNTIME, "darwin", "arm64"): RuntimeArtifact(
        runtime=LLAMA_CPP_RUNTIME,
        version=LLAMA_CPP_VERSION,
        url=f"{LLAMA_CPP_RELEASES}/{LLAMA_CPP_VERSION}"
        f"/llama-{LLAMA_CPP_VERSION}-bin-macos-arm64.tar.gz",
        sha256="5ce375f59194be1482a89be97bdbd2cedb1a91f8d434ef29502787fbd0fede7e",
        strip_components=1,
        entrypoint="llama-server",
    ),
    (LLAMA_CPP_RUNTIME, "darwin", "x86_64"): RuntimeArtifact(
        runtime=LLAMA_CPP_RUNTIME,
        version=LLAMA_CPP_VERSION,
        url=f"{LLAMA_CPP_RELEASES}/{LLAMA_CPP_VERSION}"
        f"/llama-{LLAMA_CPP_VERSION}-bin-macos-x64.tar.gz",
        sha256="e64f2eaa5051e64eb1970cfd978a8d67686206b027b83c63d68fa9abe3b7f2b5",
        strip_components=1,
        entrypoint="llama-server",
    ),
    (LLAMA_SWAP_RUNTIME, "darwin", "arm64"): RuntimeArtifact(
        runtime=LLAMA_SWAP_RUNTIME,
        version=LLAMA_SWAP_VERSION,
        url=f"{LLAMA_SWAP_RELEASES}/v{LLAMA_SWAP_VERSION}"
        f"/llama-swap_{LLAMA_SWAP_VERSION}_darwin_arm64.tar.gz",
        sha256="ebad7fe9beb7b74a6574582b7180dddc6f6bfe905bed38458bf9eb07d3092eef",
        strip_components=0,
        entrypoint="llama-swap",
    ),
    (LLAMA_SWAP_RUNTIME, "darwin", "x86_64"): RuntimeArtifact(
        runtime=LLAMA_SWAP_RUNTIME,
        version=LLAMA_SWAP_VERSION,
        url=f"{LLAMA_SWAP_RELEASES}/v{LLAMA_SWAP_VERSION}"
        f"/llama-swap_{LLAMA_SWAP_VERSION}_darwin_amd64.tar.gz",
        sha256="c1a0a3148933a2d1e20f29605d23685511bacc6b7e4109e863c8ecd465529cd2",
        strip_components=0,
        entrypoint="llama-swap",
    ),
}

RUNTIMES = (LLAMA_CPP_RUNTIME, LLAMA_SWAP_RUNTIME)

_MACHINE_ALIASES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "amd64": "x86_64",
    "x86_64": "x86_64",
}


def current_platform() -> tuple[str, str]:
    system = platform.system().lower()
    machine = _MACHINE_ALIASES.get(
        platform.machine().lower(), platform.machine().lower()
    )
    return system, machine


def resolve(
    runtime: str, system: str | None = None, machine: str | None = None
) -> RuntimeArtifact:
    if system is None or machine is None:
        detected_system, detected_machine = current_platform()
        system = system or detected_system
        machine = machine or detected_machine
    artifact = ARTIFACTS.get((runtime, system, machine))
    if artifact is None:
        raise UnsupportedPlatformError(
            f"No pinned {runtime} build for {system}/{machine}. "
            "Managed runtimes are available on macOS; elsewhere, run llama-swap "
            "yourself and point swapboard at its config."
        )
    return artifact


def is_supported(system: str | None = None, machine: str | None = None) -> bool:
    try:
        for runtime in RUNTIMES:
            resolve(runtime, system, machine)
    except UnsupportedPlatformError:
        return False
    return True
