"""The on-disk layout a deployed swapboard owns.

Everything lives under a single prefix so an installation can be inspected,
backed up, or removed as one directory.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

LLAMA_CPP_RUNTIME = "llama-cpp"
LLAMA_SWAP_RUNTIME = "llama-swap"


@dataclass(frozen=True)
class Layout:
    """Resolves every path a deployment uses from one prefix."""

    prefix: Path

    @classmethod
    def default(cls) -> "Layout":
        """Derives the prefix from the virtualenv the current process runs in.

        A deployment installs swapboard into `<prefix>/venv`, so the parent of
        `sys.prefix` is the prefix that installation belongs to. This is what
        lets `swapboard-deploy` run with no path arguments at all.
        """
        return cls(Path(sys.prefix).parent)

    @property
    def venv(self) -> Path:
        return self.prefix / "venv"

    @property
    def runtimes(self) -> Path:
        return self.prefix / "runtimes"

    @property
    def models(self) -> Path:
        return self.prefix / "models"

    @property
    def config(self) -> Path:
        return self.prefix / "config"

    @property
    def log(self) -> Path:
        return self.prefix / "log"

    @property
    def llama_swap_config(self) -> Path:
        return self.config / "llama-swap.yml"

    @property
    def llama_server_bin(self) -> Path:
        return self.runtimes / LLAMA_CPP_RUNTIME / "llama-server"

    @property
    def llama_swap_bin(self) -> Path:
        return self.runtimes / LLAMA_SWAP_RUNTIME / "llama-swap"

    def log_file(self, name: str) -> Path:
        return self.log / f"{name}.log"
