import pytest

from swapboard.runtimes import manifest


@pytest.fixture(autouse=True)
def pinned_platform(monkeypatch):
    """Pin the platform so runtime tests do not depend on the host running them.

    Pinned builds only exist for macOS, so on Linux CI every command that
    resolves an artifact fails and the assertions become about the runner
    rather than the code. Tests that care about platform handling override
    this fixture explicitly.
    """
    monkeypatch.setattr(manifest, "current_platform", lambda: ("darwin", "arm64"))
