from swapboard.deployment.macos import DEFAULT_API_PORT, DEFAULT_LLAMA_SWAP_PORT
from swapboard.deployment.macos import DEFAULT_UI_PORT as DEPLOY_UI_PORT
from swapboard.dev import DEFAULT_UI_PORT as DEV_UI_PORT
from swapboard.ui.settings import UISettings

# macOS runs com.apple.sharingd on 8770 for Continuity and AirDrop. It is a
# launchd system daemon, so the port is occupied on every Mac and anything
# defaulting to it fails to bind.
MACOS_RESERVED_PORTS = {8770}


def test_default_ui_port_avoids_ports_macos_reserves() -> None:
    assert UISettings(_env_file=None).port not in MACOS_RESERVED_PORTS


def test_every_default_port_avoids_ports_macos_reserves() -> None:
    for port in (DEFAULT_API_PORT, DEFAULT_LLAMA_SWAP_PORT, DEPLOY_UI_PORT):
        assert port not in MACOS_RESERVED_PORTS


def test_ui_default_port_is_consistent_across_entry_points() -> None:
    """The dashboard must land on the same port however it is started."""
    assert UISettings(_env_file=None).port == DEPLOY_UI_PORT == DEV_UI_PORT


def test_default_ports_do_not_collide() -> None:
    ports = [DEFAULT_API_PORT, DEFAULT_LLAMA_SWAP_PORT, DEPLOY_UI_PORT]

    assert len(set(ports)) == len(ports)


def test_ui_defaults_target_the_local_api() -> None:
    settings = UISettings(_env_file=None)

    assert settings.api_url == f"http://127.0.0.1:{DEFAULT_API_PORT}"
    assert settings.host == "127.0.0.1"


def test_ui_port_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("SWAPBOARD_UI_PORT", "9999")

    assert UISettings().port == 9999
