"""The addresses a swapboard installation serves on.

These are shared by the API, the dashboard, the local runner and the macOS
deployment so that a port is defined once rather than repeated per entry point.
"""

DEFAULT_HOST = "127.0.0.1"

DEFAULT_API_PORT = 8771
DEFAULT_LLAMA_SWAP_PORT = 8772
# 8770 is deliberately avoided: macOS runs com.apple.sharingd there for
# Continuity and AirDrop, so binding it fails on any Mac.
DEFAULT_UI_PORT = 8773
