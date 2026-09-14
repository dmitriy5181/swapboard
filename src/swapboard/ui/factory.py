from flask import Flask

from swapboard.common.client import SwapboardClient
from swapboard.ui.presentation import (
    capability_badge,
    format_context,
    format_parameters,
    state_badge,
)
from swapboard.ui.settings import UISettings
from swapboard.ui.views import models_bp


def create_app(config: dict | None = None) -> Flask:
    # No static folder: the dashboard uses stock Bootstrap from a CDN and ships
    # no CSS or JavaScript of its own.
    app = Flask(__name__, template_folder="templates", static_folder=None)

    settings = UISettings()
    app.config["SWAPBOARD_API_URL"] = settings.api_url
    app.config["SWAPBOARD_CLIENT"] = SwapboardClient(settings.api_url)
    # Applied last so callers, and tests in particular, can substitute the
    # client rather than having to reach a live API.
    if config is not None:
        app.config.update(config)

    _register_presentation(app)
    app.register_blueprint(models_bp)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _register_presentation(app: Flask) -> None:
    """Exposes the badge vocabulary to templates.

    Keeping the mapping from metadata to icons and wording in Python leaves the
    template describing layout only, and makes the wording testable on its own.
    """
    for helper in (
        state_badge,
        capability_badge,
        format_parameters,
        format_context,
    ):
        app.add_template_global(helper)
