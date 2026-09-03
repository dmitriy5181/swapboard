import logging

from flask import Blueprint, current_app, render_template

from swapboard.common.client import SwapboardClient

logger = logging.getLogger(__name__)

models_bp = Blueprint("models_bp", __name__)


def _client() -> SwapboardClient:
    return current_app.config["SWAPBOARD_CLIENT"]


def _context(feedback: str | None = None, feedback_category: str = "info") -> dict:
    return {
        "gateway": _client().get_status(),
        "feedback": feedback,
        "feedback_category": feedback_category,
    }


@models_bp.get("/")
def index() -> str:
    return render_template("index.html", **_context())


@models_bp.get("/partials/models")
def partial_models() -> str:
    return render_template("partials/models.html", **_context())


@models_bp.post("/models/<name>/download")
def download_model(name: str) -> str:
    """Starts a download and re-renders the section with the outcome.

    Returning the partial rather than a redirect is what lets HTMX swap the
    table in place, and keeps an unreachable API from breaking the page.
    """
    try:
        result = _client().download_model(name)
    except Exception as exc:
        logger.error("Failed to start model download for %s: %s", name, exc)
        return render_template(
            "partials/models.html",
            **_context("Could not reach the swapboard API.", "danger"),
        )
    category = "success" if result.started else "info"
    return render_template("partials/models.html", **_context(result.message, category))
