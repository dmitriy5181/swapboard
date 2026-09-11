import logging
from collections.abc import Callable

import httpx
from flask import Blueprint, current_app, render_template, request

from swapboard.common.client import SwapboardClient

logger = logging.getLogger(__name__)

models_bp = Blueprint("models_bp", __name__)

UNREACHABLE = "Could not reach the swapboard API."


@models_bp.get("/")
def index() -> str:
    return render_template("index.html", **_context())


@models_bp.get("/partials/models")
def partial_models() -> str:
    return render_template("partials/models.html", **_context())


@models_bp.post("/models/<path:name>/download")
def download_model(name: str) -> str:
    """Starts a download and re-renders the section with the outcome.

    Returning the partial rather than a redirect is what lets HTMX swap the
    table in place, and keeps an unreachable API from breaking the page.
    """

    def start() -> tuple[bool, str]:
        result = _client().download_model(name)
        return result.started, result.message

    return _act(start, f"download '{name}'")


@models_bp.delete("/models/<path:name>")
def remove_model(name: str) -> str:
    def remove() -> tuple[bool, str]:
        result = _client().remove_model(name)
        return result.removed, result.message

    return _act(remove, f"remove '{name}'")


@models_bp.delete("/stray-models/<path:relative_path>")
def remove_stray_model(relative_path: str) -> str:
    def remove() -> tuple[bool, str]:
        result = _client().remove_stray(relative_path)
        return result.removed, result.message

    return _act(remove, f"remove '{relative_path}'")


@models_bp.get("/partials/config-editor")
def partial_config_editor() -> tuple[str, int, dict[str, str]]:
    """Loads the config only when the editor is opened.

    The models section polls itself while a download runs, and the config has
    no place in that traffic.
    """
    try:
        document = _client().get_config()
    except Exception as exc:
        return _editor("", errors=[_reason(exc, "read the configuration")])
    return _editor(document.text, warnings=document.warnings)


@models_bp.put("/config")
def save_config() -> tuple[str, int, dict[str, str]]:
    """Saves the config, keeping the submitted text on screen if it is refused.

    Re-rendering the editor with what the user typed is what lets them correct
    the reported error instead of losing the edit.
    """
    text = request.form.get("text", "")
    try:
        result = _client().save_config(text)
    except Exception as exc:
        return _editor(text, errors=[_reason(exc, "save the configuration")])
    if not result.saved:
        return _editor(text, errors=result.errors)
    return _editor(text, saved=result.message, warnings=result.warnings)


def _act(operation: Callable[[], tuple[bool, str]], action: str) -> str:
    """Runs one API call and re-renders the section with what came of it.

    Every action here answers with the same shape, so they share one place to
    turn a failure into feedback instead of each inventing its own.
    """
    try:
        changed, message = operation()
    except Exception as exc:
        return _section(_reason(exc, action), "danger")
    return _section(message, "success" if changed else "info")


def _reason(exc: Exception, action: str) -> str:
    """Says what actually went wrong, rather than blaming the connection.

    A refused request carries the API's own explanation; only a transport
    failure means the API could not be reached at all.
    """
    logger.error("Could not %s: %s", action, exc)
    if isinstance(exc, httpx.HTTPStatusError):
        return _detail(exc.response) or f"The swapboard API refused to {action}."
    return UNREACHABLE


def _detail(response: httpx.Response) -> str | None:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return detail if isinstance(detail, str) else None


def _client() -> SwapboardClient:
    return current_app.config["SWAPBOARD_CLIENT"]


def _context(feedback: str | None = None, feedback_category: str = "info") -> dict:
    return {
        "gateway": _client().get_status(),
        "feedback": feedback,
        "feedback_category": feedback_category,
    }


def _section(feedback: str, category: str) -> str:
    return render_template("partials/models.html", **_context(feedback, category))


def _editor(
    text: str,
    *,
    saved: str | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> tuple[str, int, dict[str, str]]:
    """Renders the editor, asking the table to reload only on a saved config.

    The header tells HTMX to refresh the models section, which a save may have
    changed beyond recognition.
    """
    body = render_template(
        "partials/config_editor.html",
        text=text,
        saved=saved,
        errors=errors or [],
        warnings=warnings or [],
    )
    headers = {"HX-Trigger": "models-updated"} if saved else {}
    return body, 200, headers
