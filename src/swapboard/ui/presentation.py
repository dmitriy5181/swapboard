"""How model metadata is spelled in the dashboard."""

from dataclasses import dataclass

from swapboard.common.models import ModelParameters

UNLOADED_STATE = "unloaded"
LOADED_STATES = frozenset({"ready", "loaded"})
FAILED_STATES = frozenset({"error", "failed"})


@dataclass(frozen=True)
class Badge:
    icon: str
    label: str


@dataclass(frozen=True)
class StateBadge:
    """What llama-swap is doing with a model, and how loudly to say so."""

    icon: str
    label: str
    style: str
    title: str


CAPABILITY_BADGES = {
    "vision": Badge("bi-eye", "Vision"),
    "function_calling": Badge("bi-tools", "Tool calling"),
    "audio_transcriptions": Badge("bi-mic", "Audio"),
    "reranker": Badge("bi-sort-down", "Reranking"),
}

TASK_BADGES = {
    "chat": Badge("bi-chat-dots", "Chat"),
    "embeddings": Badge("bi-diagram-3", "Embeddings"),
    "reranking": Badge("bi-sort-down", "Reranking"),
}


def state_badge(state: str | None) -> StateBadge | None:
    """Says what llama-swap is doing with a model right now.

    Anything other than `unloaded` is worth showing, and showing the state's
    own name keeps the badge truthful as llama-swap adds further ones. Only a
    loaded model is good news, though: a state still in flight, or one that
    failed outright, must not be painted as a model sitting ready in memory.
    """
    if not state or state == UNLOADED_STATE:
        return None
    if state in LOADED_STATES:
        return StateBadge(
            "bi-lightning-charge-fill",
            state.capitalize(),
            "success",
            "Held in memory by llama-swap",
        )
    if state in FAILED_STATES:
        return StateBadge(
            "bi-exclamation-octagon-fill",
            state.capitalize(),
            "danger",
            "llama-swap could not load this model",
        )
    return StateBadge(
        "bi-hourglass-split",
        state.capitalize(),
        "warning",
        f"llama-swap reports this model as {state}",
    )


def capability_badge(name: str) -> Badge:
    """Names a capability, falling back to whatever llama-swap called it.

    New capabilities appear upstream between swapboard releases; an unknown one
    is shown rather than dropped, so the dashboard never quietly omits it.
    """
    return CAPABILITY_BADGES.get(name, Badge("bi-star", name.replace("_", " ")))


def task_badge(name: str) -> Badge:
    return TASK_BADGES.get(name, Badge("bi-cpu", name))


def format_parameters(parameters: ModelParameters | None) -> str | None:
    """Renders a parameter count, noting how much of it a request actually uses.

    A mixture-of-experts model's total size says little about its cost, so the
    active or effective count is kept alongside it.
    """
    if parameters is None or parameters.total is None:
        return None
    used = parameters.active or parameters.effective
    return f"{parameters.total} ({used} active)" if used else parameters.total


def format_context(tokens: int | None) -> str | None:
    if tokens is None:
        return None
    if tokens >= 1024:
        return f"{tokens // 1024}K ctx"
    return f"{tokens} ctx"
