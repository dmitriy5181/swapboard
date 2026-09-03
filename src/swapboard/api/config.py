"""Derives downloadable model sources from a llama-swap configuration."""

import os
import shlex
from pathlib import Path

import yaml

from swapboard.common.models import ModelFile, ModelSource


def parse_model_sources(config_path: str | os.PathLike[str]) -> list[ModelSource]:
    """Reads every model whose GGUF files can be traced back to Hugging Face.

    Models whose command line cannot be resolved are skipped rather than
    reported, because swapboard can only manage files it knows how to fetch.
    """
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    sources: list[ModelSource] = []
    for name, definition in (config.get("models") or {}).items():
        source = _parse_model_source(name, definition or {})
        if source is not None:
            sources.append(source)
    return sources


def _parse_model_source(name: str, definition: dict[str, object]) -> ModelSource | None:
    cmd = definition.get("cmd")
    if not isinstance(cmd, str):
        return None

    model_paths = _extract_model_paths(cmd)
    if model_paths is None:
        return None

    files: list[ModelFile] = []
    for model_path in model_paths:
        source_parts = _derive_hf_source(model_path)
        if source_parts is None:
            return None
        repo_id, filename, relative_path = source_parts
        files.append(
            ModelFile(
                relative_path=relative_path,
                repo_id=repo_id,
                filename=filename,
            )
        )
    return ModelSource(name=name, files=tuple(files))


def _extract_model_paths(cmd: str) -> tuple[str, ...] | None:
    """Pulls the model and optional multimodal projector out of a command line."""
    tokens = shlex.split(cmd)
    model_path: str | None = None
    projector_path: str | None = None
    for index, token in enumerate(tokens):
        if token in ("-m", "--model"):
            if index + 1 >= len(tokens):
                return None
            model_path = tokens[index + 1]
        if token.startswith("--model="):
            model_path = token.split("=", 1)[1]
        if token == "--mmproj":
            if index + 1 >= len(tokens):
                return None
            projector_path = tokens[index + 1]
        if token.startswith("--mmproj="):
            projector_path = token.split("=", 1)[1]

    if model_path is None:
        return None
    if projector_path is None:
        return (model_path,)
    return model_path, projector_path


def _derive_hf_source(model_path: str) -> tuple[str, str, str] | None:
    """Reads `<org>/<repo>/<filename>` off the tail of a model path.

    Anything ahead of those three components is the local models directory,
    however it happens to be spelled or macro-expanded in the config.
    """
    parts = Path(model_path).parts
    if len(parts) < 3:
        return None
    org, repo, filename = parts[-3], parts[-2], parts[-1]
    return f"{org}/{repo}", filename, f"{org}/{repo}/{filename}"
