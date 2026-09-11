"""Derives downloadable model sources from a llama-swap configuration."""

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

import yaml

from swapboard.common.models import ModelFile, ModelSource

_MODEL_FLAGS = ("-m", "--model")
_PROJECTOR_FLAGS = ("--mmproj",)
_DRAFT_FLAGS = ("--spec-draft-model", "-md", "--model-draft")
_COMPANION_FLAGS = (_PROJECTOR_FLAGS, _DRAFT_FLAGS)
_REPOSITORY_PARTS = 2
_PRIMARY_PATH_PARTS = _REPOSITORY_PARTS + 1


@dataclass(frozen=True)
class _ModelCommand:
    """The model files one llama-server command line points at."""

    primary: str
    companions: tuple[str, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return (self.primary, *self.companions)


def load_config(config_path: str | os.PathLike[str]) -> object:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_sources(config: object) -> list[ModelSource]:
    """Reads every model whose GGUF files can be traced back to Hugging Face.

    Models whose command line cannot be resolved are skipped rather than
    reported, because swapboard can only manage files it knows how to fetch.
    """
    sources: list[ModelSource] = []
    for name, definition in _models_of(config).items():
        source = _parse_model_source(name, definition)
        if source is not None:
            sources.append(source)
    return sources


def model_paths_by_name(config: object) -> dict[str, list[str]]:
    """Maps each configured model to the files its command line points at.

    Unlike `model_sources`, a path that cannot be traced back to Hugging Face
    is still reported: swapboard cannot download that file, but it must know
    the file is spoken for before offering to delete it.
    """
    paths_by_name: dict[str, list[str]] = {}
    for name, definition in _models_of(config).items():
        command = _parse_command(definition)
        if command is not None:
            paths_by_name[name] = list(command.paths)
    return paths_by_name


def _models_of(config: object) -> dict:
    if not isinstance(config, dict):
        return {}
    models = config.get("models")
    return models if isinstance(models, dict) else {}


def _parse_model_source(name: str, definition: object) -> ModelSource | None:
    command = _parse_command(definition)
    if command is None:
        return None
    files = _model_files(command)
    if files is None:
        return None
    return ModelSource(name=name, files=files)


def _parse_command(definition: object) -> _ModelCommand | None:
    """Picks the model, the projector and the draft model off a command line."""
    cmd = definition.get("cmd") if isinstance(definition, dict) else None
    if not isinstance(cmd, str):
        return None

    try:
        tokens = shlex.split(cmd)
        primary = _value_for(tokens, _MODEL_FLAGS)
        companions = tuple(
            value
            for flags in _COMPANION_FLAGS
            if (value := _value_for(tokens, flags)) is not None
        )
    except ValueError:
        return None
    if primary is None:
        return None
    return _ModelCommand(primary=primary, companions=companions)


def _value_for(tokens: list[str], flags: tuple[str, ...]) -> str | None:
    """Reads what one option was given, under any of the names it answers to.

    llama.cpp spells a single option several ways -- a draft model arrives as
    `--spec-draft-model`, `-md` or `--model-draft` -- and accepts both
    `--flag value` and `--flag=value`. A later occurrence wins, as it does
    there.
    """
    value = None
    for index, token in enumerate(tokens):
        if token in flags:
            if index + 1 >= len(tokens):
                raise ValueError(f"{token} requires a value")
            value = tokens[index + 1]
        for flag in flags:
            if token.startswith(f"{flag}="):
                value = token.split("=", 1)[1]
    return value


def _model_files(command: _ModelCommand) -> tuple[ModelFile, ...] | None:
    """Reads every file of one model, against the directory they all share.

    The primary path ends in `<org>/<repo>/<file>`, so whatever precedes it is
    the models directory, however the config spells or macro-expands it.
    Measuring the companions from that same point is what lets one sit deeper
    in the repository -- a speculative draft model under `MTP/`, say -- instead
    of being read as a repository of its own.
    """
    parts = Path(command.primary).parts
    if len(parts) < _PRIMARY_PATH_PARTS:
        return None
    models_dir = parts[:-_PRIMARY_PATH_PARTS]

    files: list[ModelFile] = []
    for path in command.paths:
        model_file = _model_file(path, models_dir)
        if model_file is None:
            return None
        files.append(model_file)
    return tuple(files)


def _model_file(path: str, models_dir: tuple[str, ...]) -> ModelFile | None:
    parts = Path(path).parts
    if parts[: len(models_dir)] != models_dir:
        return None

    within_store = parts[len(models_dir) :]
    if len(within_store) <= _REPOSITORY_PARTS:
        return None
    if not all(_is_plain(part) for part in within_store):
        return None

    repo_id = "/".join(within_store[:_REPOSITORY_PARTS])
    filename = "/".join(within_store[_REPOSITORY_PARTS:])
    return ModelFile(
        relative_path=f"{repo_id}/{filename}", repo_id=repo_id, filename=filename
    )


def _is_plain(part: str) -> bool:
    """Rejects a path component that is not a plain name.

    `/opt/llama.gguf` splits into exactly three parts, the first being the root
    anchor, which would otherwise yield a repository named `/` and a relative
    path that resolves outside the models directory entirely.
    """
    return part not in (".", "..") and not Path(part).is_absolute()
