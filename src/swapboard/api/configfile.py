"""Reads, validates and replaces the llama-swap configuration file."""

import os
import shutil
import tempfile
from pathlib import Path

import yaml
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError, best_match

from swapboard.api.config import model_sources
from swapboard.common.models import ConfigValidation
from swapboard.runtimes.manifest import load_config_schema

MAX_REPORTED_ERRORS = 20


class ConfigFile:
    """Owns one config path: its text, its validity and its replacement."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    def read(self) -> str:
        return self._path.read_text(encoding="utf-8")

    def validate(self, text: str) -> ConfigValidation:
        """Rejects anything llama-swap would, before it reaches the file.

        llama-swap watches the file and reloads it, so an invalid save would
        break a running server with no way to see why from the dashboard.
        """
        document, parse_error = _parse(text)
        if parse_error is not None:
            return ConfigValidation(valid=False, errors=[parse_error])

        schema_errors = _schema_errors(document)
        if schema_errors:
            return ConfigValidation(valid=False, errors=schema_errors)

        return ConfigValidation(valid=True, warnings=_unmanageable(document))

    def write(self, text: str) -> ConfigValidation:
        """Replaces the file only if the new text validates.

        The replacement is a rename over a fully written temporary file, so the
        watching llama-swap never reads a half-saved config.
        """
        validation = self.validate(text)
        if not validation.valid:
            return validation

        self._back_up()
        self._replace(text)
        return validation

    def _back_up(self) -> None:
        if self._path.exists():
            shutil.copy2(self._path, self._path.with_suffix(f"{self._path.suffix}.bak"))

    def _replace(self, text: str) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            delete=False,
        )
        try:
            with handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self._path)
        except OSError:
            Path(handle.name).unlink(missing_ok=True)
            raise


def _unmanageable(document: object) -> list[str]:
    """Names models swapboard will show no download button for.

    A model whose GGUF path does not end in `<org>/<repo>/<filename>` is
    skipped by the parser, so it would vanish from the table unexplained.
    """
    if not isinstance(document, dict):
        return []
    configured = set((document.get("models") or {}).keys())
    manageable = {source.name for source in model_sources(document)}
    return [
        f"Model '{name}' is not manageable by swapboard: its model path does "
        "not end in <org>/<repo>/<filename>, so it cannot be downloaded here."
        for name in sorted(configured - manageable)
    ]


def _parse(text: str) -> tuple[object, str | None]:
    if not text.strip():
        return None, "The configuration is empty."
    try:
        return yaml.safe_load(text), None
    except yaml.YAMLError as exc:
        return None, f"Invalid YAML: {exc}"


def _schema_errors(document: object) -> list[str]:
    validator = Draft7Validator(load_config_schema())
    errors = sorted(validator.iter_errors(document), key=lambda err: list(err.path))
    return [_describe(error) for error in errors[:MAX_REPORTED_ERRORS]]


def _describe(error: ValidationError) -> str:
    """Renders one violation as `path: message`.

    A failed `oneOf` reports every branch it tried, which is unreadable, so the
    closest branch's own message is used in its place.
    """
    specific = best_match(error.context) if error.context else None
    message = specific.message if specific is not None else error.message
    location = "/".join(str(part) for part in error.absolute_path)
    return f"{location}: {message}" if location else message
