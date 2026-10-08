"""ReadOptions: validated read() settings, and their JSON-ready record (the reader config)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal, TypeVar

from tokenizerlab.errors import ConfigurationError


class OnError(StrEnum):
    """What to do with a bad row or an unreadable file."""

    WARN = "warn"  # record it and warn once per source
    SKIP = "skip"  # record it silently
    RAISE = "raise"  # stop reading


class Unit(StrEnum):
    """How a plain text file is split into documents."""

    FILE = "file"
    LINE = "line"


MetadataFields = tuple[str, ...] | Literal["all"] | None

# What read() was given, as JSON: one dict per source, or a list of child configs for a chain.
ReaderConfig = dict[str, Any] | list["ReaderConfig"]

_Choice = TypeVar("_Choice", bound=StrEnum)


def parse_choice(choices: type[_Choice], value: str, setting: str) -> _Choice:
    """The enum member for a user-supplied value, or a ConfigurationError naming the setting."""
    try:
        return choices(value)
    except ValueError:
        allowed = ", ".join(repr(choice.value) for choice in choices)
        raise ConfigurationError(f"{setting} must be one of {allowed}, got {value!r}") from None


def parse_metadata_fields(value: Sequence[str] | str | None) -> MetadataFields:
    """None, "all", or a tuple of field names."""
    if value is None:
        return None
    if not isinstance(value, str):
        return tuple(value)
    # A plain string like "src" would otherwise be iterated as "s", "r", "c".
    if value != "all":
        raise ConfigurationError(f"metadata_fields must be a list of names or 'all', got {value!r}")
    return "all"


@dataclass(frozen=True, slots=True)
class ReadOptions:
    """Every read() setting except the source and its name."""

    format: str | None = None
    text_field: str = "text"
    metadata_fields: MetadataFields = None
    lang: str | None = None
    lang_field: str | None = None
    unit: Unit = Unit.FILE
    glob: str | None = None
    delimiter: str | None = None
    split: str = "train"
    config: str | None = None
    revision: str | None = None
    on_error: OnError = OnError.WARN

    def __post_init__(self) -> None:
        """Fail fast on settings that would otherwise break every file or row later."""
        if not isinstance(self.text_field, str) or not self.text_field:
            raise ConfigurationError("text_field must be a non-empty str")
        if self.lang is not None and (not isinstance(self.lang, str) or not self.lang):
            raise ConfigurationError("lang must be None or a non-empty str")

    def to_config(self) -> dict[str, Any]:
        """The options as a JSON-ready dict."""
        config = asdict(self)
        config["unit"] = self.unit.value
        config["on_error"] = self.on_error.value
        if isinstance(self.metadata_fields, tuple):
            config["metadata_fields"] = list(self.metadata_fields)
        return config
