"""Document: the immutable unit of data, with text, source, position, lang, metadata,
and the computed id and content_hash.

- id identifies where a document came from (source + position).
- content_hash identifies exactly what it says (its text), for deduplication.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from tokenizerlab.errors import TokenizerLabError
from tokenizerlab.shared.hashing import digest_hex, encode_text

# BCP-47 "undetermined": how documents with lang=None are counted and reported.
UNDETERMINED_LANG = "und"


class InvalidDocumentError(TokenizerLabError):
    """A Document field has the wrong type or value."""


# ---------------------------------------------------------------- identities (pure)


def text_hash(text: str) -> str:
    """Hash of the exact text: equal for equal texts, whatever their source."""
    return digest_hex(encode_text(text))


def location_id(source: str, position: int) -> str:
    """Hash of where a document came from: equal for equal (source, position)."""
    # The length prefix keeps ("a1", 2) and ("a", 12) from colliding.
    return digest_hex(encode_text(f"{len(source)}:{source}:{position}"))


# ---------------------------------------------------------------- Document


@dataclass(frozen=True, slots=True)
class Document:
    """An exact text and where it came from, with two computed identities.

    Fields:
        text: Original document text, never modified.
        source: Logical source path/name (e.g., 'te/news.jsonl').
        position: Document index/line number within source.
        lang: Optional language code (e.g., 'en').
        metadata: Optional key-value metadata, frozen on creation.
        id: Computed location identity: same (source, position), same id.
        content_hash: Computed content identity: same text, same hash.
    """

    text: str
    source: str
    position: int
    lang: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate the fields, freeze metadata and compute both identities."""
        self._validate()
        # A private read-only copy, so changing the caller's dict cannot change the Document.
        self._set_frozen_field("metadata", MappingProxyType(dict(self.metadata)))
        self._set_frozen_field("id", location_id(self.source, self.position))
        self._set_frozen_field("content_hash", text_hash(self.text))

    def _validate(self) -> None:
        """Raise InvalidDocumentError for any field of the wrong type or value."""
        if not isinstance(self.text, str):
            raise InvalidDocumentError("text must be str")

        if not isinstance(self.source, str) or not self.source:
            raise InvalidDocumentError("source must be a non-empty str")

        if isinstance(self.position, bool) or not isinstance(self.position, int):
            raise InvalidDocumentError("position must be a non-negative int")

        if self.position < 0:
            raise InvalidDocumentError("position must be a non-negative int")

        if self.lang is not None and (not isinstance(self.lang, str) or not self.lang):
            raise InvalidDocumentError("lang must be None or a non-empty str")

        if not isinstance(self.metadata, Mapping):
            raise InvalidDocumentError("metadata must be a mapping")

    def _set_frozen_field(self, name: str, value: Any) -> None:
        """Assign a field on this frozen instance; only valid during __post_init__."""
        object.__setattr__(self, name, value)
