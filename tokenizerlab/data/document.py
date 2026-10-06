"""Immutable document model for TokForge data pipelines.

Preserves exact original text without modification and computes two identities:
- id: Identifies document location (source + position).
- content_hash: Identifies exact text content for deduplication.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# Version string recorded in manifests for hashing reproducibility
HASH_VERSION = "blake2b-128-v1"
_DIGEST_SIZE = 16  # 128-bit hash digest size


# Helper Functions
def _hash(data: bytes) -> str:
    """Return deterministic hexadecimal BLAKE2b hash of data."""
    return hashlib.blake2b(data, digest_size=_DIGEST_SIZE).hexdigest()


def _text_hash(text: str) -> str:
    """Hash exact text. Uses surrogatepass to safely handle lone surrogates."""
    return _hash(text.encode("utf-8", "surrogatepass"))


def _document_id(source: str, position: int) -> str:
    """Create a deterministic ID from source and position using length prefix."""
    # Length prefix so ("a1", 2) and ("a", 12) can't collide.
    key = f"{len(source)}:{source}:{position}"
    return _hash(key.encode("utf-8", "surrogatepass"))


@dataclass(frozen=True, slots=True)
class Document:
    """Immutable unit of text data in the pipeline.

    Fields:
        text: Original document text.
        source: Logical source path/name (e.g., 'te/news.jsonl').
        position: Document index/line number within source.
        lang: Optional language code (e.g., 'en').
        metadata: Optional key-value metadata mapping.
        id: Computed location-based identity hash.
        content_hash: Computed text-content hash for deduplication.
    """

    text: str
    source: str
    position: int
    lang: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate input types and compute immutable derived identities."""

        # Input validation
        if not isinstance(self.text, str):
            raise TypeError("text must be str")

        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty str")

        if isinstance(self.position, bool) or not isinstance(self.position, int):
            raise TypeError("position must be a non-negative int")

        if self.position < 0:
            raise ValueError("position must be a non-negative int")

        if self.lang is not None and (
            not isinstance(self.lang, str) or not self.lang
        ):
            raise ValueError("lang must be None or a non-empty str")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        # Shallow copy + freeze: the caller's dict can't mutate the Document.
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

        # Compute deterministic document ID (source + position)
        object.__setattr__(
            self,
            "id",
            _document_id(self.source, self.position),
        )

        # Compute deterministic content hash (text)
        
        object.__setattr__(
            self,
            "content_hash",
            _text_hash(self.text),
        )