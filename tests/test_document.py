from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from tokenizerlab import Document
from tokenizerlab.data.document import InvalidDocumentError


def test_text_is_kept_exactly() -> None:
    """Odd whitespace and lone surrogates survive untouched and can still be hashed."""
    text = "  Hé\ud800 \r\n"
    document = Document(text, "a", 0)

    assert document.text == text
    assert len(document.content_hash) == 32


def test_identities() -> None:
    """id identifies the location, content_hash the exact text."""
    document = Document("same text", "a1", 2)

    assert document.id == Document("other text", "a1", 2).id
    assert document.id != Document("same text", "a", 12).id  # "a1"+2 vs "a"+12 must not collide
    assert document.content_hash == Document("same text", "elsewhere", 9).content_hash
    assert document.content_hash != Document("same text ", "a1", 2).content_hash


def test_document_is_immutable() -> None:
    """Neither the document nor its metadata can change after creation."""
    metadata = {"k": 1}
    document = Document("x", "a", 0, metadata=metadata)

    metadata["k"] = 2
    assert document.metadata == {"k": 1}
    with pytest.raises(TypeError):
        document.metadata["k"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        document.text = "y"  # type: ignore[misc]
    assert hash(document) == hash(Document("x", "a", 0, metadata={"k": 1}))


@pytest.mark.parametrize(
    "fields",
    [
        {"text": b"bytes", "source": "a", "position": 0},
        {"text": "x", "source": "", "position": 0},
        {"text": "x", "source": "a", "position": -1},
        {"text": "x", "source": "a", "position": True},
        {"text": "x", "source": "a", "position": 0, "lang": ""},
        {"text": "x", "source": "a", "position": 0, "metadata": ["not", "a", "mapping"]},
    ],
)
def test_invalid_fields_are_rejected(fields: dict[str, Any]) -> None:
    """Wrong types or values fail at creation with a domain error."""
    with pytest.raises(InvalidDocumentError):
        Document(**fields)
