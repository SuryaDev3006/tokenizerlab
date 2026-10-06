from dataclasses import FrozenInstanceError

import pytest

from tokenizerlab.data.document import Document


def test_document() -> None:
    text = "  Hé\ud800 "  # odd whitespace + lone surrogate
    meta = {"k": 1}
    doc = Document(text, "a1", 2, metadata=meta)

    assert doc.text == text
    assert doc == Document(text, "a1", 2, metadata={"k": 1})
    assert doc.id != Document(text, "a", 12).id
    assert doc.content_hash == Document(text, "other", 0).content_hash
    hash(doc)

    meta["k"] = 2
    assert doc.metadata == {"k": 1}
    with pytest.raises(TypeError):
        doc.metadata["k"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        doc.text = "x"  # type: ignore[misc]
