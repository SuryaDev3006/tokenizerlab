"""Deterministic JSON: the same value always produces the same text.

Keys are sorted and non-ASCII text is written as-is, so stored metadata and manifests are
byte-for-byte reproducible and readable in any language.
"""

import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Compact, key-sorted JSON, e.g. for a value stored in a table cell."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def pretty_json(value: Any) -> str:
    """Indented, key-sorted JSON ending in a newline, for files people read."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
