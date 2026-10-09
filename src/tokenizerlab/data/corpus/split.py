"""Split: one side of a train/validation split that cannot leak duplicates.

A document's side depends only on the seed and its key: its content hash by default, so
identical texts always land on the same side; its source, or a function's value, on request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from tokenizerlab.data.corpus.steps import StepContext, qualified_name
from tokenizerlab.data.document import Document
from tokenizerlab.errors import ConfigurationError
from tokenizerlab.shared.hashing import unit_interval

SplitKey = Callable[[Document], str]

SPLIT_KEYS: dict[str, SplitKey] = {
    "content": lambda document: document.content_hash,
    "source": lambda document: document.source,
}


@dataclass(frozen=True, slots=True)
class Split:
    """Keep the documents on one side ("train" or "validation") of a split."""

    validation: float  # the expected share of documents on the validation side
    seed: int
    by: str | SplitKey
    side: str

    def __post_init__(self) -> None:
        """validation must be in (0, 1), and `by` a known key name or a function."""
        if not 0 < self.validation < 1:
            raise ConfigurationError(f"validation must be in (0, 1), got {self.validation}")
        if isinstance(self.by, str) and self.by not in SPLIT_KEYS:
            raise ConfigurationError(
                f"by must be one of {sorted(SPLIT_KEYS)} or a function, got {self.by!r}"
            )

    @property
    def name(self) -> str:
        """Recorded as "split"."""
        return "split"

    @property
    def needs_prepass(self) -> bool:
        """Each document's side is decided on its own."""
        return False

    @property
    def reproducible(self) -> bool:
        """A custom key function cannot be re-created from a manifest."""
        return isinstance(self.by, str)

    def parameters(self) -> dict[str, Any]:
        """validation, seed, side, and `by` by name ("content", "source" or a function's)."""
        by = self.by if isinstance(self.by, str) else qualified_name(self.by)
        return {"validation": self.validation, "seed": self.seed, "by": by, "side": self.side}

    def apply(self, documents: Iterator[Document], context: StepContext) -> Iterator[Document]:
        """Yield the documents whose side is this step's side."""
        key = SPLIT_KEYS[self.by] if isinstance(self.by, str) else self.by
        return (document for document in documents if self.side_of(key(document)) == self.side)

    def side_of(self, key: str) -> str:
        """The side of a key.

        The step index is left out of the hash on purpose (unlike sample): a document's side
        must depend only on the seed and its key, so it stays put when documents are added or
        reordered, or when steps are added earlier in the pipeline.
        """
        in_validation = unit_interval(f"split:{self.seed}:{key}") < self.validation
        return "validation" if in_validation else "train"


# ---------------------------------------------------------------- rules over the history


def check_new_split(history: list[dict[str, Any]], seed: int) -> None:
    """Refuse a split with the seed of an earlier split: every document would land on the
    side it already took, leaving the other side empty."""
    if any(entry["op"] == "split" and entry["params"]["seed"] == seed for entry in history):
        raise ConfigurationError(
            f"this corpus was already split with seed={seed}; splitting it again with the same "
            "seed puts every document on one side. Use a different seed."
        )


def check_dedup_after_split(history: list[dict[str, Any]], *, near: bool) -> None:
    """Refuse a dedup of the train side of a split that ignores the validation side.

    Only the latest split counts. Exact dedup after a content split is safe as it is:
    identical texts always land on the same side.
    """
    splits = [entry["params"] for entry in history if entry["op"] == "split"]
    if not splits or splits[-1]["side"] != "train":
        return
    if not near and splits[-1]["by"] == "content":
        return
    raise ConfigurationError(
        "this is the train side of a split: dedup it with against=<the validation corpus>, "
        "so training keeps nothing the validation side holds (or dedup before splitting)"
    )
