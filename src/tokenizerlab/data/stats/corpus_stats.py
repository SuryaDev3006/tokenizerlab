"""CorpusStats: what one pass over a corpus contained, built incrementally, and the fingerprint.

The fingerprint is BLAKE2b-128 over every document's content_hash digest, in iteration
order. It is order-sensitive on purpose: BPE breaks frequency ties by order, so the same
documents in a different order can train a different tokenizer. It is content-only: sources
do not affect it, because the trainer only sees text.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tokenizerlab.data.document import UNDETERMINED_LANG, Document
from tokenizerlab.shared.hashing import incremental_digest, utf8_size
from tokenizerlab.shared.source_names import source_group


@dataclass(frozen=True, slots=True)
class Counts:
    """A number of documents and their total UTF-8 bytes."""

    documents: int = 0
    bytes: int = 0

    def plus(self, size: int) -> Counts:
        """These counts with one more document of `size` bytes."""
        return Counts(self.documents + 1, self.bytes + size)


@dataclass(frozen=True, slots=True)
class StepCounts:
    """Documents and bytes entering and leaving one history step during a pass."""

    op: str
    docs_in: int
    docs_out: int
    bytes_in: int
    bytes_out: int

    @classmethod
    def between(cls, op: str, counts_in: Counts, counts_out: Counts) -> StepCounts:
        """The step counts for what entered and what left a step."""
        return cls(
            op=op,
            docs_in=counts_in.documents,
            docs_out=counts_out.documents,
            bytes_in=counts_in.bytes,
            bytes_out=counts_out.bytes,
        )


class Fingerprint:
    """Order-sensitive hash of a sequence of documents' content.

    Content hashes have a fixed length, so no separator is needed, and an empty corpus has
    the fingerprint of empty input.
    """

    def __init__(self) -> None:
        """Start an empty fingerprint."""
        self._hasher = incremental_digest()

    def add(self, document: Document) -> None:
        """Append one document's content hash."""
        self._hasher.update(bytes.fromhex(document.content_hash))

    def hexdigest(self) -> str:
        """The fingerprint of every document added so far."""
        return self._hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Summary of one complete pass over a corpus.

    Bytes are UTF-8 with surrogatepass, matching how Document hashes text. Lengths are in
    characters and None for an empty corpus.
    """

    documents: int
    chars: int
    bytes: int
    length_min: int | None
    length_max: int | None
    length_mean: float | None
    by_lang: dict[str, Counts]  # lang=None is counted under "und"
    by_source: dict[str, Counts]  # grouped by the first path component of the source
    steps: tuple[StepCounts, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CorpusStats:
        """Rebuild CorpusStats from to_dict() output; raises KeyError or TypeError if malformed."""
        return cls(
            documents=data["documents"],
            chars=data["chars"],
            bytes=data["bytes"],
            length_min=data["length_min"],
            length_max=data["length_max"],
            length_mean=data["length_mean"],
            by_lang=_counts_by_key(data["by_lang"]),
            by_source=_counts_by_key(data["by_source"]),
            steps=tuple(StepCounts(**step) for step in data["steps"]),
            fingerprint=data["fingerprint"],
        )


def _counts_by_key(data: Mapping[str, Mapping[str, int]]) -> dict[str, Counts]:
    """Rebuild a {key: Counts} table from its JSON form."""
    return {key: Counts(**counts) for key, counts in data.items()}


class StatsAccumulator:
    """Builds CorpusStats one document at a time, so statistics and the fingerprint come
    from the same single pass that reads (or saves) the documents."""

    def __init__(self) -> None:
        """Start with empty totals."""
        self._totals = Counts()
        self._chars = 0
        self._length_min: int | None = None
        self._length_max: int | None = None
        self._by_lang: dict[str, Counts] = {}
        self._by_source: dict[str, Counts] = {}
        self._fingerprint = Fingerprint()

    def add(self, document: Document) -> None:
        """Count one document."""
        size = utf8_size(document.text)
        lang = document.lang or UNDETERMINED_LANG
        group = source_group(document.source)
        self._totals = self._totals.plus(size)
        self._by_lang[lang] = self._by_lang.get(lang, Counts()).plus(size)
        self._by_source[group] = self._by_source.get(group, Counts()).plus(size)
        self._add_length(len(document.text))
        self._fingerprint.add(document)

    def observe(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Yield the documents unchanged, adding each one as it passes."""
        for document in documents:
            self.add(document)
            yield document

    def finish(self, steps: Sequence[StepCounts] = ()) -> CorpusStats:
        """Freeze the totals, with the step counts of the same pass if the caller has them."""
        documents = self._totals.documents
        return CorpusStats(
            documents=documents,
            chars=self._chars,
            bytes=self._totals.bytes,
            length_min=self._length_min,
            length_max=self._length_max,
            length_mean=self._chars / documents if documents else None,
            by_lang=dict(self._by_lang),
            by_source=dict(self._by_source),
            steps=tuple(steps),
            fingerprint=self._fingerprint.hexdigest(),
        )

    def _add_length(self, chars: int) -> None:
        """Track the total, shortest and longest document length."""
        self._chars += chars
        if self._length_min is None or chars < self._length_min:
            self._length_min = chars
        if self._length_max is None or chars > self._length_max:
            self._length_max = chars
