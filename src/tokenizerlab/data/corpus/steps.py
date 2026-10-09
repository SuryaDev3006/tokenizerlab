"""Selection steps: the Step contract, selection hashing, and Filter, Dedup and Sample.

A step only decides which documents to keep. None of them changes text, and none of them
repeats (upsamples) a document. Mix lives in mixing.py and Split in split.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from tokenizerlab.data.corpus.digests import DigestIndex, DigestStore, InMemoryDigests
from tokenizerlab.data.corpus.near import HASHES, METHOD, NGRAM, banding, near_keys
from tokenizerlab.data.document import Document
from tokenizerlab.data.stats import SelectionReport
from tokenizerlab.errors import ConfigurationError
from tokenizerlab.shared.hashing import unit_interval

DocumentPredicate = Callable[[Document], bool]


@dataclass(slots=True)
class StepContext:
    """What a step can use during one pass, and where it reports its result."""

    step_index: int  # position in the history; part of every selection hash
    bytes_by_lang: Mapping[str | None, int]  # measured by the pre-pass; empty if not needed
    result: SelectionReport | None = None
    # Warnings already shown by this corpus pass; a step that reads another corpus
    # (dedup against=) shares them, so the same source never warns twice in one pass.
    shown_warnings: set[str] = field(default_factory=set)


class Step(Protocol):
    """A selection step: decides which documents to keep, in their original order."""

    @property
    def name(self) -> str:
        """The operation name recorded in the history ("filter", "dedup", ...)."""
        ...

    @property
    def needs_prepass(self) -> bool:
        """Whether apply() needs the bytes per language entering the step."""
        ...

    @property
    def reproducible(self) -> bool:
        """Whether the recorded parameters are enough to re-create this step exactly."""
        ...

    def parameters(self) -> dict[str, Any]:
        """The step's parameters as JSON-ready values, as recorded in the history."""
        ...

    def apply(self, documents: Iterator[Document], context: StepContext) -> Iterator[Document]:
        """Yield the documents this step keeps."""
        ...


def selection_value(seed: object, step_index: int, document_id: str) -> float:
    """A deterministic value in [0, 1); a step keeps a document when it is below the keep fraction.

    The step index keeps selection steps independent: without it, sample(0.5) followed by a
    mix with the same seed would reuse the same values and the mix would keep everything.
    Selection depends only on the id, never on input order or timing.
    """
    return unit_interval(f"{seed}:{step_index}:{document_id}")


def validate_max_bytes(max_bytes: int | None) -> None:
    """Reject a negative byte budget."""
    if max_bytes is not None and max_bytes < 0:
        raise ConfigurationError(f"max_bytes must be >= 0, got {max_bytes}")


# ---------------------------------------------------------------- Filter


@dataclass(frozen=True, slots=True)
class Filter:
    """Keep documents with non-whitespace text, at least min_chars characters, and fn(doc) true.

    Empty and whitespace-only documents are always removed, whatever fn and min_chars say.
    """

    fn: DocumentPredicate | None = None
    min_chars: int = 0

    def __post_init__(self) -> None:
        """Reject a negative minimum length or a non-callable predicate."""
        if self.min_chars < 0:
            raise ConfigurationError(f"min_chars must be >= 0, got {self.min_chars}")
        if self.fn is not None and not callable(self.fn):
            raise ConfigurationError("fn must be a function taking a Document")

    @property
    def name(self) -> str:
        """Recorded as "filter"."""
        return "filter"

    @property
    def needs_prepass(self) -> bool:
        """Each document is judged on its own."""
        return False

    @property
    def reproducible(self) -> bool:
        """A custom function cannot be serialized, so it cannot be re-created from a manifest."""
        return self.fn is None

    def parameters(self) -> dict[str, Any]:
        """min_chars, and fn by qualified name ("<lambda>" for a lambda)."""
        return {"fn": qualified_name(self.fn), "min_chars": self.min_chars}

    def apply(self, documents: Iterator[Document], context: StepContext) -> Iterator[Document]:
        """Yield the documents that pass every check."""
        return (document for document in documents if self._keeps(document))

    def _keeps(self, document: Document) -> bool:
        """Whether the document is non-blank, long enough, and accepted by fn."""
        text = document.text
        if not text.strip() or len(text) < self.min_chars:
            return False
        return self.fn is None or self.fn(document)


def qualified_name(fn: Callable[[Document], object] | None) -> str | None:
    """A function's qualified name; callables without one (e.g. partials) by their type."""
    if fn is None:
        return None
    name = getattr(fn, "__qualname__", None)
    return name if isinstance(name, str) else type(fn).__qualname__


# ---------------------------------------------------------------- Dedup


@dataclass(frozen=True, slots=True)
class Against:
    """The documents a dedup must also remove matches of, e.g. the validation side of a
    split, with that corpus's recorded history."""

    documents: Callable[[set[str]], Iterator[Document]]  # one pass, sharing shown warnings
    history: list[dict[str, Any]]
    reproducible: bool


@dataclass(frozen=True, slots=True)
class DedupResult:
    """How many documents a dedup removed because they matched its `against` corpus."""

    removed_against: int

    def as_record(self) -> dict[str, Any]:
        """The result as JSON-ready fields."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Dedup:
    """Drop duplicates across all sources, keeping the first occurrence.

    With `threshold` None, duplicates are exact copies (same content_hash); with a threshold,
    near-duplicates as defined in near.py, which always include exact copies. A document is
    dropped only when it matches a document kept earlier in the pass, or any document of
    `against`, which is read once at the start of the pass.

    `store` decides where the keys seen so far are kept. The default in-memory set costs
    about 100 MB per million unique documents for exact dedup (about 1.1 GB for near dedup);
    OnDiskDigests keeps them in a temporary SQLite database instead. The store never changes
    which documents are kept, so it is not recorded in the history.
    """

    store: DigestStore = field(default_factory=InMemoryDigests)
    threshold: float | None = None
    against: Against | None = None

    def __post_init__(self) -> None:
        """A near-duplicate threshold must be in (0, 1)."""
        if self.threshold is not None and not 0 < self.threshold < 1:
            raise ConfigurationError(f"threshold must be in (0, 1), got {self.threshold}")

    @property
    def name(self) -> str:
        """Recorded as "dedup"."""
        return "dedup"

    @property
    def needs_prepass(self) -> bool:
        """Duplicates are found in the main pass."""
        return False

    @property
    def reproducible(self) -> bool:
        """Reproducible unless the `against` corpus is not."""
        return self.against is None or self.against.reproducible

    def parameters(self) -> dict[str, Any]:
        """Nothing for a plain dedup; the near-duplicate settings and `against` otherwise."""
        if self.threshold is None and self.against is None:
            return {}
        parameters: dict[str, Any] = {"near": self.threshold is not None}
        if self.threshold is not None:
            bands, rows = banding(self.threshold)
            parameters.update(
                threshold=self.threshold,
                bands=bands,
                rows=rows,
                hashes=HASHES,
                ngram=NGRAM,
                method=METHOD,
            )
        parameters["against"] = None if self.against is None else self.against.history
        return parameters

    def apply(self, documents: Iterator[Document], context: StepContext) -> Iterator[Document]:
        """Yield each document that matches neither a kept document nor `against`."""
        # The stores are closed when the pass ends, fails, or is abandoned midway.
        against_store = self.store if self.against is not None else InMemoryDigests()
        with self.store.open_pass() as kept, against_store.open_pass() as excluded:
            self._index_against(excluded, context)
            removed_against = 0
            for document in documents:
                keys = self.keys(document)
                if any(excluded.seen(key) for key in keys):
                    removed_against += 1
                elif not any(kept.seen(key) for key in keys):
                    _add_all(kept, keys)
                    yield document

        if self.against is not None:
            context.result = DedupResult(removed_against)

    def keys(self, document: Document) -> list[bytes]:
        """What a duplicate shares with the document: its content hash, or its near keys."""
        if self.threshold is None:
            # Raw 16-byte digests take about half the memory of 32-character hex strings.
            return [bytes.fromhex(document.content_hash)]
        return near_keys(document.text, *banding(self.threshold))

    def _index_against(self, excluded: DigestIndex, context: StepContext) -> None:
        """Add the keys of every `against` document to `excluded`."""
        if self.against is None:
            return
        for document in self.against.documents(context.shown_warnings):
            _add_all(excluded, self.keys(document))


def _add_all(index: DigestIndex, keys: list[bytes]) -> None:
    """Add every key to the index."""
    for key in keys:
        index.add(key)


# ---------------------------------------------------------------- Sample


@dataclass(frozen=True, slots=True)
class SampleResult:
    """The keep fraction a sample used; with a byte budget, the fraction derived from it."""

    fraction: float
    target_bytes: int | None

    def as_record(self) -> dict[str, Any]:
        """The result as JSON-ready fields."""
        return asdict(self)


class KeepRate(Protocol):
    """How a sample decides its keep fraction: the part of sampling that varies."""

    @property
    def needs_prepass(self) -> bool:
        """Whether resolve() needs the bytes per language entering the step."""
        ...

    def resolve(self, bytes_by_lang: Mapping[str | None, int]) -> SampleResult:
        """The keep fraction for this pass, and what it was derived from."""
        ...

    def parameters(self) -> dict[str, Any]:
        """The rate's parameters as recorded in the history."""
        ...


@dataclass(frozen=True, slots=True)
class FixedFraction:
    """Keep a fixed fraction of documents."""

    fraction: float

    def __post_init__(self) -> None:
        """The fraction must be in [0, 1]."""
        if not 0 <= self.fraction <= 1:
            raise ConfigurationError(f"fraction must be in [0, 1], got {self.fraction}")

    @property
    def needs_prepass(self) -> bool:
        """The fraction is known up front."""
        return False

    def resolve(self, bytes_by_lang: Mapping[str | None, int]) -> SampleResult:
        """Always the given fraction."""
        return SampleResult(fraction=self.fraction, target_bytes=None)

    def parameters(self) -> dict[str, Any]:
        """fraction (and no byte budget)."""
        return {"fraction": self.fraction, "max_bytes": None}


@dataclass(frozen=True, slots=True)
class ByteBudget:
    """Keep roughly max_bytes: fraction = min(1, max_bytes / bytes entering the step)."""

    max_bytes: int

    def __post_init__(self) -> None:
        """The budget must not be negative."""
        validate_max_bytes(self.max_bytes)

    @property
    def needs_prepass(self) -> bool:
        """The fraction depends on the bytes entering the step."""
        return True

    def resolve(self, bytes_by_lang: Mapping[str | None, int]) -> SampleResult:
        """The fraction whose expected output is max_bytes (all documents if they fit)."""
        total_bytes = sum(bytes_by_lang.values())
        fraction = min(1.0, self.max_bytes / total_bytes) if total_bytes else 1.0
        return SampleResult(fraction=fraction, target_bytes=self.max_bytes)

    def parameters(self) -> dict[str, Any]:
        """max_bytes (and no fixed fraction)."""
        return {"fraction": None, "max_bytes": self.max_bytes}


def keep_rate(fraction: float | None, max_bytes: int | None) -> KeepRate:
    """The KeepRate for sample(fraction=...) or sample(max_bytes=...); exactly one is allowed."""
    if fraction is not None and max_bytes is None:
        return FixedFraction(fraction)
    if max_bytes is not None and fraction is None:
        return ByteBudget(max_bytes)
    raise ConfigurationError("sample needs exactly one of fraction or max_bytes")


@dataclass(frozen=True, slots=True)
class Sample:
    """Keep documents whose selection value is below the keep fraction of `rate`.

    A byte budget is approximate; the manifest records target and achieved bytes.
    """

    rate: KeepRate
    seed: int = 0

    @property
    def name(self) -> str:
        """Recorded as "sample"."""
        return "sample"

    @property
    def needs_prepass(self) -> bool:
        """Only a byte budget needs the bytes entering the step."""
        return self.rate.needs_prepass

    @property
    def reproducible(self) -> bool:
        """The rate plus the seed fully determine the selection."""
        return True

    def parameters(self) -> dict[str, Any]:
        """fraction, max_bytes and seed."""
        return {**self.rate.parameters(), "seed": self.seed}

    def apply(self, documents: Iterator[Document], context: StepContext) -> Iterator[Document]:
        """Yield the documents whose selection value is below the keep fraction."""
        result = self.rate.resolve(context.bytes_by_lang)
        context.result = result
        for document in documents:
            if selection_value(self.seed, context.step_index, document.id) < result.fraction:
                yield document
