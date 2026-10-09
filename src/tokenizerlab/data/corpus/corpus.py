"""Corpus: a lazy, immutable, reproducible selection of Documents for tokenizer training.

    corpus = Corpus(read("data/")).filter(min_chars=100).dedup().mix(alpha=0.3, seed=42)
    corpus.save("artifacts/v1")
    texts = Corpus.load("artifacts/v1").texts()

Every operation returns a new Corpus with one more step; nothing is read until it is
iterated, and one full iteration is one pass. Corpus never changes document text.
"""

from __future__ import annotations

import copy
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizerlab.data.corpus.digests import digest_store
from tokenizerlab.data.corpus.mixing import Mix, share_strategy
from tokenizerlab.data.corpus.steps import (
    Dedup,
    DocumentPredicate,
    Filter,
    Sample,
    Step,
    StepContext,
    keep_rate,
)
from tokenizerlab.data.corpus.storage import (
    CorpusStore,
    OpenedCorpus,
    default_store,
    trusting_loader,
    verifying_loader,
)
from tokenizerlab.data.document import Document
from tokenizerlab.data.readers import read
from tokenizerlab.data.stats import (
    CorpusStats,
    Counts,
    HistoryEntry,
    ParentInfo,
    Provenance,
    ReaderReport,
    SelectionReport,
    StatsAccumulator,
    StepCounts,
    StepRecord,
)
from tokenizerlab.errors import TokenizerLabError
from tokenizerlab.shared.hashing import utf8_size

PathLike = str | os.PathLike[str]


class IncompletePassError(TokenizerLabError):
    """Pass results were requested before a complete pass over the corpus."""


@dataclass(frozen=True, slots=True)
class _Link:
    """How a derived Corpus is built: one step applied to its parent's output."""

    parent: Corpus
    step: Step


class Corpus:
    """A lazy, immutable pipeline of selection steps over a DocumentStream.

    Each Corpus is one node of the pipeline: the source itself, or one step applied to
    its parent Corpus. Deriving a new Corpus never modifies the existing one.
    """

    def __init__(self, source: Any):
        """Wrap a DocumentStream (anything `read` accepts also works). Nothing is read yet."""
        stream = read(source)
        self._stream = stream
        self._link: _Link | None = None  # None for the source node
        self._step_index = 0  # position in the history; part of every selection hash
        self._entry: HistoryEntry | None = None  # None: the read step, recorded from the stream
        self._parent_info: ParentInfo | None = None  # set for a corpus loaded from disk
        self._bytes_by_lang: Counter[str | None] | None = None  # cached pre-pass result
        self._last_output: Counts | None = None  # from the last complete pass
        self._last_selection: SelectionReport | None = None

    # ---------------------------------------------------------------- operations

    def filter(self, fn: DocumentPredicate | None = None, *, min_chars: int = 0) -> Corpus:
        """Keep non-blank documents with at least min_chars characters that fn(doc) accepts."""
        return self._then(Filter(fn=fn, min_chars=min_chars))

    def dedup(self, *, on_disk: bool = False) -> Corpus:
        """Drop exact duplicates by content_hash, keeping the first occurrence.

        on_disk=True keeps the digests seen so far in a temporary SQLite database instead of
        memory (about 100 MB per million unique documents); the result is identical.
        """
        return self._then(Dedup(digest_store(on_disk=on_disk)))

    def sample(
        self,
        fraction: float | None = None,
        *,
        max_bytes: int | None = None,
        seed: int = 0,
    ) -> Corpus:
        """Keep a deterministic fraction of documents, or roughly max_bytes of them."""
        return self._then(Sample(keep_rate(fraction, max_bytes), seed=seed))

    def mix(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        alpha: float | None = None,
        max_bytes: int | None = None,
        seed: int = 0,
    ) -> Corpus:
        """Downsample languages to target shares from weights, or from bytes ** alpha."""
        return self._then(Mix(share_strategy(weights, alpha), max_bytes=max_bytes, seed=seed))

    def _then(self, step: Step) -> Corpus:
        """A new Corpus that applies `step` to this one's output."""
        child = copy.copy(self)
        child._link = _Link(parent=self, step=step)
        child._step_index = self._step_index + 1
        child._entry = HistoryEntry(step.name, step.parameters(), step.reproducible)
        child._bytes_by_lang = None
        child._last_output = None
        child._last_selection = None
        return child

    # ---------------------------------------------------------------- passes

    def __iter__(self) -> Iterator[Document]:
        """One pass: run pending pre-passes first, then stream documents through every step."""
        self._run_prepasses()
        context = StepContext(self._step_index, self._bytes_by_lang or Counter())

        documents_out = bytes_out = 0
        for document in self._step_output(context):
            documents_out += 1
            bytes_out += utf8_size(document.text)
            yield document

        self._last_output = Counts(documents_out, bytes_out)
        self._last_selection = context.result

    def _step_output(self, context: StepContext) -> Iterator[Document]:
        """This node's documents: the source itself, or this node's step applied to its parent."""
        if self._link is None:
            return iter(self._stream)
        return self._link.step.apply(iter(self._link.parent), context)

    def _run_prepasses(self) -> None:
        """Measure, once, the bytes per language entering every step that needs it, upstream
        first. This finishes before the main pass touches the source, so the reader errors
        reported afterwards come from that one pass."""
        if self._link is None:
            return

        parent = self._link.parent
        parent._run_prepasses()
        if self._link.step.needs_prepass and self._bytes_by_lang is None:
            self._bytes_by_lang = parent._bytes_per_language()

    def _bytes_per_language(self) -> Counter[str | None]:
        """One pass, totalling UTF-8 bytes per language."""
        totals: Counter[str | None] = Counter()
        for document in self:
            totals[document.lang] += utf8_size(document.text)
        return totals

    def _lineage(self) -> list[Corpus]:
        """This corpus and its ancestors, source first."""
        lineage = [self]
        while lineage[-1]._link is not None:
            lineage.append(lineage[-1]._link.parent)
        return lineage[::-1]

    def _step_records(self) -> tuple[StepRecord, ...]:
        """Per-step counts in and out, plus selection results, from the last complete pass."""
        records = []
        counts_in: Counts | None = None
        for node in self._lineage():
            counts_out = node._last_output
            if counts_out is None:
                raise IncompletePassError("No complete pass yet: iterate the corpus fully first")
            # The source step has no input of its own: what it reads is what it yields.
            op = node._history_entry().op
            counts = StepCounts.between(op, counts_in or counts_out, counts_out)
            records.append(StepRecord(counts, node._last_selection))
            counts_in = counts_out
        return tuple(records)

    # ---------------------------------------------------------------- inspection

    @property
    def history(self) -> list[HistoryEntry]:
        """Every step from the source onward, with its parameters."""
        return [node._history_entry() for node in self._lineage()]

    def _history_entry(self) -> HistoryEntry:
        """This node's history entry. The read step is built from the stream when asked,
        because whether it is reproducible can depend on its last pass."""
        if self._entry is not None:
            return self._entry
        return HistoryEntry("read", self._stream.config, self._stream.reproducible)

    def source_paths(self) -> tuple[Path, ...]:
        """The filesystem paths this corpus reads; save() refuses to write over them."""
        return self._stream.traits.paths

    def provenance(self) -> Provenance:
        """How this corpus was built and what each step did, from the last complete pass."""
        return Provenance(
            history=tuple(self.history),
            parent=self._lineage()[0]._parent_info,
            steps=self._step_records(),
            reader=ReaderReport.from_stream(self._stream),
        )

    def stats(self) -> CorpusStats:
        """Run one pass and summarize it."""
        accumulator = StatsAccumulator()
        for document in self:
            accumulator.add(document)
        return accumulator.finish(self.provenance().step_counts)

    def texts(self) -> Iterable[str]:
        """Re-iterable document texts for a trainer; each iteration re-runs the pipeline."""
        return _Texts(self)

    # ---------------------------------------------------------------- saving and loading

    def save(
        self,
        path: PathLike,
        *,
        overwrite: bool = False,
        store: CorpusStore | None = None,
    ) -> CorpusStats:
        """Write corpus.parquet and manifest.json atomically in one pass; return its stats."""
        return (store or default_store()).save(self, Path(path), overwrite=overwrite)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        verify: bool = True,
        store: CorpusStore | None = None,
    ) -> Corpus:
        """Open a saved corpus. With verify, ids, content hashes and the fingerprint are checked."""
        directory = Path(path)
        loader = verifying_loader if verify else trusting_loader
        opened = (store or default_store()).open(directory, loader)
        return cls._from_saved(opened, directory)

    @classmethod
    def _from_saved(cls, opened: OpenedCorpus, directory: Path) -> Corpus:
        """A source Corpus over saved documents, whose history starts with "load"."""
        fingerprint = opened.manifest.stats.fingerprint
        corpus = cls(opened.stream)
        corpus._entry = HistoryEntry("load", {"path": str(directory), "fingerprint": fingerprint})
        corpus._parent_info = opened.manifest.as_parent()
        # Continue the step numbering of everything before the save, so a sample after
        # loading never reuses the selection values of a sample applied before saving.
        corpus._step_index = opened.manifest.steps_so_far
        return corpus


class _Texts:
    """Re-iterable view of a corpus's texts; every iteration re-runs the pipeline."""

    def __init__(self, corpus: Corpus):
        """Wrap the corpus whose texts to yield."""
        self._corpus = corpus

    def __iter__(self) -> Iterator[str]:
        """One pass, yielding only document text."""
        return (document.text for document in self._corpus)
