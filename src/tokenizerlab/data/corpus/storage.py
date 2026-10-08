"""Saving and opening corpora on disk.

    path/
    ├── corpus.parquet   one row per document
    └── manifest.json    how the corpus was built, and its stats

Row conversion is pure; CorpusStore is the I/O orchestrator, with Parquet access injected.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizerlab.adapters import ColumnType, ParquetPort, PyArrowParquet, Row
from tokenizerlab.data.document import Document
from tokenizerlab.data.readers import DocumentStream, OnError, PassReport, SourceTraits
from tokenizerlab.data.stats import (
    CorpusStats,
    FileInfo,
    Fingerprint,
    Manifest,
    ManifestError,
    Provenance,
    SavedManifest,
    StatsAccumulator,
    decode_manifest,
    encode_manifest,
)
from tokenizerlab.errors import TokenizerLabError
from tokenizerlab.shared.json_format import canonical_json

CORPUS_FILE = "corpus.parquet"
MANIFEST_FILE = "manifest.json"

CORPUS_SCHEMA: tuple[tuple[str, ColumnType], ...] = (
    ("id", ColumnType.STRING),
    ("text", ColumnType.STRING),
    ("source", ColumnType.STRING),
    ("position", ColumnType.INT64),
    ("content_hash", ColumnType.STRING),
    ("lang", ColumnType.STRING),
    ("metadata", ColumnType.STRING),
)

# A row group ends at whichever limit comes first, so memory stays bounded for large documents.
ROW_GROUP_MAX_DOCUMENTS = 10_000
ROW_GROUP_MAX_CHARS = 64 * 2**20


class CorpusExistsError(TokenizerLabError):
    """save() would overwrite an existing corpus without overwrite=True."""


class UnsafeDestinationError(TokenizerLabError):
    """save() would replace something that is not a saved corpus, or the data being read."""


class MetadataNotSerializableError(TokenizerLabError):
    """A document's metadata cannot be stored as JSON."""


class IntegrityError(TokenizerLabError):
    """A saved corpus does not match its own hashes or fingerprint."""


# ---------------------------------------------------------------- rows (pure)


def document_to_row(document: Document) -> Row:
    """One row for a document; metadata becomes key-sorted JSON so all sources share a schema."""
    try:
        metadata = canonical_json(dict(document.metadata))
    except (TypeError, ValueError) as error:
        raise MetadataNotSerializableError(
            f"Document {document.id} ({document.source}#{document.position}): "
            f"metadata is not JSON-serializable: {error}"
        ) from None
    return {
        "id": document.id,
        "text": document.text,
        "source": document.source,
        "position": document.position,
        "content_hash": document.content_hash,
        "lang": document.lang,
        "metadata": metadata,
    }


def row_to_document(row: Row) -> Document:
    """Rebuild a document from its row; id and content_hash are recomputed, not copied."""
    return Document(
        text=row["text"],
        source=row["source"],
        position=row["position"],
        lang=row["lang"],
        metadata=json.loads(row["metadata"]),
    )


def row_groups(documents: Iterable[Document]) -> Iterator[list[Row]]:
    """Rows in groups bounded by document count and by total characters."""
    group: list[Row] = []
    chars = 0
    for document in documents:
        group.append(document_to_row(document))
        chars += len(document.text)
        if len(group) >= ROW_GROUP_MAX_DOCUMENTS or chars >= ROW_GROUP_MAX_CHARS:
            yield group
            group, chars = [], 0
    if group:
        yield group


# ---------------------------------------------------------------- destinations


def check_destination(
    destination: Path,
    source_paths: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    """Refuse a save that could destroy data.

    The destination may not contain, be, or lie inside any path being read, and an existing
    destination is replaced only with overwrite=True and only if it is itself a saved corpus.
    """
    _refuse_overlap(destination, source_paths)
    if not destination.exists():
        return
    if not overwrite:
        raise CorpusExistsError(f"{destination} exists; pass overwrite=True to replace it")
    _require_saved_corpus(destination)


def _refuse_overlap(destination: Path, source_paths: Sequence[Path]) -> None:
    """The destination must not overlap any source path."""
    target = destination.resolve()
    for source in source_paths:
        source_path = source.resolve()
        if target.is_relative_to(source_path) or source_path.is_relative_to(target):
            raise UnsafeDestinationError(
                f"Refusing to save to {destination}: it overlaps the source {source}, "
                "which is being read. Save somewhere else."
            )


def _require_saved_corpus(destination: Path) -> None:
    """An existing destination must hold a saved corpus and nothing else."""
    corpus_files = {CORPUS_FILE, MANIFEST_FILE}
    entries = {entry.name for entry in destination.iterdir()} if destination.is_dir() else None
    if entries is None or MANIFEST_FILE not in entries or not entries <= corpus_files:
        raise UnsafeDestinationError(
            f"Refusing to replace {destination}: it is not a saved tokenizerlab corpus "
            f"(expected only {sorted(corpus_files)})."
        )


@contextmanager
def replace_directory_atomically(path: Path) -> Iterator[Path]:
    """Yield an empty temporary directory that replaces `path` on success and is deleted on
    failure, so a crash never leaves a half-written corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created next to path, so the final rename never crosses filesystems.
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        yield temporary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if path.exists():
        shutil.rmtree(path)
    os.replace(temporary, path)


# ---------------------------------------------------------------- loading strategies


class RowLoader(Protocol):
    """How saved rows become Documents: the part of loading that varies."""

    def __call__(self, rows: Iterable[Row], manifest: SavedManifest) -> Iterator[Document]:
        """One pass of Documents rebuilt from rows."""
        ...


def trusting_loader(rows: Iterable[Row], manifest: SavedManifest) -> Iterator[Document]:
    """Rebuild every row without checking it against the stored hashes."""
    return (row_to_document(row) for row in rows)


def verifying_loader(rows: Iterable[Row], manifest: SavedManifest) -> Iterator[Document]:
    """Rebuild every row, raising IntegrityError if a stored id/content_hash differs from the
    recomputed one, or if the pass's fingerprint differs from the manifest's."""
    fingerprint = Fingerprint()
    for row in rows:
        document = row_to_document(row)
        if (document.id, document.content_hash) != (row["id"], row["content_hash"]):
            raise IntegrityError(
                f"{document.source}#{document.position}: stored id/content_hash do not "
                "match the text"
            )
        fingerprint.add(document)
        yield document

    expected = manifest.stats.fingerprint
    if fingerprint.hexdigest() != expected:
        raise IntegrityError(
            f"fingerprint {fingerprint.hexdigest()} does not match the manifest's {expected}"
        )


# ---------------------------------------------------------------- the store (I/O)


class SavableCorpus(Protocol):
    """What the store needs from a corpus: where it reads, its documents, and then how
    they were selected."""

    def __iter__(self) -> Iterator[Document]:
        """One pass over the documents."""
        ...

    def source_paths(self) -> tuple[Path, ...]:
        """The filesystem paths the corpus reads from."""
        ...

    def provenance(self) -> Provenance:
        """How the corpus was built, from the last complete pass."""
        ...


@dataclass(frozen=True, slots=True)
class OpenedCorpus:
    """A saved corpus's documents, as a stream, and its manifest."""

    stream: DocumentStream
    manifest: SavedManifest


class CorpusStore:
    """Saves corpora as Parquet plus a manifest, and opens them again."""

    def __init__(self, parquet: ParquetPort) -> None:
        """Read and write Parquet through `parquet`."""
        self._parquet = parquet

    def save(self, corpus: SavableCorpus, destination: Path, *, overwrite: bool) -> CorpusStats:
        """Write the corpus in one pass, atomically; return the stats of that pass."""
        check_destination(destination, corpus.source_paths(), overwrite=overwrite)

        accumulator = StatsAccumulator()
        with replace_directory_atomically(destination) as directory:
            corpus_file = directory / CORPUS_FILE
            groups = row_groups(accumulator.observe(corpus))
            self._parquet.write_row_groups(corpus_file, CORPUS_SCHEMA, groups)

            provenance = corpus.provenance()
            stats = accumulator.finish(provenance.step_counts)
            file_info = FileInfo(rows=stats.documents, bytes=corpus_file.stat().st_size)
            manifest = Manifest(provenance, stats, files={CORPUS_FILE: file_info})
            (directory / MANIFEST_FILE).write_text(encode_manifest(manifest), encoding="utf-8")
        return stats

    def open(self, directory: Path, loader: RowLoader) -> OpenedCorpus:
        """A stream over a saved corpus; `loader` rebuilds the Documents on every pass."""
        manifest_file = directory / MANIFEST_FILE
        if not manifest_file.is_file():
            raise ManifestError(f"{manifest_file}: no manifest; is this a saved corpus?")
        manifest = decode_manifest(manifest_file.read_text(encoding="utf-8"), str(manifest_file))
        corpus_file = directory / CORPUS_FILE

        def run(report: PassReport) -> Iterator[Document]:
            """One pass over the saved rows."""
            return loader(self._parquet.read_rows(corpus_file), manifest)

        # on_error=RAISE: a damaged saved corpus must fail loudly, never lose rows quietly.
        traits = SourceTraits(paths=(directory,))
        stream = DocumentStream(run, lambda: [str(corpus_file)], OnError.RAISE, traits)
        return OpenedCorpus(stream, manifest)


@functools.cache
def default_store() -> CorpusStore:
    """Composition root: a store writing Parquet with pyarrow, built once on first use."""
    return CorpusStore(PyArrowParquet())
