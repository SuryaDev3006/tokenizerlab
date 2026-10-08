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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizerlab.adapters import ColumnType, ParquetPort, PyArrowParquet, Row
from tokenizerlab.data.document import Document
from tokenizerlab.data.readers import DocumentStream, OnError
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
    """What the store needs from a corpus: its documents, then how they were selected."""

    def __iter__(self) -> Iterator[Document]:
        """One pass over the documents."""
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
        if destination.exists() and not overwrite:
            raise CorpusExistsError(f"{destination} exists; pass overwrite=True to replace it")

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

        def run(stream: DocumentStream) -> Iterator[Document]:
            """One pass over the saved rows."""
            return loader(self._parquet.read_rows(corpus_file), manifest)

        # on_error=RAISE: a damaged saved corpus must fail loudly, never lose rows quietly.
        stream = DocumentStream(run, lambda: [str(corpus_file)], OnError.RAISE)
        return OpenedCorpus(stream, manifest)


@functools.cache
def default_store() -> CorpusStore:
    """Composition root: a store writing Parquet with pyarrow, built once on first use."""
    return CorpusStore(PyArrowParquet())
