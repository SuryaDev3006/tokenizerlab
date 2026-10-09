"""File formats: a registry of formats and their parsers, one parser per format."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from tokenizerlab.adapters import ParquetPort
from tokenizerlab.data.document import Document
from tokenizerlab.data.readers.options import ReadOptions, Unit
from tokenizerlab.data.readers.rows import MissingTextFieldError, Source
from tokenizerlab.errors import ConfigurationError

# The default CSV field limit is 128 KB; documents are often larger. The cap is the largest
# value csv accepts on every platform (a C long is 32 bits on Windows).
CSV_FIELD_LIMIT = min(sys.maxsize, 2**31 - 1)


class UnsupportedFormatError(ConfigurationError):
    """A file type or format name that no registered parser handles."""


class Parser(Protocol):
    """Yields the Documents of one file."""

    def __call__(self, path: Path, source: Source) -> Iterator[Document]:
        """Parse the file at `path` as `source`."""
        ...


@dataclass(frozen=True, slots=True)
class FileFormat:
    """A named format, the file extensions that identify it, and its parser."""

    name: str
    extensions: tuple[str, ...]
    parser: Parser


class FormatRegistry:
    """Registered formats; adding a format never changes the code that reads files."""

    def __init__(self) -> None:
        """Start with no formats."""
        self._formats: dict[str, FileFormat] = {}

    def register(self, name: str, extensions: Sequence[str], parser: Parser) -> None:
        """Add a format recognized by any of `extensions` (e.g. ".jsonl", ".jsonl.gz")."""
        lowercase = tuple(extension.lower() for extension in extensions)
        self._formats[name] = FileFormat(name, lowercase, parser)

    @property
    def names(self) -> list[str]:
        """Every registered format name, sorted."""
        return sorted(self._formats)

    def require(self, name: str) -> str:
        """`name` if it is registered; otherwise raise UnsupportedFormatError."""
        if name not in self._formats:
            raise UnsupportedFormatError(f"format must be one of {self.names}, got {name!r}")
        return name

    def supports(self, filename: str) -> bool:
        """Whether any format recognizes this file name."""
        return any(True for _ in self._matches(filename))

    def detect(self, filename: str) -> str:
        """The format of a file name; the longest matching extension wins (".txt.gz" over ".gz")."""
        matches = sorted(self._matches(filename), key=lambda match: len(match[0]), reverse=True)
        if not matches:
            raise UnsupportedFormatError(
                f"Unknown file type: {filename!r}. Pass format= one of {self.names}."
            )
        return matches[0][1]

    def parser(self, name: str) -> Parser:
        """The parser of a registered format."""
        return self._formats[self.require(name)].parser

    def _matches(self, filename: str) -> Iterator[tuple[str, str]]:
        """(extension, format name) for every registered extension the file name ends with."""
        lowercase = filename.lower()
        for file_format in self._formats.values():
            for extension in file_format.extensions:
                if lowercase.endswith(extension):
                    yield extension, file_format.name


# ---------------------------------------------------------------- text-based parsers


def _open_text(path: Path, newline: str = "\n") -> TextIO:
    """Open a plain or gzipped file as UTF-8 text without newline translation.

    newline="\\n": lines split only on "\\n" and nothing is translated.
    utf-8-sig: a leading BOM is stripped; otherwise this is plain, strict UTF-8.
    """
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline=newline)
    return open(path, encoding="utf-8-sig", newline=newline)


TextSplitter = Callable[[TextIO, Source], Iterator[Document]]


def parse_text(path: Path, source: Source) -> Iterator[Document]:
    """Yield the file's Documents, split according to the unit= setting."""
    with _open_text(path) as file:
        yield from _TEXT_SPLITTERS[source.options.unit](file, source)


def _whole_file(file: TextIO, source: Source) -> Iterator[Document]:
    """unit="file": the whole file is one Document."""
    yield source.document(file.read(), position=0)


def _each_line(file: TextIO, source: Source) -> Iterator[Document]:
    """unit="line": one Document per physical line, positioned by line number."""
    for line_number, line in enumerate(file):
        yield source.document(_strip_line_ending(line), line_number)


_TEXT_SPLITTERS: dict[Unit, TextSplitter] = {Unit.FILE: _whole_file, Unit.LINE: _each_line}


def _strip_line_ending(line: str) -> str:
    """Remove a trailing "\\n" or "\\r\\n"; a lone "\\r" is text, not a line ending."""
    if line.endswith("\n"):
        return line[:-1].removesuffix("\r")
    return line


def _offering_raw_text(documents: Iterator[Document]) -> Iterator[Document]:
    """A CSV, TSV or JSONL file can also be read raw, so its missing-text-field error says so."""
    try:
        yield from documents
    except MissingTextFieldError as error:
        raise error.offering_raw_text() from None


def parse_jsonl(path: Path, source: Source) -> Iterator[Document]:
    """Yield one Document per JSON line; blank lines are skipped but still count as positions."""
    yield from _offering_raw_text(source.documents_from_rows(_json_lines(path, source)))


def _json_lines(path: Path, source: Source) -> Iterator[tuple[int, Any]]:
    """Yield (line number, parsed object) pairs, recording invalid JSON as row errors."""
    with _open_text(path) as file:
        for line_number, line in enumerate(file):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                source.record_row_error(f"invalid JSON: {error.msg}", line_number)
                continue
            yield line_number, row


def parse_delimited(path: Path, source: Source) -> Iterator[Document]:
    """Yield one Document per CSV/TSV row; `delimiter` defaults to tab for .tsv, else comma."""
    default_delimiter = "\t" if path.name.lower().endswith(".tsv") else ","
    delimiter = source.options.delimiter or default_delimiter

    _allow_large_csv_fields()
    with _open_text(path, newline="") as file:  # the csv module needs newline=""
        rows = csv.DictReader(file, delimiter=delimiter)
        yield from _offering_raw_text(source.documents_from_rows(enumerate(rows)))


def _allow_large_csv_fields() -> None:
    """Raise the process-wide CSV field limit to CSV_FIELD_LIMIT if it is lower.

    Done when a CSV file is read, never at import, so importing tokenizerlab leaves the
    limit alone. The old limit is not restored afterwards: passes are generators that can
    interleave, and restoring it would break another CSV pass that is still running.
    """
    if csv.field_size_limit() < CSV_FIELD_LIMIT:
        csv.field_size_limit(CSV_FIELD_LIMIT)


# ---------------------------------------------------------------- Parquet


class ParquetParser:
    """Parses Parquet files through a ParquetPort, reading only the columns needed."""

    def __init__(self, parquet: ParquetPort) -> None:
        """Read files through `parquet`."""
        self._parquet = parquet

    def __call__(self, path: Path, source: Source) -> Iterator[Document]:
        """Yield one Document per row; fail the file at once if text_field is not a column."""
        available_columns = self._parquet.column_names(path)
        if source.options.text_field not in available_columns:
            raise source.missing_text_field_error(available_columns)

        rows = self._parquet.read_rows(path, _needed_columns(source.options, available_columns))
        yield from source.documents_from_rows(enumerate(rows))


def _needed_columns(options: ReadOptions, available_columns: list[str]) -> list[str] | None:
    """All columns for metadata_fields="all" (None), else the text, lang and metadata columns."""
    if options.metadata_fields == "all":
        return None

    wanted = [options.text_field, options.lang_field, *(options.metadata_fields or ())]
    unique_wanted = dict.fromkeys(wanted)
    return [column for column in unique_wanted if column and column in available_columns]


def default_formats(parquet: ParquetPort) -> FormatRegistry:
    """Text, JSON Lines, CSV/TSV and Parquet, with gzip support for the text formats."""
    formats = FormatRegistry()
    formats.register("text", [".txt", ".txt.gz"], parse_text)
    formats.register("jsonl", [".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz"], parse_jsonl)
    formats.register("csv", [".csv", ".tsv"], parse_delimited)
    formats.register("parquet", [".parquet"], ParquetParser(parquet))
    return formats
