"""Ports and adapters: how core code reaches third-party libraries.

Core modules depend only on the Protocols (ports) below. The adapters implement them with
pyarrow and Hugging Face, importing those libraries lazily, so a missing optional extra only
matters when it is actually used.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from tokenizerlab.errors import MissingDependencyError, TokenizerLabError

Row = dict[str, Any]


class RevisionResolutionError(TokenizerLabError):
    """A dataset hub could not pin a dataset to an exact revision."""


def import_optional(module_name: str, extra: str) -> ModuleType:
    """Import an optional dependency, or explain which tokenizerlab extra installs it."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise MissingDependencyError(
            f"{module_name} is not installed. Install it with: pip install 'tokenizerlab[{extra}]'"
        ) from None


# ---------------------------------------------------------------- Parquet


class ColumnType(StrEnum):
    """Column types a Parquet schema can use; values are pyarrow type factory names."""

    STRING = "string"
    INT64 = "int64"


Schema = Sequence[tuple[str, ColumnType]]


class ParquetPort(Protocol):
    """Reads and writes Parquet files as plain rows."""

    def column_names(self, path: Path) -> list[str]:
        """The names of the file's columns."""
        ...

    def read_rows(self, path: Path, columns: Sequence[str] | None = None) -> Iterator[Row]:
        """Stream the file's rows, optionally only some columns."""
        ...

    def write_row_groups(self, path: Path, schema: Schema, row_groups: Iterable[list[Row]]) -> None:
        """Write one Parquet row group per list of rows, holding only one group at a time."""
        ...


class PyArrowParquet:
    """ParquetPort backed by pyarrow (the 'parquet' extra)."""

    def __init__(self, batch_size: int = 1024) -> None:
        """Read `batch_size` rows at a time."""
        self._batch_size = batch_size

    def column_names(self, path: Path) -> list[str]:
        """The names of the file's columns."""
        parquet = import_optional("pyarrow.parquet", extra="parquet")
        with open(path, "rb") as handle:
            return list(parquet.ParquetFile(handle).schema_arrow.names)

    def read_rows(self, path: Path, columns: Sequence[str] | None = None) -> Iterator[Row]:
        """Stream the file's rows in batches, optionally only some columns."""
        parquet = import_optional("pyarrow.parquet", extra="parquet")
        selected = None if columns is None else list(columns)
        with open(path, "rb") as handle:
            batches = parquet.ParquetFile(handle).iter_batches(
                batch_size=self._batch_size,
                columns=selected,
            )
            for batch in batches:
                yield from batch.to_pylist()

    def write_row_groups(self, path: Path, schema: Schema, row_groups: Iterable[list[Row]]) -> None:
        """Write one Parquet row group per list of rows."""
        pyarrow = import_optional("pyarrow", extra="parquet")
        parquet = import_optional("pyarrow.parquet", extra="parquet")
        fields = [(name, getattr(pyarrow, column_type.value)()) for name, column_type in schema]
        arrow_schema = pyarrow.schema(fields)
        with parquet.ParquetWriter(path, arrow_schema) as writer:
            for rows in row_groups:
                writer.write_table(pyarrow.Table.from_pylist(rows, arrow_schema))


# ---------------------------------------------------------------- dataset hubs


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    """One split of a hub dataset at an exact revision."""

    dataset: str
    config: str | None
    split: str
    revision: str | None


class DatasetHubPort(Protocol):
    """Streams rows of datasets hosted on a dataset hub."""

    def resolve_revision(self, dataset: str, revision: str | None) -> str:
        """The exact commit for `revision`; raises RevisionResolutionError if unreachable."""
        ...

    def stream_rows(self, request: DatasetRequest) -> Iterable[Row]:
        """Stream the rows of one dataset split."""
        ...


class HuggingFaceHub:
    """DatasetHubPort backed by huggingface_hub and datasets (the 'hf' extra)."""

    def resolve_revision(self, dataset: str, revision: str | None) -> str:
        """The exact commit sha for `revision` (the default branch if None)."""
        hub = import_optional("huggingface_hub", extra="hf")
        try:
            sha = hub.HfApi().dataset_info(dataset, revision=revision).sha
        except Exception as error:  # offline, unknown dataset, no access, ...
            raise RevisionResolutionError(f"{dataset}@{revision}: {error}") from error
        return str(sha)

    def stream_rows(self, request: DatasetRequest) -> Iterable[Row]:
        """Stream the rows of one dataset split without downloading it all."""
        datasets = import_optional("datasets", extra="hf")
        rows: Iterable[Row] = datasets.load_dataset(
            request.dataset,
            request.config,
            split=request.split,
            streaming=True,
            revision=request.revision,
        )
        return rows
