"""Readers: turn files, directories, Hugging Face datasets and Python iterables into lazy,
deterministic, re-iterable streams of Documents."""

from tokenizerlab.data.readers.formats import FormatRegistry, Parser, UnsupportedFormatError
from tokenizerlab.data.readers.options import OnError, ReaderConfig, ReadOptions, Unit
from tokenizerlab.data.readers.reader import DuplicateSourceError, Reader, read
from tokenizerlab.data.readers.sources import (
    HubHandler,
    IterableHandler,
    PathHandler,
    SourceHandler,
    SourceNotFoundError,
    SourceResolver,
)
from tokenizerlab.data.readers.stream import (
    DocumentStream,
    ErrorLevel,
    ErrorPolicy,
    ReadError,
    Reuse,
    StreamConsumedError,
)

__all__ = [
    "DocumentStream",
    "DuplicateSourceError",
    "ErrorLevel",
    "ErrorPolicy",
    "FormatRegistry",
    "HubHandler",
    "IterableHandler",
    "OnError",
    "Parser",
    "PathHandler",
    "ReadError",
    "ReadOptions",
    "Reader",
    "ReaderConfig",
    "Reuse",
    "SourceHandler",
    "SourceNotFoundError",
    "SourceResolver",
    "StreamConsumedError",
    "Unit",
    "UnsupportedFormatError",
    "read",
]
