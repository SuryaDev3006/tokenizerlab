"""read(): the entry point, and the Reader it delegates to.

read("data/")                                     # directory, formats auto-detected
read("data/te/news.jsonl", lang="te")             # single file
read("hf:ai4bharat/sangraha", split="train")      # Hugging Face dataset
read(["hello", "world"], name="toy")              # Python iterable (name required)
read([read("data/te", name="te", lang="te"),      # several sources, in this order
      read("data/en", name="en", lang="en")])
"""

from __future__ import annotations

import functools
import os
from collections import Counter
from collections.abc import Iterator, Sequence

from tokenizerlab.adapters import HuggingFaceHub, PyArrowParquet
from tokenizerlab.data.document import Document
from tokenizerlab.data.readers.formats import FormatRegistry, default_formats
from tokenizerlab.data.readers.options import (
    OnError,
    ReadOptions,
    Unit,
    parse_choice,
    parse_metadata_fields,
)
from tokenizerlab.data.readers.sources import (
    FileDiscovery,
    HubHandler,
    IterableHandler,
    PathHandler,
    SourceResolver,
)
from tokenizerlab.data.readers.stream import DocumentStream, PassReport, SourceTraits
from tokenizerlab.errors import ConfigurationError


class DuplicateSourceError(ConfigurationError):
    """Two sources read together would produce the same source names, and so the same ids."""


class Reader:
    """Opens sources with injected handlers, and records what was read in `stream.config`."""

    def __init__(self, resolver: SourceResolver, formats: FormatRegistry) -> None:
        """Open sources with `resolver`; check format= against `formats`."""
        self._resolver = resolver
        self._formats = formats

    def open(self, source: object, name: str | None, options: ReadOptions) -> DocumentStream:
        """A stream over a source, or over a list of sources read in order."""
        if isinstance(source, DocumentStream):
            return source
        if options.format is not None:
            self._formats.require(options.format)

        # A list without name= means "several sources"; with name= it is data.
        if isinstance(source, (list, tuple)) and name is None:
            return self._open_several(list(source), options)

        stream = self._resolver.open(source, name, options)
        stream.config = {"source": _describe(source), "name": name, **options.to_config()}
        return stream

    def _open_several(self, sources: list[object], options: ReadOptions) -> DocumentStream:
        """Open each source with the same options and chain them in order."""
        children = [self.open(child, None, options) for child in sources]
        stream = chain(children, options.on_error)
        stream.config = [child.config for child in children]
        return stream


def _describe(source: object) -> str:
    """A JSON-friendly description of a source: its path, or the type of iterable."""
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    return type(source).__name__


def chain(streams: list[DocumentStream], on_error: OnError) -> DocumentStream:
    """Concatenate streams in the given order; raise if two would produce the same source name."""
    names = [name for stream in streams for name in stream.source_names()]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise DuplicateSourceError(
            f"Sources would share names {duplicates[:5]}; give each source a distinct name=..."
        )

    def run(report: PassReport) -> Iterator[Document]:
        """Yield each stream in turn, then add its pass report to this one."""
        for stream in streams:
            yield from stream.one_pass(report.shown_warnings)
            report.absorb(stream.last_pass)

    traits = SourceTraits.combine(stream.traits for stream in streams)
    return DocumentStream(run, lambda: names, on_error, traits)


@functools.cache
def default_reader() -> Reader:
    """Composition root: the Reader behind read(), wired with the real adapters on first use."""
    formats = default_formats(PyArrowParquet())
    resolver = SourceResolver(
        [
            HubHandler(HuggingFaceHub()),  # before PathHandler: "hf:..." is also a str
            PathHandler(FileDiscovery(formats), formats),
            IterableHandler(),
        ]
    )
    return Reader(resolver, formats)


def read(  # noqa: PLR0913 - the documented keyword API; parsed into ReadOptions at once
    source: object,
    *,
    name: str | None = None,
    format: str | None = None,
    text_field: str = "text",
    metadata_fields: Sequence[str] | str | None = None,
    lang: str | None = None,
    lang_field: str | None = None,
    unit: str = "file",
    glob: str | None = None,
    delimiter: str | None = None,
    split: str = "train",
    config: str | None = None,
    revision: str | None = None,
    on_error: str = "warn",
) -> DocumentStream:
    """Read a path, directory, "hf:<dataset>", iterable, or a list of sources."""
    options = ReadOptions(
        format=format,
        text_field=text_field,
        metadata_fields=parse_metadata_fields(metadata_fields),
        lang=lang,
        lang_field=lang_field,
        unit=parse_choice(Unit, unit, "unit"),
        glob=glob,
        delimiter=delimiter,
        split=split,
        config=config,
        revision=revision,
        on_error=parse_choice(OnError, on_error, "on_error"),
    )
    return default_reader().open(source, name, options)
