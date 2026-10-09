"""Source handlers: one per kind of source read() understands, chosen through a registry."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from tokenizerlab.adapters import DatasetHubPort, DatasetRequest, RevisionResolutionError
from tokenizerlab.data.document import Document
from tokenizerlab.data.readers.formats import FormatRegistry
from tokenizerlab.data.readers.options import ReadOptions
from tokenizerlab.data.readers.rows import Source
from tokenizerlab.data.readers.stream import DocumentStream, PassReport, Reuse, SourceTraits
from tokenizerlab.errors import ConfigurationError
from tokenizerlab.shared.source_names import HUB_PREFIX, hub_source_name, is_hub_source

MAX_LISTED_EXTENSIONS = 5  # a skipped-files warning names this many extensions at most


class SourceNotFoundError(ConfigurationError):
    """A path given to read() does not exist."""


class SourceHandler(Protocol):
    """Opens one kind of source (paths, hub datasets, iterables, ...) as a DocumentStream."""

    def accepts(self, source: object) -> bool:
        """Whether this handler reads this kind of source."""
        ...

    def open(self, source: object, name: str | None, options: ReadOptions) -> DocumentStream:
        """A stream over the source."""
        ...


class SourceResolver:
    """A registry of handlers; each source goes to the first one that accepts it."""

    def __init__(self, handlers: Sequence[SourceHandler]) -> None:
        """Try `handlers` in order: more specific handlers first."""
        self._handlers = tuple(handlers)

    def open(self, source: object, name: str | None, options: ReadOptions) -> DocumentStream:
        """A stream over the source from the first handler that accepts it."""
        for handler in self._handlers:
            if handler.accepts(source):
                return handler.open(source, name, options)
        raise ConfigurationError(f"Cannot read a source of type {type(source).__name__}")


# ---------------------------------------------------------------- files and directories


@dataclass(frozen=True, slots=True)
class File:
    """A readable file: its path relative to the read root, its real path, and its format."""

    relative_path: str
    path: Path
    format: str


@dataclass(frozen=True, slots=True)
class Discovery:
    """Readable files in reading order, and the relative paths of unsupported files."""

    files: tuple[File, ...]
    skipped: tuple[str, ...] = ()


class FileDiscovery:
    """Finds readable files; the format registry decides what is readable."""

    def __init__(self, formats: FormatRegistry) -> None:
        """Recognize files with `formats`."""
        self._formats = formats

    def discover(self, path: Path, options: ReadOptions) -> Discovery:
        """The files at path: a whole directory, or one file."""
        if path.is_dir():
            return self._discover_directory(path, options)
        if not path.is_file():
            raise SourceNotFoundError(f"No such file or directory: {path}")
        return Discovery((File(path.name, path, self._format_of(path.name, options)),))

    def _discover_directory(self, root: Path, options: ReadOptions) -> Discovery:
        """Readable files under root matching glob, sorted, plus the sorted unsupported ones."""
        files: list[File] = []
        skipped: list[str] = []
        for path in _visible_files(root):
            relative_path = path.relative_to(root).as_posix()
            if options.glob and not PurePosixPath(relative_path).match(options.glob):
                continue
            if options.format is None and not self._formats.supports(path.name):
                skipped.append(relative_path)
                continue
            files.append(File(relative_path, path, self._format_of(path.name, options)))

        # Plain code point order, not locale or filesystem order, so it is identical on every OS.
        files.sort(key=lambda file: file.relative_path)
        return Discovery(tuple(files), tuple(sorted(skipped)))

    def _format_of(self, filename: str, options: ReadOptions) -> str:
        """The format= setting if given, else the format detected from the file name."""
        if options.format is not None:
            return options.format
        return self._formats.detect(filename)


def _visible_files(root: Path) -> Iterator[Path]:
    """Every file under root, skipping hidden files and directories (names starting with ".")."""
    for directory, subdirectories, filenames in os.walk(root):  # does not follow directory symlinks
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
        for filename in filenames:
            if not filename.startswith("."):
                yield Path(directory, filename)


class PathHandler:
    """Reads paths; source names are paths relative to the read root, prefixed with `name/`."""

    def __init__(self, discovery: FileDiscovery, formats: FormatRegistry) -> None:
        """Find files with `discovery` and parse them with the parsers in `formats`."""
        self._discovery = discovery
        self._formats = formats

    def accepts(self, source: object) -> bool:
        """Strings and path-like objects."""
        return isinstance(source, (str, os.PathLike))

    def open(self, source: object, name: str | None, options: ReadOptions) -> DocumentStream:
        """A stream that re-discovers and parses the files on every pass."""
        if not isinstance(source, (str, os.PathLike)):
            raise ConfigurationError(f"Not a path: {source!r}")
        path = Path(source)
        if not path.is_dir():
            self._discovery.discover(path, options)  # fail now on a missing path or unknown type
        prefix = f"{name}/" if name else ""

        def run(report: PassReport) -> Iterator[Document]:
            """Parse each discovered file in order, after noting skipped files or none found."""
            discovery = self._discovery.discover(path, options)
            report.skipped = [prefix + relative_path for relative_path in discovery.skipped]
            for notice in _discovery_notices(name or str(path), discovery, options.glob):
                report.notice(notice)
            for file in discovery.files:
                file_source = Source(prefix + file.relative_path, options, report)
                parser = self._formats.parser(file.format)
                yield from file_source.guard(parser(file.path, file_source))

        def list_source_names() -> list[str]:
            """The prefixed relative path of every readable file."""
            discovery = self._discovery.discover(path, options)
            return [prefix + file.relative_path for file in discovery.files]

        traits = SourceTraits(paths=(path,))
        return DocumentStream(run, list_source_names, options.on_error, traits)


def _discovery_notices(label: str, discovery: Discovery, glob: str | None) -> list[str]:
    """What a read should say about files it skipped, or about finding nothing to read.

    Skipped files are not errors: they stay out of the error list and the manifest, and the
    error policy decides only whether the user hears about them.
    """
    notices = []
    if discovery.skipped:
        notices.append(_skipped_files_notice(label, discovery.skipped))
    if not discovery.files:
        matching = f" matching glob {glob!r}" if glob else ""
        notices.append(f"{label}: found no readable files{matching}; the stream is empty.")
    return notices


def _skipped_files_notice(label: str, skipped: Sequence[str]) -> str:
    """How many files were skipped, their extensions, and how to read them anyway."""
    extensions = sorted({_extension(relative_path) for relative_path in skipped})
    listed = ", ".join(extensions[:MAX_LISTED_EXTENSIONS])
    if len(extensions) > MAX_LISTED_EXTENSIONS:
        listed += ", ..."
    return (
        f"{label}: skipped {len(skipped)} file(s) with unsupported extensions ({listed}). "
        "Pass format='text' to read them as raw text, one document per file."
    )


def _extension(relative_path: str) -> str:
    """A file's last extension in lower case, e.g. ".py"; "(none)" for a name without one."""
    return PurePosixPath(relative_path).suffix.lower() or "(none)"


# ---------------------------------------------------------------- dataset hubs


class HubHandler:
    """Reads "hf:<dataset>" as one split, source name "hf:{dataset}/{config}:{split}"."""

    def __init__(self, hub: DatasetHubPort) -> None:
        """Stream rows through `hub`."""
        self._hub = hub

    def accepts(self, source: object) -> bool:
        """Strings starting with "hf:"."""
        return is_hub_source(source)

    def open(self, source: object, name: str | None, options: ReadOptions) -> DocumentStream:
        """A stream that re-pins the revision and re-reads the split on every pass."""
        dataset = str(source).removeprefix(HUB_PREFIX)
        source_name = hub_source_name(dataset, options.config, options.split)

        def run(report: PassReport) -> Iterator[Document]:
            """Turn the dataset rows into Documents."""
            hub_source = Source(source_name, options, report)
            request = self._request(dataset, hub_source)
            rows = enumerate(self._hub.stream_rows(request))
            yield from hub_source.guard(hub_source.documents_from_rows(rows))

        return DocumentStream(run, lambda: [source_name], options.on_error)

    def _request(self, dataset: str, source: Source) -> DatasetRequest:
        """The split at its exact commit, so 'the same dataset' stays the same data.

        Records the revision in the pass report; an unpinned read is warned about and marked,
        because the manifest cannot promise the same data next time.
        """
        options, report, source_name = source.options, source.report, source.name
        try:
            revision: str | None = self._hub.resolve_revision(dataset, options.revision)
        except RevisionResolutionError as error:
            # Warned whatever on_error says: an unpinned read weakens what the manifest promises.
            report.warn(
                f"Could not pin {dataset} to a commit ({error}); "
                f"reading revision {options.revision!r} unpinned"
            )
            revision = options.revision
            report.unpinned.append(source_name)
        report.info[source_name] = {
            "revision": revision,
            "pinned": source_name not in report.unpinned,
        }
        return DatasetRequest(dataset, options.config, options.split, revision)


# ---------------------------------------------------------------- Python iterables


class IterableHandler:
    """Reads an iterable as source `name`. A function is called on every pass and a list
    re-iterates; an iterator or generator can be read only once."""

    def accepts(self, source: object) -> bool:
        """Iterables, and functions that return one."""
        return callable(source) or isinstance(source, Iterable)

    def open(self, source: object, name: str | None, options: ReadOptions) -> DocumentStream:
        """A stream over the iterable's items."""
        if name is None:
            raise ConfigurationError(
                "Iterable sources need name=..., so document ids from different iterables "
                "can't collide"
            )
        source_name = name

        def run(report: PassReport) -> Iterator[Document]:
            """Turn the items of one pass into Documents."""
            item_source = Source(source_name, options, report)
            items = _items_of_one_pass(source)
            yield from item_source.guard(item_source.documents_from_items(items))

        # In-memory data is not recorded anywhere, so the manifest cannot re-read it.
        traits = SourceTraits(
            reuse=Reuse.ONE_SHOT if isinstance(source, Iterator) else Reuse.REPEATABLE,
            reproducible=False,
        )
        return DocumentStream(run, lambda: [source_name], options.on_error, traits)


def _items_of_one_pass(source: object) -> Iterable[Any]:
    """The source itself, or what the source function returns for this pass."""
    items = source() if callable(source) else source
    if not isinstance(items, Iterable):
        raise ConfigurationError(f"Expected an iterable, got {type(items).__name__}")
    return items
