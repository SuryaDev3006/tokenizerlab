"""
Readers: turn external sources into lazy, deterministic, re-iterable streams of Documents.

    read("data/")                                     # directory, formats auto-detected
    read("data/te/news.jsonl", lang="te")             # single file
    read("hf:ai4bharat/sangraha", split="train")      # Hugging Face dataset
    read(["hello", "world"], name="toy")              # Python iterable (name required)
    read([read("data/te", name="te", lang="te"),      # several sources, in this order
    read("data/en", name="en", lang="en")])

Every pass re-reads the source and starts a fresh `errors` list, which is
complete once the pass has been fully consumed.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .document import Document

__all__ = ["read", "DocumentStream", "ReadError"]

OnError = Literal["warn", "skip", "raise"]

_EXTENSIONS = {
    ".txt.gz": "text", ".jsonl.gz": "jsonl", ".ndjson.gz": "jsonl",
    ".txt": "text", ".jsonl": "jsonl", ".ndjson": "jsonl",
    ".csv": "csv", ".tsv": "csv", ".parquet": "parquet",
}

# Default CSV field limit is 128 KB; documents are often larger.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class ReadError(Exception):
    """A failed row (level="row") or a failed file/source (level="file")."""

    def __init__(self, source: str, message: str, position: int | None = None, level: str = "row"):
        """Format the message as "source#position: message" and keep each part as an attribute."""
        where = source if position is None else f"{source}#{position}"
        super().__init__(f"{where}: {message}")
        self.source, self.message, self.position, self.level = source, message, position, level


class DocumentStream:
    """A re-iterable stream of Documents. Iterate it as many times as you like
    (unless it wraps a one-shot iterator); each pass resets `errors`, `skipped` and `info`."""

    def __init__(
        self, 
        run: Callable[[DocumentStream], Iterator[Document]],
        sources: Callable[[], list[str]], 
        on_error: OnError, 
        reiterable: bool = True
    ):
        """Wrap `run`, which yields one pass of Documents, and `sources`, which lists the source names."""
        self._run, self._sources = run, sources
        self.on_error, self.reiterable = on_error, reiterable
        self.errors: list[ReadError] = []
        self.skipped: list[str] = []          # unsupported files found during discovery
        self.info: dict[str, dict[str, Any]] = {}  # per-source facts, e.g. resolved HF revision
        self._passes = 0

    def __iter__(self) -> Iterator[Document]:
        """Start a fresh pass with empty errors/skipped/info; raise if a one-shot stream is reused."""
        if self._passes and not self.reiterable:
            raise RuntimeError(
                "This stream wraps a one-shot iterator that was already consumed. Pass a list, "
                "or a function that returns a fresh iterator, to allow multiple passes."
            )
        self._passes += 1
        self.errors, self.skipped, self.info = [], [], {}
        return self._run(self)

    def _record(self, err: ReadError) -> None:
        """Apply on_error to one error: raise it, or store it (warning at once if a whole file failed)."""
        if self.on_error == "raise":
            raise err
        self.errors.append(err)
        if err.level == "file" and self.on_error == "warn":
            warnings.warn(str(err), stacklevel=3)


def read(
    source: Any,
    *,
    name: str | None = None,
    format: str | None = None,
    text_field: str = "text",
    metadata_fields: Sequence[str] | Literal["all"] | None = None,
    lang: str | None = None,
    lang_field: str | None = None,
    unit: Literal["file", "line"] = "file",
    glob: str | None = None,
    delimiter: str | None = None,
    split: str = "train",
    config: str | None = None,
    revision: str | None = None,
    on_error: OnError = "warn",
) -> DocumentStream:
    """Read a path, directory, "hf:<dataset>", iterable, or a list of sources."""
    if on_error not in ("warn", "skip", "raise"):
        raise ValueError(f"on_error must be 'warn', 'skip' or 'raise', got {on_error!r}")
    if unit not in ("file", "line"):
        raise ValueError(f"unit must be 'file' or 'line', got {unit!r}")
    if format is not None and format not in _PARSERS:
        raise ValueError(f"format must be one of {sorted(_PARSERS)}, got {format!r}")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty str")
    if isinstance(metadata_fields, str) and metadata_fields != "all":  # "src" would iterate as "s","r","c"
        raise ValueError(f"metadata_fields must be a list of names or 'all', got {metadata_fields!r}")
    if lang is not None and (not isinstance(lang, str) or not lang):  # else every file fails at Document()
        raise ValueError("lang must be None or a non-empty str")

    opts = dict(text_field=text_field, metadata_fields=metadata_fields, lang=lang,
                lang_field=lang_field, unit=unit, delimiter=delimiter)

    if isinstance(source, DocumentStream):
        return source
    # A list without name= means "several sources"; with name= it is data.
    if isinstance(source, (list, tuple)) and name is None:
        kw = dict(format=format, glob=glob, split=split, config=config,
                  revision=revision, on_error=on_error, **opts)
        return _chain([read(s, **kw) for s in source], on_error)
    if isinstance(source, str) and source.startswith("hf:"):
        return _hf_stream(source[3:], config, split, revision, opts, on_error)
    if isinstance(source, (str, os.PathLike)):
        return _path_stream(Path(source), name, format, glob, opts, on_error)
    if name is None:
        raise ValueError("Iterable sources need name=..., so document ids from different iterables can't collide")
    return _iterable_stream(source, name, opts, on_error)


# ---------------------------------------------------------------- source kinds

def _path_stream(path: Path, name, fmt, glob, opts, on_error) -> DocumentStream:
    """Stream a file or directory; sources are paths relative to it, prefixed with `name/` if given."""
    if path.is_dir():
        discover = lambda: _discover(path, glob, fmt)
    elif path.is_file():
        f = fmt or _detect(path.name)
        if f is None:
            raise ValueError(f"Unknown file type: {path.name!r}. Pass format= one of {sorted(_PARSERS)}.")
        discover = lambda: ([(path.name, path, f)], [])
    else:
        raise FileNotFoundError(path)
    prefix = f"{name}/" if name else ""

    def run(stream: DocumentStream) -> Iterator[Document]:
        """Re-discover the files and parse each one in sorted order."""
        files, skipped = discover()
        stream.skipped = [prefix + s for s in skipped]
        for rel, p, f in files:
            src = prefix + rel
            yield from _guarded(stream, src, _PARSERS[f](p, src, opts, stream))

    return DocumentStream(run, lambda: [prefix + rel for rel, _, _ in discover()[0]], on_error)


def _hf_stream(dataset: str, config, split, revision, opts, on_error) -> DocumentStream:
    """Stream a Hugging Face dataset split under the source name "hf:{dataset}/{config}:{split}"."""
    src = f"hf:{dataset}/{config or 'default'}:{split}"

    def rows(stream: DocumentStream):
        """Yield (index, row) pairs from the pinned revision, recording that revision in `stream.info`."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Reading Hugging Face datasets needs: pip install 'tokenizerlab[hf]'") from None
        resolved = _resolve_revision(dataset, revision)
        stream.info[src] = {"revision": resolved}
        ds = load_dataset(dataset, config, split=split, streaming=True, revision=resolved)
        yield from enumerate(ds)

    def run(stream: DocumentStream) -> Iterator[Document]:
        """Turn the dataset rows into Documents."""
        yield from _guarded(stream, src, _docs_from_rows(rows(stream), src, opts, stream))

    return DocumentStream(run, lambda: [src], on_error)


def _iterable_stream(obj, name: str, opts, on_error) -> DocumentStream:
    """Stream a Python iterable as source `name`. A function is called on every pass (re-iterable);
    a list re-iterates; an iterator/generator is one-shot."""
    reiterable = callable(obj) or iter(obj) is not obj

    def run(stream: DocumentStream) -> Iterator[Document]:
        """Turn the items of one pass into Documents."""
        items = obj() if callable(obj) else obj
        yield from _guarded(stream, name, _docs_from_rows(enumerate(items), name, opts, stream, raw=True))

    return DocumentStream(run, lambda: [name], on_error, reiterable)


def _chain(streams: list[DocumentStream], on_error) -> DocumentStream:
    """Concatenate streams in the given order; raise if two would produce the same source name."""
    names = [s for st in streams for s in st._sources()]
    dupes = sorted(s for s, n in Counter(names).items() if n > 1)
    if dupes:
        raise ValueError(f"Sources would share names {dupes[:5]}; give each source a distinct name=...")

    def run(stream: DocumentStream) -> Iterator[Document]:
        """Yield each stream in turn, then collect its errors, skipped files and info."""
        for st in streams:
            yield from st
            stream.errors += st.errors
            stream.skipped += st.skipped
            stream.info.update(st.info)

    return DocumentStream(run, lambda: names, on_error, all(st.reiterable for st in streams))


# ---------------------------------------------------------------- shared helpers

def _detect(filename: str) -> str | None:
    """Return the format for a filename by extension, or None if unsupported."""
    low = filename.lower()
    return next((f for ext, f in _EXTENSIONS.items() if low.endswith(ext)), None)


def _discover(root: Path, glob, fmt) -> tuple[list[tuple[str, Path, str]], list[str]]:
    """Walk root, skipping hidden entries; return sorted (rel, path, format) files and sorted unsupported paths."""
    found, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(root):  # does not follow directory symlinks
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = Path(dirpath, fn)
            rel = p.relative_to(root).as_posix()
            if glob and not PurePosixPath(rel).match(glob):
                continue
            f = fmt or _detect(fn)
            if f:
                found.append((rel, p, f))
            else:
                skipped.append(rel)
    found.sort(key=lambda t: t[0])  # plain codepoint order: identical on every OS
    return found, sorted(skipped)


def _guarded(stream: DocumentStream, src: str, docs: Iterator[Document]) -> Iterator[Document]:
    """Turn file-level failures into recorded errors and emit one warning per source for bad rows."""
    start = len(stream.errors)
    try:
        yield from docs
    except ImportError:
        raise
    except ReadError as e:
        if e.level != "file":  # only reachable with on_error="raise"
            raise
        stream._record(e)
    except Exception as e:  # unreadable file, bad encoding, corrupt gzip/parquet, ...
        stream._record(ReadError(src, f"{type(e).__name__}: {e}", level="file"))
    bad = [e for e in stream.errors[start:] if e.level == "row"]
    if bad and stream.on_error == "warn":
        warnings.warn(f"{src}: skipped {len(bad)} row(s); first: {bad[0]}", stacklevel=2)


def _docs_from_rows(
    rows: Iterable[tuple[int, Any]], 
    src: str, o: dict, 
    stream: DocumentStream,
    raw: bool = False
    ) -> Iterator[Document]:
    """Numbered rows -> Documents. With raw=True, strings and Documents pass through (iterables only)."""
    checked = False
    for pos, row in rows:
        if raw and isinstance(row, Document):
            yield row
            continue
        if raw and isinstance(row, str):
            yield Document(text=row, source=src, position=pos, lang=o["lang"])
            continue
        if not checked and isinstance(row, Mapping):
            if o["text_field"] not in row:  # likely a wrong text_field: fail once, not once per row
                fields = ", ".join(map(str, row)) or "(none)"
                raise ReadError(src, f"field {o['text_field']!r} not found; available fields: {fields}", level="file")
            checked = True
        try:
            doc = _row_to_doc(row, src, pos, o)
        except ReadError as e:
            stream._record(e)
            continue
        yield doc


def _row_to_doc(row: Any, src: str, pos: int, o: dict) -> Document:
    """Build a Document from one row; raise a row-level ReadError for a bad text or lang field."""
    if not isinstance(row, Mapping):
        raise ReadError(src, f"expected an object/row, got {type(row).__name__}", pos)
    tf = o["text_field"]
    text = row.get(tf)
    if text is None:
        raise ReadError(src, f"missing or null field {tf!r}", pos)
    if not isinstance(text, str):
        raise ReadError(src, f"field {tf!r} is {type(text).__name__}, expected str", pos)

    lang = o["lang"]
    if o["lang_field"]:
        value = row.get(o["lang_field"])
        if isinstance(value, str) and value:
            lang = value
        elif value not in (None, ""):
            raise ReadError(src, f"field {o['lang_field']!r} is {type(value).__name__}, expected str", pos)

    mf = o["metadata_fields"]
    if mf == "all":
        meta = {k: v for k, v in row.items() if k != tf and k is not None}
    elif mf:
        meta = {k: row[k] for k in mf if k in row and k != tf}
    else:
        meta = {}
    return Document(text=text, source=src, position=pos, lang=lang, metadata=meta)


def _resolve_revision(dataset: str, revision: str | None) -> str | None:
    """Pin to the exact commit so 'the same dataset' stays the same data."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(dataset, revision=revision).sha
    except Exception:
        return revision


# ---------------------------------------------------------------- file parsers

def _open(path: Path, newline: str = "\n"):
    """Open a plain or gzipped file as UTF-8 text without newline translation."""
    # newline="\n": lines split only on "\n" and nothing is translated.
    # utf-8-sig: strips a leading BOM, otherwise plain UTF-8 (strict).
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", newline=newline)


def _parse_text(path: Path, src: str, o: dict, stream: DocumentStream) -> Iterator[Document]:
    """Yield the whole file as one Document, or one Document per physical line with unit="line"."""
    with _open(path) as f:
        if o["unit"] == "file":
            yield Document(text=f.read(), source=src, position=0, lang=o["lang"])
            return
        for i, line in enumerate(f):
            if line.endswith("\n"):
                line = line[:-1].removesuffix("\r")
            yield Document(text=line, source=src, position=i, lang=o["lang"])


def _parse_jsonl(path: Path, src: str, o: dict, stream: DocumentStream) -> Iterator[Document]:
    """Yield one Document per JSON line; blank lines are skipped but still count toward positions."""
    def rows():
        """Yield (line number, parsed object) pairs, recording invalid JSON as row errors."""
        with _open(path) as f:
            for i, line in enumerate(f):  # i = physical line number
                if not line.strip():
                    continue
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError as e:
                    stream._record(ReadError(src, f"invalid JSON: {e.msg}", i))
    yield from _docs_from_rows(rows(), src, o, stream)


def _parse_csv(path: Path, src: str, o: dict, stream: DocumentStream) -> Iterator[Document]:
    """Yield one Document per row after the header, split on `delimiter` (tab for .tsv, else comma)."""
    delimiter = o["delimiter"] or ("\t" if path.name.lower().endswith(".tsv") else ",")
    with _open(path, newline="") as f:  # the csv module needs newline=""
        yield from _docs_from_rows(enumerate(csv.DictReader(f, delimiter=delimiter)), src, o, stream)


def _parse_parquet(path: Path, src: str, o: dict, stream: DocumentStream) -> Iterator[Document]:
    """Yield one Document per row, reading only the needed columns in batches of 1024 rows."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Reading Parquet needs: pip install 'tokenizerlab[parquet]'") from None
    with open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        names = pf.schema_arrow.names
        if o["text_field"] not in names:
            raise ReadError(src, f"field {o['text_field']!r} not found; available fields: {', '.join(names)}",
                            level="file")
        mf = o["metadata_fields"]
        wanted = [o["text_field"], o["lang_field"], *([] if mf in (None, "all") else mf)]
        columns = None if mf == "all" else [c for c in dict.fromkeys(wanted) if c and c in names]
        rows = (row for b in pf.iter_batches(batch_size=1024, columns=columns) for row in b.to_pylist())
        yield from _docs_from_rows(enumerate(rows), src, o, stream)


_PARSERS: dict[str, Callable[..., Iterator[Document]]] = {
    "text": _parse_text, "jsonl": _parse_jsonl, "csv": _parse_csv, "parquet": _parse_parquet,
}