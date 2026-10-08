import gzip
import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from tokenizerlab import ConfigurationError, ReadError, read
from tokenizerlab.adapters import DatasetRequest, RevisionResolutionError, Row
from tokenizerlab.data.readers import (
    DuplicateSourceError,
    FormatRegistry,
    HubHandler,
    Reader,
    ReadOptions,
    SourceNotFoundError,
    SourceResolver,
    StreamConsumedError,
    UnsupportedFormatError,
)


def _write(root: Path, files: dict[str, bytes]) -> None:
    """Create files (and their directories) under root."""
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)


def test_directory_order_encoding_and_bad_rows(tmp_path: Path) -> None:
    """Files are read in code point order with exact text; bad rows are reported, not fatal."""
    with gzip.open(tmp_path / "d.jsonl.gz", "wb") as file:
        file.write(b'{"text": "x", "lang": "te"}\n\nnot json\n{"text": 5}\n{"text": ""}\n')
    _write(
        tmp_path,
        {
            "B.txt": b"\xef\xbb\xbfa\r\n\nb\rc",  # BOM, CRLF, blank line, lone \r
            "a.txt": b"z",
            "c.csv": b"text,k\r\nhi,1\r\n",
            "notes.md": b"unsupported",
        },
    )

    stream = read(tmp_path, unit="line", lang="en", lang_field="lang", metadata_fields="all")
    with pytest.warns(UserWarning, match="skipped 2 row"):  # one warning per source, not per row
        documents = list(stream)

    assert [(doc.source, doc.position, doc.text, doc.lang) for doc in documents] == [
        ("B.txt", 0, "a", "en"),
        ("B.txt", 1, "", "en"),
        ("B.txt", 2, "b\rc", "en"),
        ("a.txt", 0, "z", "en"),  # code point order: "B" < "a" on every OS
        ("c.csv", 0, "hi", "en"),
        ("d.jsonl.gz", 0, "x", "te"),
        ("d.jsonl.gz", 4, "", "en"),  # positions are physical line numbers
    ]
    assert documents[4].metadata == {"k": "1"}
    assert [(error.source, error.position) for error in stream.errors] == [
        ("d.jsonl.gz", 2),
        ("d.jsonl.gz", 3),
    ]
    assert stream.skipped == ["notes.md"]


def test_hidden_folders_are_never_read(tmp_path: Path) -> None:
    """.git, .venv and other hidden entries are skipped entirely."""
    _write(
        tmp_path,
        {".git/x.txt": b"hidden", ".venv/lib/y.txt": b"hidden", ".z.txt": b"hidden", "a.txt": b"a"},
    )

    stream = read(tmp_path)
    assert [doc.text for doc in stream] == ["a"]
    assert stream.skipped == []


def test_non_latin_jsonl(tmp_path: Path) -> None:
    """Telugu JSONL, ASCII-escaped or plain UTF-8, yields the real text, one document per line."""
    lines = ["నమస్కారం ప్రపంచం", "తెలుగు"]
    for ensure_ascii in (True, False):
        path = tmp_path / f"te-{ensure_ascii}.jsonl"
        rows = (json.dumps({"text": line}, ensure_ascii=ensure_ascii) for line in lines)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        assert [doc.text for doc in read(path, lang="te")] == lines


def test_broken_file_is_warned_and_skipped(tmp_path: Path) -> None:
    """One unreadable file is warned about and recorded; the other files still load."""
    _write(tmp_path, {"a.txt": b"fine", "b.jsonl.gz": b"not gzip at all", "c.txt": b"also fine"})

    stream = read(tmp_path)
    with pytest.warns(UserWarning, match=r"b\.jsonl\.gz"):
        assert [doc.text for doc in stream] == ["fine", "also fine"]
    assert [(error.source, str(error.level)) for error in stream.errors] == [("b.jsonl.gz", "file")]


def test_wrong_text_field_fails_the_file_once(tmp_path: Path) -> None:
    """A text_field missing from the first row names the fields that do exist."""
    _write(tmp_path, {"c.csv": b"text,k\r\nhi,1\r\n"})
    with pytest.raises(ReadError, match="available fields: text, k"):
        list(read(tmp_path / "c.csv", text_field="body", on_error="raise"))


def test_invalid_arguments_fail_before_reading(tmp_path: Path) -> None:
    """Bad settings, missing paths and unknown file types fail at read(), not mid-pass."""
    _write(tmp_path, {"a.txt": b"a", "notes.md": b"x"})
    with pytest.raises(ConfigurationError, match="on_error"):
        read(tmp_path, on_error="ignore")
    with pytest.raises(ConfigurationError, match="metadata_fields"):
        read(tmp_path, metadata_fields="k")
    with pytest.raises(UnsupportedFormatError):
        read(tmp_path, format="xml")
    with pytest.raises(UnsupportedFormatError):
        read(tmp_path / "notes.md")
    with pytest.raises(SourceNotFoundError):
        read(tmp_path / "missing")
    with pytest.raises(DuplicateSourceError):
        read([tmp_path, tmp_path])


def test_parquet(tmp_path: Path) -> None:
    """Parquet positions run across row groups; bad rows are recorded."""
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "p.parquet"
    table = pyarrow.table({"text": ["a", None, "c"], "k": [1, 2, 3], "big": ["x"] * 3})
    parquet.write_table(table, path, row_group_size=2)

    stream = read(path, metadata_fields=["k"], on_error="skip")
    assert [(doc.position, doc.text, dict(doc.metadata)) for doc in stream] == [
        (0, "a", {"k": 1}),
        (2, "c", {"k": 3}),
    ]
    assert [error.position for error in stream.errors] == [1]


def test_iterables() -> None:
    """Iterables need a name; lists re-iterate; a generator can be read only once."""
    with pytest.raises(ConfigurationError, match="name="):
        read(iter(["a"]))

    repeatable = read(["a", {"text": "b"}], name="toy")
    assert [doc.text for doc in repeatable] == [doc.text for doc in repeatable] == ["a", "b"]

    one_shot = read((item for item in ["a"]), name="once")
    assert [doc.text for doc in one_shot] == ["a"]
    with pytest.raises(StreamConsumedError):
        list(one_shot)


def test_config_records_read_arguments(tmp_path: Path) -> None:
    """read() keeps its arguments as JSON; a chained read keeps one config per source."""
    _write(tmp_path, {"a.txt": b"a"})
    single = read(tmp_path, name="te", lang="te", metadata_fields=("k",))
    chained = read([single, read(["x"], name="toy")])

    assert isinstance(single.config, dict)
    assert json.loads(json.dumps(single.config))["metadata_fields"] == ["k"]
    assert isinstance(chained.config, list)
    assert [child["name"] for child in chained.config if isinstance(child, dict)] == ["te", "toy"]


class _OfflineHub:
    """A dataset hub that serves rows but cannot pin revisions."""

    def resolve_revision(self, dataset: str, revision: str | None) -> str:
        """Always unreachable."""
        raise RevisionResolutionError(f"{dataset}: offline")

    def stream_rows(self, request: DatasetRequest) -> Iterable[Row]:
        """Two fixed rows."""
        return [{"text": "one"}, {"text": "two"}]


class _PinnedHub(_OfflineHub):
    """A dataset hub that pins every revision to the same commit."""

    def resolve_revision(self, dataset: str, revision: str | None) -> str:
        """Always the same commit."""
        return "abc123"


def test_hub_datasets_are_pinned_or_warned() -> None:
    """Hub reads record the resolved revision, and warn (not fail) when it cannot be pinned."""
    options = ReadOptions()

    def reader(hub: _OfflineHub) -> Reader:
        """A reader that only knows hub sources."""
        return Reader(SourceResolver([HubHandler(hub)]), FormatRegistry())

    pinned = reader(_PinnedHub()).open("hf:org/data", None, options)
    assert [doc.source for doc in pinned] == ["hf:org/data/default:train"] * 2
    assert pinned.info == {"hf:org/data/default:train": {"revision": "abc123"}}

    offline = reader(_OfflineHub()).open("hf:org/data", None, options)
    with pytest.warns(UserWarning, match="unpinned"):
        assert [doc.text for doc in offline] == ["one", "two"]
