"""Readers: files, directories, formats, iterables, hub datasets and read errors."""

import csv
import gzip
import json
import subprocess
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

import pytest

from tokenizerlab import ConfigurationError, Corpus, ReadError, read
from tokenizerlab.adapters import DatasetRequest, RevisionResolutionError, Row, import_optional
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
from tokenizerlab.errors import MissingDependencyError


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
    with (
        pytest.warns(UserWarning, match="skipped 2 row"),  # one warning per source, not per row
        pytest.warns(UserWarning, match=r"skipped 1 file\(s\)"),
    ):
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


def test_skipped_files_warn_once_per_pass_with_a_hint(tmp_path: Path) -> None:
    """Unsupported files give one warning naming their extensions and format='text'; they
    stay out of the errors and in `skipped`, as before."""
    repo = tmp_path / "repo"
    _write(repo, {"a.py": b"x", "b.json": b"{}", "c.md": b"#", "d.py": b"y", "e.txt": b"text"})
    stream = read(repo)

    with pytest.warns(UserWarning, match="unsupported extensions") as warned:
        assert [doc.text for doc in stream] == ["text"]
    assert [str(warning.message) for warning in warned] == [
        f"{repo}: skipped 4 file(s) with unsupported extensions (.json, .md, .py). "
        "Pass format='text' to read them as raw text, one document per file."
    ]
    assert (stream.errors, stream.skipped) == ([], ["a.py", "b.json", "c.md", "d.py"])

    with pytest.warns(UserWarning, match="unsupported extensions"):
        list(stream)  # every pass warns again


def test_skipped_files_warning_lists_at_most_five_extensions(tmp_path: Path) -> None:
    """Six different extensions are cut to the first five, then "..."."""
    _write(tmp_path, {f"file.{extension}": b"x" for extension in "abcdef"} | {"g.txt": b"g"})
    with pytest.warns(UserWarning, match=r"\(\.a, \.b, \.c, \.d, \.e, \.\.\.\)"):
        list(read(tmp_path))


def test_skipped_files_follow_on_error(tmp_path: Path) -> None:
    """on_error="skip" stays silent; on_error="raise" warns but does not raise, since a skipped
    file is not an error."""
    _write(tmp_path, {"a.py": b"x", "b.txt": b"b"})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [doc.text for doc in read(tmp_path, on_error="skip")] == ["b"]
    with pytest.warns(UserWarning, match=r"skipped 1 file\(s\)"):
        assert [doc.text for doc in read(tmp_path, on_error="raise")] == ["b"]


def test_reading_no_readable_files_warns(tmp_path: Path) -> None:
    """An empty folder, or a glob that matches nothing, warns that the stream is empty."""
    _write(tmp_path, {"data/a.txt": b"a"})
    (tmp_path / "empty").mkdir()

    with pytest.warns(UserWarning, match="empty: found no readable files"):
        assert list(read(tmp_path / "empty", name="empty")) == []
    with pytest.warns(UserWarning, match=r"no readable files matching glob '\*\.py'"):
        assert list(read(tmp_path / "data", glob="*.py")) == []


def test_missing_text_field_in_csv_or_jsonl_suggests_both_fixes(tmp_path: Path) -> None:
    """CSV and JSONL errors say to set text_field= or to read the file raw with format='text'."""
    _write(tmp_path, {"c.csv": b"id,name\r\n1,x\r\n", "d.jsonl": b'{"id": 1}\n'})
    fixes = (
        "set text_field= to the field that holds the text, "
        "or pass format='text' to read the whole file as one raw document"
    )

    with pytest.raises(ReadError) as csv_error:
        list(read(tmp_path / "c.csv", on_error="raise"))
    with pytest.raises(ReadError) as jsonl_error:
        list(read(tmp_path / "d.jsonl", on_error="raise"))

    assert (
        str(csv_error.value)
        == f"c.csv: field 'text' not found; available fields: id, name; {fixes}"
    )
    assert (
        str(jsonl_error.value) == f"d.jsonl: field 'text' not found; available fields: id; {fixes}"
    )


def test_missing_text_field_elsewhere_suggests_only_text_field(tmp_path: Path) -> None:
    """Parquet files and in-memory rows cannot be read raw, so format='text' is not offered."""
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    parquet.write_table(pyarrow.table({"body": ["a"]}), tmp_path / "p.parquet")

    for stream in (
        read(tmp_path / "p.parquet", on_error="raise"),
        read([{"body": "a"}], name="rows", on_error="raise"),
    ):
        with pytest.raises(ReadError, match="set text_field= to the field") as error:
            list(stream)
        assert "format='text'" not in str(error.value)


def test_importing_leaves_the_csv_field_limit_alone() -> None:
    """`import tokenizerlab` does not change Python's process-wide CSV field limit."""
    script = "import csv, tokenizerlab; print(csv.field_size_limit())"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=True)
    assert result.stdout.decode().strip() == "131072"


def test_csv_fields_larger_than_the_default_limit_are_read(tmp_path: Path) -> None:
    """A field over csv's default 128 KB limit is read whole, even starting from that default."""
    text = "y" * 200_000
    _write(tmp_path, {"big.csv": f"text\n{text}\n".encode()})
    original_limit = csv.field_size_limit(131072)
    try:
        assert [doc.text for doc in read(tmp_path / "big.csv", on_error="raise")] == [text]
    finally:
        csv.field_size_limit(original_limit)


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
    with pytest.raises(ConfigurationError, match="text_field must be a non-empty str"):
        read(tmp_path, text_field="")
    with pytest.raises(ConfigurationError, match="lang must be None or a non-empty str"):
        read(tmp_path, lang="")
    with pytest.raises(ConfigurationError, match="Cannot read a source of type int"):
        read(42)


def test_rows_that_are_not_objects_or_have_a_bad_lang_are_skipped() -> None:
    """In-memory rows must be mappings, and a lang_field value must be a string."""
    rows = [{"text": "a"}, 5, {"text": "b", "lang": 3}, {"text": "c", "lang": "te"}]
    stream = read(rows, name="toy", lang_field="lang", on_error="skip")

    assert [(doc.text, doc.lang) for doc in stream] == [("a", None), ("c", "te")]
    assert [str(error) for error in stream.errors] == [
        "toy#1: expected an object/row, got int",
        "toy#2: field 'lang' is int, expected str",
    ]


def test_a_source_function_must_return_an_iterable() -> None:
    """A function source that returns a non-iterable fails the pass with a ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Expected an iterable, got int"):
        list(read(lambda: 5, name="f"))


def test_missing_optional_dependency_names_the_extra() -> None:
    """A missing optional library fails with the pip command that installs it."""
    with pytest.raises(MissingDependencyError, match=r"pip install 'tokenizerlab\[parquet\]'"):
        import_optional("tokenizerlab_no_such_module", extra="parquet")


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
    assert pinned.info == {"hf:org/data/default:train": {"revision": "abc123", "pinned": True}}
    assert pinned.reproducible

    offline = reader(_OfflineHub()).open("hf:org/data", None, options)
    with pytest.warns(UserWarning, match="unpinned"):
        assert [doc.text for doc in offline] == ["one", "two"]
    assert not offline.reproducible  # the same revision cannot be promised next time

    with pytest.warns(UserWarning, match="unpinned") as caught:
        list(Corpus(offline).sample(max_bytes=10))  # a pre-pass, then the main pass
    assert len(caught) == 1
    assert not offline.reproducible


def test_only_sources_that_can_be_read_again_are_reproducible(tmp_path: Path) -> None:
    """Files can be re-read from the reader config; in-memory data cannot."""
    _write(tmp_path, {"a.txt": b"a"})

    assert read(tmp_path).reproducible
    assert not read(["a"], name="toy").reproducible
    assert not read(lambda: ["a"], name="toy").reproducible
    assert not read([read(tmp_path, name="files"), read(["a"], name="toy")]).reproducible


def test_overlapping_passes_keep_separate_reports(tmp_path: Path) -> None:
    """Two passes over one stream at the same time each see only their own errors."""
    lines = [b'{"text": "a"}', b"not json", b'{"text": "b"}', b"not json"]
    _write(tmp_path, {"d.jsonl": b"\n".join(lines) + b"\n"})
    stream = read(tmp_path / "d.jsonl", on_error="skip")

    first, second = iter(stream), iter(stream)
    interleaved = [(a.text, b.text) for a, b in zip(first, second, strict=True)]
    assert interleaved == [("a", "a"), ("b", "b")]
    assert list(first) == list(second) == []  # finish both passes
    assert [error.position for error in stream.errors] == [1, 3]
