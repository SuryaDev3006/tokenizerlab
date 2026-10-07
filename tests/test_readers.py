import gzip

import pytest

from tokenizerlab.data.readers import ReadError, read


def test_directory(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.txt").write_bytes(b"hidden")
    (tmp_path / "B.txt").write_bytes(b"\xef\xbb\xbfa\r\n\nb\rc")  # BOM, CRLF, blank line, lone \r
    (tmp_path / "a.txt").write_bytes(b"z")
    (tmp_path / "c.csv").write_bytes(b"text,k\r\nhi,1\r\n")
    (tmp_path / "notes.md").write_bytes(b"unsupported")
    with gzip.open(tmp_path / "sub" / "d.jsonl.gz", "wb") as f:
        f.write(b'{"text": "x", "lang": "te"}\n\nnot json\n{"text": 5}\n{"text": ""}\n')

    s = read(tmp_path, unit="line", lang="en", lang_field="lang", metadata_fields="all")
    with pytest.warns(UserWarning, match="skipped 2 row"):  # one warning per source, not per row
        docs = list(s)

    assert [(d.source, d.position, d.text, d.lang) for d in docs] == [
        ("B.txt", 0, "a", "en"), ("B.txt", 1, "", "en"), ("B.txt", 2, "b\rc", "en"),
        ("a.txt", 0, "z", "en"),  # codepoint order: "B" < "a" on every OS
        ("c.csv", 0, "hi", "en"),
        ("sub/d.jsonl.gz", 0, "x", "te"), ("sub/d.jsonl.gz", 4, "", "en"),
    ]
    assert docs[4].metadata == {"k": "1"}
    assert [(e.source, e.position) for e in s.errors] == [("sub/d.jsonl.gz", 2), ("sub/d.jsonl.gz", 3)]
    assert s.skipped == ["notes.md"]
    with pytest.warns(UserWarning):
        assert [d.id for d in s] == [d.id for d in docs]  # re-iterable, same identities

    with pytest.raises(ReadError, match="available fields: text, k"):
        list(read(tmp_path / "c.csv", text_field="body", on_error="raise"))
    with pytest.raises(ValueError, match="distinct name"):
        read([tmp_path, tmp_path])


def test_parquet(tmp_path) -> None:
    pa, pq = pytest.importorskip("pyarrow"), pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "p.parquet"
    pq.write_table(pa.table({"text": ["a", None, "c"], "k": [1, 2, 3], "big": ["x"] * 3}),
                   path, row_group_size=2)  # positions must run across row groups

    s = read(path, metadata_fields=["k"], on_error="skip")
    assert [(d.position, d.text, dict(d.metadata)) for d in s] == [(0, "a", {"k": 1}), (2, "c", {"k": 3})]
    assert [e.position for e in s.errors] == [1]
    with pytest.raises(ReadError, match="available fields: text, k, big"):
        list(read(path, text_field="body", on_error="raise"))


def test_iterables() -> None:
    with pytest.raises(ValueError, match="name="):
        read(iter(["a"]))
    with pytest.raises(ValueError, match="metadata_fields"):
        read(["a"], name="toy", metadata_fields="k")

    s = read((x for x in ["a", {"text": "b"}]), name="toy")
    assert [(d.source, d.position, d.text) for d in s] == [("toy", 0, "a"), ("toy", 1, "b")]
    with pytest.raises(RuntimeError, match="one-shot"):
        list(s)
