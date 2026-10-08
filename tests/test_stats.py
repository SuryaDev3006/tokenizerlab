from pathlib import Path

from tokenizerlab import Corpus, CorpusStats, Document, read
from tokenizerlab.data.stats import Counts


def test_counts_and_lengths() -> None:
    """Documents, characters, UTF-8 bytes and length statistics."""
    texts = ["ab", "అ", "hello"]  # "అ" is 1 character and 3 UTF-8 bytes
    stats = Corpus(read(texts, name="toy")).stats()

    assert (stats.documents, stats.chars, stats.bytes) == (3, 8, 10)
    assert (stats.length_min, stats.length_max) == (1, 5)
    assert stats.length_mean == 8 / 3


def test_by_lang_and_by_source(tmp_path: Path) -> None:
    """Untagged documents count as "und"; sources group by their first path component."""
    for name in ("te/news/a.txt", "te/blogs/b.txt", "en/c.txt"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("abc", encoding="utf-8")
    corpus = Corpus(
        read([read(tmp_path / "te", name="te", lang="te"), read(tmp_path / "en", name="en")])
    )

    stats = corpus.stats()
    assert stats.by_lang == {"te": Counts(2, 6), "und": Counts(1, 3)}
    assert stats.by_source == {"te": Counts(2, 6), "en": Counts(1, 3)}


def test_fingerprint_is_order_sensitive_and_content_only() -> None:
    """Reordering documents changes the fingerprint; changing only their sources does not."""
    texts = ["one", "two", "three"]

    def fingerprint(documents: list[Document]) -> str:
        """The fingerprint of a corpus of exactly these documents."""
        return Corpus(read(documents, name="toy")).stats().fingerprint

    original = fingerprint([Document(text, "a", i) for i, text in enumerate(texts)])
    reordered = fingerprint([Document(text, "a", i) for i, text in enumerate(reversed(texts))])
    moved = fingerprint([Document(text, "elsewhere", i + 10) for i, text in enumerate(texts)])

    assert reordered != original
    assert moved == original


def test_empty_corpus() -> None:
    """An empty corpus has zero counts, None lengths, and a valid fingerprint."""
    stats = Corpus(read([], name="empty")).stats()

    assert (stats.documents, stats.chars, stats.bytes) == (0, 0, 0)
    assert (stats.length_min, stats.length_max, stats.length_mean) == (None, None, None)
    assert (stats.by_lang, stats.by_source) == ({}, {})
    assert len(stats.fingerprint) == 32
    assert [(step.op, step.docs_out) for step in stats.steps] == [("read", 0)]


def test_same_directory_twice_is_identical(tmp_path: Path) -> None:
    """Reading a directory twice gives the same ids, order and fingerprint."""
    for name in ("b.txt", "a.txt", "sub/c.txt"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(name, encoding="utf-8")
    first, second = Corpus(read(tmp_path)), Corpus(read(tmp_path))

    assert [doc.id for doc in first] == [doc.id for doc in second]
    assert first.stats().fingerprint == second.stats().fingerprint


def test_stats_round_trip_through_dicts() -> None:
    """to_dict() output rebuilds the same CorpusStats."""
    stats = Corpus(read(["a", "bb"], name="toy")).dedup().stats()
    assert CorpusStats.from_dict(stats.to_dict()) == stats


def _section(printed: str, title: str) -> list[str]:
    """The stripped lines under one section title of a printed CorpusStats."""
    for block in printed.split("\n\n"):
        lines = block.splitlines()
        if lines[0] == title:
            return [line.strip() for line in lines[1:]]
    raise AssertionError(f"no section titled {title!r}")


TELUGU = ["తెలుగు" * 50, "భాష" * 40]
ENGLISH = ["hello world"]


def _utf8_bytes(texts: list[str]) -> int:
    """Total UTF-8 size of some texts."""
    return sum(len(text.encode("utf-8")) for text in texts)


def _two_sources() -> Corpus:
    """A small corpus: a larger Telugu source and a smaller English one."""
    telugu = read(TELUGU, name="te", lang="te")
    english = read(ENGLISH, name="en", lang="en")
    return Corpus(read([english, telugu])).dedup()


def test_print_shows_readable_tables() -> None:
    """print(stats) shows labeled sections with separators, sizes, shares and step counts."""
    text = str(_two_sources().stats())
    total_bytes = _utf8_bytes(TELUGU + ENGLISH)  # 1,271: Telugu letters take 3 bytes each
    lengths = [len(text) for text in TELUGU + ENGLISH]

    lines = text.splitlines()
    for section in ("Totals", "By language", "By source", "Steps"):
        assert section in lines  # each section title sits on its own line
    assert f"characters    {sum(lengths)}" in text
    assert f"1.3 kB ({total_bytes:,} bytes)" in text
    mean = sum(lengths) / len(lengths)
    assert f"min {min(lengths)}, mean {mean:.1f}, max {max(lengths)} characters" in text
    by_source = _section(text, "By source")
    assert [row.split()[0] for row in by_source[1:]] == ["te", "en"]  # largest first
    assert by_source[1].endswith(f"{_utf8_bytes(TELUGU) / total_bytes:.1%}")
    assert [row.split()[0] for row in _section(text, "Steps")[1:]] == ["read", "dedup"]


def test_repr_and_fields_stay_programmatic() -> None:
    """repr(), the fields and to_dict() are unchanged by the readable views."""
    stats = _two_sources().stats()
    total_bytes = _utf8_bytes(TELUGU + ENGLISH)

    assert repr(stats).startswith(f"CorpusStats(documents=3, chars=431, bytes={total_bytes},")
    assert stats.by_lang["te"] == Counts(2, _utf8_bytes(TELUGU))
    assert CorpusStats.from_dict(stats.to_dict()) == stats


def test_readable_views_handle_empty_and_huge_corpora() -> None:
    """An empty corpus shows "-" and "none"; big sizes use units plus the exact count."""
    empty = str(Corpus(read([], name="empty")).stats())
    assert "length        -" in empty
    assert _section(empty, "By language")[-1] == "none"

    huge = CorpusStats.from_dict(
        {**Corpus(read(["a"], name="toy")).stats().to_dict(), "bytes": 5_234_123_456}
    )
    assert "5.2 GB (5,234,123,456 bytes)" in str(huge)


def test_notebook_view_escapes_names_and_caps_long_tables() -> None:
    """Jupyter's HTML view escapes source names, and long tables end with a count."""
    sources = [read([f"text {i}"], name=f"source{i:02d}") for i in range(25)]
    stats = Corpus(read([read(["<b>bold</b>"], name="<b>"), *sources])).stats()

    page = stats._repr_html_()
    assert "<table" in page
    assert "&lt;b&gt;" in page
    assert "<b>bold" not in page
    assert "... and 6 more" in page
    assert "... and 6 more" in str(stats)
