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
