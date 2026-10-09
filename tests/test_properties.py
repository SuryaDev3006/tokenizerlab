"""Property-based tests: rules that must hold for any input, not just hand-picked examples."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tokenizerlab import Corpus, Document, read

texts = st.lists(st.text(max_size=20), max_size=40)
seeds = st.integers(min_value=0, max_value=2**32)


def _corpus(items: list[str]) -> Corpus:
    """A corpus over the given texts, in order."""
    return Corpus(read(items, name="toy"))


@given(st.text(), st.text(min_size=1), st.integers(min_value=0), st.text())
def test_identities_depend_only_on_what_they_identify(
    text: str,
    source: str,
    position: int,
    other_text: str,
) -> None:
    """id ignores the text; content_hash ignores the location."""
    document = Document(text, source, position)
    assert document.id == Document(other_text, source, position).id
    assert document.content_hash == Document(text, source + "x", position + 1).content_hash
    assert (document.content_hash == Document(other_text, source, 0).content_hash) == (
        text == other_text
    )


@given(texts)
def test_dedup_keeps_the_first_occurrence_of_each_text(items: list[str]) -> None:
    """dedup yields each distinct text once, at its first position, in order."""
    assert [doc.text for doc in _corpus(items).dedup()] == list(dict.fromkeys(items))


@settings(max_examples=25, deadline=None)
@given(texts)
def test_on_disk_dedup_keeps_the_same_documents(items: list[str]) -> None:
    """dedup(on_disk=True) yields exactly the documents of the in-memory dedup()."""
    corpus = _corpus(items)
    assert list(corpus.dedup(on_disk=True)) == list(corpus.dedup())


@given(texts, st.integers(min_value=0, max_value=50))
def test_filter_only_removes_and_never_changes_text(items: list[str], min_chars: int) -> None:
    """filter keeps exactly the non-blank, long-enough texts, unchanged and in order."""
    kept = [doc.text for doc in _corpus(items).filter(min_chars=min_chars)]
    assert kept == [text for text in items if text.strip() and len(text) >= min_chars]


@given(texts, st.floats(min_value=0, max_value=1), seeds)
def test_sample_is_a_deterministic_ordered_subset(
    items: list[str],
    fraction: float,
    seed: int,
) -> None:
    """The same seed selects the same documents, in input order; 0 keeps none, 1 keeps all."""
    corpus = _corpus(items)
    first = [doc.position for doc in corpus.sample(fraction, seed=seed)]

    assert first == [doc.position for doc in corpus.sample(fraction, seed=seed)]
    assert first == sorted(first)
    assert [doc.position for doc in corpus.sample(0.0, seed=seed)] == []
    assert len(list(corpus.sample(1.0, seed=seed))) == len(items)


@given(st.lists(st.text(max_size=20), min_size=2, max_size=20, unique=True))
def test_fingerprint_depends_on_content_order_not_sources(items: list[str]) -> None:
    """Swapping two different texts changes the fingerprint; renaming every source does not."""
    swapped = [items[1], items[0], *items[2:]]
    renamed = [Document(text, "renamed", i + 7) for i, text in enumerate(items)]
    fingerprint = _corpus(items).stats().fingerprint

    assert _corpus(swapped).stats().fingerprint != fingerprint
    assert Corpus(read(renamed, name="other")).stats().fingerprint == fingerprint


@settings(max_examples=25, deadline=None)
@given(texts)
def test_save_and_load_round_trip_any_text(items: list[str]) -> None:
    """Any texts survive a save/load round trip with ids, order and fingerprint intact."""
    pytest.importorskip("pyarrow")
    corpus = _corpus(items)
    with tempfile.TemporaryDirectory() as directory:
        saved = corpus.save(Path(directory) / "corpus")
        loaded = Corpus.load(Path(directory) / "corpus")

        assert [(doc.id, doc.text) for doc in loaded] == [(doc.id, doc.text) for doc in corpus]
        assert loaded.stats().fingerprint == saved.fingerprint
