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


repeating_texts = st.lists(st.text(alphabet="ab ", max_size=4), max_size=40)
word_texts = st.lists(
    st.lists(st.sampled_from(["a", "b", "c", "D"]), max_size=10).map(" ".join),
    max_size=30,
)
validation_shares = st.floats(min_value=0.01, max_value=0.99)


@given(repeating_texts, validation_shares, seeds)
def test_split_never_puts_one_text_on_both_sides(
    items: list[str],
    validation: float,
    seed: int,
) -> None:
    """Repeated texts always land on the same side, and every document lands on one side."""
    train, held_out = _corpus(items).split(validation=validation, seed=seed)
    train_texts = {doc.text for doc in train}
    held_out_texts = {doc.text for doc in held_out}

    assert not train_texts & held_out_texts
    assert len(list(train)) + len(list(held_out)) == len(items)


@settings(deadline=None)
@given(word_texts)
def test_near_dedup_keeps_no_identical_texts_and_keeps_the_first(items: list[str]) -> None:
    """Kept texts are all different, each kept document is its text's first occurrence, and
    the first document is always kept."""
    kept = list(_corpus(items).dedup(near=True))
    kept_texts = [doc.text for doc in kept]

    assert len(kept_texts) == len(set(kept_texts))
    assert all(items.index(doc.text) == doc.position for doc in kept)
    assert not items or kept[0].position == 0


@settings(max_examples=25, deadline=None)
@given(word_texts)
def test_on_disk_near_dedup_keeps_the_same_documents(items: list[str]) -> None:
    """dedup(near=True, on_disk=True) yields exactly the documents of the in-memory store."""
    corpus = _corpus(items)
    assert list(corpus.dedup(near=True, on_disk=True)) == list(corpus.dedup(near=True))
