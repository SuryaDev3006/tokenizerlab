import json
from pathlib import Path
from typing import Any

import pytest

from tokenizerlab import ConfigurationError, Corpus, CorpusStats, Document, read
from tokenizerlab.data.corpus import UnreachableMixError
from tokenizerlab.data.readers import StreamConsumedError


def _texts(corpus: Corpus) -> list[str]:
    """The texts of one pass over corpus."""
    return [document.text for document in corpus]


def _ids(corpus: Corpus) -> set[str]:
    """The ids of one pass over corpus."""
    return {document.id for document in corpus}


def _two_languages(a_count: int, b_count: int) -> list[Document]:
    """10-byte documents in languages "a" and "b", plus one untagged document."""
    return (
        [Document("x" * 10, "a", position, lang="a") for position in range(a_count)]
        + [Document("y" * 10, "b", position, lang="b") for position in range(b_count)]
        + [Document("z", "untagged", 0)]
    )


def _byte_shares(stats: CorpusStats) -> dict[str, float]:
    """The share of bytes per language."""
    return {lang: counts.bytes / stats.bytes for lang, counts in stats.by_lang.items()}


def _saved_last_step(corpus: Corpus, directory: Path) -> dict[str, Any]:
    """Save the corpus and return its last step as recorded in manifest.json."""
    corpus.save(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    step: dict[str, Any] = manifest["steps"][-1]
    return step


def test_operations_return_independent_corpora() -> None:
    """Each operation returns a new corpus; the original and its siblings are unaffected."""
    corpus = Corpus(read(["a", " ", "a", "bb", ""], name="toy"))
    filtered = corpus.filter()
    deduped = corpus.dedup()

    assert _texts(filtered) == ["a", "a", "bb"]
    assert _texts(deduped) == ["a", " ", "bb", ""]
    assert _texts(corpus) == ["a", " ", "a", "bb", ""]


def test_filter() -> None:
    """Blank documents always go; min_chars counts characters; fn decides the rest."""
    corpus = Corpus(read(["a", " ", "\t\n", "అఆ", ""], name="toy"))

    assert _texts(corpus.filter()) == ["a", "అఆ"]
    assert _texts(corpus.filter(min_chars=2)) == ["అఆ"]  # 2 characters, 6 bytes
    assert _texts(corpus.filter(lambda doc: doc.text != "a")) == ["అఆ"]
    with pytest.raises(ConfigurationError):
        corpus.filter(min_chars=-1)


def test_dedup_keeps_first_and_reports_count() -> None:
    """Exact duplicates are removed across sources, the first is kept, and the count reported."""
    corpus = Corpus(read([read(["a", "b"], name="s1"), read(["a", "a", "c"], name="s2")])).dedup()

    assert [(doc.source, doc.position) for doc in corpus] == [("s1", 0), ("s1", 1), ("s2", 2)]
    dedup_step = corpus.stats().steps[-1]
    assert (dedup_step.op, dedup_step.docs_in - dedup_step.docs_out) == ("dedup", 2)


def test_sample_is_seeded_and_independent_per_step() -> None:
    """Same seed, same documents; another seed, another set; later steps are not degenerate."""
    corpus = Corpus(read(_two_languages(3000, 1000), name="toy"))

    assert _ids(corpus.sample(0.5, seed=7)) == _ids(corpus.sample(0.5, seed=7))
    assert _ids(corpus.sample(0.5, seed=7)) != _ids(corpus.sample(0.5, seed=8))

    # Reusing the sample's selection values would keep 2/3 of "a" instead of 1/3 and skew this.
    mixed = corpus.sample(0.5, seed=7).mix(weights={"a": 1, "b": 1}, seed=7).stats()
    assert _byte_shares(mixed)["a"] == pytest.approx(0.5, abs=0.04)

    budget = corpus.sample(max_bytes=20_000).stats()
    assert budget.bytes == pytest.approx(20_000, rel=0.05)
    with pytest.raises(ConfigurationError, match="exactly one"):
        corpus.sample()


def test_mix_reaches_target_shares(tmp_path: Path) -> None:
    """alpha and weights both land within a small tolerance and record dropped untagged docs."""
    pytest.importorskip("pyarrow")
    corpus = Corpus(read(_two_languages(3000, 1000), name="toy"))
    alpha_weights = {"a": 30_000**0.3, "b": 10_000**0.3}  # bytes per language ** alpha
    alpha_total = sum(alpha_weights.values())
    alpha_targets = {lang: weight / alpha_total for lang, weight in alpha_weights.items()}
    cases = [
        (corpus.mix(alpha=0.3, seed=1), alpha_targets),
        (corpus.mix(weights={"a": 2, "b": 1}, seed=1), {"a": 2 / 3, "b": 1 / 3}),
    ]

    for index, (mixed, targets) in enumerate(cases):
        step = _saved_last_step(mixed, tmp_path / str(index))
        assert step["target_shares"] == pytest.approx(targets)
        assert step["achieved_shares"] == pytest.approx(targets, abs=0.02)
        assert _byte_shares(mixed.stats()) == pytest.approx(targets, abs=0.02)
        assert step["dropped"] == {"und": 1}


def test_mix_never_upsamples_and_rejects_unreachable_targets() -> None:
    """The scarcest language limits the size; a weighted language with no data is an error."""
    corpus = Corpus(read(_two_languages(3000, 1000), name="toy"))

    mixed = corpus.mix(weights={"a": 1, "b": 3}).stats()
    assert mixed.by_lang["b"].documents == 1000  # all of the scarce language, nothing repeated
    with pytest.raises(UnreachableMixError, match="fr"):
        list(corpus.mix(weights={"a": 1, "fr": 1}))
    with pytest.raises(ConfigurationError):
        corpus.mix(alpha=1.5)


def test_one_shot_source_is_never_silently_empty() -> None:
    """A pre-pass consumes a one-shot iterator, so the main pass must raise, not yield nothing."""
    one_shot = Corpus(read(iter(_two_languages(10, 10)), name="once")).sample(max_bytes=10)
    with pytest.raises(StreamConsumedError):
        list(one_shot)


def test_texts_is_re_iterable() -> None:
    """A trainer can make several passes over texts() and sees the same texts each time."""
    texts = Corpus(read(["a", "b", "a"], name="toy")).dedup().texts()
    assert list(texts) == list(texts) == ["a", "b"]


def test_mix_explains_missing_language_tags() -> None:
    """Mixing untagged documents raises a clear error instead of returning nothing."""
    untagged = Corpus(read(["a", "b"], name="toy"))

    for mixed in (untagged.mix(alpha=0.3), untagged.mix(weights={"en": 1})):
        with pytest.raises(UnreachableMixError, match="no documents with a language tag"):
            list(mixed)
    assert list(Corpus(read([], name="empty")).mix(alpha=0.3)) == []  # nothing to mix is fine
