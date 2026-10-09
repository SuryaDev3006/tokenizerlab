"""Near-duplicate dedup and the train/validation split: keys, guards and leakage."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tokenizerlab import ConfigurationError, Corpus, Document, read
from tokenizerlab.data.corpus.near import banding, near_keys

VOCABULARY = [f"w{index}" for index in range(2000)]


def _words(rng: random.Random, count: int) -> list[str]:
    """`count` random vocabulary words."""
    return [rng.choice(VOCABULARY) for _ in range(count)]


def _replaced(rng: random.Random, text_words: list[str], share: float) -> list[str]:
    """A copy with exactly `share` of its positions, chosen at random, given a random word."""
    copy = list(text_words)
    for position in rng.sample(range(len(copy)), round(len(copy) * share)):
        copy[position] = rng.choice(VOCABULARY)
    return copy


def _match_rate(length: int, share: float, pairs: int, threshold: float = 0.75) -> float:
    """How often a `length`-word text and a copy with `share` replaced share a near key."""
    rng = random.Random(f"{length}:{share}:{threshold}")
    bands, rows = banding(threshold)
    matches = 0
    for _ in range(pairs):
        original = _words(rng, length)
        copy = _replaced(rng, original, share)
        original_keys = set(near_keys(" ".join(original), bands, rows))
        matches += bool(original_keys & set(near_keys(" ".join(copy), bands, rows)))
    return matches / pairs


def _texts(corpus: Corpus) -> list[str]:
    """The texts of one pass over corpus."""
    return [document.text for document in corpus]


def _manifest(directory: Path) -> dict[str, Any]:
    """The parsed manifest.json of a saved corpus."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text("utf-8"))
    return manifest


def _near_pairs(count: int, length: int) -> Corpus:
    """`count` random texts, each followed later by a copy with 1% of its words replaced;
    every document carries its pair number in metadata["pair"]."""
    rng = random.Random(7)
    originals = [_words(rng, length) for _ in range(count)]
    copies = [_replaced(rng, original, 0.01) for original in originals]
    documents = [
        Document(" ".join(text_words), "pairs", position, metadata={"pair": position % count})
        for position, text_words in enumerate(originals + copies)
    ]
    return Corpus(read(documents, name="pairs"))


def _leaked_pairs(train: Corpus, held_out: Corpus) -> int:
    """How many pairs have a document on each side."""
    train_pairs = {document.metadata["pair"] for document in train}
    return len(train_pairs & {document.metadata["pair"] for document in held_out})


# ---------------------------------------------------------------- split


def test_a_side_depends_only_on_the_seed_and_the_key() -> None:
    """Adding or reordering documents, or filtering before the split, moves no document."""
    texts = [f"document {index}" for index in range(200)]

    def held_out(corpus: Corpus) -> set[str]:
        """The validation texts of a 30% split."""
        return set(_texts(corpus.split(validation=0.3, seed=5)[1]))

    base = held_out(Corpus(read(texts, name="toy")))
    more = held_out(Corpus(read([*reversed(texts), "new 1", "new 2"], name="toy")))
    filtered = held_out(Corpus(read(texts, name="toy")).filter(min_chars=1))

    assert base == more - {"new 1", "new 2"} == filtered
    assert 0 < len(base) < len(texts)


def test_dedup_before_or_after_a_content_split_keeps_the_same_documents() -> None:
    """dedup then split equals split then dedup on each side, in the same order."""
    corpus = Corpus(read([f"text {index % 40}" for index in range(120)], name="toy"))
    before = corpus.dedup().split(validation=0.25, seed=1)
    after = [side.dedup() for side in corpus.split(validation=0.25, seed=1)]

    for dedup_first, dedup_after in zip(before, after, strict=True):
        assert [doc.id for doc in dedup_first] == [doc.id for doc in dedup_after]


def test_validation_share_is_close_to_the_requested_share() -> None:
    """10,000 distinct documents with validation=0.1 put 8-12% on the validation side."""
    corpus = Corpus(read([f"document {index}" for index in range(10_000)], name="toy"))
    held_out = sum(1 for _ in corpus.split(validation=0.1, seed=3)[1])
    assert 800 <= held_out <= 1200


def test_split_by_source_keeps_each_source_on_one_side() -> None:
    """With by="source", every document of a source is on the same side."""
    documents = [Document(f"text {index}", f"s{index % 30}", index) for index in range(300)]
    train, held_out = Corpus(read(documents, name="toy")).split(validation=0.3, by="source")
    train_sources = {doc.source for doc in train}
    held_out_sources = {doc.source for doc in held_out}

    assert not train_sources & held_out_sources
    assert train_sources | held_out_sources == {f"s{index}" for index in range(30)}


def test_invalid_split_arguments_and_repeated_seeds_raise() -> None:
    """Shares outside (0, 1), unknown key names and a repeated seed are refused."""
    corpus = Corpus(read(["a", "b"], name="toy"))
    with pytest.raises(ConfigurationError, match="validation must be in"):
        corpus.split(validation=1.0)
    with pytest.raises(ConfigurationError, match="by must be one of"):
        corpus.split(validation=0.5, by="language")

    train, _ = corpus.split(validation=0.5, seed=1)
    with pytest.raises(ConfigurationError, match="already split with seed=1"):
        train.split(validation=0.5, seed=1)
    train.split(validation=0.5, seed=2)


def test_manifest_records_the_split(tmp_path: Path) -> None:
    """validation, seed, by and side are recorded; a key function is not reproducible."""
    pytest.importorskip("pyarrow")
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("a", encoding="utf-8")
    corpus = Corpus(read(data, unit="line"))
    corpus.split(validation=0.5, seed=4)[0].save(tmp_path / "train")
    corpus.split(validation=0.5, by=lambda doc: doc.text)[1].save(tmp_path / "custom")

    train = _manifest(tmp_path / "train")
    assert train["history"][-1]["op"] == "split"
    assert train["history"][-1]["params"] == {
        "validation": 0.5,
        "seed": 4,
        "by": "content",
        "side": "train",
    }
    assert train["reproducible"] is True
    custom = _manifest(tmp_path / "custom")
    assert custom["history"][-1]["params"]["by"].endswith("<lambda>")
    assert custom["reproducible"] is False


# ---------------------------------------------------------------- near-duplicate keys


def test_banding_matches_fineweb_detection_at_each_threshold() -> None:
    """Rows per band are chosen so the threshold is caught as often as FineWeb's 0.75."""
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    assert [banding(threshold) for threshold in thresholds] == [
        (28, 4),
        (22, 5),
        (16, 7),
        (14, 8),
        (12, 9),
        (9, 12),
        (7, 16),
        (4, 23),
    ]


def test_short_texts_rarely_match_after_one_word_changes() -> None:
    """Regression: one replaced word in an 8- or 12-word text seldom looks like a duplicate."""
    assert _match_rate(8, 1 / 8, pairs=400) < 0.20
    assert _match_rate(12, 1 / 12, pairs=400) < 0.35


def test_long_copies_match_by_how_much_changed() -> None:
    """400-word copies with 1% replaced are caught; with 8% replaced they almost never are."""
    assert _match_rate(400, 0.01, pairs=300) >= 0.97
    assert _match_rate(400, 0.08, pairs=300) <= 0.15


def test_threshold_sets_how_similar_a_copy_must_be() -> None:
    """2% replaced is caught at threshold 0.6 and mostly missed at 0.9."""
    assert _match_rate(400, 0.02, pairs=300, threshold=0.6) >= 0.95
    assert _match_rate(400, 0.02, pairs=300, threshold=0.9) <= 0.45


# ---------------------------------------------------------------- near dedup


def test_near_dedup_ignores_case_and_spacing_only_for_comparing() -> None:
    """A copy differing in case and spacing is dropped, an unrelated text kept, and the kept
    text is unchanged."""
    original = "The quick brown fox jumps over the lazy dog today"
    copy = "the  QUICK brown\tfox jumps over the lazy dog   today"
    other = "An entirely different sentence about tokenizers and corpora"
    kept = _texts(Corpus(read([original, copy, other], name="toy")).dedup(near=True))
    assert kept == [original, other]


def test_texts_under_five_words_are_compared_exactly_ignoring_case_and_spacing() -> None:
    """Short texts are near-duplicates only when equal ignoring case and spacing."""
    texts = ["hello big world", "Hello  big WORLD", "hello big planet", "hello big world"]
    assert _texts(Corpus(read(texts, name="toy")).dedup(near=True)) == texts[:1] + texts[2:3]


def test_near_dedup_arguments_are_checked() -> None:
    """threshold must be in (0, 1) and only applies to near dedup."""
    corpus = Corpus(read(["a"], name="toy"))
    with pytest.raises(ConfigurationError, match="threshold must be in"):
        corpus.dedup(near=True, threshold=1.0)
    with pytest.raises(ConfigurationError, match="only applies to dedup"):
        corpus.dedup(threshold=0.9)


def test_near_dedup_is_the_same_in_every_process() -> None:
    """Two fresh processes with different hash seeds keep exactly the same documents."""
    script = (
        "from tokenizerlab import Corpus, read\n"
        "texts = [f'w{i % 7} w{i % 5} w{i % 3} w{i % 11} w{i % 2} w{i % 13}' for i in range(300)]\n"
        "print([d.position for d in Corpus(read(texts, name='toy')).dedup(near=True)])\n"
    )
    outputs = []
    for hash_seed in ("1", "2"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=True,
            env=environment,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert outputs[0].count(b",") > 10  # some documents kept, some dropped


def test_plain_dedup_records_what_it_always_has(tmp_path: Path) -> None:
    """dedup() keeps empty parameters and a step record without extra fields."""
    pytest.importorskip("pyarrow")
    Corpus(read(["a", "b", "a"], name="toy")).dedup().save(tmp_path / "out")
    manifest = _manifest(tmp_path / "out")

    assert manifest["history"][-1] == {"op": "dedup", "params": {}, "reproducible": True}
    assert manifest["steps"][-1] == {
        "op": "dedup",
        "docs_in": 3,
        "docs_out": 2,
        "bytes_in": 3,
        "bytes_out": 2,
    }


def test_near_dedup_records_its_settings(tmp_path: Path) -> None:
    """The history entry names the method and its banding."""
    pytest.importorskip("pyarrow")
    Corpus(read(["a"], name="toy")).dedup(near=True, threshold=0.9).save(tmp_path / "out")

    assert _manifest(tmp_path / "out")["history"][-1]["params"] == {
        "near": True,
        "threshold": 0.9,
        "bands": 7,
        "rows": 16,
        "hashes": 112,
        "ngram": 5,
        "method": "minhash-oph-v1",
        "against": None,
    }


# ---------------------------------------------------------------- dedup after a split


def test_train_side_dedup_needs_against_unless_it_is_safe() -> None:
    """Near dedup of the train side, or exact dedup after a source split, needs against=;
    the validation side and exact dedup after a content split do not."""
    documents = [Document(f"text {index}", f"s{index % 5}", index) for index in range(50)]
    corpus = Corpus(read(documents, name="toy"))
    train, held_out = corpus.split(validation=0.3)
    by_source_train, _ = corpus.split(validation=0.3, by="source")

    with pytest.raises(ConfigurationError, match="against="):
        train.dedup(near=True)
    with pytest.raises(ConfigurationError, match="against="):
        by_source_train.dedup()
    held_out.dedup(near=True)
    train.dedup()
    train.filter().dedup(near=True, against=held_out)


def test_against_removes_everything_the_validation_side_holds(tmp_path: Path) -> None:
    """No kept training document shares a near key or content hash with validation, and the
    manifest reports how many documents against= removed."""
    pytest.importorskip("pyarrow")
    train, held_out = _near_pairs(count=100, length=300).split(validation=0.3, seed=2)
    held_out = held_out.dedup(near=True)
    train = train.dedup(near=True, against=held_out)

    def keys(corpus: Corpus) -> set[bytes]:
        """Every near key and content hash of the corpus's documents."""
        bands, rows = banding(0.75)
        return {
            key
            for document in corpus
            for key in [*near_keys(document.text, bands, rows), document.content_hash.encode()]
        }

    assert not keys(train) & keys(held_out)
    train.save(tmp_path / "train")
    step = _manifest(tmp_path / "train")["steps"][-1]
    assert step["removed_against"] > 0
    assert step["docs_in"] - step["docs_out"] >= step["removed_against"]


def test_near_copies_do_not_leak_across_the_split() -> None:
    """With 1% edited copies of 300 texts, at most 3 pairs end up on both sides, whether dedup
    runs before the split or on both sides after it."""
    corpus = _near_pairs(count=300, length=300)
    dedup_first = corpus.dedup(near=True).split(validation=0.2)
    train, held_out = corpus.split(validation=0.2)
    held_out = held_out.dedup(near=True)
    split_first = (train.dedup(near=True, against=held_out), held_out)

    assert _leaked_pairs(*dedup_first) <= 3
    assert _leaked_pairs(*split_first) <= 3


def test_split_rules_survive_save_and_load(tmp_path: Path) -> None:
    """A loaded train side still needs against=, and a loaded validation side works as it."""
    pytest.importorskip("pyarrow")
    train, held_out = _near_pairs(count=20, length=50).split(validation=0.3)
    train.save(tmp_path / "train")
    held_out.save(tmp_path / "validation")
    loaded_train = Corpus.load(tmp_path / "train")
    loaded_validation = Corpus.load(tmp_path / "validation")

    with pytest.raises(ConfigurationError, match="against="):
        loaded_train.dedup(near=True)
    with pytest.raises(ConfigurationError, match="already split"):
        loaded_train.split(validation=0.3)
    deduped = loaded_train.dedup(near=True, against=loaded_validation)
    assert {doc.text for doc in deduped}.isdisjoint(doc.text for doc in loaded_validation)


def test_split_rules_reach_through_every_saved_generation(tmp_path: Path) -> None:
    """A train side saved and reloaded three times over still needs against=, and still
    refuses the seed of its split."""
    pytest.importorskip("pyarrow")
    train, _ = Corpus(read([f"text {index}" for index in range(30)], name="toy")).split(
        validation=0.3,
        seed=8,
    )
    train.save(tmp_path / "first")
    Corpus.load(tmp_path / "first").filter().save(tmp_path / "second")
    Corpus.load(tmp_path / "second").filter().save(tmp_path / "third")
    reloaded = Corpus.load(tmp_path / "third")

    with pytest.raises(ConfigurationError, match="against="):
        reloaded.dedup(near=True)
    with pytest.raises(ConfigurationError, match="already split with seed=8"):
        reloaded.split(validation=0.3, seed=8)
