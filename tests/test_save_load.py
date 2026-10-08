import json
from pathlib import Path
from typing import Any

import pytest

from tokenizerlab import Corpus, Document, read
from tokenizerlab.data.corpus import CorpusExistsError, IntegrityError, MetadataNotSerializableError
from tokenizerlab.data.stats import UnsupportedManifestVersionError

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")


def _manifest(directory: Path) -> dict[str, Any]:
    """The parsed manifest.json of a saved corpus."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text("utf-8"))
    return manifest


def _without_run_details(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest without the fields allowed to differ between two saves."""
    return {key: value for key, value in manifest.items() if key not in ("created_at", "versions")}


def test_round_trip_and_tampering(tmp_path: Path) -> None:
    """A saved corpus loads back with the same fingerprint; a tampered row fails verification."""
    documents = [
        Document("hi", "s", 0, lang="te", metadata={"k": 1}),
        Document("yo", "s", 1),
        Document(" ", "s", 2),
    ]
    corpus = Corpus(read(documents, name="toy")).filter()
    saved = corpus.save(tmp_path / "v1")
    with pytest.raises(CorpusExistsError):
        corpus.save(tmp_path / "v1")

    loaded = Corpus.load(tmp_path / "v1")
    identities = [(doc.id, doc.lang, dict(doc.metadata)) for doc in corpus]
    assert [(doc.id, doc.lang, dict(doc.metadata)) for doc in loaded] == identities
    assert loaded.stats().fingerprint == saved.fingerprint

    corpus_file = tmp_path / "v1" / "corpus.parquet"
    table = parquet.read_table(corpus_file)
    parquet.write_table(table.set_column(1, "text", pyarrow.array(["HI", "yo"])), corpus_file)
    with pytest.raises(IntegrityError):
        list(Corpus.load(tmp_path / "v1"))


def test_manifest_records_how_the_corpus_was_built(tmp_path: Path) -> None:
    """History, per-step counts, files and stats; identical across saves but for run details."""
    corpus = Corpus(read(["a", "b", "a"], name="toy")).dedup()
    corpus.save(tmp_path / "first")
    corpus.save(tmp_path / "second")
    manifest = _manifest(tmp_path / "first")

    assert manifest["format_version"] == 1
    assert manifest["reproducible"] is True
    assert manifest["parent"] is None
    assert [entry["op"] for entry in manifest["history"]] == ["read", "dedup"]
    assert manifest["history"][0]["params"]["name"] == "toy"
    assert manifest["steps"][1] == {
        "op": "dedup",
        "docs_in": 3,
        "docs_out": 2,
        "bytes_in": 3,
        "bytes_out": 2,
    }
    assert manifest["stats"]["documents"] == manifest["files"]["corpus.parquet"]["rows"] == 2
    assert set(manifest["versions"]) == {"tokenizerlab", "python", "pyarrow", "hash"}
    assert _without_run_details(manifest) == _without_run_details(_manifest(tmp_path / "second"))


def test_sample_records_target_and_achieved_bytes(tmp_path: Path) -> None:
    """A byte-budget sample records what it aimed for and what it got."""
    texts = [f"document {i:04d}" for i in range(500)]
    Corpus(read(texts, name="toy")).sample(max_bytes=3000).save(tmp_path / "out")
    step = _manifest(tmp_path / "out")["steps"][-1]

    assert step["target_bytes"] == 3000
    assert step["achieved_bytes"] == step["bytes_out"]
    assert step["achieved_bytes"] == pytest.approx(3000, rel=0.15)


def test_parent_chain(tmp_path: Path) -> None:
    """Saving a loaded corpus records its parent's fingerprint and history."""
    stats = Corpus(read(["a", "b"], name="toy")).save(tmp_path / "v1")
    Corpus.load(tmp_path / "v1").dedup().save(tmp_path / "v2")
    manifest = _manifest(tmp_path / "v2")

    assert [entry["op"] for entry in manifest["history"]] == ["load", "dedup"]
    assert manifest["history"][0]["params"]["fingerprint"] == stats.fingerprint
    assert manifest["parent"]["fingerprint"] == stats.fingerprint
    assert [entry["op"] for entry in manifest["parent"]["history"]] == ["read"]


def test_empty_corpus_saves_and_loads(tmp_path: Path) -> None:
    """An empty corpus produces a valid manifest and loads back empty."""
    stats = Corpus(read([], name="empty")).save(tmp_path / "empty")
    manifest = _manifest(tmp_path / "empty")

    assert manifest["stats"]["documents"] == 0
    assert manifest["stats"]["length_mean"] is None
    assert manifest["files"]["corpus.parquet"]["rows"] == 0
    assert list(Corpus.load(tmp_path / "empty")) == []
    assert Corpus.load(tmp_path / "empty").stats().fingerprint == stats.fingerprint


def test_lambda_filter_is_not_reproducible(tmp_path: Path) -> None:
    """A custom filter function is recorded by name and marks the manifest not reproducible."""
    corpus = Corpus(read(["a", "bb"], name="toy")).filter(lambda doc: len(doc.text) > 1)
    corpus.save(tmp_path / "out")
    manifest = _manifest(tmp_path / "out")

    filter_entry = manifest["history"][1]
    assert filter_entry["params"]["fn"].endswith("<lambda>")
    assert filter_entry["reproducible"] is False
    assert manifest["reproducible"] is False


def test_broken_file_appears_in_manifest(tmp_path: Path) -> None:
    """A skipped broken file is counted and listed under the manifest's reader section."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "good.txt").write_text("fine", encoding="utf-8")
    (data / "bad.jsonl.gz").write_bytes(b"not gzip")
    (data / "notes.md").write_text("unsupported", encoding="utf-8")

    with pytest.warns(UserWarning, match="bad.jsonl.gz"):
        Corpus(read(data)).save(tmp_path / "out")
    reader = _manifest(tmp_path / "out")["reader"]

    assert reader["errors_by_level"] == {"file": 1}
    assert reader["errors_by_source"] == {"bad.jsonl.gz": 1}
    assert reader["first_errors"][0].startswith("bad.jsonl.gz: ")
    assert (reader["skipped"], reader["skipped_total"]) == (["notes.md"], 1)


def test_error_lists_are_capped(tmp_path: Path) -> None:
    """A thousand bad rows keep only the first 100 messages, but the totals stay complete."""
    rows = "\n".join("not json" for _ in range(1000))
    (tmp_path / "bad.jsonl").write_text(rows + '\n{"text": "ok"}\n', encoding="utf-8")

    Corpus(read(tmp_path / "bad.jsonl", on_error="skip")).save(tmp_path / "out")
    reader = _manifest(tmp_path / "out")["reader"]

    assert len(reader["first_errors"]) == 100
    assert reader["errors_by_level"] == {"row": 1000}


def test_unknown_format_version_is_rejected(tmp_path: Path) -> None:
    """Loading refuses a manifest whose major format version this code does not know."""
    directory = tmp_path / "out"
    Corpus(read(["a"], name="toy")).save(directory)
    manifest = _manifest(directory)
    manifest["format_version"] = 2
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UnsupportedManifestVersionError):
        Corpus.load(directory)


def test_failed_save_leaves_nothing_behind(tmp_path: Path) -> None:
    """A save that fails names the bad document and leaves no half-written directory."""
    document = Document("x", "s", 0, metadata={"k": object()})
    with pytest.raises(MetadataNotSerializableError, match=document.id):
        Corpus(read([document], name="bad")).save(tmp_path / "bad")
    assert list(tmp_path.iterdir()) == []
