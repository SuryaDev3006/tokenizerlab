"""Saving and loading corpora: manifests, verification, provenance and safe destinations."""

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tokenizerlab import Corpus, Document, read
from tokenizerlab.data.corpus import (
    CorpusExistsError,
    IntegrityError,
    MetadataNotSerializableError,
    UnsafeDestinationError,
)
from tokenizerlab.data.stats import ManifestError, UnsupportedManifestVersionError

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
    data = tmp_path / "data"
    data.mkdir()
    for name, text in {"1.txt": "a", "2.txt": "b", "3.txt": "a"}.items():
        (data / name).write_text(text, encoding="utf-8")
    corpus = Corpus(read(data, name="toy")).dedup()
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

    with (
        pytest.warns(UserWarning, match="bad.jsonl.gz"),
        pytest.warns(UserWarning, match=r"skipped 1 file\(s\)"),
    ):
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


@pytest.mark.parametrize(
    ("manifest_text", "message"),
    [
        (None, "no manifest"),
        ("not json", "not valid JSON"),
        ("[]", "not a tokenizerlab corpus manifest"),
        ('{"format_version": 1}', "malformed corpus manifest"),
    ],
)
def test_load_rejects_a_directory_that_is_not_a_saved_corpus(
    tmp_path: Path,
    manifest_text: str | None,
    message: str,
) -> None:
    """A missing, unparsable or incomplete manifest.json fails at load() with ManifestError."""
    if manifest_text is not None:
        (tmp_path / "manifest.json").write_text(manifest_text, encoding="utf-8")
    with pytest.raises(ManifestError, match=message):
        Corpus.load(tmp_path)


def test_verify_checks_the_fingerprint_and_can_be_turned_off(tmp_path: Path) -> None:
    """A fingerprint that does not match the rows fails a verified pass; verify=False trusts
    the rows and loads them as they are."""
    directory = tmp_path / "out"
    Corpus(read(["a", "b"], name="toy")).save(directory)
    manifest = _manifest(directory)
    manifest["stats"]["fingerprint"] = "0" * 32
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IntegrityError, match="does not match the manifest"):
        list(Corpus.load(directory))
    assert [doc.text for doc in Corpus.load(directory, verify=False)] == ["a", "b"]


def test_failed_save_leaves_nothing_behind(tmp_path: Path) -> None:
    """A save that fails names the bad document and leaves no half-written directory."""
    document = Document("x", "s", 0, metadata={"k": object()})
    with pytest.raises(MetadataNotSerializableError, match=document.id):
        Corpus(read([document], name="bad")).save(tmp_path / "bad")
    assert list(tmp_path.iterdir()) == []


def test_in_memory_sources_are_not_reproducible(tmp_path: Path) -> None:
    """A corpus read from a Python list cannot be recreated from its manifest."""
    Corpus(read(["a", "b"], name="toy")).save(tmp_path / "out")
    manifest = _manifest(tmp_path / "out")

    assert manifest["history"][0]["reproducible"] is False
    assert manifest["reproducible"] is False


def test_sampling_stays_independent_across_save_and_load(tmp_path: Path) -> None:
    """A sample after loading keeps about half of a previously sampled corpus, not all of it."""
    texts = [f"document {i}" for i in range(10_000)]
    Corpus(read(texts, name="toy")).sample(0.5, seed=0).save(tmp_path / "v1")
    loaded = Corpus.load(tmp_path / "v1")
    loaded.sample(0.5, seed=0).save(tmp_path / "v2")
    resampled = Corpus.load(tmp_path / "v2").sample(0.5, seed=0)

    first = sum(1 for _ in loaded)
    assert sum(1 for _ in loaded.sample(0.5, seed=0)) == pytest.approx(first / 2, rel=0.06)
    assert sum(1 for _ in resampled) == pytest.approx(first / 4, rel=0.1)  # two loads deep


def test_save_never_replaces_source_data(tmp_path: Path) -> None:
    """save() refuses destinations that overlap what it reads or that are not saved corpora."""
    raw = tmp_path / "rawdata"
    raw.mkdir()
    (raw / "a.txt").write_text("precious", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep me", encoding="utf-8")
    corpus = Corpus(read(raw))

    for destination in (raw, raw / "inside", tmp_path):
        with pytest.raises(UnsafeDestinationError, match="overlaps"):
            corpus.save(destination, overwrite=True)
    with pytest.raises(UnsafeDestinationError, match="not a saved tokenizerlab corpus"):
        corpus.save(other, overwrite=True)

    assert (raw / "a.txt").read_text(encoding="utf-8") == "precious"
    assert (other / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["other", "rawdata"]


def test_overwrite_replaces_a_saved_corpus(tmp_path: Path) -> None:
    """overwrite=True replaces a saved corpus and leaves no temporary or backup directory."""
    Corpus(read(["old"], name="toy")).save(tmp_path / "out")
    Corpus(read(["new"], name="toy")).save(tmp_path / "out", overwrite=True)

    assert [doc.text for doc in Corpus.load(tmp_path / "out")] == ["new"]
    assert [path.name for path in tmp_path.iterdir()] == ["out"]


RenameFilter = Callable[[Path, Path], bool]


def _moves_new_corpus_in(destination: Path) -> RenameFilter:
    """Matches the rename of the new corpus from its temporary sibling onto the destination."""
    return lambda source, target: target == destination and source.parent == destination.parent


def _moves_old_corpus_aside(destination: Path) -> RenameFilter:
    """Matches the rename of the old corpus away from the destination (e.g. a locked file)."""
    return lambda source, target: source == destination


@pytest.mark.parametrize("failing_rename", [_moves_new_corpus_in, _moves_old_corpus_aside])
def test_failed_overwrite_keeps_the_old_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_rename: Callable[[Path], RenameFilter],
) -> None:
    """If a rename of the swap fails, the old corpus still loads and nothing is left over."""
    destination = tmp_path / "out"
    Corpus(read(["old"], name="toy")).save(destination)
    fails = failing_rename(destination)
    real_replace = os.replace

    def replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        """os.replace, except for the one rename this test makes fail."""
        if fails(Path(source), Path(target)):
            raise OSError("simulated rename failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        Corpus(read(["new"], name="toy")).save(destination, overwrite=True)
    monkeypatch.undo()

    assert [doc.text for doc in Corpus.load(destination)] == ["old"]
    assert [path.name for path in tmp_path.iterdir()] == ["out"]


def test_a_leftover_after_overwrite_is_reported_at_the_save_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the replaced corpus cannot be deleted, the warning names the user's save() line."""
    Corpus(read(["old"], name="toy")).save(tmp_path / "out")
    real_rmtree = shutil.rmtree

    def rmtree(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        """shutil.rmtree, except that the moved-aside old corpus is locked."""
        if Path(path).name.startswith(".out.old."):
            raise OSError("simulated: a file is in use")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    with pytest.warns(UserWarning, match="could not delete the old copy") as caught:
        Corpus(read(["new"], name="toy")).save(tmp_path / "out", overwrite=True)
    monkeypatch.undo()

    assert Path(caught[0].filename) == Path(__file__)
    assert [doc.text for doc in Corpus.load(tmp_path / "out")] == ["new"]
