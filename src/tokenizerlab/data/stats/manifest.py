"""Provenance and the manifest: how a corpus was built and what a saved corpus contains.

Building, encoding and decoding are pure; the corpus store writes and reads the file.
Apart from `created_at` and `versions`, saving the same corpus twice gives an identical
manifest. There is no hash of the Parquet file, because its bytes can differ between
pyarrow versions for the same data; the fingerprint is the content check.
"""

from __future__ import annotations

import json
import platform
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from tokenizerlab.data.readers import DocumentStream
from tokenizerlab.data.stats.corpus_stats import CorpusStats, StepCounts
from tokenizerlab.errors import TokenizerLabError
from tokenizerlab.shared.hashing import HASH_VERSION
from tokenizerlab.shared.json_format import pretty_json

MANIFEST_FORMAT_VERSION = 1
UNKNOWN_VERSION = "unknown"
# A million bad rows must not produce a million-line manifest; totals are always complete.
MAX_LISTED = 100


class ManifestError(TokenizerLabError):
    """A manifest.json is missing, malformed, or not a tokenizerlab manifest."""


class UnsupportedManifestVersionError(ManifestError):
    """A manifest was written in a format version this tokenizerlab cannot read."""


# ---------------------------------------------------------------- provenance


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """An operation ("read", "load", "filter", ...) and the parameters it was given."""

    op: str
    # JSON-ready: a mapping, or for a chained read the list of each source's reader config.
    params: Any
    reproducible: bool = True  # False if the parameters cannot re-create the step

    def to_dict(self) -> dict[str, Any]:
        """The entry as written to the manifest."""
        return {"op": self.op, "params": self.params, "reproducible": self.reproducible}


@dataclass(frozen=True, slots=True)
class ParentInfo:
    """The saved corpus a loaded corpus came from, so provenance chains across saves."""

    fingerprint: str
    history: list[dict[str, Any]]
    parent: dict[str, Any] | None  # the parent's own parent, as recorded in its manifest
    reproducible: bool


class SelectionReport(Protocol):
    """What a selection step (sample, mix) reports about its decisions."""

    def as_record(self) -> dict[str, Any]:
        """The report as JSON-ready fields."""
        ...


@dataclass(frozen=True, slots=True)
class StepRecord:
    """A step's counts, plus target vs achieved results for sample and mix."""

    counts: StepCounts
    selection: SelectionReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """One flat JSON object per step."""
        record = asdict(self.counts)
        if self.selection is None:
            return record
        return {**record, **self.selection.as_record(), "achieved_bytes": self.counts.bytes_out}


@dataclass(frozen=True, slots=True)
class ReaderReport:
    """Error counts, the first error messages, skipped files, and per-source reader info."""

    errors_by_level: dict[str, int]
    errors_by_source: dict[str, int]
    first_errors: list[str]
    skipped: list[str]  # the first MAX_LISTED unsupported files
    skipped_total: int
    info: dict[str, Any]  # e.g. resolved Hugging Face revisions

    @classmethod
    def from_stream(cls, stream: DocumentStream) -> ReaderReport:
        """Summarize the errors, skipped files and info of the stream's last pass."""
        errors = stream.errors
        return cls(
            errors_by_level=dict(Counter(str(error.level) for error in errors)),
            errors_by_source=dict(Counter(error.source for error in errors)),
            first_errors=[str(error) for error in errors[:MAX_LISTED]],
            skipped=stream.skipped[:MAX_LISTED],
            skipped_total=len(stream.skipped),
            info=dict(stream.info),
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything about how a corpus was built, from one complete pass."""

    history: tuple[HistoryEntry, ...]
    parent: ParentInfo | None
    steps: tuple[StepRecord, ...]
    reader: ReaderReport

    @property
    def step_counts(self) -> tuple[StepCounts, ...]:
        """The per-step counts alone, as CorpusStats records them."""
        return tuple(step.counts for step in self.steps)

    @property
    def reproducible(self) -> bool:
        """False if any step, here or in the parent chain, cannot be re-created from records."""
        return chain_is_reproducible(self.history, self.parent)


def chain_is_reproducible(history: Iterable[HistoryEntry], parent: ParentInfo | None) -> bool:
    """Whether every step of `history`, and of the saved corpus it was loaded from, can be
    re-created from the records."""
    parent_reproducible = parent is None or parent.reproducible
    return parent_reproducible and all(entry.reproducible for entry in history)


# ---------------------------------------------------------------- manifest


def _installed_version(package: str) -> str:
    """The installed version of a package, or "unknown"."""
    try:
        return version(package)
    except PackageNotFoundError:
        return UNKNOWN_VERSION


def _utc_now() -> str:
    """The current UTC time in ISO 8601."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Versions:
    """Versions of the software that wrote a manifest."""

    tokenizerlab: str
    python: str
    pyarrow: str
    hash: str

    @classmethod
    def current(cls) -> Versions:
        """The versions running now."""
        return cls(
            tokenizerlab=_installed_version("tokenizerlab"),
            python=platform.python_version(),
            pyarrow=_installed_version("pyarrow"),
            hash=HASH_VERSION,
        )


@dataclass(frozen=True, slots=True)
class FileInfo:
    """A file saved next to the manifest."""

    rows: int
    bytes: int


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything recorded next to a saved corpus."""

    provenance: Provenance
    stats: CorpusStats
    files: dict[str, FileInfo]
    format_version: int = MANIFEST_FORMAT_VERSION
    created_at: str = field(default_factory=_utc_now)
    versions: Versions = field(default_factory=Versions.current)

    def to_dict(self) -> dict[str, Any]:
        """The JSON-ready manifest."""
        provenance = self.provenance
        return {
            "format_version": self.format_version,
            "created_at": self.created_at,
            "versions": asdict(self.versions),
            "reproducible": provenance.reproducible,
            "parent": asdict(provenance.parent) if provenance.parent else None,
            "history": [entry.to_dict() for entry in provenance.history],
            "steps": [step.to_dict() for step in provenance.steps],
            "reader": asdict(provenance.reader),
            "stats": self.stats.to_dict(),
            "files": {name: asdict(info) for name, info in self.files.items()},
        }


def encode_manifest(manifest: Manifest) -> str:
    """The manifest as indented, key-sorted UTF-8 JSON text."""
    return pretty_json(manifest.to_dict())


@dataclass(frozen=True, slots=True)
class SavedManifest:
    """The parts of a saved manifest that loading a corpus needs."""

    stats: CorpusStats
    history: list[dict[str, Any]]
    parent: dict[str, Any] | None
    reproducible: bool
    # History steps that led here, across every saved ancestor. A loaded corpus continues
    # the step numbering from here, so its selection hashes never repeat earlier steps'.
    steps_so_far: int

    def as_parent(self) -> ParentInfo:
        """How a corpus loaded from this manifest records where it came from."""
        return ParentInfo(self.stats.fingerprint, self.history, self.parent, self.reproducible)


def recorded_history(
    history: list[dict[str, Any]],
    parent: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The history entries of a manifest's whole parent chain, then its own, oldest first."""
    entries = list(history)
    while parent is not None:
        entries = parent["history"] + entries
        parent = parent["parent"]
    return entries


def decode_manifest(text: str, origin: str) -> SavedManifest:
    """Parse manifest text; errors name `origin` (usually the file path)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ManifestError(f"{origin}: not valid JSON ({error.msg})") from None
    if not isinstance(data, dict):
        raise ManifestError(f"{origin}: not a tokenizerlab corpus manifest")
    _check_format_version(origin, data.get("format_version"))

    try:
        history = list(data["history"])
        return SavedManifest(
            stats=CorpusStats.from_dict(data["stats"]),
            history=history,
            parent=data["parent"],
            reproducible=bool(data["reproducible"]),
            steps_so_far=len(recorded_history(history, data["parent"])),
        )
    except (KeyError, TypeError) as error:
        raise ManifestError(f"{origin}: malformed corpus manifest ({error!r})") from None


def _check_format_version(origin: str, format_version: object) -> None:
    """Reject manifests with an unknown major format version."""
    major = str(format_version).split(".", 1)[0]
    if major != str(MANIFEST_FORMAT_VERSION):
        raise UnsupportedManifestVersionError(
            f"{origin}: unsupported manifest format_version {format_version!r}; "
            f"this tokenizerlab reads version {MANIFEST_FORMAT_VERSION}"
        )
