"""Where Dedup keeps the digests (content hashes or near-duplicate keys) of a pass.

The in-memory set is the fastest and costs about 100 MB per million unique documents. The
on-disk store keeps the digests in a private temporary SQLite database instead, so memory
stays flat at the cost of speed. Both remember exactly the same digests, so they keep exactly
the same documents.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DigestIndex:
    """One pass's record of digests. Asking and adding are separate, so a document with
    several keys can check all of them before adding any."""

    seen: Callable[[bytes], bool]
    add: Callable[[bytes], None]


class DigestStore(Protocol):
    """Where a dedup pass keeps the digests it has seen: the part of dedup that varies."""

    def open_pass(self) -> AbstractContextManager[DigestIndex]:
        """An empty record of digests for one pass, released when the pass ends."""
        ...


@dataclass(frozen=True, slots=True)
class InMemoryDigests:
    """Seen digests in a Python set: fast, about 100 MB per million unique documents."""

    @contextmanager
    def open_pass(self) -> Iterator[DigestIndex]:
        """A fresh set for this pass."""
        digests: set[bytes] = set()
        yield DigestIndex(seen=digests.__contains__, add=digests.add)


@dataclass(frozen=True, slots=True)
class OnDiskDigests:
    """Seen digests in a private temporary SQLite database: flat memory, but slower."""

    @contextmanager
    def open_pass(self) -> Iterator[DigestIndex]:
        """A fresh database for this pass, closed (and so deleted) however the pass ends."""
        database = sqlite3.connect("")  # "": a temporary on-disk database, deleted on close
        try:
            # Nothing has to survive a crash: the database is thrown away after the pass.
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")

            def seen(digest: bytes) -> bool:
                """Whether the table holds the digest."""
                query = database.execute("SELECT 1 FROM seen WHERE digest = ?", (digest,))
                return query.fetchone() is not None

            def add(digest: bytes) -> None:
                """Insert the digest; adding one twice keeps a single row."""
                database.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,))

            yield DigestIndex(seen=seen, add=add)
        finally:
            database.close()


def digest_store(*, on_disk: bool) -> DigestStore:
    """The DigestStore for dedup(on_disk=...)."""
    return OnDiskDigests() if on_disk else InMemoryDigests()
