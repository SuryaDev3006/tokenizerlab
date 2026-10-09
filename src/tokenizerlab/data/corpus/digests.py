"""Where Dedup keeps the content digests it has already seen during a pass.

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

FirstSighting = Callable[[bytes], bool]  # records a digest; True the first time it is seen


class DigestStore(Protocol):
    """Where a dedup pass keeps the digests it has seen: the part of dedup that varies."""

    def open_pass(self) -> AbstractContextManager[FirstSighting]:
        """An empty record of seen digests for one pass, released when the pass ends."""
        ...


@dataclass(frozen=True, slots=True)
class InMemoryDigests:
    """Seen digests in a Python set: fast, about 100 MB per million unique documents."""

    @contextmanager
    def open_pass(self) -> Iterator[FirstSighting]:
        """A fresh set for this pass."""
        seen: set[bytes] = set()

        def first_sighting(digest: bytes) -> bool:
            """Add the digest; True if the set did not hold it yet."""
            if digest in seen:
                return False
            seen.add(digest)
            return True

        yield first_sighting


@dataclass(frozen=True, slots=True)
class OnDiskDigests:
    """Seen digests in a private temporary SQLite database: flat memory, but slower."""

    @contextmanager
    def open_pass(self) -> Iterator[FirstSighting]:
        """A fresh database for this pass, closed (and so deleted) however the pass ends."""
        database = sqlite3.connect("")  # "": a temporary on-disk database, deleted on close
        try:
            # Nothing has to survive a crash: the database is thrown away after the pass.
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")

            def first_sighting(digest: bytes) -> bool:
                """Insert the digest; True if the table did not hold it yet."""
                insert = database.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,))
                return insert.rowcount == 1

            yield first_sighting
        finally:
            database.close()


def digest_store(*, on_disk: bool) -> DigestStore:
    """The DigestStore for dedup(on_disk=...)."""
    return OnDiskDigests() if on_disk else InMemoryDigests()
