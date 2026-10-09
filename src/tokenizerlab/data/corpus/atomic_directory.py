"""Replacing a directory without ever losing it: the new directory is written next to the
destination, and the old one is moved aside, not deleted, until the new one is in place."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tokenizerlab.shared.user_warnings import warn_user


@contextmanager
def replace_directory_atomically(path: Path) -> Iterator[Path]:
    """Yield an empty temporary directory that replaces `path` once the block succeeds.

    The temporary directory is created next to `path`, so every rename stays on one
    filesystem. An existing `path` stays intact and in place until the new directory is
    complete. Then the old directory is moved aside, the new one is moved in, and only then
    is the old one deleted. If moving the new one in fails, the old one is moved back.

    Whenever an exception escapes, the temporary directory is removed and `path` still holds
    the old directory (or nothing, if there was none). On success the parent holds only
    `path`. A process killed between the two renames leaves both copies complete next to
    `path`: the new one in ".{name}.*" and the old one in ".{name}.old.*/{name}".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        yield temporary
        _swap_in(temporary, path)
    finally:
        # Gone already after a successful swap. After a failure, a leftover that cannot be
        # removed must not hide the exception that caused the failure.
        shutil.rmtree(temporary, ignore_errors=True)


def _swap_in(new: Path, path: Path) -> None:
    """Move `new` to `path`, moving an existing `path` aside first and deleting it last.

    Windows cannot rename a directory onto an existing one, so the old directory is moved
    into a fresh, empty holder directory next to it instead of being renamed over.
    """
    if not path.exists():
        os.replace(new, path)
        return

    holder = Path(tempfile.mkdtemp(prefix=f".{path.name}.old.", dir=path.parent))
    old = holder / path.name
    try:
        os.replace(path, old)
    except BaseException:
        holder.rmdir()
        raise
    try:
        os.replace(new, path)
    except BaseException:
        os.replace(old, path)  # put the old directory back where it was
        holder.rmdir()
        raise
    _delete_old(holder, path)


def _delete_old(holder: Path, path: Path) -> None:
    """Delete the replaced directory. The new one is already in place, so a failure here is
    reported with a warning naming the leftover rather than failing the save."""
    try:
        shutil.rmtree(holder)
    except OSError as error:
        warn_user(
            f"Replaced {path}, but could not delete the old copy in {holder} ({error}); "
            "delete it by hand."
        )
