"""DocumentStream: a lazy, re-iterable stream of Documents, and how its errors are handled.

Each pass resets a stream's `errors`, `skipped` and `info`, which are complete once the pass
has been fully consumed.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from enum import Enum, StrEnum
from typing import Any, Protocol

from tokenizerlab.data.document import Document
from tokenizerlab.data.readers.options import OnError, ReaderConfig
from tokenizerlab.errors import TokenizerLabError

RunPass = Callable[["DocumentStream"], Iterator[Document]]


class ErrorLevel(StrEnum):
    """How much of a source a read error affects."""

    ROW = "row"
    FILE = "file"


class ReadError(TokenizerLabError):
    """A failed row (level ROW) or a failed file or source (level FILE)."""

    def __init__(
        self,
        source: str,
        message: str,
        position: int | None = None,
        level: ErrorLevel = ErrorLevel.ROW,
    ):
        """Format the message as "source#position: message" and keep each part as an attribute."""
        location = source if position is None else f"{source}#{position}"
        super().__init__(f"{location}: {message}")
        self.source = source
        self.message = message
        self.position = position
        self.level = level


class StreamConsumedError(TokenizerLabError):
    """A one-shot stream was iterated a second time."""


class Reuse(Enum):
    """Whether a stream's source can be read more than once."""

    REPEATABLE = "repeatable"  # files, lists, functions that return a fresh iterator
    ONE_SHOT = "one-shot"  # iterators and generators


class _PassState(Enum):
    """Lifecycle of a stream: READY until a one-shot stream starts its only pass."""

    READY = "ready"
    CONSUMED = "consumed"


# ---------------------------------------------------------------- error policies


class ErrorPolicy(Protocol):
    """What happens to a bad row or file: the part of error handling that varies."""

    def record(self, error: ReadError, errors: list[ReadError]) -> None:
        """Handle one error as it happens; `errors` is the pass's error list."""
        ...

    def summarize_bad_rows(self, source_name: str, bad_rows: list[ReadError]) -> None:
        """Report a source's bad rows once the source has been read."""
        ...


class RaisePolicy:
    """on_error="raise": stop reading at the first error."""

    def record(self, error: ReadError, errors: list[ReadError]) -> None:
        """Raise the error."""
        raise error

    def summarize_bad_rows(self, source_name: str, bad_rows: list[ReadError]) -> None:
        """Nothing to summarize: the first bad row already stopped the pass."""


class SkipPolicy:
    """on_error="skip": record errors silently."""

    def record(self, error: ReadError, errors: list[ReadError]) -> None:
        """Keep the error for the pass's report."""
        errors.append(error)

    def summarize_bad_rows(self, source_name: str, bad_rows: list[ReadError]) -> None:
        """Stay silent; the errors are in the stream's report."""


class WarnPolicy:
    """on_error="warn": record errors, warn at once about failed files and once per source
    about bad rows."""

    def record(self, error: ReadError, errors: list[ReadError]) -> None:
        """Keep the error; a whole failed file is worth an immediate warning."""
        errors.append(error)
        if error.level is ErrorLevel.FILE:
            warnings.warn(str(error), stacklevel=4)

    def summarize_bad_rows(self, source_name: str, bad_rows: list[ReadError]) -> None:
        """One warning per source rather than one per row."""
        if bad_rows:
            warnings.warn(
                f"{source_name}: skipped {len(bad_rows)} row(s); first: {bad_rows[0]}",
                stacklevel=4,
            )


ERROR_POLICIES: dict[OnError, ErrorPolicy] = {
    OnError.RAISE: RaisePolicy(),
    OnError.SKIP: SkipPolicy(),
    OnError.WARN: WarnPolicy(),
}


# ---------------------------------------------------------------- DocumentStream


class DocumentStream:
    """A lazy, re-iterable stream of Documents."""

    def __init__(
        self,
        run: RunPass,
        list_source_names: Callable[[], list[str]],
        on_error: OnError,
        reuse: Reuse = Reuse.REPEATABLE,
    ):
        """Wrap `run`, which yields one pass of Documents, and `list_source_names`."""
        self._run = run
        self._list_source_names = list_source_names
        self.on_error = on_error
        self.error_policy = ERROR_POLICIES[on_error]
        self.reuse = reuse
        self.errors: list[ReadError] = []
        self.skipped: list[str] = []  # unsupported files found during discovery
        self.info: dict[str, dict[str, Any]] = {}  # per-source facts, e.g. resolved HF revision
        self.config: ReaderConfig = {}  # read()'s arguments, recorded in corpus manifests
        self._state = _PassState.READY

    def __iter__(self) -> Iterator[Document]:
        """Start a fresh pass with empty errors, skipped files and info."""
        self._start_pass()
        self.errors, self.skipped, self.info = [], [], {}
        return self._run(self)

    def source_names(self) -> list[str]:
        """The names of every source this stream reads, in reading order."""
        return self._list_source_names()

    def record_error(self, error: ReadError) -> None:
        """Handle one error according to the stream's error policy."""
        self.error_policy.record(error, self.errors)

    def _start_pass(self) -> None:
        """READY → READY for repeatable streams; READY → CONSUMED for a one-shot stream."""
        if self._state is _PassState.CONSUMED:
            raise StreamConsumedError(
                "This stream wraps a one-shot iterator that was already consumed. Pass a list, "
                "or a function that returns a fresh iterator, to allow multiple passes."
            )
        if self.reuse is Reuse.ONE_SHOT:
            self._state = _PassState.CONSUMED
