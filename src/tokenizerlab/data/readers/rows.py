"""Rows and sources: how the content of one source becomes Documents during a pass."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from tokenizerlab.data.document import Document
from tokenizerlab.data.readers.options import ReadOptions
from tokenizerlab.data.readers.stream import ErrorLevel, PassReport, ReadError
from tokenizerlab.errors import TokenizerLabError

SET_TEXT_FIELD = "set text_field= to the field that holds the text"
READ_AS_RAW_TEXT = "or pass format='text' to read the whole file as one raw document"


class MissingTextFieldError(ReadError):
    """A source has no text_field, so the whole source fails (level FILE)."""

    def __init__(
        self,
        source: str,
        text_field: str,
        available_fields: Iterable[Any],
        fixes: tuple[str, ...] = (SET_TEXT_FIELD,),
    ):
        """List the fields the source does have, then how to fix the read."""
        self.text_field = text_field
        self.available_fields = tuple(map(str, available_fields))
        self.fixes = fixes
        fields = ", ".join(self.available_fields) or "(none)"
        message = f"field {text_field!r} not found; available fields: {fields}; {', '.join(fixes)}"
        super().__init__(source, message, level=ErrorLevel.FILE)

    def offering_raw_text(self) -> MissingTextFieldError:
        """The same error, also suggesting format='text', for a file that can be read raw."""
        fixes = (*self.fixes, READ_AS_RAW_TEXT)
        return MissingTextFieldError(self.source, self.text_field, self.available_fields, fixes)


class RowParser:
    """Turns the rows (mappings such as JSON objects) of one source into Documents."""

    def __init__(self, source_name: str, options: ReadOptions):
        """Bind the source name and the read options that say which fields to use."""
        self._source_name = source_name
        self._options = options
        self._text_field_checked = False

    def parse(self, position: int, row: Any) -> Document:
        """Build a Document; raise a row-level ReadError if this row is unusable,
        or a file-level one if the source's first row lacks text_field."""
        if not isinstance(row, Mapping):
            raise self._row_error(position, f"expected an object/row, got {type(row).__name__}")

        self._check_text_field_once(row)
        return Document(
            text=self._text(position, row),
            source=self._source_name,
            position=position,
            lang=self._lang(position, row),
            metadata=self._metadata(row),
        )

    def missing_text_field_error(self, available_fields: Iterable[Any]) -> MissingTextFieldError:
        """The file-level error for a text_field that the source does not have."""
        text_field = self._options.text_field
        return MissingTextFieldError(self._source_name, text_field, available_fields)

    def _check_text_field_once(self, row: Mapping[Any, Any]) -> None:
        """A missing text_field in the first row is almost always a wrong text_field setting,
        so fail the source once instead of failing every row."""
        if self._text_field_checked:
            return
        if self._options.text_field not in row:
            raise self.missing_text_field_error(row)
        self._text_field_checked = True

    def _text(self, position: int, row: Mapping[Any, Any]) -> str:
        """The row's text field, which must be a string."""
        text_field = self._options.text_field
        text = row.get(text_field)
        if text is None:
            raise self._row_error(position, f"missing or null field {text_field!r}")
        if not isinstance(text, str):
            message = f"field {text_field!r} is {type(text).__name__}, expected str"
            raise self._row_error(position, message)
        return text

    def _lang(self, position: int, row: Mapping[Any, Any]) -> str | None:
        """The row's lang_field if set and non-empty, else the default language."""
        lang_field = self._options.lang_field
        if not lang_field:
            return self._options.lang

        value = row.get(lang_field)
        if value in (None, ""):
            return self._options.lang
        if not isinstance(value, str):
            message = f"field {lang_field!r} is {type(value).__name__}, expected str"
            raise self._row_error(position, message)
        return value

    def _metadata(self, row: Mapping[Any, Any]) -> dict[Any, Any]:
        """The fields chosen by metadata_fields, never including the text field."""
        text_field = self._options.text_field
        metadata_fields = self._options.metadata_fields

        if metadata_fields == "all":
            return {
                key: value for key, value in row.items() if key != text_field and key is not None
            }
        if metadata_fields:
            return {key: row[key] for key in metadata_fields if key in row and key != text_field}
        return {}

    def _row_error(self, position: int, message: str) -> ReadError:
        """A row-level error at `position` of this source."""
        return ReadError(self._source_name, message, position)


# ---------------------------------------------------------------- one source during a pass


class Source:
    """One named source during a pass: builds its Documents and records its errors."""

    def __init__(self, name: str, options: ReadOptions, report: PassReport):
        """Bind the source name, its read options and the report of the current pass."""
        self.name = name
        self.options = options
        self.report = report
        self._row_parser = RowParser(name, options)

    def document(self, text: str, position: int) -> Document:
        """A Document from this source in the default language, without metadata."""
        return Document(text=text, source=self.name, position=position, lang=self.options.lang)

    def documents_from_items(self, items: Iterable[Any]) -> Iterator[Document]:
        """Documents pass through, strings become Documents, anything else is a row."""
        for position, item in enumerate(items):
            if isinstance(item, Document):
                yield item
            elif isinstance(item, str):
                yield self.document(item, position)
            else:
                yield from self.documents_from_rows([(position, item)])

    def documents_from_rows(self, rows: Iterable[tuple[int, Any]]) -> Iterator[Document]:
        """Convert numbered rows to Documents; bad rows are recorded and skipped."""
        for position, row in rows:
            try:
                document = self._row_parser.parse(position, row)
            except ReadError as error:
                if error.level is ErrorLevel.FILE:
                    raise
                self.report.record_error(error)
                continue
            yield document

    def record_row_error(self, message: str, position: int) -> None:
        """Record one bad row of this source."""
        self.report.record_error(ReadError(self.name, message, position))

    def missing_text_field_error(self, available_fields: Iterable[Any]) -> MissingTextFieldError:
        """The file-level error for a text_field that this source does not have."""
        return self._row_parser.missing_text_field_error(available_fields)

    def guard(self, documents: Iterator[Document]) -> Iterator[Document]:
        """Record a whole-source failure instead of ending the pass; warn once about bad rows."""
        first_new_error = len(self.report.errors)
        try:
            yield from documents
        except ReadError as error:
            if error.level is not ErrorLevel.FILE:  # row errors only propagate with on_error=raise
                raise
            self.report.record_error(error)
        except TokenizerLabError:
            raise  # e.g. a missing optional dependency: a setup problem, not a bad file
        except Exception as error:  # noqa: BLE001 - unreadable/corrupt file: recorded, not silenced
            message = f"{type(error).__name__}: {error}"
            self.report.record_error(ReadError(self.name, message, level=ErrorLevel.FILE))
        new_errors = self.report.errors[first_new_error:]
        bad_rows = [error for error in new_errors if error.level is ErrorLevel.ROW]
        self.report.policy.summarize_bad_rows(self.name, bad_rows)
