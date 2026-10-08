"""Human-readable views of CorpusStats: aligned text for print() and HTML for notebooks.

The views only present the numbers. CorpusStats itself (its fields, repr() and to_dict())
stays the programmatic interface. Both views are built from the same tables, so they
always show the same thing. Text output uses only ASCII symbols, so printing works on any
console encoding.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizerlab.data.stats.corpus_stats import CorpusStats, Counts, StepCounts

MAX_ROWS = 20  # longer tables end with "... and N more" instead of flooding the screen
NO_VALUE = "-"
_BYTE_UNITS = ("kB", "MB", "GB", "TB", "PB")  # decimal (SI) units, as storage is sold
_COLUMN_GAP = "   "
_INDENT = "  "


class Align(Enum):
    """How the value columns of a table are aligned (the first column is always left)."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class Table:
    """A titled table of already-formatted cells; the first column labels each row."""

    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str = ""  # shown under the rows, e.g. "none" or "... and 12 more"
    values: Align = Align.RIGHT


# ---------------------------------------------------------------- formatting (pure)


def human_bytes(size: int) -> str:
    """A byte count in decimal units: "730 B", "4.7 kB", "5.2 GB"."""
    if size < 1000:
        return f"{size} B"
    value = size / 1000
    unit = 0
    while round(value, 1) >= 1000 and unit < len(_BYTE_UNITS) - 1:
        value /= 1000
        unit += 1
    return f"{value:.1f} {_BYTE_UNITS[unit]}"


def _bytes_with_exact(size: int) -> str:
    """A readable size, followed by the exact byte count once units are rounded."""
    if size < 1000:
        return human_bytes(size)
    return f"{human_bytes(size)} ({size:,} bytes)"


def _share(part: int, whole: int) -> str:
    """part / whole as a percentage, or NO_VALUE when there is no whole."""
    return f"{part / whole:.1%}" if whole else NO_VALUE


def _length_summary(stats: CorpusStats) -> str:
    """Shortest, mean and longest document length, or NO_VALUE for an empty corpus."""
    if stats.length_min is None or stats.length_max is None or stats.length_mean is None:
        return NO_VALUE
    return (
        f"min {stats.length_min:,}, mean {stats.length_mean:,.1f}, "
        f"max {stats.length_max:,} characters"
    )


# ---------------------------------------------------------------- tables (pure)


def stats_tables(stats: CorpusStats) -> list[Table]:
    """Every section of the summary, in reading order."""
    return [
        _totals_table(stats),
        _counts_table("By language", "lang", stats.by_lang, stats.bytes),
        _counts_table("By source", "source", stats.by_source, stats.bytes),
        _steps_table(stats.steps),
    ]


def summary_line(stats: CorpusStats) -> str:
    """The one-line headline: how many documents and how much text."""
    return f"CorpusStats: {stats.documents:,} documents, {human_bytes(stats.bytes)}"


def _totals_table(stats: CorpusStats) -> Table:
    """The corpus-wide totals as label/value rows."""
    rows = (
        ("documents", f"{stats.documents:,}"),
        ("characters", f"{stats.chars:,}"),
        ("bytes", _bytes_with_exact(stats.bytes)),
        ("length", _length_summary(stats)),
        ("fingerprint", stats.fingerprint),
    )
    return Table("Totals", (), rows, values=Align.LEFT)


def _counts_table(
    title: str,
    key_header: str,
    counts: Mapping[str, Counts],
    total_bytes: int,
) -> Table:
    """Documents, bytes and byte share per key, largest first, capped at MAX_ROWS."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1].bytes, item[0]))
    rows = tuple(
        (key, f"{entry.documents:,}", human_bytes(entry.bytes), _share(entry.bytes, total_bytes))
        for key, entry in ordered[:MAX_ROWS]
    )
    return Table(title, (key_header, "documents", "bytes", "share"), rows, _note(len(ordered)))


def _steps_table(steps: Sequence[StepCounts]) -> Table:
    """What each step received and kept."""
    rows = tuple(
        (
            step.op,
            f"{step.docs_in:,}",
            f"{step.docs_out:,}",
            _share(step.docs_out, step.docs_in),
            human_bytes(step.bytes_in),
            human_bytes(step.bytes_out),
        )
        for step in steps
    )
    headers = ("step", "docs in", "docs out", "kept", "bytes in", "bytes out")
    return Table("Steps", headers, rows, "none" if not rows else "")


def _note(total_rows: int) -> str:
    """ "none" for an empty table, "... and N more" when rows were cut, else nothing."""
    if total_rows == 0:
        return "none"
    hidden = total_rows - MAX_ROWS
    return f"... and {hidden:,} more" if hidden > 0 else ""


# ---------------------------------------------------------------- text view


def render_text(stats: CorpusStats) -> str:
    """Aligned plain-text tables, for print(stats)."""
    sections = [_text_table(table) for table in stats_tables(stats)]
    return "\n\n".join([summary_line(stats), *sections])


def _text_table(table: Table) -> str:
    """One table as an indented block of aligned columns."""
    grid = ([table.headers] if table.headers else []) + list(table.rows)
    widths = (
        [max(len(row[column]) for row in grid) for column in range(len(grid[0]))] if grid else []
    )
    lines = [table.title]
    lines += [_INDENT + _text_row(row, widths, table.values) for row in grid]
    if table.note:
        lines.append(_INDENT + table.note)
    return "\n".join(lines)


def _text_row(row: tuple[str, ...], widths: list[int], values: Align) -> str:
    """One row: the label left-aligned, the values aligned as the table asks."""
    cells = [row[0].ljust(widths[0])]
    for cell, width in zip(row[1:], widths[1:], strict=True):
        cells.append(cell.rjust(width) if values is Align.RIGHT else cell.ljust(width))
    return _COLUMN_GAP.join(cells).rstrip()


# ---------------------------------------------------------------- HTML view


def render_html(stats: CorpusStats) -> str:
    """HTML tables for Jupyter and other rich displays; every value is escaped."""
    tables = "".join(_html_table(table) for table in stats_tables(stats))
    headline = html.escape(summary_line(stats))
    return (
        f'<div class="tokenizerlab-corpus-stats"><p><strong>{headline}</strong></p>{tables}</div>'
    )


def _html_table(table: Table) -> str:
    """One table, with a caption, an optional header row and an optional note row."""
    columns = len(table.headers) or (len(table.rows[0]) if table.rows else 1)
    head = ""
    if table.headers:
        head = "<thead>" + _html_row(table.headers, table.values, cell="th") + "</thead>"
    body = "".join(_html_row(row, table.values, cell="td") for row in table.rows)
    if table.note:
        body += f'<tr><td colspan="{columns}"><em>{html.escape(table.note)}</em></td></tr>'
    caption = (
        f'<caption style="text-align:left"><strong>{html.escape(table.title)}</strong></caption>'
    )
    return f'<table style="margin-bottom:1em">{caption}{head}<tbody>{body}</tbody></table>'


def _html_row(row: tuple[str, ...], values: Align, cell: str) -> str:
    """One table row; the first cell is left-aligned, the rest as the table asks."""
    aligns = ["left"] + [values.value] * (len(row) - 1)
    cells = "".join(
        f'<{cell} style="text-align:{align}">{html.escape(text)}</{cell}>'
        for text, align in zip(row, aligns, strict=True)
    )
    return f"<tr>{cells}</tr>"
