"""Rule-based markdown-table → bluebell-TABLE converter.

Used by ``__init__.py``: each rescued ``table_region`` is rendered to a
bluebell ``TABLE`` block via :func:`render_region`. The result is held in
memory and spliced into the section-LLM output at sentinel positions.
"""

from __future__ import annotations

import re
from typing import TypedDict


class ParsedTable(TypedDict):
    headers: list[str]
    rows: list[list[str]]


_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")


def _split_pipe_row(line: str) -> list[str]:
    """Split a `| a | b | c |` row into ``["a", "b", "c"]``.

    Trims whitespace per cell. Drops the leading and trailing empty fields
    that ``str.split("|")`` produces from rows that start and end with ``|``.
    """
    parts = line.strip().split("|")
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return [p.strip() for p in parts]


def _is_separator_row(cells: list[str]) -> bool:
    """True if every cell is a markdown header-separator (``---`` or ``:---:``)."""
    if not cells:
        return False
    return all(_SEPARATOR_CELL_RE.fullmatch(c) for c in cells)


def parse_markdown_tables(md: str) -> list[ParsedTable]:
    """Extract all tables from a markdown blob.

    A "table" is a maximal contiguous run of pipe-delimited lines, optionally
    containing a ``|---|`` separator row that splits header from body. If no
    separator is present, the first row is treated as the header.
    """
    out: list[ParsedTable] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and line.endswith("|"):
            # Start of a pipe-block; collect contiguous pipe-delimited lines.
            block_rows: list[list[str]] = []
            while i < len(lines):
                cur = lines[i].strip()
                if cur.startswith("|") and cur.endswith("|"):
                    block_rows.append(_split_pipe_row(cur))
                    i += 1
                else:
                    break
            table = _block_rows_to_table(block_rows)
            if table is not None:
                out.append(table)
        else:
            i += 1
    return out


def _block_rows_to_table(rows: list[list[str]]) -> ParsedTable | None:
    if not rows:
        return None
    sep_idx = next(
        (j for j, r in enumerate(rows) if _is_separator_row(r)), None
    )
    if sep_idx is None:
        # No separator row → first row is the header, rest is body.
        return {"headers": rows[0], "rows": rows[1:]}
    headers = rows[sep_idx - 1] if sep_idx > 0 else []
    body = rows[sep_idx + 1:]
    return {"headers": headers, "rows": body}


def merge_continuation_tables(tables: list[ParsedTable]) -> list[ParsedTable]:
    """Merge adjacent tables with structurally identical headers.
    """
    if not tables:
        return []
    merged: list[ParsedTable] = [
        {"headers": list(tables[0]["headers"]), "rows": list(tables[0]["rows"])}
    ]
    for t in tables[1:]:
        if t["headers"] == merged[-1]["headers"] and t["headers"]:
            merged[-1]["rows"].extend(t["rows"])
        else:
            merged.append(
                {"headers": list(t["headers"]), "rows": list(t["rows"])}
            )
    return merged


def render_bluebell_table(table: ParsedTable, indent: str = "") -> str:
    """Render one parsed table as a bluebell ``TABLE`` block at ``indent``.

    Empty cells are preserved as empty TC/TH lines (bluebell allows them).
    """
    out: list[str] = [f"{indent}TABLE"]
    if table["headers"]:
        out.append(f"{indent}  TR")
        for h in table["headers"]:
            out.append(f"{indent}    TH")
            out.append(f"{indent}      {h}")
    for row in table["rows"]:
        out.append(f"{indent}  TR")
        for cell in row:
            out.append(f"{indent}    TC")
            out.append(f"{indent}      {cell}")
    return "\n".join(out)


def render_region(markdown: str, indent: str = "") -> str:
    """Convert one rescued region's OCR markdown into bluebell at ``indent``.
    """
    tables = merge_continuation_tables(parse_markdown_tables(markdown))
    if not tables:
        # Defensive fallback — preserve the OCR text rather than dropping it.
        return (
            f"{indent}TABLE\n"
            f"{indent}  TR\n"
            f"{indent}    TC\n"
            + "\n".join(f"{indent}      {ln}" for ln in markdown.splitlines() if ln.strip())
        )
    return ("\n\n").join(
        render_bluebell_table(t, indent=indent) for t in tables
    )
