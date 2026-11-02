"""Deduplication rules shared by the in-memory deduplicator and the database.

The same paper often appears in several sources (an arXiv preprint is also in
OpenAlex and Semantic Scholar). We decide that two records describe the same
paper by checking identifiers in priority order:

    1. arXiv ID       2. DOI       3. Semantic Scholar ID
    4. OpenAlex ID    5. normalized title + publication year

Identifiers are exact; title+year is a fuzzy fallback, so it is rejected when
the two records carry *different* DOIs or arXiv IDs (they are then genuinely
different papers that happen to share a title).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from cyberresearch.ingestion.schema import PaperRecord

# Paper columns that hold a unique identifier, in matching priority.
ID_FIELDS = ("arxiv_id", "doi", "s2_paper_id", "openalex_id")

# Descriptive columns filled from later sources when the paper lacks them.
FILLABLE_FIELDS = (
    "title",
    "abstract",
    "authors",
    "year",
    "publication_date",
    "venue",
    "url",
    "pdf_url",
    "is_open_access",
)


def record_identity(record: PaperRecord) -> dict[str, Any]:
    """Return the matching values of a record (identifiers + title/year)."""
    identity: dict[str, Any] = {name: getattr(record, name) for name in ID_FIELDS}
    identity["normalized_title"] = record.normalized_title
    identity["year"] = record.year
    return identity


def match_reason(paper: Mapping[str, Any], record: PaperRecord) -> str | None:
    """Explain why ``record`` is the same paper as ``paper``, or return None.

    ``paper`` is a dict with the papers-table columns (from the DB or memory).
    """
    for name in ID_FIELDS:
        value = getattr(record, name)
        if value and paper.get(name) == value:
            return name

    title = record.normalized_title
    if title and record.year and paper.get("normalized_title") == title and paper.get("year") == record.year:
        for name in ("doi", "arxiv_id"):
            mine, theirs = getattr(record, name), paper.get(name)
            if mine and theirs and mine != theirs:
                return None  # same title but provably different papers
        return "title_year"
    return None


def find_match(candidates: list[Mapping[str, Any]], record: PaperRecord) -> tuple[Mapping[str, Any], str] | None:
    """Pick the best candidate paper for a record, by identifier priority."""
    priority = {name: i for i, name in enumerate((*ID_FIELDS, "title_year"))}
    best: tuple[Mapping[str, Any], str] | None = None
    for paper in candidates:
        reason = match_reason(paper, record)
        if reason and (best is None or priority[reason] < priority[best[1]]):
            best = (paper, reason)
    return best


def fill_missing(paper: Mapping[str, Any], record: PaperRecord) -> dict[str, Any]:
    """Return the fields the paper is missing that the record can provide.

    Existing values are never overwritten: the first source that supplied a
    value keeps it, which makes results reproducible across runs.
    """
    updates: dict[str, Any] = {}
    for name in (*FILLABLE_FIELDS, *ID_FIELDS):
        current = paper.get(name)
        value = getattr(record, name)
        if current in (None, "", []) and value not in (None, "", []):
            updates[name] = value
    if "title" in updates:
        updates["normalized_title"] = record.normalized_title
    return updates


def paper_from_record(record: PaperRecord) -> dict[str, Any]:
    """Build a new papers-table row from the first record of a paper."""
    row: dict[str, Any] = {name: getattr(record, name) for name in (*FILLABLE_FIELDS, *ID_FIELDS)}
    row["normalized_title"] = record.normalized_title
    return row


@dataclass
class MergedPaper:
    """A deduplicated paper and every source record that describes it."""

    fields: dict[str, Any]
    records: list[PaperRecord] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        return sorted({r.source for r in self.records})


def deduplicate(records: list[PaperRecord]) -> list[MergedPaper]:
    """Group records that describe the same paper (pure in-memory version).

    The database store applies exactly the same rules via ``find_match`` and
    ``fill_missing``; this function is used in tests and benchmarks.
    """
    papers: list[MergedPaper] = []
    for record in records:
        if not record.is_valid():
            continue
        match = find_match([p.fields for p in papers], record)
        if match is None:
            papers.append(MergedPaper(fields=paper_from_record(record), records=[record]))
            continue
        merged = next(p for p in papers if p.fields is match[0])
        merged.fields.update(fill_missing(merged.fields, record))
        merged.records.append(record)
    return papers
