"""Common metadata schema shared by all sources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from cyberresearch.ingestion.normalize import is_junk_title, is_mostly_english, normalize_title


@dataclass
class PaperRecord:
    """One paper as returned by one source, already normalized.

    ``source`` + ``source_id`` identify where the record came from;
    ``raw`` keeps the original API payload so nothing is lost.
    Any field except ``source``/``source_id`` may be missing (None or empty).
    """

    source: str
    source_id: str
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    publication_date: date | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    s2_paper_id: str | None = None
    openalex_id: str | None = None
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    is_open_access: bool | None = None
    categories: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def normalized_title(self) -> str | None:
        """Title key used for title+year deduplication."""
        return normalize_title(self.title)

    def is_valid(self) -> bool:
        """A record is usable if it has a title or at least one identifier,
        and is an English paper: the embedding model is English-only, and
        exam-dump PDFs and proceedings front matter are not papers."""
        has_identity = bool(self.source_id) and bool(
            self.title or self.doi or self.arxiv_id or self.s2_paper_id or self.openalex_id
        )
        # Title only: abstracts of English papers may quote other scripts.
        return has_identity and is_mostly_english(self.title or "") and not is_junk_title(self.title)

    def content_hash(self) -> str:
        """Hash of the normalized metadata (not ``raw``).

        Used to detect whether a source record changed since the last run.
        ``raw`` is excluded because it contains volatile values such as
        citation counts that would force needless reprocessing.
        """
        data = asdict(self)
        data.pop("raw")
        payload = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
