"""OpenAlex API client (JSON).

Docs: https://docs.openalex.org/
Limits: up to 10 requests/second; sending ``mailto`` joins the polite pool.
Pagination (verified 2026-11): cursor paging past the first page requires an
API key (HTTP 401 otherwise), so we use
  * with ``OPENALEX_API_KEY``: cursor paging (``cursor=*``), unlimited depth;
  * without a key: page-number paging, stored as ``"page:N"``, which OpenAlex
    allows for the first 10,000 results of a query.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from cyberresearch.ingestion.clients.base import BaseClient, Page
from cyberresearch.ingestion.http import HttpClient
from cyberresearch.ingestion.normalize import (
    clean_text,
    normalize_arxiv_id,
    normalize_doi,
    parse_date,
    parse_year,
    reconstruct_abstract,
)
from cyberresearch.ingestion.schema import PaperRecord

logger = logging.getLogger(__name__)

PAGE_PREFIX = "page:"
MAX_BASIC_PAGING_RESULTS = 10_000
PAPER_TYPES = "article|preprint|review|book-chapter|conference-paper|dissertation"


class OpenAlexClient(BaseClient):
    """Fetch works from OpenAlex with full-text search and date filters."""

    name = "openalex"

    def __init__(
        self,
        http: HttpClient,
        base_url: str,
        page_size: int,
        mailto: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(http, base_url, page_size)
        self.mailto = mailto
        self.api_key = api_key

    def _page_number(self, cursor: str | None) -> int | None:
        """Page number for keyless paging, or None when cursor paging is used."""
        if self.api_key:
            return None
        if cursor and cursor.startswith(PAGE_PREFIX):
            return int(cursor[len(PAGE_PREFIX):])
        if cursor:
            # A cursor saved while a key was configured cannot be used without
            # it; restart the window (stored records will count as unchanged).
            logger.warning("openalex cursor needs an API key; restarting window at page 1", extra={"source": self.name})
        return 1

    def build_params(
        self, query: str, start: date, end: date, cursor: str | None, page_number: int | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "search": query,
            # English papers only: datasets, paratext (front matter) and other
            # non-paper types made up much of the noise in the 2026-11 backfill.
            "filter": (
                f"from_publication_date:{start.isoformat()},to_publication_date:{end.isoformat()}"
                f",language:en,type:{PAPER_TYPES}"
            ),
            "per-page": self.page_size,
        }
        if page_number is None:
            params["cursor"] = cursor or "*"
        else:
            params["page"] = page_number
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key  # redacted by the logging formatter
        return params

    def fetch_page(self, query: str, start: date, end: date, cursor: str | None) -> Page:
        page_number = self._page_number(cursor)
        response = self.http.get(self.base_url, params=self.build_params(query, start, end, cursor, page_number))
        return self.parse_response(response.json(), page_number=page_number)

    def parse_response(self, payload: dict[str, Any], page_number: int | None = None) -> Page:
        """Parse a works response; ``page_number`` is set in keyless paging mode."""
        results = payload.get("results") or []
        meta = payload.get("meta") or {}
        records: list[PaperRecord] = []
        skipped = 0
        for work in results:
            record = self.parse_work(work)
            if record and record.is_valid():
                records.append(record)
            else:
                skipped += 1

        total = meta.get("count")
        if not results:
            next_cursor = None
        elif page_number is None:
            next_cursor = meta.get("next_cursor")
        else:
            seen = page_number * self.page_size
            more = seen < min(total or 0, MAX_BASIC_PAGING_RESULTS)
            next_cursor = f"{PAGE_PREFIX}{page_number + 1}" if more else None
        return Page(records=records, next_cursor=next_cursor, total=total, skipped=skipped, raw_count=len(results))

    def parse_work(self, work: dict[str, Any]) -> PaperRecord | None:
        """Convert one OpenAlex ``Work`` object into a PaperRecord."""
        openalex_url = work.get("id") or ""
        openalex_id = openalex_url.rsplit("/", 1)[-1] or None
        if not openalex_id:
            return None

        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        open_access = work.get("open_access") or {}
        best_oa = work.get("best_oa_location") or {}

        # OpenAlex has no arXiv ID field, but arXiv appears as a location URL.
        arxiv_id = None
        for location in work.get("locations") or []:
            for key in ("landing_page_url", "pdf_url"):
                url = (location or {}).get(key) or ""
                if "arxiv.org/" in url and (arxiv_id := normalize_arxiv_id(url)):
                    break
            if arxiv_id:
                break

        authors = [
            name
            for authorship in work.get("authorships") or []
            if (name := clean_text(((authorship or {}).get("author") or {}).get("display_name")))
        ]
        categories = [
            name for t in (work.get("topics") or [])[:5] if (name := clean_text((t or {}).get("display_name")))
        ]
        return PaperRecord(
            source=self.name,
            source_id=openalex_id,
            title=clean_text(work.get("title") or work.get("display_name")),
            abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
            authors=authors,
            year=parse_year(work.get("publication_year")),
            publication_date=parse_date(work.get("publication_date")),
            doi=normalize_doi(work.get("doi")),
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
            venue=clean_text(source.get("display_name")),
            url=primary.get("landing_page_url") or openalex_url,
            pdf_url=best_oa.get("pdf_url") or None,
            is_open_access=open_access.get("is_oa"),
            categories=categories,
            raw=work,
        )
