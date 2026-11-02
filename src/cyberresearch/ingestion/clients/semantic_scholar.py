"""Semantic Scholar Academic Graph API client (JSON, bulk search).

Docs: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_bulk_search
Limits: 1 request/second with an API key; the keyless public pool is shared
and frequently returns HTTP 429 (handled by retries with backoff).
Pagination: continuation ``token``; each page holds up to 1000 papers.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cyberresearch.ingestion.clients.base import BaseClient, Page
from cyberresearch.ingestion.normalize import (
    clean_text,
    normalize_arxiv_id,
    normalize_doi,
    parse_date,
    parse_year,
)
from cyberresearch.ingestion.schema import PaperRecord

FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "publicationDate",
        "authors",
        "isOpenAccess",
        "openAccessPdf",
        "fieldsOfStudy",
    ]
)


class SemanticScholarClient(BaseClient):
    """Fetch papers from the Semantic Scholar bulk search endpoint."""

    name = "semantic_scholar"

    def build_params(self, query: str, start: date, end: date, cursor: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": query,
            "fields": FIELDS,
            "publicationDateOrYear": f"{start.isoformat()}:{end.isoformat()}",
            "fieldsOfStudy": "Computer Science",
        }
        if cursor:
            params["token"] = cursor
        return params

    def fetch_page(self, query: str, start: date, end: date, cursor: str | None) -> Page:
        response = self.http.get(self.base_url, params=self.build_params(query, start, end, cursor))
        return self.parse_response(response.json())

    def parse_response(self, payload: dict[str, Any]) -> Page:
        items = payload.get("data") or []
        records: list[PaperRecord] = []
        skipped = 0
        for item in items:
            record = self.parse_paper(item)
            if record and record.is_valid():
                records.append(record)
            else:
                skipped += 1
        next_cursor = payload.get("token") if items else None
        return Page(records=records, next_cursor=next_cursor, total=payload.get("total"), skipped=skipped, raw_count=len(items))

    def parse_paper(self, item: dict[str, Any]) -> PaperRecord | None:
        """Convert one Semantic Scholar paper object into a PaperRecord."""
        paper_id = clean_text(item.get("paperId"))
        if not paper_id:
            return None
        external = item.get("externalIds") or {}
        oa_pdf = item.get("openAccessPdf") or {}
        authors = [name for a in item.get("authors") or [] if (name := clean_text((a or {}).get("name")))]
        return PaperRecord(
            source=self.name,
            source_id=paper_id,
            title=clean_text(item.get("title")),
            abstract=clean_text(item.get("abstract")),
            authors=authors,
            year=parse_year(item.get("year")),
            publication_date=parse_date(item.get("publicationDate")),
            doi=normalize_doi(external.get("DOI")),
            arxiv_id=normalize_arxiv_id(external.get("ArXiv")),
            s2_paper_id=paper_id,
            venue=clean_text(item.get("venue")),
            url=item.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}",
            pdf_url=oa_pdf.get("url") or None,
            is_open_access=item.get("isOpenAccess"),
            categories=[c for c in item.get("fieldsOfStudy") or [] if c],
            raw=item,
        )
