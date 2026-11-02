"""arXiv API client (Atom XML).

Docs: https://info.arxiv.org/help/api/user-manual.html
Terms: at most one request every three seconds (enforced by the rate limiter).
Pagination: offset-based (``start`` parameter); cursor = offset as a string.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date

from cyberresearch.ingestion.clients.base import BaseClient, Page
from cyberresearch.ingestion.http import HttpError
from cyberresearch.ingestion.normalize import (
    clean_text,
    normalize_arxiv_id,
    normalize_doi,
    parse_date,
)
from cyberresearch.ingestion.schema import PaperRecord

logger = logging.getLogger(__name__)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


class ArxivClient(BaseClient):
    """Fetch paper metadata from the arXiv export API."""

    name = "arxiv"

    def build_params(self, query: str, start: date, end: date, offset: int) -> dict[str, str | int]:
        """Combine the topic query with a submission-date range filter."""
        date_filter = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
        return {
            "search_query": f"({query}) AND {date_filter}",
            "start": offset,
            "max_results": self.page_size,
            # Ascending order keeps offsets stable while new papers arrive.
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }

    def fetch_page(self, query: str, start: date, end: date, cursor: str | None) -> Page:
        offset = int(cursor or 0)
        response = self.http.get(self.base_url, params=self.build_params(query, start, end, offset))
        return self.parse_feed(response.text, offset)

    def parse_feed(self, xml_text: str, offset: int) -> Page:
        """Parse an Atom feed into records and compute the next offset."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise HttpError(f"arxiv: invalid XML response ({exc})") from exc

        total_text = root.findtext("opensearch:totalResults", default="0", namespaces=NS)
        total = int(total_text or 0)
        entries = root.findall("atom:entry", NS)

        records: list[PaperRecord] = []
        skipped = 0
        for entry in entries:
            entry_id = entry.findtext("atom:id", default="", namespaces=NS)
            if "api/errors" in entry_id:  # arXiv reports query errors as an entry
                message = entry.findtext("atom:summary", default="", namespaces=NS)
                raise HttpError(f"arxiv: API error: {clean_text(message)}")
            record = self.parse_entry(entry)
            if record and record.is_valid():
                records.append(record)
            else:
                skipped += 1

        next_offset = offset + len(entries)
        # Stop when the API returns an empty page, even if totalResults says
        # more exist (arXiv occasionally reports inaccurate totals).
        next_cursor = str(next_offset) if entries and next_offset < total else None
        return Page(records=records, next_cursor=next_cursor, total=total, skipped=skipped, raw_count=len(entries))

    def parse_entry(self, entry: ET.Element) -> PaperRecord | None:
        """Convert one Atom ``<entry>`` into a PaperRecord."""
        abs_url = clean_text(entry.findtext("atom:id", namespaces=NS))
        arxiv_id = normalize_arxiv_id(abs_url)
        if not arxiv_id:
            logger.warning("arxiv entry without a valid id skipped", extra={"source": self.name, "id": abs_url})
            return None

        published = parse_date(entry.findtext("atom:published", namespaces=NS))
        pdf_url = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        authors = [
            name
            for author in entry.findall("atom:author", NS)
            if (name := clean_text(author.findtext("atom:name", namespaces=NS)))
        ]
        categories = [c.get("term", "") for c in entry.findall("atom:category", NS) if c.get("term")]
        primary = entry.find("arxiv:primary_category", NS)

        raw = {
            "id": abs_url,
            "title": entry.findtext("atom:title", namespaces=NS),
            "published": entry.findtext("atom:published", namespaces=NS),
            "updated": entry.findtext("atom:updated", namespaces=NS),
            "authors": authors,
            "doi": entry.findtext("arxiv:doi", namespaces=NS),
            "journal_ref": entry.findtext("arxiv:journal_ref", namespaces=NS),
            "primary_category": primary.get("term") if primary is not None else None,
            "categories": categories,
        }
        return PaperRecord(
            source=self.name,
            source_id=arxiv_id,
            title=clean_text(raw["title"]),
            abstract=clean_text(entry.findtext("atom:summary", namespaces=NS)),
            authors=authors,
            year=published.year if published else None,
            publication_date=published,
            doi=normalize_doi(raw["doi"]),
            arxiv_id=arxiv_id,
            venue=clean_text(raw["journal_ref"]),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=pdf_url,
            is_open_access=True,  # every arXiv paper is freely readable
            categories=categories,
            raw=raw,
        )
