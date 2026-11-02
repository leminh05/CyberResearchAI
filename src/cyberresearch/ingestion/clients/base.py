"""Common interface implemented by every source client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from cyberresearch.ingestion.http import HttpClient
from cyberresearch.ingestion.schema import PaperRecord


@dataclass
class Page:
    """One page of search results.

    ``next_cursor`` is an opaque string (offset, cursor or token, depending on
    the API) that is saved in the checkpoint; ``None`` means no more pages.
    """

    records: list[PaperRecord]
    next_cursor: str | None
    total: int | None = None
    skipped: int = 0  # raw items dropped because they had no usable metadata
    raw_count: int = 0  # items returned by the API before validation


class SourceClient(Protocol):
    """What the pipeline needs from a source: fetch one page for a time window."""

    name: str
    page_size: int

    def fetch_page(self, query: str, start: date, end: date, cursor: str | None) -> Page:
        """Fetch records published between ``start`` and ``end`` (inclusive)."""
        ...


def user_agent(contact_email: str | None) -> str:
    """User-Agent identifying the project, as requested by API providers."""
    contact = f"; mailto:{contact_email}" if contact_email else ""
    return f"CyberResearchAI/0.1 (university research project{contact})"


class BaseClient:
    """Shared constructor for the concrete clients."""

    name = "base"

    def __init__(self, http: HttpClient, base_url: str, page_size: int) -> None:
        self.http = http
        self.base_url = base_url
        self.page_size = page_size
