"""Source API clients and a factory that builds them from settings."""

from __future__ import annotations

from cyberresearch.config import Settings
from cyberresearch.ingestion.clients.arxiv import ArxivClient
from cyberresearch.ingestion.clients.base import Page, SourceClient, user_agent
from cyberresearch.ingestion.clients.openalex import OpenAlexClient
from cyberresearch.ingestion.clients.semantic_scholar import SemanticScholarClient
from cyberresearch.ingestion.http import HttpClient

__all__ = ["ArxivClient", "OpenAlexClient", "Page", "SemanticScholarClient", "SourceClient", "build_clients"]


def build_clients(settings: Settings, only: list[str] | None = None) -> dict[str, SourceClient]:
    """Create one client per enabled source (optionally restricted to ``only``)."""
    headers = {"User-Agent": user_agent(settings.contact_email)}
    clients: dict[str, SourceClient] = {}
    for name, src in settings.sources.items():
        if not src.enabled or (only and name not in only):
            continue
        source_headers = dict(headers)
        if name == "semantic_scholar" and settings.semantic_scholar_api_key:
            source_headers["x-api-key"] = settings.semantic_scholar_api_key
        http = HttpClient(name, src.min_interval_seconds, settings.http, headers=source_headers)
        if name == "arxiv":
            clients[name] = ArxivClient(http, src.base_url, src.page_size)
        elif name == "openalex":
            clients[name] = OpenAlexClient(
                http, src.base_url, src.page_size, mailto=settings.contact_email, api_key=settings.openalex_api_key
            )
        elif name == "semantic_scholar":
            clients[name] = SemanticScholarClient(http, src.base_url, src.page_size)
    return clients
