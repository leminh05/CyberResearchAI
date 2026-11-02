"""Polite PDF downloader: robots.txt, per-host rate limits, PDF validation."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from cyberresearch.config import HttpSettings, ProcessingSettings
from cyberresearch.ingestion.http import HttpClient, HttpError
from cyberresearch.processing.access import PdfCandidate

logger = logging.getLogger(__name__)

ROBOTS_AGENT = "CyberResearchAI"
ARXIV_HOSTS = ("arxiv.org", "export.arxiv.org")


@dataclass
class DownloadResult:
    """Outcome of one download attempt (stored in ``pdf_documents``)."""

    status: str  # downloaded | robots_disallowed | not_pdf | too_large | failed
    file_path: Path | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    http_status: int | None = None
    error: str | None = None


class PdfDownloader:
    """Download PDFs while respecting robots.txt and per-host rate limits."""

    def __init__(
        self,
        settings: ProcessingSettings,
        http_settings: HttpSettings,
        user_agent: str,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.http_settings = http_settings
        self.headers = {"User-Agent": user_agent}
        self.session = session or requests.Session()
        self._sleep = sleep
        self._clients: dict[str, HttpClient] = {}
        self._robots: dict[str, RobotFileParser | None] = {}

    def _client_for(self, host: str, crawl_delay: float | None) -> HttpClient:
        """One HttpClient (and rate limiter) per host.

        The interval is the larger of our own politeness delay and the
        ``Crawl-delay`` the site requests in robots.txt.
        """
        if host not in self._clients:
            interval = 3.0 if host.endswith(ARXIV_HOSTS) else self.settings.host_min_interval_seconds
            interval = max(interval, crawl_delay or 0.0)
            self._clients[host] = HttpClient(
                host, interval, self.http_settings, headers=self.headers, session=self.session, sleep=self._sleep
            )
        return self._clients[host]

    def crawl_delay(self, url: str) -> float | None:
        """Crawl-delay from robots.txt for this URL's origin (after robots_allowed)."""
        parts = urlsplit(url)
        parser = self._robots.get(f"{parts.scheme}://{parts.netloc}")
        delay = parser.crawl_delay(ROBOTS_AGENT) if parser else None
        return float(delay) if delay else None

    def robots_allowed(self, url: str) -> bool:
        """Check robots.txt (RFC 9309).

        4xx (no robots.txt) -> allowed; 5xx or network error -> treated as
        disallowed, which is the conservative reading of the standard.
        """
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                response = self.session.get(
                    f"{origin}/robots.txt", headers=self.headers, timeout=self.http_settings.timeout_seconds
                )
                if response.status_code >= 500:
                    parser = None
                elif response.status_code >= 400:
                    parser.allow_all = True
                else:
                    parser.parse(response.text.splitlines())
            except requests.RequestException:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is not None and parser.can_fetch(ROBOTS_AGENT, url)

    def download(self, paper_id: int, candidate: PdfCandidate) -> DownloadResult:
        """Download one PDF to ``<pdf_directory>/<paper_id>.pdf``."""
        if not self.robots_allowed(candidate.url):
            return DownloadResult("robots_disallowed", error="disallowed by robots.txt or robots.txt unavailable")

        host = urlsplit(candidate.url).netloc
        try:
            response = self._client_for(host, self.crawl_delay(candidate.url)).get(candidate.url)
        except HttpError as exc:
            return DownloadResult("failed", http_status=exc.status_code, error=str(exc))

        content: bytes = response.content
        max_bytes = int(self.settings.max_pdf_megabytes * 1024 * 1024)
        if len(content) > max_bytes:
            return DownloadResult("too_large", size_bytes=len(content), http_status=response.status_code)
        # Many "PDF" links are HTML landing pages; check the PDF magic bytes.
        if b"%PDF-" not in content[:1024]:
            content_type = response.headers.get("Content-Type", "unknown")
            return DownloadResult("not_pdf", http_status=response.status_code, error=f"content-type {content_type}")

        self.settings.pdf_directory.mkdir(parents=True, exist_ok=True)
        path = self.settings.pdf_directory / f"{paper_id}.pdf"
        path.write_bytes(content)
        logger.info("pdf downloaded", extra={"paper_id": paper_id, "host": host, "bytes": len(content)})
        return DownloadResult(
            "downloaded",
            file_path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            http_status=response.status_code,
        )
