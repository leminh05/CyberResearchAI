"""Test doubles: a fake HTTP session and helpers to load fixture data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cyberresearch.config import HttpSettings
from cyberresearch.ingestion.clients import ArxivClient, OpenAlexClient, SemanticScholarClient
from cyberresearch.ingestion.http import HttpClient
from cyberresearch.ingestion.schema import PaperRecord

FIXTURES = Path(__file__).parent / "fixtures"
FAST_HTTP = HttpSettings(timeout_seconds=5, max_retries=3, backoff_base_seconds=1.0, backoff_max_seconds=10.0)


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "", json_data: Any = None, headers: dict | None = None):
        self.status_code = status_code
        self.text = text if json_data is None else json.dumps(json_data)
        self.content = self.text.encode("utf-8")
        self._json = json_data
        self.headers = headers or {}

    def json(self) -> Any:
        return self._json if self._json is not None else json.loads(self.text)


class FakeSession:
    """Returns (or raises) queued items in order and records every call."""

    def __init__(self, *items: FakeResponse | Exception):
        self.items = list(items)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params=None, headers=None, timeout=None) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_http(session: FakeSession, source: str = "test", sleeps: list[float] | None = None) -> HttpClient:
    """HttpClient that never really sleeps; sleeps are recorded in ``sleeps``."""
    record = sleeps if sleeps is not None else []
    return HttpClient(source, 0.0, FAST_HTTP, session=session, sleep=record.append, jitter=lambda: 0.0)


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> Any:
    return json.loads(fixture_text(name))


def all_fixture_records() -> list[PaperRecord]:
    """Parse the three fixture files: 3 arXiv + 2 OpenAlex + 3 Semantic Scholar records."""
    arxiv = ArxivClient(None, "", 3).parse_feed(fixture_text("arxiv_feed.xml"), 0).records  # type: ignore[arg-type]
    oa = OpenAlexClient(None, "", 100).parse_response(fixture_json("openalex_works.json")).records  # type: ignore[arg-type]
    s2 = SemanticScholarClient(None, "", 1000).parse_response(fixture_json("s2_bulk.json")).records  # type: ignore[arg-type]
    return arxiv + oa + s2
