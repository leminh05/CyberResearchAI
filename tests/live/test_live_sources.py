"""Live tests against the real APIs (network required).

Run with:  python -m pytest -m live -v
Skipped by default because they are slow and depend on external services.
"""

from datetime import date, timedelta

import pytest

from cyberresearch.config import load_settings
from cyberresearch.ingestion.clients import build_clients
from cyberresearch.ingestion.pipeline import IngestionPipeline

pytestmark = pytest.mark.live

END = date.today() - timedelta(days=7)
START = END - timedelta(days=60)


@pytest.fixture(scope="module")
def clients():
    return build_clients(load_settings())


@pytest.mark.parametrize("source", ["arxiv", "openalex", "semantic_scholar"])
def test_source_returns_normalized_records(clients, source: str) -> None:
    settings = load_settings()
    topic = next(t for t in settings.topics if t.key == "intrusion_detection")
    page = clients[source].fetch_page(topic.queries[source], START, END, None)
    assert page.records, f"{source} returned no records"
    for record in page.records[:20]:
        assert record.source == source and record.source_id
        assert record.is_valid()
        if record.publication_date:
            assert START - timedelta(days=366) <= record.publication_date <= date.today()
    assert page.next_cursor is None or isinstance(page.next_cursor, str)


@pytest.mark.parametrize("source", ["arxiv", "openalex", "semantic_scholar"])
def test_source_second_page_can_be_fetched(clients, source: str) -> None:
    """Regression: resuming from a saved cursor must work (OpenAlex returned 401)."""
    settings = load_settings()
    topic = next(t for t in settings.topics if t.key == "intrusion_detection")
    client = clients[source]
    first = client.fetch_page(topic.queries[source], START, END, None)
    if first.next_cursor is None:
        pytest.skip(f"{source}: only one page in the test window")
    second = client.fetch_page(topic.queries[source], START, END, first.next_cursor)
    assert second.records
    assert {r.source_id for r in second.records}.isdisjoint({r.source_id for r in first.records})


def test_end_to_end_two_sources_into_database(clients, store) -> None:
    settings = load_settings()
    topics = [t for t in settings.topics if t.key == "intrusion_detection"]
    selected = {name: clients[name] for name in ("arxiv", "openalex")}
    pipeline = IngestionPipeline(store, selected, topics, settings.ingestion)

    first = pipeline.run_backfill(START, END, max_records=50)
    assert all(r.status in ("completed", "partial") for r in first), [(r.source, r.error) for r in first]
    papers = store.conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"]
    assert papers > 0
