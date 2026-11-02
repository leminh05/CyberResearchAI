"""Database persistence: schema, inserts, deduplication, change detection."""

from dataclasses import replace
from datetime import date

import pytest

from cyberresearch.ingestion.schema import PaperRecord
from cyberresearch.ingestion.storage import Outcome, PostgresStore
from fakes import all_fixture_records

pytestmark = pytest.mark.db


def store_all(store: PostgresStore, records: list[PaperRecord], topic: str = "intrusion_detection") -> list[Outcome]:
    return [store.upsert_record(r, topic) for r in records]


def count(store: PostgresStore, table: str) -> int:
    return store.conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]


def test_init_schema_is_idempotent(store: PostgresStore) -> None:
    store.init_schema()
    store.init_schema()
    tables = {
        r["table_name"]
        for r in store.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
        )
    }
    assert {"papers", "paper_sources", "ingestion_logs", "ingestion_checkpoints"} <= tables


def test_cross_source_duplicates_are_merged(store: PostgresStore) -> None:
    outcomes = store_all(store, all_fixture_records())
    assert outcomes.count(Outcome.INSERTED) == 5
    assert outcomes.count(Outcome.MERGED) == 3
    assert count(store, "papers") == 5
    assert count(store, "paper_sources") == 8


def test_metadata_is_stored_correctly(store: PostgresStore) -> None:
    store_all(store, all_fixture_records())
    paper = store.conn.execute("SELECT * FROM papers WHERE arxiv_id = '2401.01234'").fetchone()
    assert paper["title"] == "Deep Learning for Network Intrusion Detection"
    assert paper["doi"] == "10.1109/tifs.2024.0001"
    assert paper["openalex_id"] == "W111"  # filled in from OpenAlex
    assert paper["authors"] == ["Alice Nguyen", "Bob Tran"]
    assert paper["publication_date"] == date(2024, 1, 3)
    assert paper["topics"] == ["intrusion_detection"]

    sources = store.conn.execute(
        "SELECT source, source_id, raw FROM paper_sources WHERE paper_id = %s ORDER BY source",
        (paper["paper_id"],),
    ).fetchall()
    assert [(s["source"], s["source_id"]) for s in sources] == [("arxiv", "2401.01234"), ("openalex", "W111")]
    assert sources[1]["raw"]["cited_by_count"] == 12  # original payload preserved


def test_merge_fills_gaps_from_later_sources(store: PostgresStore) -> None:
    store_all(store, all_fixture_records())
    gnn = store.conn.execute("SELECT * FROM papers WHERE arxiv_id = '2401.05678'").fetchone()
    assert gnn["s2_paper_id"] == "abc123"
    assert gnn["venue"] == "ACM CCS"  # arXiv had no venue; Semantic Scholar did
    fl = store.conn.execute("SELECT * FROM papers WHERE openalex_id = 'W222'").fetchone()
    assert fl["s2_paper_id"] == "def456"
    assert fl["abstract"] == "FL for smart grids."  # OpenAlex had no abstract


def test_rerun_does_not_reprocess_unchanged_records(store: PostgresStore) -> None:
    records = all_fixture_records()
    store_all(store, records)
    before = store.conn.execute("SELECT paper_id, updated_at FROM papers ORDER BY paper_id").fetchall()
    outcomes = store_all(store, records)
    assert set(outcomes) == {Outcome.UNCHANGED}
    after = store.conn.execute("SELECT paper_id, updated_at FROM papers ORDER BY paper_id").fetchall()
    assert before == after
    assert count(store, "papers") == 5


def test_changed_source_record_is_updated(store: PostgresStore) -> None:
    record = PaperRecord(source="openalex", source_id="W9", title="Draft", openalex_id="W9", raw={"v": 1})
    assert store.upsert_record(record) == Outcome.INSERTED
    revised = replace(record, abstract="Now with an abstract", raw={"v": 2})
    assert store.upsert_record(revised) == Outcome.UPDATED
    row = store.conn.execute(
        "SELECT p.abstract, s.raw FROM papers p JOIN paper_sources s USING (paper_id) WHERE s.source_id = 'W9'"
    ).fetchone()
    assert row["abstract"] == "Now with an abstract"
    assert row["raw"] == {"v": 2}


def test_topics_accumulate_without_duplicates(store: PostgresStore) -> None:
    record = PaperRecord(source="arxiv", source_id="2401.00001", arxiv_id="2401.00001", title="T")
    store.upsert_record(record, "network_security")
    store.upsert_record(record, "intrusion_detection")
    store.upsert_record(record, "intrusion_detection")
    topics = store.conn.execute("SELECT topics FROM papers").fetchone()["topics"]
    assert topics == ["network_security", "intrusion_detection"]


def test_record_with_missing_metadata_is_stored(store: PostgresStore) -> None:
    bare = PaperRecord(source="semantic_scholar", source_id="ghi789", s2_paper_id="ghi789", doi="10.5555/xyz")
    assert store.upsert_record(bare) == Outcome.INSERTED
    paper = store.conn.execute("SELECT * FROM papers").fetchone()
    assert paper["title"] is None and paper["authors"] == [] and paper["topics"] == []


def test_conflicting_identifier_is_not_copied(store: PostgresStore) -> None:
    store.upsert_record(PaperRecord(source="openalex", source_id="W1", openalex_id="W1", doi="10.1/a", title="A"))
    store.upsert_record(PaperRecord(source="openalex", source_id="W2", openalex_id="W2", title="B", year=2024))
    # Matches W2 by title+year, but its DOI already belongs to W1's paper.
    tricky = PaperRecord(source="semantic_scholar", source_id="s1", s2_paper_id="s1", doi="10.1/a", title="A")
    assert store.upsert_record(tricky) == Outcome.MERGED  # merged into W1 via DOI
    assert count(store, "papers") == 2
