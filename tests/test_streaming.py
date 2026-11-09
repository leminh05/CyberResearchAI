"""Streaming rules that need no broker; Spark with Kafka is covered by the live run in Docker."""

from datetime import date

import psycopg
import pytest

from cyberresearch.bigdata.streaming import KafkaStore, parse_message, store_messages, to_message
from cyberresearch.config import IngestionSettings, Topic
from cyberresearch.ingestion.clients.base import Page
from cyberresearch.ingestion.pipeline import IngestionPipeline
from cyberresearch.ingestion.schema import PaperRecord

RECORD = PaperRecord(source="arxiv", source_id="2401.00001", title="Graph IDS", abstract="We study IDS.",
                     authors=["Ann Lee"], year=2024, publication_date=date(2024, 1, 2), arxiv_id="2401.00001",
                     raw={"note": "kept"})


class Future:
    def __init__(self, error=None):
        self.error = error

    def get(self, timeout=None):
        if self.error:
            raise self.error


class Producer:
    def __init__(self, error=None):
        self.error = error
        self.sent = []
        self.flushes = 0

    def send(self, topic, key, value):
        self.sent.append((topic, key, value))
        return Future(self.error)

    def flush(self):
        self.flushes += 1


class Client:
    name = "arxiv"
    page_size = 10

    def fetch_page(self, query, start, end, cursor):
        return Page(records=[RECORD], next_cursor=None, raw_count=1)


def run_producer(store, producer):
    kafka_store = KafkaStore(store.conn, producer, "papers.raw")
    pipeline = IngestionPipeline(kafka_store, {"arxiv": Client()}, [Topic("ids", "IDS", {"arxiv": "q"})],
                                 IngestionSettings(), today=date(2026, 11, 7))
    return pipeline.run_daily()[0]


def test_message_round_trip_keeps_content_hash() -> None:
    topic, record = parse_message(to_message(RECORD, "ids"))
    assert topic == "ids" and record == RECORD and record.content_hash() == RECORD.content_hash()


@pytest.mark.parametrize("value", [b"not json", b"[1]", b"7", b'{"topic": "x"}', b'{"record": {"unknown": 1}}',
                                   b'{"record": {"source": "a", "source_id": ""}}',
                                   b'{"record": {"source": "a", "source_id": "1", "title": "t", "publication_date": "soon"}}'])
def test_bad_messages_are_rejected(value) -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        parse_message(value)


@pytest.mark.db
def test_replayed_batch_changes_nothing_and_bad_data_is_kept_aside(store) -> None:
    good = to_message(RECORD, None)
    counts, rejected = store_messages([good, b"not json"], store)
    assert counts == {"inserted": 1} and len(rejected) == 1 and rejected[0]["value"] == "not json"
    counts, rejected = store_messages([good], store)
    assert counts == {"unchanged": 1} and not rejected
    assert store.conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"] == 1


@pytest.mark.db
def test_database_outage_fails_the_batch_instead_of_rejecting(store, monkeypatch) -> None:
    def down(*args, **kwargs):
        raise psycopg.OperationalError("connection lost")

    monkeypatch.setattr(store, "upsert_record", down)
    with pytest.raises(psycopg.OperationalError):
        store_messages([to_message(RECORD, None)], store)


@pytest.mark.db
def test_producer_sends_and_completes_the_window(store) -> None:
    producer = Producer()
    result = run_producer(store, producer)
    assert result.status == "completed" and producer.flushes >= 1
    assert [(t, k) for t, k, _ in producer.sent] == [("papers.raw", b"arxiv:2401.00001")]
    assert store.conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"] == 0  # consumer's job


@pytest.mark.db
def test_failed_send_does_not_complete_the_window(store) -> None:
    result = run_producer(store, Producer(error=RuntimeError("broker gone")))
    assert result.status == "failed" and "broker gone" in result.error
    done = store.conn.execute("SELECT count(*) AS n FROM ingestion_checkpoints WHERE status = 'completed'").fetchone()
    assert done["n"] == 0


def test_stream_produce_backfills_a_range_once_or_takes_new_papers():
    import argparse

    from cyberresearch.processing.cli import add_processing_commands, produce_round

    class Pipeline:
        def run_backfill(self, start, end, max_records):
            return [("backfill", start, end, max_records)]

        def run_daily(self, max_records):
            return [("daily", max_records)]

    parser = argparse.ArgumentParser()
    add_processing_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["stream-produce", "--start", "2024-01-01", "--end", "2024-06-30", "--max-records", "500"])
    assert produce_round(Pipeline(), args) == [("backfill", date(2024, 1, 1), date(2024, 6, 30), 500)]
    args = parser.parse_args(["stream-produce", "--start", "2024-01-01"])
    assert produce_round(Pipeline(), args)[0][2] == date.today()
    assert produce_round(Pipeline(), parser.parse_args(["stream-produce"])) == [("daily", None)]
