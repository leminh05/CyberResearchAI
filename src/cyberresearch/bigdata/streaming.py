"""Streaming ingestion: crawler -> Kafka -> Spark Structured Streaming -> PostgreSQL.

The producer reuses ``IngestionPipeline`` (windows, resume, logs) with a store
that sends records to Kafka instead of saving them. The consumer stores each
micro-batch with the normal deduplicating upsert, so a batch replayed after a
crash changes nothing, then embeds the new papers' abstracts.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Callable

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from cyberresearch.ingestion.schema import PaperRecord
from cyberresearch.ingestion.storage import Outcome, PostgresStore


def to_message(record: PaperRecord, topic: str | None) -> bytes:
    return json.dumps({"topic": topic, "record": asdict(record)}, default=str, ensure_ascii=False).encode("utf-8")


def parse_message(value: bytes) -> tuple[str | None, PaperRecord]:
    """Validate one message. Raises ValueError, TypeError or KeyError when it is unusable."""
    data = json.loads(value)
    fields = dict(data["record"])
    if fields.get("publication_date"):
        fields["publication_date"] = date.fromisoformat(fields["publication_date"])
    record = PaperRecord(**fields)
    if not record.is_valid():
        raise ValueError("record has no source id, title or identifier")
    return data.get("topic"), record


class KafkaStore(PostgresStore):
    """Sends records to Kafka; checkpoints and run logs stay in PostgreSQL."""

    def __init__(self, conn: psycopg.Connection, producer: Any, kafka_topic: str) -> None:
        super().__init__(conn)
        self.producer = producer
        self.kafka_topic = kafka_topic
        self._pending: list[Any] = []

    def upsert_record(self, record: PaperRecord, topic: str | None = None) -> Outcome:
        key = f"{record.source}:{record.source_id}".encode("utf-8")
        self._pending.append(self.producer.send(self.kafka_topic, key=key, value=to_message(record, topic)))
        return Outcome.INSERTED  # counted as "inserted", meaning accepted for sending

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        # A cursor must not move past records the broker has not confirmed:
        # a failed send raises here, so the page is fetched again on the next run.
        self.producer.flush()
        pending, self._pending = self._pending, []
        for future in pending:
            future.get(timeout=60)
        super().save_checkpoint(*args, **kwargs)


def store_messages(values: list[bytes], store: PostgresStore) -> tuple[Counter, list[dict[str, str]]]:
    """Upsert a batch. Returns (outcome counts, rejected messages with a reason)."""
    counts: Counter = Counter()
    rejected: list[dict[str, str]] = []

    def reject(value: bytes | None, reason: Exception) -> None:
        rejected.append({"value": (value or b"").decode("utf-8", "replace"), "reason": f"{type(reason).__name__}: {reason}"})

    for value in values:
        try:
            topic, record = parse_message(value)
        except (ValueError, TypeError, KeyError) as exc:
            reject(value, exc)
            continue
        try:
            with store.transaction():
                counts[store.upsert_record(record, topic).value] += 1
        except psycopg.OperationalError:
            raise  # database down: fail the batch so Spark replays it, do not reject good data
        except Exception as exc:
            reject(value, exc)
    return counts, rejected


def ensure_database(url: str) -> None:
    """Create the database in ``url`` if it does not exist yet."""
    params = conninfo_to_dict(url)
    name = params["dbname"]
    with psycopg.connect(make_conninfo(**{**params, "dbname": "postgres"}), autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def consume(
    spark: Any,
    bootstrap_servers: str,
    kafka_topic: str,
    lake_root: str,
    store: PostgresStore,
    pipeline: Any,
    max_offsets_per_trigger: int,
    trigger_seconds: int | None,
    report: Callable[[dict[str, Any]], None] = print,
) -> None:
    """Run the streaming query. ``trigger_seconds=None`` handles the backlog, then stops."""
    root = lake_root.rstrip("/")

    def handle_batch(df: Any, batch_id: int) -> None:
        # The batch is bounded by maxOffsetsPerTrigger, and the embedding model lives in the driver.
        rows = df.select("value", "timestamp").collect()
        if not rows:
            return
        # One directory per batch id: a replayed batch overwrites itself instead of duplicating.
        df.write.mode("overwrite").parquet(f"{root}/bronze/papers_stream/batch_id={batch_id}")
        counts, rejected = store_messages([row["value"] for row in rows], store)
        if rejected:
            spark.createDataFrame(rejected, "value string, reason string").write.mode("overwrite").parquet(
                f"{root}/rejected/papers_stream/batch_id={batch_id}"
            )
        processed = pipeline.run()
        report({
            "batch": batch_id,
            "messages": len(rows),
            "stored": dict(counts),
            "rejected": len(rejected),
            "papers_processed": processed["papers"],
            "chunks": processed["chunks"],
            "processing_errors": processed["errors"],
            "max_seconds_since_sent": round((datetime.now() - min(row["timestamp"] for row in rows)).total_seconds(), 1),
        })

    writer = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", kafka_topic)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", max_offsets_per_trigger)
        .load()
        .selectExpr("value", "key", "topic", "partition", "offset", "timestamp")
        .writeStream.foreachBatch(handle_batch)
        .option("checkpointLocation", f"{root}/streaming/papers_stream_checkpoint")
    )
    trigger = {"availableNow": True} if trigger_seconds is None else {"processingTime": f"{trigger_seconds} seconds"}
    writer.trigger(**trigger).start().awaitTermination()
