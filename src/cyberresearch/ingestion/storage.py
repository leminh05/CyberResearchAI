"""PostgreSQL persistence for papers, source records, logs and checkpoints."""

from __future__ import annotations

import logging
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from cyberresearch.ingestion.dedup import ID_FIELDS, fill_missing, find_match, paper_from_record
from cyberresearch.ingestion.schema import PaperRecord

logger = logging.getLogger(__name__)

SCHEMA_SQL = (Path(__file__).parent / "sql" / "schema.sql").read_text(encoding="utf-8")


class Outcome(str, Enum):
    """What happened to one incoming record."""

    INSERTED = "inserted"    # new paper
    MERGED = "merged"        # new source record linked to an existing paper
    UPDATED = "updated"      # known source record whose metadata changed
    UNCHANGED = "unchanged"  # known source record, identical metadata -> skipped


def _db_value(name: str, value: Any) -> Any:
    """Adapt Python values to column types (JSONB for authors)."""
    return Jsonb(value) if name == "authors" else value


class PostgresStore:
    """All SQL used by the ingestion pipeline lives in this class.

    The connection runs in autocommit mode; ``transaction()`` groups several
    statements (one API page + its checkpoint) into one atomic unit.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    @classmethod
    def connect(cls, database_url: str) -> "PostgresStore":
        # connect_timeout: fail with a clear error instead of hanging forever
        # (on Windows "localhost" resolves to ::1 first, which Docker's
        # 127.0.0.1-only port does not answer).
        return cls(psycopg.connect(database_url, autocommit=True, row_factory=dict_row, connect_timeout=15))

    def close(self) -> None:
        self.conn.close()

    def transaction(self) -> Any:
        return self.conn.transaction()

    def init_schema(self) -> None:
        """Create tables and indexes if they do not exist yet."""
        self.conn.execute(SCHEMA_SQL)

    # ------------------------------------------------------------------ papers

    def upsert_record(self, record: PaperRecord, topic: str | None = None) -> Outcome:
        """Store one source record, deduplicating against existing papers."""
        content_hash = record.content_hash()
        known = self.conn.execute(
            "SELECT paper_id, content_hash FROM paper_sources WHERE source = %s AND source_id = %s",
            (record.source, record.source_id),
        ).fetchone()

        if known and known["content_hash"] == content_hash:
            # Seen before with identical metadata: only record that we saw it.
            self.conn.execute(
                "UPDATE paper_sources SET last_seen_at = now() WHERE source = %s AND source_id = %s",
                (record.source, record.source_id),
            )
            self._add_topic(known["paper_id"], topic)
            return Outcome.UNCHANGED

        if known:
            paper = self._get_paper(known["paper_id"])
            self._merge_into(paper, record, topic)
            self.conn.execute(
                """UPDATE paper_sources
                   SET raw = %s, content_hash = %s, source_url = %s,
                       last_seen_at = now(), last_changed_at = now()
                   WHERE source = %s AND source_id = %s""",
                (Jsonb(record.raw), content_hash, record.url, record.source, record.source_id),
            )
            return Outcome.UPDATED

        match = find_match(self._candidates(record), record)
        if match:
            paper, reason = match
            self._merge_into(paper, record, topic)
            paper_id = paper["paper_id"]
            outcome = Outcome.MERGED
            logger.debug(
                "record merged into existing paper",
                extra={"source": record.source, "source_id": record.source_id, "paper_id": paper_id, "match": reason},
            )
        else:
            paper_id = self._insert_paper(record, topic)
            outcome = Outcome.INSERTED

        self.conn.execute(
            """INSERT INTO paper_sources (paper_id, source, source_id, source_url, content_hash, raw)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (paper_id, record.source, record.source_id, record.url, content_hash, Jsonb(record.raw)),
        )
        return outcome

    def _candidates(self, record: PaperRecord) -> list[dict[str, Any]]:
        """Papers sharing any identifier or the title+year key with the record."""
        return self.conn.execute(
            """SELECT * FROM papers
               WHERE arxiv_id = %(arxiv_id)s OR doi = %(doi)s
                  OR s2_paper_id = %(s2_paper_id)s OR openalex_id = %(openalex_id)s
                  OR (normalized_title = %(title)s AND year = %(year)s)
               ORDER BY paper_id""",
            {
                "arxiv_id": record.arxiv_id,
                "doi": record.doi,
                "s2_paper_id": record.s2_paper_id,
                "openalex_id": record.openalex_id,
                "title": record.normalized_title,
                "year": record.year,
            },
        ).fetchall()

    def _get_paper(self, paper_id: int) -> dict[str, Any]:
        return self.conn.execute("SELECT * FROM papers WHERE paper_id = %s", (paper_id,)).fetchone()

    def _insert_paper(self, record: PaperRecord, topic: str | None) -> int:
        row = paper_from_record(record)
        row["topics"] = [topic] if topic else []
        columns = list(row)
        query = sql.SQL("INSERT INTO papers ({}) VALUES ({}) RETURNING paper_id").format(
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        )
        return self.conn.execute(query, [_db_value(c, row[c]) for c in columns]).fetchone()["paper_id"]

    def _merge_into(self, paper: dict[str, Any], record: PaperRecord, topic: str | None) -> None:
        """Fill the paper's missing fields from the record (never overwrite)."""
        updates = fill_missing(paper, record)
        for name in ID_FIELDS:
            # An identifier may already belong to another paper (rare data
            # conflicts between sources); keep the unique constraint intact.
            if name in updates and self.conn.execute(
                sql.SQL("SELECT 1 FROM papers WHERE {} = %s AND paper_id <> %s").format(sql.Identifier(name)),
                (updates[name], paper["paper_id"]),
            ).fetchone():
                logger.warning(
                    "identifier owned by another paper, not merged",
                    extra={"field": name, "value": updates.pop(name), "paper_id": paper["paper_id"]},
                )
        if updates:
            assignments = sql.SQL(", ").join(
                sql.SQL("{} = %s").format(sql.Identifier(name)) for name in updates
            )
            self.conn.execute(
                sql.SQL("UPDATE papers SET {}, updated_at = now() WHERE paper_id = %s").format(assignments),
                [*(_db_value(n, v) for n, v in updates.items()), paper["paper_id"]],
            )
        self._add_topic(paper["paper_id"], topic)

    def _add_topic(self, paper_id: int, topic: str | None) -> None:
        if topic:
            self.conn.execute(
                "UPDATE papers SET topics = array_append(topics, %s) WHERE paper_id = %s AND NOT (%s = ANY(topics))",
                (topic, paper_id, topic),
            )

    # ------------------------------------------------------------ checkpoints

    def get_checkpoint(self, key: str) -> dict[str, Any] | None:
        return self.conn.execute(
            "SELECT * FROM ingestion_checkpoints WHERE checkpoint_key = %s", (key,)
        ).fetchone()

    def save_checkpoint(
        self,
        key: str,
        *,
        mode: str,
        source: str,
        topic: str,
        window_start: date,
        window_end: date,
        cursor: str | None,
        status: str,
        records_fetched: int,
    ) -> None:
        self.conn.execute(
            """INSERT INTO ingestion_checkpoints
                   (checkpoint_key, mode, source, topic, window_start, window_end, cursor, status, records_fetched)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (checkpoint_key) DO UPDATE
               SET cursor = EXCLUDED.cursor, status = EXCLUDED.status,
                   records_fetched = EXCLUDED.records_fetched, updated_at = now()""",
            (key, mode, source, topic, window_start, window_end, cursor, status, records_fetched),
        )

    def open_checkpoint(self, mode: str, source: str, topic: str) -> dict[str, Any] | None:
        """The oldest unfinished checkpoint for this task, if any."""
        return self.conn.execute(
            """SELECT * FROM ingestion_checkpoints
               WHERE mode = %s AND source = %s AND topic = %s AND status = 'in_progress'
               ORDER BY window_start LIMIT 1""",
            (mode, source, topic),
        ).fetchone()

    def last_completed_window_end(self, mode: str, source: str, topic: str) -> date | None:
        """High-water mark used by incremental (daily) collection."""
        row = self.conn.execute(
            """SELECT max(window_end) AS last_end FROM ingestion_checkpoints
               WHERE mode = %s AND source = %s AND topic = %s AND status = 'completed'""",
            (mode, source, topic),
        ).fetchone()
        return row["last_end"] if row else None

    # ------------------------------------------------------------------- logs

    def start_log(
        self, run_id: UUID, mode: str, source: str, topic: str, window_start: date, window_end: date
    ) -> int:
        return self.conn.execute(
            """INSERT INTO ingestion_logs (run_id, mode, source, topic, window_start, window_end, status)
               VALUES (%s, %s, %s, %s, %s, %s, 'running') RETURNING log_id""",
            (run_id, mode, source, topic, window_start, window_end),
        ).fetchone()["log_id"]

    def finish_log(self, log_id: int, status: str, counts: dict[str, int], error: str | None = None) -> None:
        columns = ("pages", "fetched", "inserted", "merged", "updated", "unchanged", "skipped", "failed")
        assignments = sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(c)) for c in columns)
        self.conn.execute(
            sql.SQL(
                "UPDATE ingestion_logs SET {}, status = %s, error_message = %s, finished_at = now() WHERE log_id = %s"
            ).format(assignments),
            [*(counts.get(c, 0) for c in columns), status, error, log_id],
        )

    # ----------------------------------------------------------------- status

    def summary(self) -> dict[str, Any]:
        """Row counts and per-source totals, for the ``status`` command."""
        one = lambda q: self.conn.execute(q).fetchone()["n"]  # noqa: E731
        return {
            "papers": one("SELECT count(*) AS n FROM papers"),
            "paper_sources": one("SELECT count(*) AS n FROM paper_sources"),
            "multi_source_papers": one(
                "SELECT count(*) AS n FROM (SELECT paper_id FROM paper_sources GROUP BY paper_id HAVING count(*) > 1) t"
            ),
            "by_source": {
                r["source"]: r["n"]
                for r in self.conn.execute("SELECT source, count(*) AS n FROM paper_sources GROUP BY source ORDER BY source")
            },
            "checkpoints": {
                r["status"]: r["n"]
                for r in self.conn.execute(
                    "SELECT status, count(*) AS n FROM ingestion_checkpoints GROUP BY status ORDER BY status"
                )
            },
            "recent_logs": self.conn.execute(
                """SELECT mode, source, topic, window_start, window_end, status, fetched, inserted,
                          merged, updated, unchanged, failed, error_message
                   FROM ingestion_logs ORDER BY log_id DESC LIMIT 10"""
            ).fetchall(),
        }
