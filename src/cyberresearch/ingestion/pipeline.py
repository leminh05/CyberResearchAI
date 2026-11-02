"""Ingestion pipeline: daily (incremental) and backfill (historical) modes.

Unit of work = one (mode, source, topic, time window). Its progress is stored
in ``ingestion_checkpoints`` after every API page, in the same database
transaction as the page's records. If the process stops (crash, Ctrl+C,
network failure, record limit), the next run resumes from the saved cursor
without fetching or storing the finished pages again.

* daily    - window = [last completed day - overlap, today]
* backfill - a historical range split into fixed windows (e.g. one year);
             completed windows are skipped on later runs.
"""

from __future__ import annotations

import calendar
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

from cyberresearch.config import IngestionSettings, Topic
from cyberresearch.ingestion.clients.base import SourceClient
from cyberresearch.ingestion.storage import PostgresStore

logger = logging.getLogger(__name__)

DAILY = "daily"
BACKFILL = "backfill"


@dataclass
class TaskResult:
    """Outcome of one (mode, source, topic, window) task."""

    mode: str
    source: str
    topic: str
    window_start: date
    window_end: date
    status: str  # completed | partial | failed | skipped
    counts: Counter = field(default_factory=Counter)
    error: str | None = None


def add_months(day: date, months: int) -> date:
    """Shift a date by whole months, clamping the day (Jan 31 + 1 -> Feb 28/29)."""
    month_index = day.month - 1 + months
    year, month = day.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def split_windows(start: date, end: date, months: int) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive windows of ``months`` months, newest first.

    Newest first, because recent research is usually the most useful when a
    backfill is interrupted part way.
    """
    if start > end:
        raise ValueError("start must be on or before end")
    windows: list[tuple[date, date]] = []
    current = start
    while current <= end:
        window_end = min(add_months(current, months) - timedelta(days=1), end)
        windows.append((current, window_end))
        current = window_end + timedelta(days=1)
    return list(reversed(windows))


def checkpoint_key(mode: str, source: str, topic: str, start: date, end: date) -> str:
    return f"{mode}:{source}:{topic}:{start.isoformat()}:{end.isoformat()}"


class IngestionPipeline:
    """Runs source clients for every topic and stores the results."""

    def __init__(
        self,
        store: PostgresStore,
        clients: dict[str, SourceClient],
        topics: list[Topic],
        settings: IngestionSettings,
        today: date | None = None,
    ) -> None:
        self.store = store
        self.clients = clients
        self.topics = topics
        self.settings = settings
        self.today = today or date.today()
        self.run_id = uuid.uuid4()

    def _tasks(self):
        """Yield (client, topic, query) for every source/topic combination."""
        for source, client in self.clients.items():
            for topic in self.topics:
                query = topic.queries.get(source)
                if query:
                    yield client, topic, query

    # ------------------------------------------------------------------ modes

    def daily_window(self, source: str, topic: str) -> tuple[date, date]:
        """Next incremental window for a source/topic.

        An unfinished daily window is resumed first; otherwise the window
        starts at the last completed end minus an overlap (papers are often
        indexed a day or two after their publication date).
        """
        unfinished = self.store.open_checkpoint(DAILY, source, topic)
        if unfinished:
            return unfinished["window_start"], unfinished["window_end"]
        last_end = self.store.last_completed_window_end(DAILY, source, topic)
        if last_end:
            start = last_end - timedelta(days=self.settings.daily_overlap_days)
        else:
            start = self.today - timedelta(days=self.settings.daily_initial_lookback_days)
        return min(start, self.today), self.today

    def run_daily(self, max_records: int | None = None) -> list[TaskResult]:
        """Collect recently published papers for every source and topic."""
        limit = max_records or self.settings.max_records_per_topic
        results = []
        for client, topic, query in self._tasks():
            start, end = self.daily_window(client.name, topic.key)
            results.append(self._run_window(DAILY, client, topic, query, start, end, limit))
        return results

    def run_backfill(self, start: date, end: date, max_records: int | None = None) -> list[TaskResult]:
        """Collect a historical range, window by window, newest window first."""
        limit = max_records or self.settings.max_records_per_topic
        windows = split_windows(start, end, self.settings.backfill_window_months)
        results = []
        for client, topic, query in self._tasks():
            budget = limit
            for window_start, window_end in windows:
                result = self._run_window(BACKFILL, client, topic, query, window_start, window_end, budget)
                results.append(result)
                budget -= result.counts["fetched"]
                # Stop this source/topic on failure or when the budget is used;
                # the checkpoint lets the next run continue exactly here.
                if result.status in ("failed", "partial") or budget <= 0:
                    break
        return results

    # ------------------------------------------------------------ one window

    def _run_window(
        self,
        mode: str,
        client: SourceClient,
        topic: Topic,
        query: str,
        start: date,
        end: date,
        budget: int,
    ) -> TaskResult:
        source = client.name
        key = checkpoint_key(mode, source, topic.key, start, end)
        result = TaskResult(mode, source, topic.key, start, end, status="completed")
        log_extra = {"mode": mode, "source": source, "topic": topic.key, "window": f"{start}..{end}"}

        checkpoint = self.store.get_checkpoint(key)
        if checkpoint and checkpoint["status"] == "completed":
            result.status = "skipped"
            logger.info("window already completed, skipped", extra=log_extra)
            return result

        cursor = checkpoint["cursor"] if checkpoint else None
        fetched_total = checkpoint["records_fetched"] if checkpoint else 0
        checkpoint_fields = dict(mode=mode, source=source, topic=topic.key, window_start=start, window_end=end)
        if checkpoint is None:
            self.store.save_checkpoint(key, **checkpoint_fields, cursor=None, status="in_progress", records_fetched=0)
        else:
            logger.info("resuming from checkpoint", extra={**log_extra, "cursor": cursor})

        log_id = self.store.start_log(self.run_id, mode, source, topic.key, start, end)
        counts = result.counts
        try:
            while True:
                page = client.fetch_page(query, start, end, cursor)
                counts["pages"] += 1
                counts["fetched"] += len(page.records)
                counts["skipped"] += page.skipped
                next_cursor = page.next_cursor if page.next_cursor != cursor else None  # loop guard
                done = next_cursor is None

                # Records and the new cursor are committed together, so a crash
                # can never save the cursor without the records (or vice versa).
                with self.store.transaction():
                    for record in page.records:
                        try:
                            with self.store.transaction():  # savepoint per record
                                counts[self.store.upsert_record(record, topic.key).value] += 1
                        except Exception:
                            counts["failed"] += 1
                            logger.exception(
                                "record could not be stored",
                                extra={**log_extra, "source_id": record.source_id},
                            )
                    fetched_total += page.raw_count
                    self.store.save_checkpoint(
                        key,
                        **checkpoint_fields,
                        cursor=next_cursor,
                        status="completed" if done else "in_progress",
                        records_fetched=fetched_total,
                    )
                cursor = next_cursor
                logger.info(
                    "page stored",
                    extra={**log_extra, "page_records": len(page.records), "total": page.total, **counts},
                )
                if done:
                    break
                if counts["fetched"] >= budget:
                    result.status = "partial"
                    logger.info("record limit reached; will resume next run", extra=log_extra)
                    break
        except Exception as exc:  # network errors, API errors, bad payloads
            result.status = "failed"
            result.error = str(exc)
            logger.error("window failed", extra={**log_extra, "error": str(exc)})

        self.store.finish_log(log_id, result.status, counts, result.error)
        return result
