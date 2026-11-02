"""Pipeline: daily vs backfill windows, checkpoints, resume, logs."""

from datetime import date

import pytest

from cyberresearch.config import IngestionSettings, Topic
from cyberresearch.ingestion.clients.base import Page
from cyberresearch.ingestion.http import HttpError
from cyberresearch.ingestion.pipeline import IngestionPipeline, add_months, checkpoint_key, split_windows
from cyberresearch.ingestion.schema import PaperRecord

TODAY = date(2024, 6, 15)
TOPIC = Topic(key="intrusion_detection", name="Intrusion Detection", queries={"fake": "ids"})
SETTINGS = IngestionSettings(
    max_records_per_topic=100, daily_initial_lookback_days=3, daily_overlap_days=1, backfill_window_months=12
)


def make_records(prefix: str, n: int) -> list[PaperRecord]:
    return [
        PaperRecord(source="fake", source_id=f"{prefix}{i}", title=f"Paper {prefix}{i}", openalex_id=f"{prefix}{i}")
        for i in range(n)
    ]


class FakeClient:
    """Serves pages keyed by cursor: None -> "1" -> "2" -> end."""

    name = "fake"
    page_size = 2

    def __init__(self, fail_on_cursor: str | None = None):
        self.pages = {
            None: Page(make_records("a", 2), "1", raw_count=2),
            "1": Page(make_records("b", 2), "2", raw_count=2),
            "2": Page(make_records("c", 1), None, raw_count=1),
        }
        self.fail_on_cursor = fail_on_cursor
        self.calls: list[tuple[date, date, str | None]] = []

    def fetch_page(self, query, start, end, cursor):
        self.calls.append((start, end, cursor))
        if self.fail_on_cursor is not None and cursor == self.fail_on_cursor:
            raise HttpError("fake: giving up after 6 attempts (HTTP 503)", 503)
        return self.pages[cursor]


def pipeline(store, client, today=TODAY) -> IngestionPipeline:
    return IngestionPipeline(store, {"fake": client}, [TOPIC], SETTINGS, today=today)


def logs(store):
    return store.conn.execute("SELECT * FROM ingestion_logs ORDER BY log_id").fetchall()


# ------------------------------------------------------------ pure helpers

def test_add_months_clamps_day() -> None:
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2023, 11, 15), 3) == date(2024, 2, 15)


def test_split_windows_newest_first() -> None:
    assert split_windows(date(2022, 1, 1), date(2024, 3, 31), 12) == [
        (date(2024, 1, 1), date(2024, 3, 31)),
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2022, 1, 1), date(2022, 12, 31)),
    ]
    with pytest.raises(ValueError):
        split_windows(date(2024, 1, 2), date(2024, 1, 1), 12)


# ----------------------------------------------------------------- daily

@pytest.mark.db
def test_daily_run_completes_and_logs(store) -> None:
    client = FakeClient()
    [result] = pipeline(store, client).run_daily()
    assert result.status == "completed"
    assert (result.window_start, result.window_end) == (date(2024, 6, 12), TODAY)  # 3-day lookback
    assert result.counts["inserted"] == 5 and result.counts["pages"] == 3

    cp = store.get_checkpoint(checkpoint_key("daily", "fake", TOPIC.key, date(2024, 6, 12), TODAY))
    assert cp["status"] == "completed" and cp["cursor"] is None and cp["records_fetched"] == 5
    [log] = logs(store)
    assert (log["status"], log["fetched"], log["inserted"]) == ("completed", 5, 5)
    assert log["finished_at"] is not None


@pytest.mark.db
def test_daily_is_incremental_and_skips_unchanged(store) -> None:
    pipeline(store, FakeClient()).run_daily()
    client = FakeClient()
    [result] = pipeline(store, client, today=date(2024, 6, 16)).run_daily()
    # Next window starts at the last completed end minus 1 overlap day.
    assert (result.window_start, result.window_end) == (date(2024, 6, 14), date(2024, 6, 16))
    assert result.counts["unchanged"] == 5 and result.counts["inserted"] == 0


@pytest.mark.db
def test_record_limit_saves_checkpoint_and_next_run_resumes(store) -> None:
    first_client = FakeClient()
    [first] = pipeline(store, first_client).run_daily(max_records=2)
    assert first.status == "partial"
    assert [c[2] for c in first_client.calls] == [None]  # stopped after page 1

    second_client = FakeClient()
    [second] = pipeline(store, second_client).run_daily(max_records=100)
    assert second.status == "completed"
    assert [c[2] for c in second_client.calls] == ["1", "2"]  # resumed, page 1 not refetched
    assert second.counts["inserted"] == 3
    assert store.conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"] == 5


@pytest.mark.db
def test_api_failure_keeps_progress_and_resume_continues(store) -> None:
    [failed] = pipeline(store, FakeClient(fail_on_cursor="2")).run_daily()
    assert failed.status == "failed" and "503" in failed.error
    assert failed.counts["inserted"] == 4  # pages 1-2 were committed

    log = logs(store)[-1]
    assert log["status"] == "failed" and "503" in log["error_message"]

    retry_client = FakeClient()
    [resumed] = pipeline(store, retry_client).run_daily()
    assert [c[2] for c in retry_client.calls] == ["2"]
    assert resumed.status == "completed" and resumed.counts["inserted"] == 1


# --------------------------------------------------------------- backfill

@pytest.mark.db
def test_backfill_uses_windows_and_skips_completed_ones(store) -> None:
    client = FakeClient()
    results = pipeline(store, client).run_backfill(date(2022, 1, 1), date(2023, 12, 31))
    assert [(r.window_start.year, r.status) for r in results] == [(2023, "completed"), (2022, "completed")]

    again = pipeline(store, FakeClient()).run_backfill(date(2022, 1, 1), date(2023, 12, 31))
    assert [r.status for r in again] == ["skipped", "skipped"]

    modes = {r["mode"] for r in store.conn.execute("SELECT mode FROM ingestion_checkpoints")}
    assert modes == {"backfill"}  # backfill never touches daily checkpoints


@pytest.mark.db
def test_backfill_budget_stops_and_resumes_in_same_window(store) -> None:
    [partial] = pipeline(store, FakeClient()).run_backfill(date(2023, 1, 1), date(2023, 12, 31), max_records=2)
    assert partial.status == "partial"
    client = FakeClient()
    [done] = pipeline(store, client).run_backfill(date(2023, 1, 1), date(2023, 12, 31))
    assert done.status == "completed"
    assert client.calls[0][2] == "1"
