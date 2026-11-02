"""Command-line commands for ingestion (wired into ``main.py``)."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import date

from cyberresearch.config import Settings
from cyberresearch.db import resolve_database_url
from cyberresearch.ingestion.clients import build_clients
from cyberresearch.ingestion.pipeline import IngestionPipeline, TaskResult
from cyberresearch.ingestion.storage import PostgresStore

logger = logging.getLogger(__name__)


def add_ingestion_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register init-db, daily, backfill and status sub-commands."""
    subparsers.add_parser("init-db", help="create database tables")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sources", nargs="+", help="subset of sources, e.g. arxiv openalex")
        p.add_argument("--topics", nargs="+", help="subset of topic keys, e.g. intrusion_detection")
        p.add_argument("--max-records", type=int, help="max records per source/topic in this run")

    daily = subparsers.add_parser("daily", help="incremental collection of recent papers")
    add_common(daily)

    backfill = subparsers.add_parser("backfill", help="historical collection for a date range")
    backfill.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    backfill.add_argument("--end", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD (default today)")
    add_common(backfill)

    subparsers.add_parser("status", help="show table counts, checkpoints and recent logs")


def _open_store(settings: Settings) -> PostgresStore:
    store = PostgresStore.connect(resolve_database_url(settings))
    store.init_schema()
    return store


def _print_results(results: list[TaskResult]) -> None:
    totals: Counter = Counter()
    print(f"{'source':<17}{'topic':<23}{'window':<24}{'status':<10}{'fetched':>8}{'new':>6}{'merged':>7}{'upd':>5}{'same':>6}{'fail':>5}")
    for r in results:
        c = r.counts
        totals.update(c)
        window = f"{r.window_start}..{r.window_end}"
        print(
            f"{r.source:<17}{r.topic:<23}{window:<24}{r.status:<10}{c['fetched']:>8}{c['inserted']:>6}"
            f"{c['merged']:>7}{c['updated']:>5}{c['unchanged']:>6}{c['failed']:>5}"
        )
        if r.error:
            print(f"    error: {r.error}")
    print(
        f"TOTAL fetched={totals['fetched']} inserted={totals['inserted']} merged={totals['merged']} "
        f"updated={totals['updated']} unchanged={totals['unchanged']} failed={totals['failed']}"
    )


def run_ingestion_command(args: argparse.Namespace, settings: Settings) -> int:
    """Execute one ingestion sub-command. Returns a process exit code."""
    store = _open_store(settings)
    try:
        if args.command == "init-db":
            print("Database schema is ready.")
            return 0
        if args.command == "status":
            summary = store.summary()
            for name in ("papers", "paper_sources", "multi_source_papers"):
                print(f"{name:<20}: {summary[name]}")
            print(f"{'by_source':<20}: {summary['by_source']}")
            print(f"{'checkpoints':<20}: {summary['checkpoints']}")
            print("recent logs:")
            for row in summary["recent_logs"]:
                print(
                    f"  {row['mode']:<8} {row['source']:<16} {row['topic']:<22} "
                    f"{row['window_start']}..{row['window_end']} {row['status']:<9} "
                    f"fetched={row['fetched']} new={row['inserted']} merged={row['merged']} same={row['unchanged']}"
                )
            return 0

        clients = build_clients(settings, only=args.sources)
        topics = [t for t in settings.topics if not args.topics or t.key in args.topics]
        if not clients or not topics:
            print("No matching sources or topics.")
            return 2
        pipeline = IngestionPipeline(store, clients, topics, settings.ingestion)
        if args.command == "daily":
            results = pipeline.run_daily(args.max_records)
        else:
            results = pipeline.run_backfill(args.start, args.end, args.max_records)
        _print_results(results)
        return 1 if any(r.status == "failed" for r in results) else 0
    finally:
        store.close()
