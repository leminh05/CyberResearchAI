"""Command-line commands for Phase 2 (wired into ``main.py``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyberresearch.config import PROJECT_ROOT, Settings
from cyberresearch.db import resolve_database_url
from cyberresearch.ingestion.clients.base import user_agent
from cyberresearch.ingestion.storage import PostgresStore

BENCHMARK_DATA_DIR = PROJECT_ROOT / "data" / "benchmark"
BENCHMARK_REPORT = PROJECT_ROOT / "docs" / "benchmark_results.md"


def add_processing_commands(subparsers: argparse._SubParsersAction) -> None:
    process = subparsers.add_parser("process", help="download legal PDFs, extract, chunk, embed and store")
    process.add_argument("--limit", type=int, help="max papers to process in this run")
    process.add_argument("--no-download", action="store_true", help="do not download PDFs (abstracts only)")
    process.add_argument("--retry-failed", action="store_true", help="retry papers whose download failed")
    process.add_argument("--reprocess-pdfs", action="store_true",
                         help="re-extract and re-chunk already downloaded PDFs (no new downloads)")

    search = subparsers.add_parser("search", help="semantic search over stored chunks")
    search.add_argument("query", help="natural-language query")
    search.add_argument("-k", type=int, default=5, help="number of results")

    bench = subparsers.add_parser("benchmark", help="plain Python vs PySpark benchmark")
    bench.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000, 500_000])
    bench.add_argument("--repeats", type=int, default=3)


def _stores(settings: Settings):
    from cyberresearch.processing.store import ChunkStore

    base = PostgresStore.connect(resolve_database_url(settings))
    base.init_schema()
    chunks = ChunkStore(base.conn, settings.embedding.dimension)
    chunks.init_schema()
    return base, chunks


def run_processing_command(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "benchmark":
        return _benchmark(args, settings)

    from cyberresearch.processing.embeddings import SentenceTransformerEmbedder

    base, chunk_store = _stores(settings)
    try:
        embedder = SentenceTransformerEmbedder(settings.embedding)
        if args.command == "search":
            for i, row in enumerate(chunk_store.search(embedder.embed([args.query])[0], args.k), 1):
                pages = f"p.{row['page_start']}" if row["page_start"] == row["page_end"] else f"pp.{row['page_start']}-{row['page_end']}"
                where = pages if row["content_source"] == "pdf" else "abstract"
                print(f"{i}. [{row['similarity']:.3f}] {row['title']} ({row['year']}) - {row['section']}, {where}")
                print(f"   paper_id={row['paper_id']} chunk_id={row['chunk_id']} url={row['source_url'] or row['paper_url']}")
                print(f"   {row['text'][:220]}...")
            return 0

        from cyberresearch.processing.downloader import PdfDownloader
        from cyberresearch.processing.pipeline import ProcessingPipeline

        downloader = None if args.no_download else PdfDownloader(
            settings.processing, settings.http, user_agent(settings.contact_email)
        )
        pipeline = ProcessingPipeline(chunk_store, embedder, settings.processing, downloader)
        if args.reprocess_pdfs:
            totals = pipeline.reprocess_downloaded()
        else:
            totals = pipeline.run(args.limit, args.retry_failed)
        print("This run:", dict(sorted(totals.items())))
        print("Database:", json.dumps(chunk_store.summary(), default=str, indent=2))
        return 1 if totals["errors"] else 0
    finally:
        base.close()


def _benchmark(args: argparse.Namespace, settings: Settings) -> int:
    from cyberresearch.bigdata.benchmark import format_markdown, run_benchmark
    from cyberresearch.bigdata.dataset import export_base_records

    base = PostgresStore.connect(resolve_database_url(settings))
    try:
        records = export_base_records(base.conn)
    finally:
        base.close()
    if not records:
        print("No source records found; run `python main.py daily` first.")
        return 2
    result = run_benchmark(records, sorted(args.sizes), BENCHMARK_DATA_DIR, settings.spark, args.repeats)
    report = format_markdown(result)
    BENCHMARK_REPORT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_REPORT.write_text(report, encoding="utf-8")
    Path(BENCHMARK_REPORT.with_suffix(".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(report)
    print(f"Saved to {BENCHMARK_REPORT.relative_to(PROJECT_ROOT)}")
    return 0 if all(r["results_match"] for r in result["rows"]) else 1
