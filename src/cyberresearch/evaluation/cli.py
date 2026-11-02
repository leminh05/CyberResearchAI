"""`python main.py evaluate` - Phase 4 measurements, saved as JSON under docs/evaluation/."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cyberresearch.config import PROJECT_ROOT, Settings
from cyberresearch.db import resolve_database_url
from cyberresearch.ingestion.storage import PostgresStore

RESULTS_DIR = PROJECT_ROOT / "docs" / "evaluation"
PARTS = ("data", "retrieval", "performance", "answers")


def add_evaluation_commands(subparsers: argparse._SubParsersAction) -> None:
    ev = subparsers.add_parser("evaluate", help="Phase 4 system evaluation (results in docs/evaluation/)")
    ev.add_argument("--part", nargs="+", choices=PARTS, default=["data", "retrieval", "performance"],
                    help="what to measure; 'answers' calls the LLM and is slow (default: all but answers)")
    ev.add_argument("--provider", choices=["claude", "ollama"], help="answer generator for --part answers")
    ev.add_argument("--ids", nargs="+", help="only these golden-set question ids")
    ev.add_argument("--pdfs", type=int, default=20, help="PDFs used for the processing throughput test")
    ev.add_argument("--repeats", type=int, default=5, help="repeats per query for latency tests")


def _save(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _meta(settings: Settings, **extra: Any) -> dict[str, Any]:
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}",
        "python": platform.python_version(),
        "embedding_model": settings.embedding.model_name,
        **extra,
    }


def run_evaluation_command(args: argparse.Namespace, settings: Settings) -> int:
    from cyberresearch.evaluation.dataset import key_index, load_golden_set

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    questions = [q for q in load_golden_set() if not args.ids or q["id"] in args.ids]
    store = PostgresStore.connect(resolve_database_url(settings))
    conn = store.conn
    try:
        if "data" in args.part:
            from cyberresearch.evaluation.data_quality import data_quality

            print("Saved", _save("data_quality", {"meta": _meta(settings), **data_quality(conn)}))
        if not set(args.part) - {"data"}:
            return 0

        from cyberresearch.processing.embeddings import SentenceTransformerEmbedder
        from cyberresearch.rag.retrieval import Retriever

        embedder = SentenceTransformerEmbedder(settings.embedding)
        retriever = Retriever(conn, embedder, settings.rag)
        retriever.check_embedding_model()
        keys = key_index(conn)
        missing = {k for q in questions for k in q["relevant"]} - set(keys.values())
        if missing:
            print(f"WARNING: {len(missing)} judged papers are not in the database (corpus changed?): {sorted(missing)[:5]}")
        indexed = {keys[r["paper_id"]] for r in conn.execute("SELECT DISTINCT paper_id FROM paper_chunks")}

        if "retrieval" in args.part:
            from cyberresearch.evaluation.retrieval_eval import evaluate_retrieval

            result = evaluate_retrieval(retriever, questions, keys, indexed)
            print(json.dumps(result["aggregate"], indent=2))
            print("Saved", _save("retrieval", {"meta": _meta(settings, rag=_rag_meta(settings)), **result}))

        if "performance" in args.part:
            from cyberresearch.evaluation.performance import processing_throughput, search_latency, spark_benchmark

            result = {
                "processing": processing_throughput(conn, embedder, settings, args.pdfs),
                "search": search_latency(conn, embedder, settings, [q["question"] for q in questions], args.repeats),
                "python_vs_pyspark": spark_benchmark(),
                "claude_api_latency": "not measured: no ANTHROPIC_API_KEY on this machine (see answers.json for Ollama)",
            }
            print(json.dumps({k: result[k] for k in ("processing", "search")}, indent=2))
            print("Saved", _save("performance", {"meta": _meta(settings), **result}))

        if "answers" in args.part:
            return _answers(args, settings, retriever, questions, keys)
        return 0
    finally:
        store.close()


def _rag_meta(settings: Settings) -> dict[str, Any]:
    r = settings.rag
    return {"top_k": r.top_k, "candidate_k": r.candidate_k, "max_chunks_per_paper": r.max_chunks_per_paper,
            "min_similarity": r.min_similarity, "hnsw_ef_search": r.hnsw_ef_search}


def _answers(args: argparse.Namespace, settings: Settings, retriever, questions, keys) -> int:
    from cyberresearch.evaluation.answer_eval import TimedGenerator, aggregate_answers, evaluate_answers
    from cyberresearch.rag.engine import RAGEngine

    provider = args.provider or settings.rag.provider
    if provider == "ollama":
        from cyberresearch.rag.ollama_client import OllamaGenerator

        generator, model = OllamaGenerator(settings.rag), settings.rag.ollama_model
    else:
        from cyberresearch.rag.llm import ClaudeClient

        if not settings.anthropic_api_key_present:
            print("ANTHROPIC_API_KEY is not set; use --provider ollama")
            return 2
        generator, model = ClaudeClient(settings.rag), settings.rag.model
    timer = TimedGenerator(generator)
    engine = RAGEngine(retriever, timer, settings.rag, translator=timer)
    meta = _meta(settings, provider=provider, llm_model=model, rag=_rag_meta(settings))
    rows: list[dict[str, Any]] = []

    def on_row(row: dict[str, Any]) -> None:
        rows.append(row)
        # Saved after every question: a full local-model run takes over an hour.
        _save("answers", {"meta": meta, "aggregate": aggregate_answers(rows), "questions": rows})
        print(f"  {row['id']:<20} {row['status']:<22} {row['end_to_end_s']:>7.1f} s", flush=True)

    evaluate_answers(engine, timer, questions, keys, on_row)
    print(json.dumps(aggregate_answers(rows), indent=2))
    print("Saved", RESULTS_DIR / "answers.json")
    return 0
