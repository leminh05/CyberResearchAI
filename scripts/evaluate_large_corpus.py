"""Retrieval on the large corpus with its own labels: exact scan vs HNSW.

    python scripts/evaluate_large_corpus.py            # docs/bigdata/large_corpus_retrieval.json
    python scripts/evaluate_large_corpus.py --answers  # local model; docs/bigdata/large_corpus_answers_ollama_reranked.json

Uses the database in DATABASE_URL with its name replaced by cyberresearch_large.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import replace

sys.path.insert(0, "src")

from dotenv import dotenv_values  # noqa: E402
from psycopg.conninfo import conninfo_to_dict, make_conninfo  # noqa: E402

from cyberresearch.config import PROJECT_ROOT, load_settings  # noqa: E402
from cyberresearch.evaluation.dataset import GOLDEN_SET, key_index, load_golden_set  # noqa: E402
from cyberresearch.evaluation.retrieval_eval import evaluate_retrieval  # noqa: E402
from cyberresearch.ingestion.storage import PostgresStore  # noqa: E402
from cyberresearch.processing.embeddings import SentenceTransformerEmbedder  # noqa: E402
from cyberresearch.rag.retrieval import Retriever  # noqa: E402

LABELS = PROJECT_ROOT / "config" / "eval" / "golden_set_large.yaml"
OUTPUT = PROJECT_ROOT / "docs" / "bigdata" / "large_corpus_retrieval.json"
ANSWERS_OUTPUT = PROJECT_ROOT / "docs" / "bigdata" / "large_corpus_answers_ollama_reranked.json"


def run_answers(settings, conn, embedder, questions, keys) -> int:
    from cyberresearch.evaluation.answer_eval import TimedGenerator, aggregate_answers, evaluate_answers
    from cyberresearch.rag.engine import RAGEngine
    from cyberresearch.rag.ollama_client import OllamaGenerator

    generator = OllamaGenerator(settings.rag)
    if not generator.is_ready():
        print("Ollama or its model is not available")
        return 2
    timer = TimedGenerator(generator)
    engine = RAGEngine(Retriever(conn, embedder, settings.rag), timer, settings.rag, translator=timer)
    meta = {
        "provider": "ollama",
        "llm_model": settings.rag.ollama_model,
        "rerank_model": settings.rag.rerank_model if settings.rag.rerank else None,
        "labels": LABELS.name,
        "database": "cyberresearch_large",
    }
    rows: list[dict] = []

    def on_row(row: dict) -> None:
        rows.append(row)
        # Saved after every question: the run takes about an hour.
        ANSWERS_OUTPUT.write_text(
            json.dumps({"meta": meta, "aggregate": aggregate_answers(rows), "questions": rows}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"{row['id']:<20} {row['status']:<22} {row['end_to_end_s']:>7.1f} s", flush=True)

    evaluate_answers(engine, timer, questions, keys, on_row)
    print(json.dumps(aggregate_answers(rows), indent=2))
    return 0


def main() -> int:
    settings = load_settings()
    url = settings.database_url or dotenv_values(PROJECT_ROOT / ".env")["DATABASE_URL"]
    conn = PostgresStore.connect(make_conninfo(**{**conninfo_to_dict(url), "dbname": "cyberresearch_large"})).conn
    embedder = SentenceTransformerEmbedder(settings.embedding)
    keys = key_index(conn)
    indexed = {keys[r["paper_id"]] for r in conn.execute("SELECT DISTINCT paper_id FROM paper_chunks")}
    new_labels, old_labels = load_golden_set(LABELS), load_golden_set(GOLDEN_SET)
    if "--answers" in sys.argv:
        return run_answers(settings, conn, embedder, new_labels, keys)
    assert not {k for q in new_labels for k in q["relevant"]} - set(keys.values()), "judged paper not in the database"

    def count(table: str) -> int:
        return conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]

    report = {
        "corpus": {"papers": count("papers"), "chunks": count("paper_chunks")},
        "labels": "config/eval/golden_set_large.yaml (silver, pooled: recall is optimistic; reranked top 6-10 only partly judged)",
        "old_labels_exact_scan": None,
        "variants": {},
    }
    exact = Retriever(conn, embedder, replace(settings.rag, exact_vector_search=True, rerank=False))
    report["old_labels_exact_scan"] = evaluate_retrieval(exact, old_labels, keys, indexed)["aggregate"]
    variants = {
        "exact": exact,
        "hnsw_ef200": Retriever(conn, embedder, replace(settings.rag, exact_vector_search=False, rerank=False)),
        "exact_reranked": Retriever(conn, embedder, replace(settings.rag, exact_vector_search=True, rerank=True)),
    }
    for name, retriever in variants.items():
        aggregate = evaluate_retrieval(retriever, new_labels, keys, indexed)["aggregate"]
        timings = []
        for question in new_labels[:30]:
            for _ in range(3):
                start = time.perf_counter()
                retriever.retrieve(question["question"])
                timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        report["variants"][name] = {
            "aggregate": aggregate,
            "retrieve_ms": {"n": len(timings), "p50": round(statistics.median(timings), 1), "p95": round(timings[int(len(timings) * 0.95)], 1)},
        }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
