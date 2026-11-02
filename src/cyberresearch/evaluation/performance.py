"""Processing throughput and search latency, measured on this machine."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import psycopg

from cyberresearch.config import PROJECT_ROOT, Settings
from cyberresearch.evaluation.metrics import summarize
from cyberresearch.processing.embeddings import Embedder
from cyberresearch.processing.pipeline import ProcessingPipeline, embedding_input
from cyberresearch.processing.store import to_vector_literal
from cyberresearch.rag.retrieval import Retriever

BENCHMARK_JSON = PROJECT_ROOT / "docs" / "benchmark_results.json"


class FixedEmbedder:
    """Returns precomputed vectors so only the database search is timed."""

    def __init__(self, model_name: str, vectors: dict[str, list[float]]) -> None:
        self.model_name = model_name
        self.dimension = len(next(iter(vectors.values())))
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[t] for t in texts]


def processing_throughput(conn: psycopg.Connection, embedder: Embedder, settings: Settings, pdfs: int) -> dict[str, Any]:
    """Extract + clean + chunk ``pdfs`` downloaded PDFs (lowest paper_ids), then embed their chunks."""
    docs = conn.execute(
        """SELECT d.paper_id, d.file_path, p.title FROM pdf_documents d JOIN papers p USING (paper_id)
           WHERE d.pdf_status = 'downloaded' AND d.file_path IS NOT NULL ORDER BY d.paper_id LIMIT %s""",
        (pdfs,),
    ).fetchall()
    pipeline = ProcessingPipeline(None, embedder, settings.processing)  # type: ignore[arg-type]  # store unused here
    pages = size = 0
    inputs: list[str] = []
    start = time.perf_counter()
    for doc in docs:
        path = Path(doc["file_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        status: dict[str, Any] = {"file_path": path}
        chunks = pipeline.pdf_chunks(status)
        pages += status.get("page_count") or 0
        size += path.stat().st_size
        inputs += [embedding_input(doc["title"], c) for c in chunks]
    extract_s = time.perf_counter() - start

    start = time.perf_counter()
    embedder.embed(inputs)
    embed_s = time.perf_counter() - start
    return {
        "pdfs": len(docs),
        "pages": pages,
        "megabytes": round(size / 1e6, 1),
        "chunks": len(inputs),
        "extract_clean_chunk_s": round(extract_s, 2),
        "pages_per_second": round(pages / extract_s, 1),
        "pdfs_per_second": round(len(docs) / extract_s, 2),
        "embedding_s": round(embed_s, 2),
        "chunks_embedded_per_second": round(len(inputs) / embed_s, 1),
        "embedding_device": "cpu",
    }


ANN_SQL = "SELECT chunk_id FROM paper_chunks ORDER BY embedding <=> %s::vector LIMIT %s"


def _timed_ms(fn, items: list[Any], repeats: int) -> dict[str, Any]:
    times = []
    for item in items:
        fn(item)  # warm-up (buffer cache)
        for _ in range(repeats):
            start = time.perf_counter()
            fn(item)
            times.append((time.perf_counter() - start) * 1000)
    return summarize(times, 2)


def search_latency(conn: psycopg.Connection, embedder: Embedder, settings: Settings, questions: list[str], repeats: int) -> dict[str, Any]:
    """Query embedding time; production retrieval time; HNSW vs exact nearest-neighbour search.

    The production query joins papers and filters sections, and PostgreSQL's
    planner decides itself whether to use the HNSW index, so the plan it picks
    is reported too. HNSW and exact search are then timed on the bare
    nearest-neighbour query with the other plan disabled, and HNSW recall is
    the overlap of its top-k with the exact top-k.
    """
    embed_times, vectors = [], {}
    for q in questions:
        for _ in range(repeats):
            start = time.perf_counter()
            vectors[q] = embedder.embed([q])[0]
            embed_times.append((time.perf_counter() - start) * 1000)
    literals = {q: to_vector_literal(v) for q, v in vectors.items()}
    k = settings.rag.candidate_k

    retriever = Retriever(conn, FixedEmbedder(embedder.model_name, vectors), settings.rag)
    production = _timed_ms(retriever.retrieve, questions, repeats)  # also sets hnsw.ef_search
    plan = conn.execute("EXPLAIN (FORMAT JSON) " + ANN_SQL, (literals[questions[0]], k)).fetchone()
    uses_index = "idx_paper_chunks_embedding" in json.dumps(next(iter(plan.values())))

    def ann(q: str) -> list[int]:
        return [r["chunk_id"] for r in conn.execute(ANN_SQL, (literals[q], k))]

    results: dict[str, dict[str, list[int]]] = {}
    timings: dict[str, Any] = {}
    for label, disable in (("hnsw", "enable_seqscan"), ("exact", "enable_indexscan")):
        conn.execute(f"SET {disable} = off")
        try:
            timings[label] = _timed_ms(ann, questions, repeats)
            results[label] = {q: ann(q) for q in questions}
        finally:
            conn.execute(f"RESET {disable}")
    recall = [len(set(results["hnsw"][q]) & set(results["exact"][q])) / k for q in questions]

    chunks = next(iter(conn.execute("SELECT count(*) FROM paper_chunks").fetchone().values()))
    return {
        "chunks_searched": chunks,
        "queries": len(questions),
        "repeats": repeats,
        "unit": "ms",
        "query_embedding_ms": summarize(embed_times, 2),
        "retrieval_query_db_ms": production,
        "planner_uses_hnsw_by_default": uses_index,
        "hnsw_ef_search": settings.rag.hnsw_ef_search,
        f"ann_top{k}_hnsw_ms": timings["hnsw"],
        f"ann_top{k}_exact_ms": timings["exact"],
        f"hnsw_recall_at_{k}_vs_exact": summarize(recall, 3),
    }


def spark_benchmark() -> dict[str, Any] | None:
    """Latest `python main.py benchmark` result, if it was run."""
    if not BENCHMARK_JSON.exists():
        return None
    return json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
