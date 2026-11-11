"""Print the candidate pool to judge for each golden-set question.

Pool = top-15 papers from dense retrieval (the system under test) + top-10
papers from PostgreSQL full-text search (an independent lexical ranker), so
the judgments are not limited to what the embedding model already finds.

    python scripts/build_judgment_pool.py [--ids gnn-ids ...] > data/eval/pool.md
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from cyberresearch.config import load_settings  # noqa: E402
from cyberresearch.db import resolve_database_url  # noqa: E402
from cyberresearch.evaluation.dataset import key_index, load_golden_set  # noqa: E402
from cyberresearch.ingestion.storage import PostgresStore  # noqa: E402
from cyberresearch.processing.embeddings import SentenceTransformerEmbedder  # noqa: E402
from cyberresearch.rag.retrieval import Retriever  # noqa: E402

DENSE_PAPERS, FTS_PAPERS = 15, 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = load_settings()
    store = PostgresStore.connect(resolve_database_url(settings))
    conn = store.conn
    keys = key_index(conn)
    retriever = Retriever(conn, SentenceTransformerEmbedder(settings.embedding), settings.rag)
    for q in load_golden_set():
        if q["type"] != "answerable" or (args.ids and q["id"] not in args.ids):
            continue
        dense = []
        for e in retriever.retrieve(q["question"], top_k=40):
            if e.paper_id not in dense:
                dense.append(e.paper_id)
        fts = [
            r["paper_id"]
            for r in conn.execute(
                """SELECT paper_id FROM papers,
                          to_tsquery('english', %(q)s) query
                   WHERE to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')) @@ query
                   ORDER BY ts_rank(to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')), query) DESC
                   LIMIT %(n)s""",
                {"q": q["fts"], "n": FTS_PAPERS},
            )
        ]
        pool = list(dict.fromkeys(dense[:DENSE_PAPERS] + fts))
        print(f"\n## {q['id']}: {q['question']}")
        for pid in pool:
            row = conn.execute(
                """SELECT p.title, coalesce(p.abstract, (SELECT text FROM paper_chunks c WHERE c.paper_id = p.paper_id
                          ORDER BY chunk_index LIMIT 1), '') AS text FROM papers p WHERE paper_id = %s""",
                (pid,),
            ).fetchone()
            origin = ("D" if pid in dense[:DENSE_PAPERS] else "") + ("F" if pid in fts else "")
            print(f"- [{origin}] {keys[pid]} | {row['title']}\n    {' '.join(row['text'].split()[:45])}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
