"""PostgreSQL + pgvector storage for PDF status, chunks and embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

from cyberresearch.processing.chunking import Chunk

SCHEMA_TEMPLATE = (Path(__file__).parent / "sql" / "processing_schema.sql").read_text(encoding="utf-8")

DOCUMENT_COLUMNS = (
    "access_reason", "pdf_status", "extraction_status", "content_source", "pdf_url", "pdf_origin",
    "license", "http_status", "file_path", "sha256", "size_bytes", "page_count", "chunk_count", "error",
)


def to_vector_literal(vector: list[float]) -> str:
    """Format a Python list as pgvector text input: ``[0.1,0.2,...]``."""
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def parse_vector(text: str) -> list[float]:
    """Parse pgvector text output back into floats."""
    return [float(x) for x in text.strip("[]").split(",") if x]


class ChunkStore:
    """SQL for Phase 2. Shares the connection style of ``PostgresStore``."""

    def __init__(self, conn: psycopg.Connection, dimension: int) -> None:
        self.conn = conn
        self.dimension = dimension

    def transaction(self) -> Any:
        return self.conn.transaction()

    def init_schema(self) -> None:
        self.conn.execute(SCHEMA_TEMPLATE.replace("{dimension}", str(int(self.dimension))))

    # ---------------------------------------------------------------- inputs

    def papers_to_process(self, limit: int | None = None, retry_failed: bool = False) -> list[dict[str, Any]]:
        """Papers without a pdf_documents row (or with a failed download).

        Each row includes ``sources``: the original payloads, needed for the
        license check.
        """
        condition = "d.paper_id IS NULL"
        if retry_failed:
            condition += " OR d.pdf_status = 'failed'"
        return self.conn.execute(
            f"""SELECT p.paper_id, p.title, p.abstract, p.arxiv_id, p.doi, p.url,
                       COALESCE(d.attempts, 0) AS attempts,
                       (SELECT json_agg(json_build_object('source', s.source, 'raw', s.raw))
                          FROM paper_sources s WHERE s.paper_id = p.paper_id) AS sources
                FROM papers p LEFT JOIN pdf_documents d USING (paper_id)
                WHERE {condition}
                ORDER BY p.arxiv_id IS NULL, p.paper_id
                LIMIT %s""",
            (limit,),
        ).fetchall()

    def downloaded_documents(self) -> list[dict[str, Any]]:
        """Papers whose PDF is already on disk, with their stored status."""
        return self.conn.execute(
            """SELECT d.*, p.title, p.abstract, p.url
               FROM pdf_documents d JOIN papers p USING (paper_id)
               WHERE d.pdf_status = 'downloaded' AND d.file_path IS NOT NULL
               ORDER BY d.paper_id"""
        ).fetchall()

    # --------------------------------------------------------------- outputs

    def record_document(self, paper_id: int, attempts: int, **fields: Any) -> None:
        """Insert or update the processing status of one paper."""
        values = {c: fields.get(c) for c in DOCUMENT_COLUMNS}
        values["extraction_status"] = values["extraction_status"] or "not_attempted"
        values["content_source"] = values["content_source"] or "none"
        values["chunk_count"] = values["chunk_count"] or 0
        columns = ", ".join(DOCUMENT_COLUMNS)
        placeholders = ", ".join(["%s"] * len(DOCUMENT_COLUMNS))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in DOCUMENT_COLUMNS)
        self.conn.execute(
            f"""INSERT INTO pdf_documents (paper_id, attempts, {columns}) VALUES (%s, %s, {placeholders})
                ON CONFLICT (paper_id) DO UPDATE SET {updates},
                    attempts = EXCLUDED.attempts, processed_at = now()""",
            (paper_id, attempts, *values.values()),
        )

    def replace_chunks(
        self,
        paper_id: int,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        content_source: str,
        source_url: str | None,
        model_name: str,
    ) -> None:
        """Replace all chunks of a paper (idempotent re-processing)."""
        if len(chunks) != len(embeddings):
            raise ValueError("one embedding per chunk is required")
        self.conn.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))
        with self.conn.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO paper_chunks
                       (paper_id, chunk_index, content_source, section, page_start, page_end,
                        text, word_count, source_url, embedding, embedding_model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)""",
                [
                    (
                        paper_id, c.chunk_index, content_source, c.section, c.page_start, c.page_end,
                        c.text, c.word_count, source_url, to_vector_literal(v), model_name,
                    )
                    for c, v in zip(chunks, embeddings)
                ],
            )

    # ------------------------------------------------------------- retrieval

    def search(self, query_embedding: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Return the ``k`` chunks most similar to the query (cosine), with paper metadata."""
        vector = to_vector_literal(query_embedding)
        return self.conn.execute(
            """SELECT c.chunk_id, c.paper_id, c.chunk_index, c.content_source, c.section,
                      c.page_start, c.page_end, c.text, c.source_url,
                      p.title, p.year, p.doi, p.arxiv_id, p.url AS paper_url,
                      1 - (c.embedding <=> %s::vector) AS similarity
               FROM paper_chunks c JOIN papers p USING (paper_id)
               ORDER BY c.embedding <=> %s::vector
               LIMIT %s""",
            (vector, vector, k),
        ).fetchall()

    def get_embedding(self, chunk_id: int) -> list[float]:
        row = self.conn.execute(
            "SELECT embedding::text AS e FROM paper_chunks WHERE chunk_id = %s", (chunk_id,)
        ).fetchone()
        return parse_vector(row["e"])

    def summary(self) -> dict[str, Any]:
        rows = lambda q: self.conn.execute(q).fetchall()  # noqa: E731
        return {
            "pdf_status": {r["pdf_status"]: r["n"] for r in rows(
                "SELECT pdf_status, count(*) AS n FROM pdf_documents GROUP BY 1 ORDER BY 2 DESC")},
            "access_reason": {r["access_reason"]: r["n"] for r in rows(
                "SELECT access_reason, count(*) AS n FROM pdf_documents GROUP BY 1 ORDER BY 2 DESC")},
            "content_source": {r["content_source"]: r["n"] for r in rows(
                "SELECT content_source, count(*) AS n FROM pdf_documents GROUP BY 1 ORDER BY 2 DESC")},
            "chunks": rows(
                """SELECT content_source, count(*) AS chunks, count(DISTINCT paper_id) AS papers,
                          round(avg(word_count)) AS avg_words
                   FROM paper_chunks GROUP BY 1 ORDER BY 1"""),
        }
