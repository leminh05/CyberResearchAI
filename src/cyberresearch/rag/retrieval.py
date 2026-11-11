"""Hybrid retrieval of evidence chunks with metadata filters.

Steps:
  1. Embed the question with the SAME model used for the chunks (checked).
  2. pgvector cosine search over ``paper_chunks`` with SQL metadata filters,
     fetching ``candidate_k`` candidates.
  3. Full-text search for chunks containing all rare terms of the question,
     merged with step 2 by reciprocal rank fusion. Exact terms ("federated")
     recover papers that a generic long document outranks by cosine alone.
  4. Optional cross-encoder reranking of the merged candidates (``rerank``).
  5. Diversity selection: at most ``max_chunks_per_paper`` chunks per paper,
     then the best ``top_k``. Without it, one long paper can fill every slot.

``Evidence.similarity`` stays the cosine similarity for every chunk. Fusion can
only make the relevance gate of the engine stricter: the per-paper cap may drop
the top cosine chunk in favour of a full-text hit, never add a closer one.

Reranking changes only the order. ``Evidence.similarity`` stays cosine for the same
reason, so the relevance gate does not depend on the reranker's scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import psycopg

from cyberresearch.config import RagSettings
from cyberresearch.processing.embeddings import Embedder, load_reranker
from cyberresearch.processing.store import to_vector_literal


# shortcut: a long question only uses its first terms, enough for real questions.
MAX_LEXICAL_TERMS = 32


class Reranker(Protocol):
    def scores(self, query: str, texts: list[str]) -> list[float]:
        ...


class EmbeddingModelMismatch(Exception):
    """Stored chunks were embedded with a different model than the query embedder."""


@dataclass
class SearchFilters:
    """Optional metadata filters applied inside the SQL query."""

    year_min: int | None = None
    year_max: int | None = None
    topics: list[str] = field(default_factory=list)          # papers.topics overlap
    content_sources: list[str] = field(default_factory=list)  # "pdf", "abstract"
    exclude_sections: list[str] = field(default_factory=list)
    paper_ids: list[int] = field(default_factory=list)

    def describe(self) -> str:
        parts = []
        if self.year_min or self.year_max:
            parts.append(f"year {self.year_min or '...'}-{self.year_max or '...'}")
        if self.topics:
            parts.append("topics=" + ",".join(self.topics))
        if self.content_sources:
            parts.append("source=" + ",".join(self.content_sources))
        if self.paper_ids:
            parts.append("papers=" + ",".join(map(str, self.paper_ids)))
        return "; ".join(parts) or "none"


@dataclass
class Evidence:
    """One retrieved chunk plus the paper metadata needed to cite it."""

    evidence_id: str  # "E1", "E2", ... - the only handle Claude may cite
    chunk_id: int
    paper_id: int
    chunk_index: int
    title: str | None
    authors: list[str]
    year: int | None
    doi: str | None
    arxiv_id: str | None
    paper_url: str | None
    source_url: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    content_source: str
    text: str
    similarity: float

    @property
    def canonical_url(self) -> str | None:
        """DOI link, else arXiv abstract page, else the paper's landing page."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return self.paper_url

    @property
    def location(self) -> str:
        """Human-readable location: "p. 5", "pp. 5-6" or "abstract"."""
        if self.content_source != "pdf" or self.page_start is None:
            return "abstract"
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


def select_diverse(rows: list[dict[str, Any]], top_k: int, max_per_paper: int) -> list[dict[str, Any]]:
    """Keep rows in similarity order, at most ``max_per_paper`` per paper."""
    per_paper: dict[int, int] = {}
    selected = []
    for row in rows:
        if per_paper.get(row["paper_id"], 0) >= max_per_paper:
            continue
        per_paper[row["paper_id"]] = per_paper.get(row["paper_id"], 0) + 1
        selected.append(row)
        if len(selected) == top_k:
            break
    return selected


def fuse_rankings(rankings: list[list[dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    """Reciprocal rank fusion: rank positions only, as cosine and text scores are not comparable."""
    scores: dict[int, float] = {}
    rows: dict[int, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row["chunk_id"]] = scores.get(row["chunk_id"], 0.0) + 1 / (k + rank)
            rows[row["chunk_id"]] = row
    return [rows[chunk_id] for chunk_id in sorted(scores, key=scores.__getitem__, reverse=True)]


class Retriever:
    """Embeds questions and searches the chunk table."""

    def __init__(
        self, conn: psycopg.Connection, embedder: Embedder, settings: RagSettings, reranker: Reranker | None = None
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker or (load_reranker(settings.rerank_model) if settings.rerank else None)

    def check_embedding_model(self) -> None:
        """Fail loudly if chunks were embedded with another model.

        Vectors from different models live in different spaces; comparing
        them returns confident-looking but meaningless similarities.
        """
        models = {
            r["embedding_model"]
            for r in self.conn.execute("SELECT DISTINCT embedding_model FROM paper_chunks")
        }
        if models and models != {self.embedder.model_name}:
            raise EmbeddingModelMismatch(
                f"chunks use {sorted(models)}, query embedder is {self.embedder.model_name!r}"
            )

    def _lexical_query(self, query: str) -> str | None:
        """tsquery requiring every rare term of ``query``; None if it has no searchable term.

        PostgreSQL's text rank ignores how rare a term is, so frequent terms
        ("detection" in a security corpus) would only add noise.
        """
        lexemes = [
            r["lexeme"]
            for r in self.conn.execute("SELECT unnest(tsvector_to_array(to_tsvector('english', %s))) AS lexeme", (query,))
        ][:MAX_LEXICAL_TERMS]
        if not lexemes:
            return None
        terms = ["'" + w.replace("\\", "\\\\").replace("'", "''") + "'" for w in lexemes]
        total = self.conn.execute("SELECT count(*) AS n FROM paper_chunks").fetchone()["n"]
        rare = [
            term for term in terms
            if self.conn.execute("SELECT count(*) AS n FROM paper_chunks WHERE text_search @@ %s::tsquery", (term,)).fetchone()["n"]
            <= self.settings.lexical_max_doc_fraction * total
        ]
        return " & ".join(rare or terms)

    def retrieve(self, query: str, filters: SearchFilters | None = None, top_k: int | None = None) -> list[Evidence]:
        """Return up to ``top_k`` evidence chunks, most similar first."""
        filters = filters or SearchFilters()
        vector = to_vector_literal(self.embedder.embed([query])[0])
        # HNSW is approximate and SQL filters are applied after it, so a small
        # search breadth can miss the best matching chunk (see settings.yaml).
        # SET cannot take a bind parameter; the value is a validated int.
        self.conn.execute(f"SET hnsw.ef_search = {int(self.settings.hnsw_ef_search)}")
        conditions, params = ["TRUE"], {"vec": vector, "limit": self.settings.candidate_k}
        if filters.year_min is not None:
            conditions.append("p.year >= %(year_min)s")
            params["year_min"] = filters.year_min
        if filters.year_max is not None:
            conditions.append("p.year <= %(year_max)s")
            params["year_max"] = filters.year_max
        if filters.topics:
            conditions.append("p.topics && %(topics)s::text[]")
            params["topics"] = filters.topics
        if filters.content_sources:
            conditions.append("c.content_source = ANY(%(sources)s)")
            params["sources"] = filters.content_sources
        exclude = filters.exclude_sections or list(self.settings.exclude_sections)
        if exclude:
            conditions.append("COALESCE(c.section, '') <> ALL(%(exclude)s)")
            params["exclude"] = exclude
        if filters.paper_ids:
            conditions.append("c.paper_id = ANY(%(paper_ids)s)")
            params["paper_ids"] = filters.paper_ids

        def search(where: str, order: str) -> list[dict[str, Any]]:
            return self.conn.execute(
                f"""SELECT c.chunk_id, c.paper_id, c.chunk_index, c.content_source, c.section,
                           c.page_start, c.page_end, c.text, c.source_url,
                           p.title, p.authors, p.year, p.doi, p.arxiv_id, p.url AS paper_url,
                           1 - (c.embedding <=> %(vec)s::vector) AS similarity
                    FROM paper_chunks c JOIN papers p USING (paper_id)
                    WHERE {where}
                    ORDER BY {order}
                    LIMIT %(limit)s""",
                params,
            ).fetchall()

        where = " AND ".join(conditions)
        with self.conn.transaction():
            # Left to itself the planner flips between index and exact scan as table statistics change.
            if self.settings.exact_vector_search:
                self.conn.execute("SET LOCAL enable_indexscan = off")
            rows = search(where, "c.embedding <=> %(vec)s::vector")
        params["lexical"] = self._lexical_query(query) if self.settings.hybrid_search else None
        if params["lexical"]:
            lexical = search(
                f"{where} AND c.text_search @@ %(lexical)s::tsquery",
                "ts_rank(c.text_search, %(lexical)s::tsquery) DESC, c.chunk_id",
            )
            rows = fuse_rankings([rows, lexical], self.settings.rrf_k)
        if self.reranker and rows:
            scores = self.reranker.scores(query, [r["text"] for r in rows])
            rows = [rows[i] for i in sorted(range(len(rows)), key=scores.__getitem__, reverse=True)]

        selected = select_diverse(rows, top_k or self.settings.top_k, self.settings.max_chunks_per_paper)
        return [
            Evidence(
                evidence_id=f"E{i}",
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                chunk_index=r["chunk_index"],
                title=r["title"],
                authors=list(r["authors"] or []),
                year=r["year"],
                doi=r["doi"],
                arxiv_id=r["arxiv_id"],
                paper_url=r["paper_url"],
                source_url=r["source_url"],
                section=r["section"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                content_source=r["content_source"],
                text=r["text"],
                similarity=float(r["similarity"]),
            )
            for i, r in enumerate(selected, start=1)
        ]
