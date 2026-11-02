"""Data pipeline metrics computed from the database (Phases 1-2 outputs)."""

from __future__ import annotations

from typing import Any

import psycopg

from cyberresearch.evaluation.metrics import looks_like_references, rate
from cyberresearch.rag.engine import is_mostly_english

# Lexical proxy for "is this paper about security at all?" (title + abstract).
SECURITY_TERMS = (
    "secur", "attack", "intrusion", "malware", "threat", "cyber", "vulnerab",
    "privacy", "adversarial", "phishing", "fraud", "botnet", "ddos", "encrypt",
    "crypt", "authentic", "anomal", "forensic",
)
METADATA_FIELDS = {
    "title": "title IS NOT NULL AND title <> ''",
    "abstract": "abstract IS NOT NULL AND abstract <> ''",
    "authors": "jsonb_array_length(authors) > 0",
    "year": "year IS NOT NULL",
    "publication_date": "publication_date IS NOT NULL",
    "doi": "doi IS NOT NULL",
    "venue": "venue IS NOT NULL AND venue <> ''",
    "url": "url IS NOT NULL",
    "pdf_url": "pdf_url IS NOT NULL",
}


def _scalar(conn: psycopg.Connection, sql: str) -> Any:
    return next(iter(conn.execute(sql).fetchone().values()))


def data_quality(conn: psycopg.Connection) -> dict[str, Any]:
    papers = _scalar(conn, "SELECT count(*) FROM papers")
    sources = _scalar(conn, "SELECT count(*) FROM paper_sources")
    by_source = {r["source"]: r["n"] for r in conn.execute(
        "SELECT source, count(*) n FROM paper_sources GROUP BY 1 ORDER BY 1")}

    # Residual duplicates: different paper rows that still share a title.
    same_title = conn.execute(
        """SELECT normalized_title, count(*) n,
                  count(DISTINCT authors->>0) first_authors, count(authors->>0) with_author
           FROM papers WHERE normalized_title IS NOT NULL
           GROUP BY 1 HAVING count(*) > 1"""
    ).fetchall()
    # Same title AND every row has the same (non-missing) first author.
    likely_dupes = [r for r in same_title if r["first_authors"] == 1 and r["with_author"] == r["n"]]

    completeness = {
        name: rate(_scalar(conn, f"SELECT count(*) FROM papers WHERE {cond}"), papers)
        for name, cond in METADATA_FIELDS.items()
    }

    texts = conn.execute("SELECT title, abstract FROM papers").fetchall()
    on_topic = sum(
        any(t in f"{r['title'] or ''} {r['abstract'] or ''}".lower() for t in SECURITY_TERMS) for r in texts
    )
    english_titles = sum(is_mostly_english(r["title"] or "") for r in texts if r["title"])
    years = {r["year"]: r["n"] for r in conn.execute("SELECT year, count(*) n FROM papers GROUP BY 1 ORDER BY 1")}

    docs = {r["k"]: r["n"] for r in conn.execute(
        "SELECT pdf_status || '/' || extraction_status AS k, count(*) n FROM pdf_documents GROUP BY 1")}
    downloaded = sum(n for k, n in docs.items() if k.startswith("downloaded/"))
    extracted = docs.get("downloaded/extracted", 0)
    attempted = sum(n for k, n in docs.items() if k.split("/")[0] in ("downloaded", "failed", "robots_disallowed", "not_pdf", "too_large"))
    allowed = _scalar(conn, "SELECT count(*) FROM pdf_documents WHERE access_reason = 'arxiv' OR access_reason LIKE 'license:%%'")
    content = {r["content_source"]: r["n"] for r in conn.execute(
        "SELECT content_source, count(*) n FROM pdf_documents GROUP BY 1")}

    chunk_rows = conn.execute("SELECT content_source, word_count, text FROM paper_chunks").fetchall()
    reference_like = sum(looks_like_references(r["text"]) for r in chunk_rows)

    runs = conn.execute(
        """SELECT run_id, min(mode) mode, min(started_at) started, max(finished_at) finished,
                  extract(epoch FROM max(finished_at) - min(started_at))::float AS wall_seconds,
                  sum(fetched) fetched, sum(inserted) inserted, sum(merged) merged,
                  count(*) FILTER (WHERE status = 'failed') failed_units, count(*) units
           FROM ingestion_logs GROUP BY run_id ORDER BY started"""
    ).fetchall()
    total_wall = sum(r["wall_seconds"] or 0 for r in runs)
    total_fetched = sum(r["fetched"] for r in runs)

    return {
        "collection": {
            "papers": papers,
            "source_records": sources,
            "source_records_by_source": by_source,
            "multi_source_papers": _scalar(conn, "SELECT count(*) FROM (SELECT paper_id FROM paper_sources GROUP BY 1 HAVING count(*) > 1) t"),
            "publication_years": years,
        },
        "deduplication": {
            "records_merged_into_existing_papers": sources - papers,
            "cross_source_duplicate_rate": rate(sources - papers, sources),
            "residual_same_title_groups": len(same_title),
            "residual_likely_duplicates_same_title_and_first_author": len(likely_dupes),
            "residual_duplicate_rate": rate(sum(r["n"] - 1 for r in likely_dupes), papers),
            "papers_without_title": _scalar(conn, "SELECT count(*) FROM papers WHERE normalized_title IS NULL"),
        },
        "metadata_completeness": completeness,
        "topical_relevance": {
            "papers_with_security_term_in_title_or_abstract": rate(on_topic, papers),
            "english_titles": rate(english_titles, sum(1 for r in texts if r["title"])),
            "method": "lexical proxy: any of " + ", ".join(SECURITY_TERMS),
        },
        "pdf": {
            "documents_by_status": docs,
            "pdf_url_available_rate": completeness["pdf_url"],
            "legally_downloadable": allowed,
            "legally_downloadable_rate": rate(allowed, papers),
            "download_attempted": attempted,
            "downloaded": downloaded,
            "download_success_rate": rate(downloaded, attempted),
            "pdf_availability_rate": rate(downloaded, papers),
            "extraction_success_rate": rate(extracted, downloaded),
            "content_source": content,
            "papers_without_indexed_text": content.get("none", 0),
        },
        "chunks": {
            "total": len(chunk_rows),
            "pdf": sum(r["content_source"] == "pdf" for r in chunk_rows),
            "abstract": sum(r["content_source"] == "abstract" for r in chunk_rows),
            "mean_words": rate(sum(r["word_count"] for r in chunk_rows), len(chunk_rows), 1),
            "reference_like_chunks": reference_like,
            "reference_like_rate": rate(reference_like, len(chunk_rows)),
        },
        "ingestion": {
            "runs": [
                {
                    "mode": r["mode"],
                    "started": r["started"].isoformat(),
                    "wall_seconds": round(r["wall_seconds"] or 0, 1),
                    "units": r["units"],
                    "failed_units": r["failed_units"],
                    "fetched": r["fetched"],
                    "inserted": r["inserted"],
                    "merged": r["merged"],
                }
                for r in runs
            ],
            "total_wall_seconds": round(total_wall, 1),
            "records_fetched": total_fetched,
            "records_per_second": rate(total_fetched, total_wall, 2),
            "note": "wall time includes the per-source rate-limit waits (arXiv 3 s, Semantic Scholar 3 s without key)",
        },
    }
