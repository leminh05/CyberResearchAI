"""Build the benchmark dataset: raw source records as JSON lines.

The base dataset is exported from ``paper_sources`` (real API payloads, *not*
yet normalized). To measure scalability it can be enlarged synthetically:
every copy gets new identifiers and a share of copies are deliberate
duplicates with re-formatted DOIs and titles, so deduplication has real work
to do. The README states clearly that the large sizes are synthetic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from cyberresearch.ingestion.normalize import reconstruct_abstract

FIELDS = ("source", "source_id", "title", "doi", "arxiv_id", "s2_paper_id", "openalex_id", "year", "abstract", "venue")


def flatten_source_record(source: str, source_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Pick the raw (un-normalized) metadata fields of one source payload."""
    if source == "arxiv":
        return {
            "source": source, "source_id": source_id, "title": raw.get("title"), "doi": raw.get("doi"),
            "arxiv_id": raw.get("id"), "s2_paper_id": None, "openalex_id": None,
            "year": (raw.get("published") or "")[:4] or None, "abstract": None,
            "venue": raw.get("journal_ref"),
        }
    if source == "openalex":
        location = raw.get("primary_location") or {}
        return {
            "source": source, "source_id": source_id, "title": raw.get("title") or raw.get("display_name"),
            "doi": raw.get("doi"), "arxiv_id": None, "s2_paper_id": None, "openalex_id": raw.get("id"),
            "year": raw.get("publication_year"),
            "abstract": reconstruct_abstract(raw.get("abstract_inverted_index")),
            "venue": (location.get("source") or {}).get("display_name"),
        }
    external = raw.get("externalIds") or {}
    return {
        "source": source, "source_id": source_id, "title": raw.get("title"), "doi": external.get("DOI"),
        "arxiv_id": external.get("ArXiv"), "s2_paper_id": raw.get("paperId"), "openalex_id": None,
        "year": raw.get("year"), "abstract": raw.get("abstract"), "venue": raw.get("venue"),
    }


def export_base_records(conn: Any) -> list[dict[str, Any]]:
    """Read every source record from the database as a flat raw dict."""
    rows = conn.execute("SELECT source, source_id, raw FROM paper_sources ORDER BY id").fetchall()
    return [flatten_source_record(r["source"], r["source_id"], r["raw"]) for r in rows]


def _reformat_doi(doi: str, rng: random.Random) -> str:
    """Same DOI written differently, as different APIs do."""
    bare = doi.replace("https://doi.org/", "")
    return rng.choice([f"https://doi.org/{bare.upper()}", f"doi:{bare}", bare.upper(), f" {bare} "])


def scale_records(
    base: list[dict[str, Any]], target_size: int, duplicate_rate: float = 0.2, seed: int = 42
) -> list[dict[str, Any]]:
    """Return ``target_size`` records built from ``base``.

    Copies receive new identifiers; ``duplicate_rate`` of the copies instead
    reuse an earlier record's DOI/title in a different format (true duplicates).
    """
    rng = random.Random(seed)
    records = [dict(r) for r in base[:target_size]]
    copy = 0
    while len(records) < target_size:
        copy += 1
        template = base[(len(records)) % len(base)]
        record = dict(template)
        record["source_id"] = f"{template['source_id']}-syn{copy}"
        if rng.random() < duplicate_rate and template.get("doi"):
            record["doi"] = _reformat_doi(template["doi"], rng)  # duplicate of the template
            record["title"] = (template.get("title") or "").upper()
            record["arxiv_id"] = record["s2_paper_id"] = record["openalex_id"] = None
        else:
            record["doi"] = f"10.99999/syn.{copy}" if template.get("doi") else None
            record["title"] = f"{template.get('title') or 'Untitled'} [variant {copy}]"
            record["arxiv_id"] = None
            record["s2_paper_id"] = f"syn{copy}" if template.get("s2_paper_id") else None
            record["openalex_id"] = f"https://openalex.org/WSYN{copy}" if template.get("openalex_id") else None
        records.append(record)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    """Write records as JSON lines; returns the file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps({k: record.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")
    return path.stat().st_size
