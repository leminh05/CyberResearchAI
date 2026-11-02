"""Plain-Python implementation of the metadata pipeline (benchmark baseline).

It implements *exactly* the same rules as ``spark_jobs.py`` so both produce
identical results on the same input:

1. normalize  - title whitespace, matching title, DOI, arXiv ID, IDs, year
2. filter     - drop records without a title and without any identifier,
                and records with an implausible year
3. deduplicate- key = first of arXiv ID, DOI, S2 ID, OpenAlex ID,
                normalized title + year; keep the most complete record
4. quality    - counts and null rates

Note: unlike ``ingestion.normalize.normalize_title`` (used when storing
papers), these rules do not transliterate accents, because Spark SQL has no
built-in for it; both benchmark implementations use the same simpler rule.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Regexes shared with the Spark implementation (same patterns, same order).
DOI_PREFIX = r"^(https?://(dx\.)?doi\.org/|doi:)"
ARXIV_PREFIX = r"^(arxiv:|https?://(export\.)?arxiv\.org/(abs|pdf)/)"
ARXIV_VERSION = r"v[0-9]+$"
OPENALEX_PREFIX = r"^https?://openalex\.org/"
NON_ALNUM = r"[^a-z0-9]+"
# ASCII whitespace only: Java regex "\s" and Python "\s" differ on Unicode
# spaces, so both implementations use this explicit class instead.
WHITESPACE = r"[ \t\n\r\f\x0B]+"
EDGE_WHITESPACE = r"^[ \t\n\r\f\x0B]+|[ \t\n\r\f\x0B]+$"
MIN_YEAR, MAX_YEAR = 1990, 2100
SOURCE_RANK = {"arxiv": 1, "openalex": 2, "semantic_scholar": 3}
COMPLETENESS_FIELDS = ("title", "doi", "arxiv_id", "s2_paper_id", "openalex_id", "year", "abstract", "venue")

_re = {
    p: re.compile(p)
    for p in (DOI_PREFIX, ARXIV_PREFIX, ARXIV_VERSION, OPENALEX_PREFIX, NON_ALNUM, WHITESPACE, EDGE_WHITESPACE)
}


def _strip(value: Any) -> str:
    """Strip ASCII whitespace (equivalent of the Spark regexp_replace)."""
    return _re[EDGE_WHITESPACE].sub("", value) if isinstance(value, str) else ("" if value is None else str(value))


def _blank_to_none(value: str | None) -> str | None:
    return value if value else None


def normalize_record(r: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw record (Python equivalent of ``spark_jobs.normalize_metadata``)."""
    title = _blank_to_none(_strip(_re[WHITESPACE].sub(" ", r.get("title") or "")))
    norm_title = _blank_to_none(_strip(_re[NON_ALNUM].sub(" ", (title or "").lower())))
    doi = _re[DOI_PREFIX].sub("", _strip(r.get("doi")).lower())
    doi = doi if doi.startswith("10.") else None
    arxiv = _re[ARXIV_PREFIX].sub("", _strip(r.get("arxiv_id")).lower())
    arxiv = _blank_to_none(_re[ARXIV_VERSION].sub("", arxiv))
    openalex = _blank_to_none(_re[OPENALEX_PREFIX].sub("", _strip(r.get("openalex_id"))))
    s2 = _blank_to_none(_strip(r.get("s2_paper_id")))
    year_text = _strip(r.get("year"))[:4]
    year = int(year_text) if year_text.isdigit() and len(year_text) == 4 else None
    return {
        "source": r.get("source"), "source_id": r.get("source_id"), "title": title,
        "normalized_title": norm_title, "doi": doi, "arxiv_id": arxiv, "s2_paper_id": s2,
        "openalex_id": openalex, "year": year,
        "abstract": _blank_to_none(_strip(r.get("abstract"))),
        "venue": _blank_to_none(_strip(r.get("venue"))),
    }


def passes_filter(r: dict[str, Any]) -> bool:
    has_identity = any(r[k] for k in ("normalized_title", "doi", "arxiv_id", "s2_paper_id", "openalex_id"))
    year_ok = r["year"] is None or MIN_YEAR <= r["year"] <= MAX_YEAR
    return has_identity and year_ok


def dedup_key(r: dict[str, Any]) -> str:
    for name in ("arxiv_id", "doi", "s2_paper_id", "openalex_id"):
        if r[name]:
            return f"{name}:{r[name]}"
    return f"title:{r['normalized_title']}|{r['year'] if r['year'] is not None else ''}"


def completeness(r: dict[str, Any]) -> int:
    return sum(r[f] is not None for f in COMPLETENESS_FIELDS)


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best record per key: most complete, then source rank, then id."""
    best: dict[str, dict[str, Any]] = {}
    for r in records:
        key = dedup_key(r)
        rank = (-completeness(r), SOURCE_RANK.get(r["source"], 9), r["source_id"])
        if key not in best or rank < best[key]["_rank"]:
            best[key] = {**r, "dedup_key": key, "_rank": rank}
    return [{k: v for k, v in r.items() if k != "_rank"} for r in best.values()]


def quality_report(normalized: list[dict[str, Any]], filtered: list[dict[str, Any]], deduped: list[dict[str, Any]]) -> dict[str, Any]:
    """Data quality metrics (same keys as the Spark version)."""
    total = len(normalized)
    nulls = Counter()
    for r in normalized:
        for f in COMPLETENESS_FIELDS:
            nulls[f] += r[f] is None
    return {
        "input_records": total,
        "after_filter": len(filtered),
        "after_dedup": len(deduped),
        "removed_by_filter": total - len(filtered),
        "duplicates_removed": len(filtered) - len(deduped),
        "null_rate": {f: round(nulls[f] / total, 4) if total else 0.0 for f in COMPLETENESS_FIELDS},
        "by_source": dict(sorted(Counter(r["source"] for r in deduped).items())),
    }


def run_pipeline(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read JSON lines, normalize, filter, deduplicate; return records + report."""
    with open(path, encoding="utf-8") as fh:
        normalized = [normalize_record(json.loads(line)) for line in fh]
    filtered = [r for r in normalized if passes_filter(r)]
    deduped = deduplicate(filtered)
    return deduped, quality_report(normalized, filtered, deduped)
