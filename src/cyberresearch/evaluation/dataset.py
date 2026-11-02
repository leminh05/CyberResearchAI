"""The evaluation dataset: questions plus relevance judgments keyed by stable ids."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import psycopg
import yaml

from cyberresearch.config import PROJECT_ROOT

GOLDEN_SET = PROJECT_ROOT / "config" / "eval" / "golden_set.yaml"
KEY_FIELDS = ("doi", "arxiv_id", "openalex_id", "s2_paper_id")
_PREFIX = {"doi": "doi", "arxiv_id": "arxiv", "openalex_id": "openalex", "s2_paper_id": "s2"}


def paper_key(row: Mapping[str, Any]) -> str:
    """Stable key of a paper: the first identifier it has (paper_id is not stable)."""
    for name in KEY_FIELDS:
        if row.get(name):
            return f"{_PREFIX[name]}:{row[name]}"
    return f"paper_id:{row['paper_id']}"


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        questions = yaml.safe_load(fh)["questions"]
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in the golden set")
    for q in questions:
        if q["type"] not in ("answerable", "unanswerable"):
            raise ValueError(f"{q['id']}: unknown type {q['type']!r}")
        q.setdefault("relevant", [])
        q.setdefault("key_points", [])
    return questions


def key_index(conn: psycopg.Connection) -> dict[int, str]:
    """paper_id -> stable key for every paper in the database."""
    rows = conn.execute(f"SELECT paper_id, {', '.join(KEY_FIELDS)} FROM papers").fetchall()
    return {r["paper_id"]: paper_key(r) for r in rows}
