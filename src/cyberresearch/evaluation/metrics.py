"""Pure metric functions (no database, no model), unit-tested."""

from __future__ import annotations

import re
import statistics
from typing import Iterable, Sequence

_BRACKET_REF = re.compile(r"\[\d{1,3}\]")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the first ``k`` ranks that are relevant (missing ranks count as misses)."""
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float | None:
    """Share of relevant items found in the first ``k`` ranks; None if nothing is relevant."""
    if not relevant:
        return None
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1 / rank
    return 0.0


def key_point_coverage(text: str, groups: list[list[str]]) -> float | None:
    """Share of key-point groups with at least one term in ``text`` (case-insensitive substring)."""
    if not groups:
        return None
    lowered = text.lower()
    return sum(any(term.lower() in lowered for term in group) for group in groups) / len(groups)


def looks_like_references(text: str) -> bool:
    """Heuristic: a chunk that is mostly a bibliography (many [n] markers and years)."""
    return len(_BRACKET_REF.findall(text)) >= 4 and len(_YEAR.findall(text)) >= 4


def summarize(values: Iterable[float | None], digits: int = 3) -> dict[str, float | int | None]:
    """Mean, median, p95, min, max of the non-None values."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    p95 = data[min(len(data) - 1, round(0.95 * (len(data) - 1)))]
    return {
        "n": len(data),
        "mean": round(statistics.fmean(data), digits),
        "p50": round(statistics.median(data), digits),
        "p95": round(p95, digits),
        "min": round(data[0], digits),
        "max": round(data[-1], digits),
    }


def rate(numerator: int, denominator: int, digits: int = 3) -> float | None:
    return round(numerator / denominator, digits) if denominator else None
