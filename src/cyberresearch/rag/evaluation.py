"""Run the test questions and check each answer against category rules.

Pass rules (deliberately about safety, not wording):
  sufficient - answered/partial with >= 1 verified evidence citation
  weak       - never an unsupported answer: either answered/partial with
               every shown claim verified, or insufficient_evidence
  outside    - insufficient_evidence / out_of_scope (or gated), no claims
  ambiguous  - an ambiguity note, or partial/insufficient_evidence
In every category, no rejected citation may be shown to the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from cyberresearch.config import PROJECT_ROOT
from cyberresearch.rag.engine import RAGEngine, RAGResponse
from cyberresearch.rag.retrieval import Retriever

QUESTIONS_FILE = PROJECT_ROOT / "config" / "rag_test_questions.yaml"


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["questions"]


@dataclass
class EvalRow:
    id: str
    category: str
    best_similarity: float
    status: str
    gated: bool
    evidence_claims: int
    verified_citations: int
    rejected_citations: int
    ambiguity_note: bool
    passed: bool
    reason: str


def judge(category: str, response: RAGResponse) -> tuple[bool, str]:
    a = response.answer
    shown_evidence_claims = [c for c in a.claims if c.kind == "evidence"]
    verified = sum(len(c.accepted) for c in a.claims)
    # Shown claims are verified by construction; this guards the invariant.
    if any(not c.supported for c in a.claims):
        return False, "an unsupported claim was shown"
    if a.status in ("error", "refused"):
        return False, f"no answer ({a.status}): {a.summary[:120]}"
    if category == "sufficient":
        ok = a.status in ("answered", "partial") and verified > 0
        return ok, "verified answer" if ok else f"expected a cited answer, got {a.status}"
    if category == "weak":
        ok = a.status in ("insufficient_evidence", "partial", "answered")
        return ok, "no unsupported claims" if ok else f"unexpected status {a.status}"
    if category == "outside":
        ok = a.status in ("insufficient_evidence", "out_of_scope") and not shown_evidence_claims
        return ok, "refused to answer from the corpus" if ok else f"answered an out-of-corpus question ({a.status})"
    if category == "ambiguous":
        ok = bool(a.ambiguity_note) or a.status in ("partial", "insufficient_evidence")
        return ok, "ambiguity handled" if ok else "ambiguity not mentioned"
    return False, f"unknown category {category}"


def evaluate(
    engine: RAGEngine, questions: list[dict[str, Any]], on_row: Callable[[EvalRow], None] | None = None
) -> list[EvalRow]:
    """Run every question; ``on_row`` is called as soon as each one finishes
    (with a slow local model a full run takes over half an hour)."""
    rows = []
    for q in questions:
        engine.reset()  # every test question is independent
        response = engine.ask(q["question"])
        passed, reason = judge(q["category"], response)
        a = response.answer
        rows.append(
            EvalRow(
                id=q["id"],
                category=q["category"],
                best_similarity=round(response.best_similarity, 3),
                status=a.status,
                gated=response.gated,
                evidence_claims=sum(c.kind == "evidence" for c in a.claims),
                verified_citations=sum(len(c.accepted) for c in a.claims),
                rejected_citations=len(a.invalid_citations),
                ambiguity_note=bool(a.ambiguity_note),
                passed=passed,
                reason=reason,
            )
        )
        if on_row:
            on_row(rows[-1])
    return rows


def retrieval_report(retriever: Retriever, questions: list[dict[str, Any]], min_similarity: float) -> list[dict[str, Any]]:
    """Retrieval-only check (no Claude call): best similarity and top papers per question."""
    report = []
    for q in questions:
        evidence = retriever.retrieve(q["question"])
        best = max((e.similarity for e in evidence), default=0.0)
        report.append(
            {
                "id": q["id"],
                "category": q["category"],
                "best_similarity": round(best, 3),
                "gate": "send to Claude" if best >= min_similarity else "insufficient (no Claude call)",
                "top_titles": [f"{e.similarity:.2f} {(e.title or '')[:70]}" for e in evidence[:3]],
            }
        )
    return report


def format_rows(rows: list[EvalRow]) -> str:
    header = f"{'id':<18}{'category':<12}{'best':>6}  {'status':<22}{'claims':>7}{'cites':>6}{'rej':>5}  {'pass':<5}reason"
    lines = [header, "-" * len(header)]
    for r in rows:
        status = r.status + (" (gated)" if r.gated else "")
        lines.append(
            f"{r.id:<18}{r.category:<12}{r.best_similarity:>6.2f}  {status:<22}{r.evidence_claims:>7}"
            f"{r.verified_citations:>6}{r.rejected_citations:>5}  {'yes' if r.passed else 'NO':<5}{r.reason}"
        )
    passed = sum(r.passed for r in rows)
    lines.append(f"\n{passed}/{len(rows)} passed")
    return "\n".join(lines)
