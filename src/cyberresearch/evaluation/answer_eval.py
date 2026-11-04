"""Answer quality and end-to-end latency (calls the answer generator).

Every number comes from the validated answer the user would see:
  citation validity  = citations whose quote was found in the cited chunk /
                       all citations the model produced
  unsupported claims = "evidence" claims with no valid citation / all
                       evidence claims the model produced (these are hidden
                       from the user; the rate measures the model, not the UI)
  evidence relevance = cited papers judged relevant / distinct cited papers
  completeness       = key-point coverage of the shown text (lexical proxy)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cyberresearch.evaluation.metrics import key_point_coverage, rate, summarize
from cyberresearch.rag.engine import RAGEngine, RAGResponse
from cyberresearch.rag.llm import AnswerGenerator, LLMResult

ABSTAIN = ("insufficient_evidence", "out_of_scope")


@dataclass
class TimedGenerator:
    """Wraps an answer generator and records how long each call took."""

    inner: AnswerGenerator
    calls: list[tuple[str, float]] = field(default_factory=list)

    def generate(self, system: str, user_message: str, schema: dict, **kwargs: Any) -> LLMResult:
        start = time.perf_counter()
        try:
            return self.inner.generate(system, user_message, schema, **kwargs)
        finally:
            self.calls.append(("generate", time.perf_counter() - start))

    def translate_query(self, question: str) -> str:
        start = time.perf_counter()
        try:
            return self.inner.translate_query(question)
        finally:
            self.calls.append(("translate", time.perf_counter() - start))


def shown_text(response: RAGResponse) -> str:
    a = response.answer
    return " ".join([a.summary, *(c.statement for c in a.claims), a.ambiguity_note, a.limitations])


def score_response(q: dict[str, Any], response: RAGResponse, keys: dict[int, str]) -> dict[str, Any]:
    a = response.answer
    all_claims = a.claims + a.rejected_claims
    evidence_claims = [c for c in all_claims if c.kind == "evidence"]
    citations = [c for claim in all_claims for c in claim.citations]
    by_id = {e.evidence_id: e for e in response.evidence}
    cited_papers = {keys[by_id[c.evidence_id].paper_id] for claim in a.claims for c in claim.accepted}
    row: dict[str, Any] = {
        "id": q["id"],
        "type": q["type"],
        "hard": bool(q.get("hard")),
        "status": a.status,
        "gated": response.gated,
        "best_similarity": round(response.best_similarity, 3),
        "retrieval_query": response.retrieval_query,
        "citations": len(citations),
        "valid_citations": sum(c.valid for c in citations),
        "evidence_claims": len(evidence_claims),
        "unsupported_claims": sum(not c.supported for c in evidence_claims),
        "shown_claims": len(a.claims),
        "cited_papers": sorted(cited_papers),
        "answer": shown_text(response).strip(),
        "warnings": a.warnings,
    }
    if q["type"] == "answerable":
        relevant = set(q["relevant"])
        row["cited_relevant_papers"] = len(cited_papers & relevant)
        row["evidence_relevance"] = rate(len(cited_papers & relevant), len(cited_papers))
        row["completeness"] = (
            key_point_coverage(row["answer"], q["key_points"]) if a.status not in ABSTAIN else 0.0
        )
    return row


def evaluate_answers(
    engine: RAGEngine,
    timer: TimedGenerator,
    questions: list[dict[str, Any]],
    keys: dict[int, str],
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for q in questions:
        engine.reset()
        timer.calls.clear()
        start = time.perf_counter()
        response = engine.ask(q["question"])
        total = time.perf_counter() - start
        row = score_response(q, response, keys)
        llm = sum(t for kind, t in timer.calls if kind == "generate")
        translate = sum(t for kind, t in timer.calls if kind == "translate")
        row.update(
            end_to_end_s=round(total, 2),
            llm_s=round(llm, 2) if timer.calls else None,
            translate_s=round(translate, 2) if translate else None,
            retrieval_and_validation_s=round(total - llm - translate, 3),
            usage=response.llm.usage if response.llm else {},
        )
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] == "answerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    generated = [r for r in rows if r["llm_s"] is not None]

    def count(items: list[dict[str, Any]], *statuses: str) -> int:
        return sum(r["status"] in statuses for r in items)

    return {
        "questions": len(rows),
        "citation_validity_rate": rate(sum(r["valid_citations"] for r in rows), sum(r["citations"] for r in rows)),
        "citations_total": sum(r["citations"] for r in rows),
        "unsupported_claim_rate": rate(sum(r["unsupported_claims"] for r in rows), sum(r["evidence_claims"] for r in rows)),
        "evidence_claims_total": sum(r["evidence_claims"] for r in rows),
        "unsupported_claims_shown_to_user": 0,  # by construction; checked in tests/test_rag_citations.py
        "evidence_relevance": summarize(r["evidence_relevance"] for r in answerable)["mean"],
        "completeness_keypoint_coverage": summarize(r["completeness"] for r in answerable)["mean"],
        "answerable": {
            "n": len(answerable),
            "answered": count(answerable, "answered"),
            "partial": count(answerable, "partial"),
            "abstained": count(answerable, *ABSTAIN),
            "error_or_refused": count(answerable, "error", "refused"),
            "answer_rate": rate(count(answerable, "answered", "partial"), len(answerable)),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "gated_before_llm": sum(r["gated"] for r in unanswerable),
            "abstained": count(unanswerable, *ABSTAIN),
            "partial": count(unanswerable, "partial"),
            "answered": count(unanswerable, "answered"),
            "correct_abstention_rate": rate(count(unanswerable, *ABSTAIN), len(unanswerable)),
            "hard_correct_abstention": f"{count([r for r in unanswerable if r['hard']], *ABSTAIN)}/{sum(r['hard'] for r in unanswerable)}",
        },
        "latency_s": {
            "end_to_end_all": summarize((r["end_to_end_s"] for r in rows), 2),
            "end_to_end_with_llm": summarize((r["end_to_end_s"] for r in generated), 2),
            "llm_generation": summarize((r["llm_s"] for r in generated), 2),
            "query_translation": summarize((r["translate_s"] for r in rows), 2),
            "retrieval_and_validation": summarize((r["retrieval_and_validation_s"] for r in rows), 3),
        },
        "output_tokens": summarize((r["usage"].get("output_tokens") for r in generated), 0),
    }
