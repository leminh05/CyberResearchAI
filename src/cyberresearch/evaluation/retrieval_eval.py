"""Retrieval quality against the judged golden set (no LLM calls).

Relevance is judged per paper, so a retrieved chunk counts as relevant when
its paper is relevant. Ranked papers = distinct papers in the order the
retriever returns their chunks (after the per-paper diversity cap).
"""

from __future__ import annotations

import time
from typing import Any

from cyberresearch.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank, summarize
from cyberresearch.rag.engine import is_mostly_english
from cyberresearch.rag.retrieval import Retriever

KS = (1, 3, 5, 10)


def evaluate_retrieval(
    retriever: Retriever,
    questions: list[dict[str, Any]],
    keys: dict[int, str],
    indexed: set[str],
) -> dict[str, Any]:
    """``keys``: paper_id -> stable key; ``indexed``: keys of papers that have chunks."""
    context_size = retriever.settings.top_k
    rows = []
    for q in questions:
        start = time.perf_counter()
        # candidate_k rows are always fetched; a larger top_k only keeps more of them.
        evidence = retriever.retrieve(q["question"], top_k=retriever.settings.candidate_k)
        latency = time.perf_counter() - start
        ranked = list(dict.fromkeys(keys[e.paper_id] for e in evidence))
        best = max((e.similarity for e in evidence), default=0.0)
        row: dict[str, Any] = {
            "id": q["id"],
            "type": q["type"],
            "english": is_mostly_english(q["question"]),
            "latency_s": round(latency, 4),
            "best_similarity": round(best, 3),
            "gated": best < retriever.settings.min_similarity,
        }
        if q["type"] == "answerable":
            relevant = set(q["relevant"]) & indexed
            context = evidence[:context_size]  # exactly what the LLM would see
            context_papers = {keys[e.paper_id] for e in context}
            row.update(
                relevant_judged=len(q["relevant"]),
                relevant_unindexed=len(set(q["relevant"]) - indexed),
                ranked_top10=ranked[:10],
                mrr=round(reciprocal_rank(ranked, relevant), 3),
                context_chunk_precision=round(sum(keys[e.paper_id] in relevant for e in context) / max(len(context), 1), 3),
                context_relevant_papers=len(context_papers & relevant),
                context_paper_recall=recall_at_k(list(context_papers), relevant, len(context_papers)),
                context_has_relevant=bool(context_papers & relevant),
            )
            for k in KS:
                row[f"p@{k}"] = round(precision_at_k(ranked, relevant, k), 3)
                row[f"r@{k}"] = recall_at_k(ranked, relevant, k)
                row[f"hit@{k}"] = bool(relevant.intersection(ranked[:k]))
        rows.append(row)

    answerable = [r for r in rows if r["type"] == "answerable" and r["english"]]
    aggregate: dict[str, Any] = {
        "questions_answerable_english": len(answerable),
        "mrr": summarize(r["mrr"] for r in answerable)["mean"],
        "context_size_chunks": context_size,
        "evidence_coverage": round(sum(r["context_has_relevant"] for r in answerable) / max(len(answerable), 1), 3),
        "context_chunk_precision": summarize(r["context_chunk_precision"] for r in answerable)["mean"],
        "context_paper_recall": summarize(r["context_paper_recall"] for r in answerable)["mean"],
        "relevant_judged_total": sum(r["relevant_judged"] for r in answerable),
        "relevant_unindexed_total": sum(r["relevant_unindexed"] for r in answerable),
        "latency_s": summarize((r["latency_s"] for r in rows), 4),
    }
    for k in KS:
        aggregate[f"precision@{k}"] = summarize(r[f"p@{k}"] for r in answerable)["mean"]
        aggregate[f"recall@{k}"] = summarize(r[f"r@{k}"] for r in answerable)["mean"]
        aggregate[f"hit@{k}"] = round(sum(r[f"hit@{k}"] for r in answerable) / max(len(answerable), 1), 3)
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    aggregate["unanswerable_gated_by_similarity"] = f"{sum(r['gated'] for r in unanswerable)}/{len(unanswerable)}"
    aggregate["non_english_rows"] = [
        {k: r[k] for k in ("id", "best_similarity", "mrr", "hit@10")} for r in rows if not r["english"] and r["type"] == "answerable"
    ]
    return {"aggregate": aggregate, "questions": rows}
