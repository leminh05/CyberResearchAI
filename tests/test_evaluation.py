"""Phase 4 evaluation: metric functions, golden-set loading and answer scoring."""

import pytest

from cyberresearch.evaluation.answer_eval import aggregate_answers, score_response
from cyberresearch.evaluation.dataset import load_golden_set, paper_key
from cyberresearch.evaluation.metrics import (
    key_point_coverage,
    looks_like_references,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from cyberresearch.rag.citations import validate_answer
from cyberresearch.rag.engine import RAGResponse
from cyberresearch.rag.retrieval import SearchFilters
from test_rag_citations import EVIDENCE, TEXT_1, answer, claim

RANKED = ["a", "x", "b", "y", "z"]


def test_precision_counts_missing_ranks_as_misses() -> None:
    assert precision_at_k(RANKED, {"a", "b"}, 1) == 1.0
    assert precision_at_k(RANKED, {"a", "b"}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], {"a"}, 5) == 0.2


def test_recall_and_reciprocal_rank() -> None:
    assert recall_at_k(RANKED, {"a", "b", "c"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, set(), 3) is None
    assert reciprocal_rank(RANKED, {"b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"q"}) == 0.0


def test_key_point_coverage_any_of_groups() -> None:
    text = "A GNN models hosts; federated training keeps data local."
    assert key_point_coverage(text, [["graph neural", "gnn"], ["federated"], ["shap"]]) == pytest.approx(2 / 3)
    assert key_point_coverage(text, []) is None


def test_reference_heuristic() -> None:
    refs = "[1] A. B, 2021. [2] C. D, 2022. [3] E. F, 2023. [4] G. H, 2024."
    assert looks_like_references(refs)
    assert not looks_like_references(TEXT_1)


def test_summarize_ignores_none() -> None:
    s = summarize([1.0, None, 3.0, 2.0])
    assert s["n"] == 3 and s["mean"] == 2.0 and s["p50"] == 2.0 and s["max"] == 3.0
    assert summarize([None])["mean"] is None


def test_paper_key_prefers_doi_then_arxiv() -> None:
    assert paper_key({"paper_id": 1, "doi": "10.1/x", "arxiv_id": "2401.1"}) == "doi:10.1/x"
    assert paper_key({"paper_id": 1, "doi": None, "arxiv_id": "2401.1"}) == "arxiv:2401.1"
    assert paper_key({"paper_id": 7}) == "paper_id:7"


def test_golden_set_is_well_formed() -> None:
    questions = load_golden_set()
    answerable = [q for q in questions if q["type"] == "answerable"]
    assert len(answerable) >= 30 and len(questions) - len(answerable) >= 10
    assert all(q["relevant"] and q["key_points"] for q in answerable)
    assert all(k.split(":")[0] in ("doi", "arxiv", "openalex", "s2") for q in answerable for k in q["relevant"])


KEYS = {1: "doi:10.1/gnn", 2: "arxiv:2401.00001"}
Q = {"id": "q", "type": "answerable", "relevant": ["doi:10.1/gnn"], "key_points": [["graph"], ["drift"]]}


def response(data) -> RAGResponse:
    return RAGResponse("q?", "q?", SearchFilters(), EVIDENCE, validate_answer(data, EVIDENCE))


def test_score_counts_invalid_citations_and_unsupported_claims() -> None:
    data = answer(claims=[
        claim("Graph neural networks model hosts as nodes.", "evidence", ("E1", "model hosts as nodes and flows as edges")),
        claim("Made-up claim.", "evidence", ("E2", "this sentence is not in the chunk at all")),
    ])
    row = score_response(Q, response(data), KEYS)
    assert (row["citations"], row["valid_citations"]) == (2, 1)
    assert (row["evidence_claims"], row["unsupported_claims"]) == (2, 1)
    assert row["cited_papers"] == ["doi:10.1/gnn"] and row["evidence_relevance"] == 1.0
    assert row["completeness"] == 0.5  # "graph" is in the shown claim, "drift" is not
    assert row["status"] == "partial"


def test_abstention_scores_zero_completeness_and_aggregates() -> None:
    abstain = score_response(Q, response(answer(status="insufficient_evidence", summary="graph drift")), KEYS)
    assert abstain["completeness"] == 0.0
    outside = score_response({"id": "o", "type": "unanswerable"}, response(answer(status="insufficient_evidence")), KEYS)
    rows = [dict(r, end_to_end_s=1.0, llm_s=1.0, translate_s=None, retrieval_and_validation_s=0.1, usage={})
            for r in (abstain, outside)]
    agg = aggregate_answers(rows)
    assert agg["answerable"]["abstained"] == 1 and agg["unanswerable"]["correct_abstention_rate"] == 1.0
    assert agg["citation_validity_rate"] is None  # no citations at all


def test_large_corpus_labels_only_add_to_the_original_set() -> None:
    from cyberresearch.config import PROJECT_ROOT
    from cyberresearch.evaluation.dataset import load_golden_set

    original = {q["id"]: q for q in load_golden_set()}
    large = {q["id"]: q for q in load_golden_set(PROJECT_ROOT / "config" / "eval" / "golden_set_large.yaml")}
    assert large.keys() == original.keys()
    assert all(set(original[i]["relevant"]) <= set(large[i]["relevant"]) for i in original)
    assert {i for i in original if original[i]["type"] != large[i]["type"]} == {"hard-bgp", "hard-wpa3"}
