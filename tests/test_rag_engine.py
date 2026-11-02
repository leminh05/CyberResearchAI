"""Retrieval filters, RAG engine flow, Claude client request shape, terminal chat."""

import io
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from cyberresearch.config import RagSettings
from cyberresearch.ingestion.schema import PaperRecord
from cyberresearch.processing.chunking import Chunk
from cyberresearch.processing.store import ChunkStore
from cyberresearch.rag.chat import apply_filter, run_chat
from cyberresearch.rag.engine import RAGEngine
from cyberresearch.rag.evaluation import judge
from cyberresearch.rag.llm import FALLBACK_BETA, ClaudeClient, LLMError, LLMResult
from cyberresearch.rag.prompts import ANSWER_SCHEMA
from cyberresearch.rag.retrieval import EmbeddingModelMismatch, Retriever, SearchFilters, select_diverse
from fakes import HashEmbedder

SETTINGS = RagSettings(top_k=4, candidate_k=20, max_chunks_per_paper=2, min_similarity=0.2, weak_similarity=0.4)
IDS_TEXT = "graph neural networks detect network intrusion using flow graphs of hosts and lateral movement"


# ------------------------------------------------------------- fixtures

@pytest.fixture
def corpus(store):
    """Three papers: two on intrusion detection (2024, 2026), one on cooking (2020)."""
    chunk_store = ChunkStore(store.conn, 384)
    chunk_store.init_schema()
    emb = HashEmbedder()
    papers = {
        "W1": dict(title="GNN Intrusion Detection", year=2024, doi="10.1/gnn", authors=["Ann Lee", "Bo Tran"],
                   texts=[IDS_TEXT, IDS_TEXT + " evaluation", IDS_TEXT + " results", "Front page affiliations"],
                   sections=["2 Method", "4 Evaluation", "5 Results", "Front Matter"], topic="intrusion_detection"),
        "W2": dict(title="Drift in IDS", year=2026, doi=None, authors=["Cy Do"],
                   texts=["concept drift degrades network intrusion detection accuracy over time"],
                   sections=["Abstract"], topic="intrusion_detection", source="abstract"),
        "W3": dict(title="Baking Bread", year=2020, doi=None, authors=[], texts=["sourdough bread baking recipe"],
                   sections=["1 Intro"], topic="network_security"),
    }
    ids = {}
    for key, p in papers.items():
        store.upsert_record(PaperRecord(source="openalex", source_id=key, openalex_id=key, title=p["title"],
                                        year=p["year"], doi=p["doi"], authors=p["authors"], url=f"https://x/{key}"),
                            p["topic"])
        pid = store.conn.execute("SELECT paper_id FROM papers WHERE openalex_id=%s", (key,)).fetchone()["paper_id"]
        ids[key] = pid
        source = p.get("source", "pdf")
        chunks = [Chunk(i, t, s, None if source == "abstract" else i + 1, None if source == "abstract" else i + 1,
                        len(t.split())) for i, (t, s) in enumerate(zip(p["texts"], p["sections"]))]
        chunk_store.replace_chunks(pid, chunks, emb.embed(p["texts"]), source, "https://x/pdf", emb.model_name)
    return ids


@pytest.fixture
def retriever(store, corpus):
    return Retriever(store.conn, HashEmbedder(), SETTINGS)


class FakeGenerator:
    def __init__(self, data=None, stop_reason="end_turn", error=None):
        self.data, self.stop_reason, self.error = data, stop_reason, error
        self.calls = []

    def generate(self, system, user_message, schema):
        self.calls.append(user_message)
        if self.error:
            raise self.error
        return LLMResult(self.data, self.stop_reason, "claude-opus-5-5",
                         "cyber" if self.stop_reason == "refusal" else None)


def good_answer(evidence_id="E1"):
    return {"answer_status": "answered", "summary": "GNNs use flow graphs.", "ambiguity_note": "",
            "limitations": "", "follow_up_suggestions": ["What about drift?"],
            "claims": [{"statement": "GNNs build flow graphs of hosts.", "kind": "evidence",
                        "citations": [{"evidence_id": evidence_id, "quote": "detect network intrusion using flow graphs"}]}]}


# ------------------------------------------------------------ retrieval

def test_select_diverse_caps_chunks_per_paper() -> None:
    rows = [{"paper_id": 1}, {"paper_id": 1}, {"paper_id": 1}, {"paper_id": 2}]
    assert [r["paper_id"] for r in select_diverse(rows, top_k=3, max_per_paper=2)] == [1, 1, 2]


@pytest.mark.db
def test_retrieve_ranks_relevant_chunks_with_metadata(retriever, corpus) -> None:
    evidence = retriever.retrieve("graph neural networks network intrusion detection")
    assert [e.evidence_id for e in evidence] == [f"E{i}" for i in range(1, len(evidence) + 1)]
    assert evidence[0].paper_id == corpus["W1"] and evidence[0].similarity > evidence[-1].similarity
    assert sum(e.paper_id == corpus["W1"] for e in evidence) == 2  # per-paper cap
    assert all(e.section != "Front Matter" for e in evidence)     # excluded by default
    first = evidence[0]
    assert first.title == "GNN Intrusion Detection" and first.authors == ["Ann Lee", "Bo Tran"]
    assert first.canonical_url == "https://doi.org/10.1/gnn" and first.location.startswith("p.")


@pytest.mark.db
def test_metadata_filters(retriever, corpus) -> None:
    q = "network intrusion detection"
    assert {e.paper_id for e in retriever.retrieve(q, SearchFilters(year_min=2025))} == {corpus["W2"]}
    assert corpus["W2"] not in {e.paper_id for e in retriever.retrieve(q, SearchFilters(year_max=2024))}
    assert {e.paper_id for e in retriever.retrieve(q, SearchFilters(topics=["network_security"]))} == {corpus["W3"]}
    only_abstract = retriever.retrieve(q, SearchFilters(content_sources=["abstract"]))
    assert {e.content_source for e in only_abstract} == {"abstract"} and only_abstract[0].location == "abstract"


@pytest.mark.db
def test_embedding_model_mismatch_is_detected(store, corpus) -> None:
    other = HashEmbedder()
    other.model_name = "sentence-transformers/other-model"
    with pytest.raises(EmbeddingModelMismatch):
        Retriever(store.conn, other, SETTINGS).check_embedding_model()
    Retriever(store.conn, HashEmbedder(), SETTINGS).check_embedding_model()  # same model: fine


# --------------------------------------------------------------- engine

@pytest.mark.db
def test_engine_answers_with_validated_citations(retriever) -> None:
    gen = FakeGenerator(good_answer())
    response = RAGEngine(retriever, gen, SETTINGS).ask("graph neural networks network intrusion detection")
    assert response.answer.status == "answered" and len(response.answer.references) == 1
    assert response.answer.references[0].evidence.title == "GNN Intrusion Detection"
    assert '<evidence id="E1"' in gen.calls[0]


@pytest.mark.db
def test_engine_gates_irrelevant_questions_without_calling_claude(retriever) -> None:
    gen = FakeGenerator(good_answer())
    engine = RAGEngine(retriever, gen, SETTINGS)
    response = engine.ask("football world cup winner")
    assert response.gated and response.answer.status == "insufficient_evidence"
    assert gen.calls == [] and "No answer is given" in response.answer.summary


@pytest.mark.db
def test_engine_rejects_hallucinated_citation(retriever) -> None:
    response = RAGEngine(retriever, FakeGenerator(good_answer("E99")), SETTINGS).ask("graph neural networks intrusion")
    assert response.answer.status == "insufficient_evidence" and response.answer.references == []


@pytest.mark.db
def test_follow_up_uses_previous_question(retriever) -> None:
    gen = FakeGenerator(good_answer())
    engine = RAGEngine(retriever, gen, SETTINGS)
    engine.ask("graph neural networks network intrusion detection")
    second = engine.ask("what about its evaluation?")
    assert second.retrieval_query.startswith("graph neural networks") and "evaluation" in second.retrieval_query
    assert "<conversation_history>" in gen.calls[1] and "GNNs use flow graphs." in gen.calls[1]
    engine.reset()
    assert engine.retrieval_query("x") == "x"


@pytest.mark.db
def test_engine_handles_refusal_and_api_errors(retriever) -> None:
    refused = RAGEngine(retriever, FakeGenerator(stop_reason="refusal"), SETTINGS).ask("network intrusion detection")
    assert refused.answer.status == "refused" and "cyber" in refused.answer.summary
    failed = RAGEngine(retriever, FakeGenerator(error=LLMError("boom")), SETTINGS).ask("network intrusion detection")
    assert failed.answer.status == "error" and "boom" in failed.answer.summary


def test_is_mostly_english() -> None:
    from cyberresearch.rag.engine import is_mostly_english

    assert is_mostly_english("How are GNNs used for IDS?")
    assert not is_mostly_english("Mô hình ngôn ngữ được dùng như thế nào để phát hiện lỗ hổng?")
    assert is_mostly_english("1234 ?!")


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate_query(self, question):
        self.calls.append(question)
        return "graph neural networks network intrusion detection"


@pytest.mark.db
def test_non_english_question_is_searched_in_english_once(retriever) -> None:
    translator, gen = FakeTranslator(), FakeGenerator(good_answer())
    engine = RAGEngine(retriever, gen, SETTINGS, translator=translator)
    vi = "Mạng nơ-ron đồ thị phát hiện xâm nhập mạng như thế nào?"
    response = engine.ask(vi)
    assert response.retrieval_query == "graph neural networks network intrusion detection"
    assert response.answer.status == "answered"
    assert f"<question>\n{vi}\n</question>" in gen.calls[0]  # Claude still sees the original question
    engine.ask("Còn đánh giá thì sao?")
    assert translator.calls.count(vi) == 1  # cached for the follow-up
    english = RAGEngine(retriever, gen, SETTINGS, translator=translator)
    english.ask("graph neural networks intrusion")
    assert len(translator.calls) == 2  # English questions are not translated


# --------------------------------------------------------- Claude client

class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def fake_anthropic(response=None, error=None):
    messages = FakeMessages(response, error)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def fake_response(text='{"answer_status": "answered"}', stop_reason="end_turn", category=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, model="claude-opus-5-5",
        stop_details=SimpleNamespace(category=category) if category else None,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


def test_claude_client_request_shape() -> None:
    client, messages = fake_anthropic(fake_response())
    result = ClaudeClient(RagSettings(), client).generate("sys", "user", ANSWER_SCHEMA)
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "sys"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert kw["output_config"]["effort"] == "medium"
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert "thinking" not in kw and "temperature" not in kw
    assert result.data == {"answer_status": "answered"} and result.usage["output_tokens"] == 50


def test_claude_client_refusal_and_truncation() -> None:
    client, _ = fake_anthropic(fake_response(text="", stop_reason="refusal", category="cyber"))
    result = ClaudeClient(RagSettings(), client).generate("s", "u", ANSWER_SCHEMA)
    assert result.stop_reason == "refusal" and result.refusal_category == "cyber" and result.data is None
    client, _ = fake_anthropic(fake_response(text='{"answer_st', stop_reason="max_tokens"))
    assert ClaudeClient(RagSettings(), client).generate("s", "u", ANSWER_SCHEMA).data is None


def test_translate_query_uses_low_effort_and_falls_back() -> None:
    client, messages = fake_anthropic(fake_response(text='{"english_query": "LLM vulnerability detection"}'))
    assert ClaudeClient(RagSettings(), client).translate_query("phát hiện lỗ hổng") == "LLM vulnerability detection"
    assert messages.kwargs["output_config"]["effort"] == "low" and messages.kwargs["max_tokens"] == 1024
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    client, _ = fake_anthropic(error=error)
    assert ClaudeClient(RagSettings(), client).translate_query("phát hiện lỗ hổng") == "phát hiện lỗ hổng"


def test_claude_client_maps_connection_errors() -> None:
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    client, _ = fake_anthropic(error=error)
    with pytest.raises(LLMError, match="cannot reach"):
        ClaudeClient(RagSettings(), client).generate("s", "u", ANSWER_SCHEMA)


# ------------------------------------------------------------------ chat

@pytest.mark.db
def test_terminal_chat_session(retriever) -> None:
    engine = RAGEngine(retriever, FakeGenerator(good_answer()), SETTINGS)
    inputs = iter(["/citations", "graph neural networks network intrusion detection", "/citations", "/evidence",
                   "/filter year>=2025", "/filters", "/new", "/unknown", "/exit"])
    out = io.StringIO()
    run_chat(engine, read=lambda prompt: next(inputs), out=out)
    text = out.getvalue()
    assert "Ask a question first." in text
    assert "Status: ANSWERED" in text and "[1]" in text and "https://doi.org/10.1/gnn" in text
    assert 'quote: "detect network intrusion using flow graphs"' in text
    assert "Retrieval query:" in text and "E1 [" in text
    assert "Active filters: year 2025-..." in text and "New conversation started." in text
    assert "Unknown command" in text and text.rstrip().endswith("Bye.")
    assert engine.history == []  # /new cleared it


def test_chat_exits_on_eof() -> None:
    def eof(prompt):
        raise EOFError

    out = io.StringIO()
    run_chat(SimpleNamespace(), read=eof, out=out)
    assert "Bye." in out.getvalue()


def test_apply_filter_validation() -> None:
    f = SearchFilters()
    assert "Year must be a number" in apply_filter(f, "year>=abc")
    assert "pdf or abstract" in apply_filter(f, "source=html")
    apply_filter(f, "topic=intrusion_detection,network_security")
    assert f.topics == ["intrusion_detection", "network_security"]
    assert apply_filter(f, "clear") == "Filters cleared." and f.topics == []


# ------------------------------------------------------------ evaluation

def _response(status, claims=(), note="", gated=False):
    answer = SimpleNamespace(status=status, claims=list(claims), ambiguity_note=note, invalid_citations=[], summary="")
    return SimpleNamespace(answer=answer, gated=gated)


def test_judge_rules() -> None:
    supported = SimpleNamespace(kind="evidence", supported=True, accepted=[object()])
    assert judge("sufficient", _response("answered", [supported]))[0]
    assert not judge("sufficient", _response("insufficient_evidence"))[0]
    assert judge("outside", _response("insufficient_evidence", gated=True))[0]
    assert not judge("outside", _response("answered", [supported]))[0]
    assert judge("ambiguous", _response("answered", [supported], note="several meanings"))[0]
    assert not judge("weak", _response("error"))[0]
