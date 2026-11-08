"""FastAPI endpoints: real retrieval and citation validation, fake LLM and embedder."""

import uuid
from dataclasses import replace

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from cyberresearch.config import WebSettings, load_settings
from cyberresearch.rag.llm import LLMError, LLMResult
from cyberresearch.web.app import MAX_BODY_BYTES, Services, create_app
from cyberresearch.web.limits import ConversationStore, RateLimiter
from cyberresearch.web.schemas import MAX_QUESTION_CHARS, safe_url
from fakes import HashEmbedder
from test_rag_engine import IDS_TEXT, SETTINGS, corpus  # noqa: F401  (fixture)

QUOTE = "graph neural networks detect network intrusion using flow graphs"


def good_answer(quote=QUOTE):
    return {
        "answer_status": "answered", "summary": "GNNs model flows.", "ambiguity_note": "", "limitations": "",
        "follow_up_suggestions": ["What about drift?"],
        "claims": [{"statement": "GNNs detect intrusions. See https://evil.example/x", "kind": "evidence",
                    "citations": [{"evidence_id": "E1", "quote": quote}]}],
    }


class Generator:
    def __init__(self, data=None, error=None, ready=True):
        self.data, self.error, self.ready = data or good_answer(), error, ready
        self.prompts = []

    def is_ready(self):
        return self.ready

    def translate_query(self, question):
        return question

    def generate(self, system, user_message, schema):
        self.prompts.append(user_message)
        if self.error:
            raise self.error
        return LLMResult(self.data, "end_turn", "fake-model")


@pytest.fixture
def make_client(store, corpus, database_url):  # noqa: F811
    schema = store.conn.execute("SELECT current_schema() AS s").fetchone()["s"]

    def connect():
        return psycopg.connect(database_url, autocommit=True, row_factory=dict_row,
                               options=f"-c search_path={schema},public")

    def make(generator=None, web=None, connect_fn=connect, alternatives=None, **kw):
        settings = replace(load_settings(), rag=SETTINGS, web=web or WebSettings())
        services = Services(settings, "ollama", generator or Generator(), HashEmbedder(), connect_fn, alternatives or {})
        return TestClient(create_app(services, **kw), raise_server_exceptions=False)

    return make


def test_chat_returns_verified_citations_with_database_links(make_client) -> None:
    response = make_client().post("/api/chat", json={"question": f"  {IDS_TEXT}  "})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered" and body["question"] == IDS_TEXT
    claim = body["claims"][0]
    assert "evil.example" not in claim["statement"]
    assert claim["citations"] == [{"reference": 1, "quote": QUOTE}]
    ref = body["references"][0]
    assert ref["number"] == 1 and ref["title"] == "GNN Intrusion Detection"
    assert ref["url"] == "https://doi.org/10.1/gnn" and ref["authors"] == ["Ann Lee", "Bo Tran"]
    assert ref["location"].startswith("p") and body["meta"]["model"] == "fake-model"
    assert uuid.UUID(body["conversation_id"])


def test_fabricated_quote_is_not_returned_as_citation(make_client) -> None:
    client = make_client(Generator(good_answer("this sentence was never in any retrieved chunk")))
    body = client.post("/api/chat", json={"question": IDS_TEXT}).json()
    assert body["status"] == "insufficient_evidence"
    assert body["claims"] == [] and body["references"] == [] and body["warnings"]


def test_gated_question_does_not_call_the_llm(make_client, store) -> None:
    store.conn.execute("DELETE FROM paper_chunks")  # nothing to retrieve: similarity 0 is below the gate
    generator = Generator()
    body = make_client(generator).post("/api/chat", json={"question": "chocolate cake"}).json()
    assert body["status"] == "insufficient_evidence" and body["meta"]["gated"] is True
    assert generator.prompts == []


def test_follow_up_uses_conversation_context_and_delete_forgets_it(make_client) -> None:
    generator = Generator()
    client = make_client(generator)
    first = client.post("/api/chat", json={"question": IDS_TEXT}).json()
    conversation_id = first["conversation_id"]

    follow_up = client.post("/api/chat", json={"question": "and its evaluation?", "conversation_id": conversation_id})
    assert follow_up.json()["conversation_id"] == conversation_id
    assert IDS_TEXT in follow_up.json()["meta"]["retrieval_query"]
    assert IDS_TEXT in generator.prompts[-1]

    other = client.post("/api/chat", json={"question": IDS_TEXT}).json()
    assert other["conversation_id"] != conversation_id

    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    gone = client.post("/api/chat", json={"question": "again", "conversation_id": conversation_id})
    assert gone.status_code == 404 and gone.json()["error"]["code"] == "conversation_not_found"


@pytest.mark.parametrize("payload", [
    {},
    {"question": "   "},
    {"question": "x" * (MAX_QUESTION_CHARS + 1)},
    {"question": "ok", "conversation_id": "not-a-uuid"},
    {"question": "ok", "unexpected": 1},
    {"question": 5},
])
def test_invalid_input_is_rejected_before_the_engine(make_client, payload) -> None:
    generator = Generator()
    response = make_client(generator).post("/api/chat", json=payload)
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"
    assert generator.prompts == []


def test_oversized_body_is_rejected(make_client) -> None:
    response = make_client().post("/api/chat", content=b"x" * (MAX_BODY_BYTES + 1))
    assert response.status_code == 413


def test_rate_limit_returns_429_with_retry_after(make_client) -> None:
    client = make_client(web=WebSettings(rate_limit_per_minute=1))
    assert client.post("/api/chat", json={"question": IDS_TEXT}).status_code == 200
    limited = client.post("/api/chat", json={"question": IDS_TEXT})
    assert limited.status_code == 429 and limited.json()["error"]["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1


def test_all_slots_busy_returns_503(make_client) -> None:
    during = []

    class Busy(Generator):
        def generate(self, system, user_message, schema):
            # A second request arrives while this one holds the only slot.
            during.append(client.post("/api/chat", json={"question": IDS_TEXT}))
            return super().generate(system, user_message, schema)

    client = make_client(Busy(), web=WebSettings(max_concurrent_requests=1))
    assert client.post("/api/chat", json={"question": IDS_TEXT}).status_code == 200
    assert during[0].status_code == 503 and during[0].json()["error"]["code"] == "busy"


def test_llm_failure_is_502_without_internal_details(make_client) -> None:
    client = make_client(Generator(error=LLMError("connection to http://10.0.0.5:11434 refused")))
    response = client.post("/api/chat", json={"question": IDS_TEXT})
    assert response.status_code == 502 and response.json()["error"]["code"] == "llm_failed"
    assert "10.0.0.5" not in response.text


def test_database_failure_is_503(make_client) -> None:
    def broken():
        raise psycopg.OperationalError("connection to server at secret-host failed")

    client = make_client(connect_fn=broken)
    response = client.post("/api/chat", json={"question": IDS_TEXT})
    assert response.status_code == 503 and response.json()["error"]["code"] == "database_unavailable"
    assert "secret-host" not in response.text
    health = client.get("/api/health")
    assert health.status_code == 503 and health.json()["status"] == "unavailable"


def test_unexpected_error_is_a_generic_500(make_client) -> None:
    client = make_client(Generator(error=RuntimeError("secret detail")))
    response = client.post("/api/chat", json={"question": IDS_TEXT})
    assert response.status_code == 500 and "secret detail" not in response.text


def test_health_reports_components_without_secrets(make_client, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-appear")
    body = make_client().get("/api/health")
    assert body.status_code == 200 and "sk-should" not in body.text
    assert body.json()["status"] == "ok" and body.json()["papers"] == 3 and body.json()["chunks"] > 0
    assert make_client(Generator(ready=False)).get("/api/health").json()["status"] == "degraded"


def test_security_headers_and_frontend_serving(make_client, tmp_path) -> None:
    (tmp_path / "index.html").write_text("<title>site</title>")
    client = make_client(frontend_dist=tmp_path)
    page = client.get("/")
    assert "site" in page.text
    assert "default-src 'self'" in page.headers["Content-Security-Policy"]
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/api/health").status_code == 200  # not shadowed by the static mount


@pytest.mark.parametrize("url,expected", [
    ("https://doi.org/10.1/x", "https://doi.org/10.1/x"),
    ("HTTP://x.example/p", "HTTP://x.example/p"),
    ("javascript:alert(1)", None),
    ("data:text/html,x", None),
    (None, None),
])
def test_safe_url_allows_only_http(url, expected) -> None:
    assert safe_url(url) == expected


def test_unsafe_paper_url_never_reaches_the_client(make_client, store) -> None:
    store.conn.execute("UPDATE papers SET doi = NULL, url = 'javascript:alert(1)' WHERE doi = '10.1/gnn'")
    body = make_client().post("/api/chat", json={"question": IDS_TEXT}).json()
    assert body["references"][0]["url"] is None


def test_rate_limiter_window_and_cleanup() -> None:
    now = [0.0]
    limiter = RateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.retry_after("a") == 0 and limiter.retry_after("a") == 0
    assert limiter.retry_after("a") == 60
    assert limiter.retry_after("b") == 0  # limits are per client
    now[0] = 61
    assert limiter.retry_after("a") == 0
    assert "b" not in limiter._hits


def test_conversation_store_expires_and_evicts() -> None:
    now = [0.0]
    conversations = ConversationStore(lambda: "engine", max_items=2, ttl=10, clock=lambda: now[0])
    first, _ = conversations.create()
    second, _ = conversations.create()
    now[0] = 5
    assert conversations.get(first)  # refreshes first, so second is now the oldest
    third, _ = conversations.create()
    assert conversations.get(second) is None and conversations.get(first) and conversations.get(third)
    now[0] = 20
    assert conversations.get(first) is None


def test_each_question_can_use_another_provider(make_client) -> None:
    local, hosted = Generator(), Generator()
    client = make_client(local, alternatives={"gemini": hosted})
    assert client.get("/api/health").json()["providers"] == ["ollama", "gemini"]

    first = client.post("/api/chat", json={"question": IDS_TEXT, "provider": "gemini"}).json()
    client.post("/api/chat", json={"question": IDS_TEXT, "conversation_id": first["conversation_id"]})
    assert len(hosted.prompts) == 1 and len(local.prompts) == 1  # same conversation, default model again

    wrong = client.post("/api/chat", json={"question": IDS_TEXT, "provider": "claude"})
    assert wrong.status_code == 422 and "ollama, gemini" in wrong.json()["error"]["message"]
