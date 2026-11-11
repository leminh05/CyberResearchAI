"""API clients: request parameters, parsing and pagination (no network)."""

from datetime import date

import pytest

from cyberresearch.config import load_settings
from cyberresearch.ingestion.clients import ArxivClient, OpenAlexClient, SemanticScholarClient, build_clients
from cyberresearch.ingestion.http import HttpError
from fakes import FakeResponse, FakeSession, fake_http, fixture_json, fixture_text

START, END = date(2024, 1, 1), date(2024, 1, 31)


# ------------------------------------------------------------------ arXiv

def test_arxiv_request_has_query_date_filter_and_offset() -> None:
    session = FakeSession(FakeResponse(200, fixture_text("arxiv_feed.xml")))
    client = ArxivClient(fake_http(session), "https://export.arxiv.org/api/query", 3)
    client.fetch_page('abs:"intrusion detection"', START, END, cursor="6")
    params = session.calls[0]["params"]
    assert params["search_query"] == (
        '(abs:"intrusion detection") AND submittedDate:[202401010000 TO 202401312359]'
    )
    assert params["start"] == 6
    assert params["max_results"] == 3


def test_arxiv_parses_entries_and_next_offset() -> None:
    session = FakeSession(FakeResponse(200, fixture_text("arxiv_feed.xml")))
    page = ArxivClient(fake_http(session), "u", 3).fetch_page("q", START, END, None)
    assert page.total == 5
    assert page.next_cursor == "3"  # 3 entries read, 5 in total
    first = page.records[0]
    assert first.source == "arxiv"
    assert first.source_id == first.arxiv_id == "2401.01234"  # version removed
    assert first.title == "Deep Learning for Network Intrusion Detection"
    assert first.authors == ["Alice Nguyen", "Bob Tran"]
    assert first.doi == "10.1109/tifs.2024.0001"
    assert first.publication_date == date(2024, 1, 3)
    assert first.pdf_url == "http://arxiv.org/pdf/2401.01234v2"
    assert first.categories == ["cs.CR", "cs.LG"]
    assert first.raw["journal_ref"] == "IEEE TIFS 2024"  # original info preserved


def test_arxiv_last_page_has_no_cursor() -> None:
    page = ArxivClient(None, "u", 3).parse_feed(fixture_text("arxiv_feed.xml"), offset=2)  # type: ignore[arg-type]
    assert page.next_cursor is None  # 2 + 3 >= 5


def test_arxiv_api_error_entry_raises() -> None:
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/api/errors#incorrect_id_format</id>
      <summary>incorrect id format</summary></entry></feed>"""
    with pytest.raises(HttpError, match="incorrect id format"):
        ArxivClient(None, "u", 3).parse_feed(xml, 0)  # type: ignore[arg-type]


def test_arxiv_invalid_xml_raises() -> None:
    with pytest.raises(HttpError):
        ArxivClient(None, "u", 3).parse_feed("<html>maintenance", 0)  # type: ignore[arg-type]


# --------------------------------------------------------------- OpenAlex

def test_openalex_request_params() -> None:
    session = FakeSession(FakeResponse(200, json_data=fixture_json("openalex_works.json")))
    client = OpenAlexClient(fake_http(session), "https://api.openalex.org/works", 100, mailto="me@x.org", api_key="k")
    client.fetch_page('"intrusion detection"', START, END, None)
    params = session.calls[0]["params"]
    assert params["cursor"] == "*"  # first page
    assert params["filter"] == (
        "from_publication_date:2024-01-01,to_publication_date:2024-01-31"
        ",language:en,type:article|preprint|review|book-chapter|conference-paper|dissertation"
    )
    assert params["mailto"] == "me@x.org"
    assert params["api_key"] == "k"


def test_openalex_parses_works_and_cursor() -> None:
    page = OpenAlexClient(None, "u", 100).parse_response(fixture_json("openalex_works.json"))  # type: ignore[arg-type]
    assert page.next_cursor == "IlsxNjA5MzcyODAwMDAwXSI="
    assert len(page.records) == 2 and page.skipped == 1  # item without id skipped
    work = page.records[0]
    assert work.source_id == work.openalex_id == "W111"
    assert work.doi == "10.1109/tifs.2024.0001"
    assert work.arxiv_id == "2401.01234"  # found in locations
    assert work.abstract == "We propose a model."
    assert work.venue == "IEEE Transactions on Information Forensics and Security"
    assert work.is_open_access is True
    assert work.raw["cited_by_count"] == 12


def test_openalex_without_key_uses_page_numbers() -> None:
    payload = {"meta": {"count": 250, "next_cursor": None}, "results": [{"id": "https://openalex.org/W1", "title": "T"}]}
    session = FakeSession(*[FakeResponse(200, json_data=payload) for _ in range(3)])
    client = OpenAlexClient(fake_http(session), "u", 100)  # no api_key

    first = client.fetch_page("q", START, END, None)
    assert session.calls[0]["params"]["page"] == 1 and "cursor" not in session.calls[0]["params"]
    assert first.next_cursor == "page:2"

    last = client.fetch_page("q", START, END, "page:3")
    assert session.calls[1]["params"]["page"] == 3
    assert last.next_cursor is None  # 3 * 100 >= 250

    client.fetch_page("q", START, END, "IlsxNjA5MzcyXSI=")  # cursor saved while a key was set
    assert session.calls[2]["params"]["page"] == 1


def test_openalex_keyless_paging_stops_at_10000_results() -> None:
    client = OpenAlexClient(None, "u", 100)  # type: ignore[arg-type]
    payload = {"meta": {"count": 50_000}, "results": [{"id": "https://openalex.org/W1", "title": "T"}]}
    assert client.parse_response(payload, page_number=99).next_cursor == "page:100"
    assert client.parse_response(payload, page_number=100).next_cursor is None


def test_openalex_empty_results_stop_pagination() -> None:
    page = OpenAlexClient(None, "u", 100).parse_response({"meta": {"next_cursor": "x"}, "results": []})  # type: ignore[arg-type]
    assert page.next_cursor is None


# -------------------------------------------------------- Semantic Scholar

def test_semantic_scholar_params_and_token() -> None:
    session = FakeSession(FakeResponse(200, json_data=fixture_json("s2_bulk.json")))
    client = SemanticScholarClient(fake_http(session), "u", 1000)
    page = client.fetch_page('"intrusion detection"', START, END, cursor="tok1")
    params = session.calls[0]["params"]
    assert params["publicationDateOrYear"] == "2024-01-01:2024-01-31"
    assert params["token"] == "tok1"
    assert "externalIds" in params["fields"]
    assert page.next_cursor == "PCOA3RZZB2ADADAEYCX2BLJJRDEGL"


def test_semantic_scholar_parses_papers() -> None:
    page = SemanticScholarClient(None, "u", 1000).parse_response(fixture_json("s2_bulk.json"))  # type: ignore[arg-type]
    first = page.records[0]
    assert first.source_id == first.s2_paper_id == "abc123"
    assert first.arxiv_id == "2401.05678"
    assert first.pdf_url == "https://arxiv.org/pdf/2401.05678"
    assert first.venue == "ACM CCS"


# --------------------------------------------------------------- factory

def test_build_clients_sends_s2_key_as_header_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "dummy-key")
    settings = load_settings(env_file=env)
    clients = build_clients(settings)
    assert set(clients) == {"arxiv", "openalex", "semantic_scholar"}
    assert clients["semantic_scholar"].http.headers["x-api-key"] == "dummy-key"
    assert "x-api-key" not in clients["arxiv"].http.headers
    assert "CyberResearchAI" in clients["arxiv"].http.headers["User-Agent"]

    only = build_clients(settings, only=["arxiv"])
    assert list(only) == ["arxiv"]
