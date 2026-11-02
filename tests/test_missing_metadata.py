"""Records with missing or invalid metadata are handled, not crashed on."""

from cyberresearch.ingestion.clients import ArxivClient, OpenAlexClient, SemanticScholarClient
from cyberresearch.ingestion.schema import PaperRecord
from fakes import fixture_json, fixture_text


def test_arxiv_entry_without_abstract_or_authors() -> None:
    page = ArxivClient(None, "u", 3).parse_feed(fixture_text("arxiv_feed.xml"), 0)  # type: ignore[arg-type]
    minimal = page.records[2]
    assert minimal.title == "Minimal Entry"
    assert minimal.abstract is None
    assert minimal.authors == []
    assert minimal.doi is None and minimal.venue is None


def test_arxiv_entry_without_valid_id_is_skipped() -> None:
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://example.org/not-arxiv</id><title>No id</title></entry></feed>"""
    page = ArxivClient(None, "u", 3).parse_feed(xml, 0)  # type: ignore[arg-type]
    assert page.records == [] and page.skipped == 1


def test_openalex_work_with_null_fields() -> None:
    page = OpenAlexClient(None, "u", 100).parse_response(fixture_json("openalex_works.json"))  # type: ignore[arg-type]
    work = page.records[1]
    assert work.source_id == "W222"
    assert work.abstract is None and work.doi is None and work.publication_date is None
    assert work.authors == [] and work.venue is None and work.pdf_url is None
    assert work.year == 2024


def test_semantic_scholar_paper_with_only_an_identifier() -> None:
    page = SemanticScholarClient(None, "u", 1000).parse_response(fixture_json("s2_bulk.json"))  # type: ignore[arg-type]
    bare = page.records[2]
    assert bare.title is None and bare.year is None and bare.authors == []
    assert bare.doi == "10.5555/xyz"
    assert bare.is_valid()  # an identifier is enough to keep it


def test_semantic_scholar_skips_items_without_paper_id() -> None:
    page = SemanticScholarClient(None, "u", 1000).parse_response(  # type: ignore[arg-type]
        {"data": [{"paperId": None, "title": "x"}, {"title": "y"}], "token": None}
    )
    assert page.records == [] and page.skipped == 2


def test_record_without_title_or_ids_is_invalid() -> None:
    assert not PaperRecord(source="openalex", source_id="W1").is_valid()
    assert PaperRecord(source="openalex", source_id="W1", title="Only a title").is_valid()


def test_content_hash_is_stable_and_ignores_raw() -> None:
    a = PaperRecord(source="s", source_id="1", title="T", raw={"citations": 1})
    b = PaperRecord(source="s", source_id="1", title="T", raw={"citations": 99})
    c = PaperRecord(source="s", source_id="1", title="T2")
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()
