"""Deduplication rules (in-memory; the database uses the same functions)."""

from cyberresearch.ingestion.dedup import deduplicate, fill_missing, match_reason, paper_from_record
from cyberresearch.ingestion.schema import PaperRecord
from fakes import all_fixture_records


def rec(source: str, source_id: str, **kw) -> PaperRecord:
    return PaperRecord(source=source, source_id=source_id, **kw)


def test_fixture_records_collapse_to_five_papers() -> None:
    records = all_fixture_records()
    assert len(records) == 8
    papers = deduplicate(records)
    groups = sorted(sorted(f"{r.source}:{r.source_id}" for r in p.records) for p in papers)
    assert groups == [
        ["arxiv:2401.01234", "openalex:W111"],          # same DOI and arXiv ID
        ["arxiv:2401.05678", "semantic_scholar:abc123"],  # same arXiv ID
        ["arxiv:2401.09999"],
        ["openalex:W222", "semantic_scholar:def456"],     # same normalized title + year
        ["semantic_scholar:ghi789"],
    ]


def test_match_by_arxiv_id_ignores_version() -> None:
    a = rec("arxiv", "2401.1", arxiv_id="2401.00001")
    b = rec("semantic_scholar", "s1", arxiv_id="2401.00001", s2_paper_id="s1")
    assert match_reason(paper_from_record(a), b) == "arxiv_id"


def test_match_by_doi() -> None:
    a = rec("openalex", "W1", doi="10.1/abc", openalex_id="W1")
    b = rec("semantic_scholar", "s1", doi="10.1/abc", s2_paper_id="s1")
    assert match_reason(paper_from_record(a), b) == "doi"


def test_match_by_semantic_scholar_id() -> None:
    a = rec("semantic_scholar", "s1", s2_paper_id="s1", title="A")
    b = rec("semantic_scholar", "s1", s2_paper_id="s1", title="A (revised)")
    assert match_reason(paper_from_record(a), b) == "s2_paper_id"


def test_match_by_title_and_year() -> None:
    a = rec("openalex", "W1", title="Zero-Day Attack Detection", year=2023, openalex_id="W1")
    b = rec("semantic_scholar", "s1", title="zero day attack detection.", year=2023, s2_paper_id="s1")
    assert match_reason(paper_from_record(a), b) == "title_year"


def test_same_title_different_year_is_not_a_duplicate() -> None:
    a = rec("openalex", "W1", title="Intrusion Detection Survey", year=2020)
    b = rec("openalex", "W2", title="Intrusion Detection Survey", year=2023)
    assert len(deduplicate([a, b])) == 2


def test_same_title_but_conflicting_doi_is_not_a_duplicate() -> None:
    a = rec("openalex", "W1", title="A Survey", year=2023, doi="10.1/a")
    b = rec("openalex", "W2", title="A Survey", year=2023, doi="10.1/b")
    assert len(deduplicate([a, b])) == 2


def test_identifier_beats_title_when_choosing_a_match() -> None:
    by_title = rec("openalex", "W1", title="Shared Title", year=2024)
    by_doi = rec("openalex", "W2", title="Other", year=2024, doi="10.1/x")
    incoming = rec("semantic_scholar", "s1", title="Shared Title", year=2024, doi="10.1/x")
    papers = deduplicate([by_title, by_doi, incoming])
    target = next(p for p in papers if p.fields["doi"] == "10.1/x")
    assert {r.source_id for r in target.records} == {"W2", "s1"}


def test_merge_fills_missing_fields_without_overwriting() -> None:
    existing = paper_from_record(rec("arxiv", "1", arxiv_id="2401.00001", title="Original", abstract=None))
    incoming = rec("semantic_scholar", "s1", title="Different title", abstract="Filled", s2_paper_id="s1", venue="CCS")
    updates = fill_missing(existing, incoming)
    assert updates == {"abstract": "Filled", "venue": "CCS", "s2_paper_id": "s1"}
