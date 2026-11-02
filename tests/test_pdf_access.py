"""PDF access policy and the polite downloader."""

from dataclasses import replace
from pathlib import Path

import pytest
import requests

from cyberresearch.config import ProcessingSettings
from cyberresearch.processing.access import PdfCandidate, decide_access, normalize_license, pdf_candidates
from cyberresearch.processing.downloader import PdfDownloader
from fakes import FAST_HTTP, FakeResponse, FakeSession

SETTINGS = ProcessingSettings()
PDF_BYTES = b"%PDF-1.7\n" + b"0" * 2000


def openalex_source(url: str | None, license_: str | None) -> dict:
    return {"source": "openalex", "raw": {"best_oa_location": {"pdf_url": url, "license": license_}}}


# ------------------------------------------------------------------ policy

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("cc-by", "cc-by"),
        ("CC-BY-NC", "cc-by-nc"),
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "cc-by-nc-nd"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("other-oa", "other-oa"),
        (None, None),
    ],
)
def test_normalize_license(raw, expected) -> None:
    assert normalize_license(raw) == expected


def test_arxiv_paper_is_allowed_and_preferred() -> None:
    paper = {"arxiv_id": "2401.01234"}
    decision = decide_access(paper, [openalex_source("https://pub.org/x.pdf", None)], SETTINGS)
    assert decision.allowed and decision.reason == "arxiv"
    assert decision.candidate.url == "https://arxiv.org/pdf/2401.01234"


def test_open_license_is_allowed() -> None:
    decision = decide_access({}, [openalex_source("https://pub.org/x.pdf", "cc-by")], SETTINGS)
    assert decision.allowed and decision.reason == "license:cc-by"


def test_unknown_license_is_refused_by_default() -> None:
    sources = [openalex_source("https://pub.org/x.pdf", None),
               {"source": "semantic_scholar", "raw": {"openAccessPdf": {"url": "https://b.org/y.pdf", "license": None}}}]
    decision = decide_access({}, sources, SETTINGS)
    assert not decision.allowed and decision.reason == "license_unknown"
    assert decision.candidate.url == "https://pub.org/x.pdf"  # recorded for transparency

    permissive = replace(SETTINGS, allow_unknown_license=True)
    assert decide_access({}, sources, permissive).allowed


def test_restrictive_license_is_refused() -> None:
    decision = decide_access({}, [openalex_source("https://pub.org/x.pdf", "all-rights-reserved")], SETTINGS)
    assert not decision.allowed and decision.reason == "license_not_allowed"


def test_no_pdf_url() -> None:
    decision = decide_access({}, [openalex_source(None, "cc-by")], SETTINGS)
    assert decision == decide_access({}, [], SETTINGS)
    assert decision.reason == "no_pdf_url" and not decision.allowed


def test_candidates_are_unique() -> None:
    sources = [openalex_source("https://x.org/a.pdf", "cc-by"), openalex_source("https://x.org/a.pdf", "cc-by")]
    assert len(pdf_candidates({}, sources)) == 1


# -------------------------------------------------------------- downloader

def downloader(tmp_path: Path, *responses) -> tuple[PdfDownloader, FakeSession]:
    session = FakeSession(*responses)
    settings = replace(SETTINGS, pdf_directory=tmp_path, max_pdf_megabytes=0.01)
    return PdfDownloader(settings, FAST_HTTP, "test-agent", session=session, sleep=lambda s: None), session


CANDIDATE = PdfCandidate("https://pub.org/paper.pdf", "openalex", "cc-by")


def test_successful_download_is_validated_and_hashed(tmp_path: Path) -> None:
    dl, session = downloader(tmp_path, FakeResponse(200, "User-agent: *\nAllow: /"), FakeResponse(200, ""))
    session.items[-1].content = PDF_BYTES
    result = dl.download(7, CANDIDATE)
    assert result.status == "downloaded"
    assert result.file_path == tmp_path / "7.pdf" and result.file_path.read_bytes() == PDF_BYTES
    assert len(result.sha256) == 64 and result.size_bytes == len(PDF_BYTES)
    assert session.calls[0]["url"] == "https://pub.org/robots.txt"


def test_robots_disallow_prevents_download(tmp_path: Path) -> None:
    dl, session = downloader(tmp_path, FakeResponse(200, "User-agent: *\nDisallow: /"))
    assert dl.download(1, CANDIDATE).status == "robots_disallowed"
    assert len(session.calls) == 1  # the PDF itself was never requested


def test_missing_robots_allows_but_server_error_blocks(tmp_path: Path) -> None:
    dl, session = downloader(tmp_path, FakeResponse(404), FakeResponse(200))
    session.items[-1].content = PDF_BYTES
    assert dl.download(1, CANDIDATE).status == "downloaded"

    dl, _ = downloader(tmp_path, FakeResponse(503))
    assert dl.download(1, CANDIDATE).status == "robots_disallowed"

    dl, _ = downloader(tmp_path, requests.ConnectionError())
    assert dl.download(1, CANDIDATE).status == "robots_disallowed"


def test_html_landing_page_is_rejected(tmp_path: Path) -> None:
    html = FakeResponse(200, "<html>Landing page</html>", headers={"Content-Type": "text/html"})
    html.content = b"<html>Landing page</html>"
    dl, _ = downloader(tmp_path, FakeResponse(404), html)
    result = dl.download(1, CANDIDATE)
    assert result.status == "not_pdf" and "text/html" in result.error
    assert not (tmp_path / "1.pdf").exists()


def test_too_large_and_http_errors(tmp_path: Path) -> None:
    big = FakeResponse(200)
    big.content = b"%PDF-" + b"0" * 20_000  # limit is 0.01 MB
    dl, _ = downloader(tmp_path, FakeResponse(404), big)
    assert dl.download(1, CANDIDATE).status == "too_large"

    dl, _ = downloader(tmp_path, FakeResponse(404), FakeResponse(403))
    result = dl.download(1, CANDIDATE)
    assert result.status == "failed" and result.http_status == 403


def test_crawl_delay_from_robots_sets_rate_limit(tmp_path: Path) -> None:
    dl, session = downloader(tmp_path, FakeResponse(200, "User-agent: *\nCrawl-delay: 15\nAllow: /"), FakeResponse(200))
    session.items[-1].content = PDF_BYTES
    dl.download(1, CANDIDATE)
    assert dl._clients["pub.org"].rate_limiter.min_interval == 15
