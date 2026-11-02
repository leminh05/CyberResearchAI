"""Metadata normalization helpers."""

from datetime import date

import pytest

from cyberresearch.ingestion.normalize import (
    clean_text,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    parse_date,
    parse_year,
    reconstruct_abstract,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("10.1109/TIFS.2024.0001", "10.1109/tifs.2024.0001"),
        ("https://doi.org/10.1109/TIFS.2024.0001", "10.1109/tifs.2024.0001"),
        ("http://dx.doi.org/10.1/ABC", "10.1/abc"),
        ("doi: 10.1/abc", "10.1/abc"),
        ("not-a-doi", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_doi(raw, expected) -> None:
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2401.01234", "2401.01234"),
        ("2401.01234v3", "2401.01234"),
        ("arXiv:2401.01234v1", "2401.01234"),
        ("http://arxiv.org/abs/2401.01234v2", "2401.01234"),
        ("https://arxiv.org/pdf/2401.01234v2.pdf", "2401.01234"),
        ("http://arxiv.org/abs/cs/0112017v1", "cs/0112017"),
        ("0704.0001", "0704.0001"),
        ("https://doi.org/10.1/x", None),
        (None, None),
    ],
)
def test_normalize_arxiv_id(raw, expected) -> None:
    assert normalize_arxiv_id(raw) == expected


def test_normalize_title_ignores_case_punctuation_and_accents() -> None:
    a = normalize_title("Deep-Learning for  Intrusion Détection!")
    b = normalize_title("deep learning for intrusion detection")
    assert a == b == "deep learning for intrusion detection"
    assert normalize_title("  !!! ") is None


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("Deep Learning for Network\n      Intrusion Detection") == (
        "Deep Learning for Network Intrusion Detection"
    )
    assert clean_text("   ") is None


def test_parse_date_and_year() -> None:
    assert parse_date("2024-01-03T18:30:00Z") == date(2024, 1, 3)
    assert parse_date("2024-01-03") == date(2024, 1, 3)
    assert parse_date("January 2024") is None
    assert parse_year(2024) == 2024
    assert parse_year("2024-05-01") == 2024
    assert parse_year("unknown") is None
    assert parse_year(1700) is None


def test_reconstruct_openalex_abstract() -> None:
    index = {"detection": [2], "Intrusion": [0], "is": [1], "hard.": [3]}
    assert reconstruct_abstract(index) == "Intrusion is detection hard."
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None
