"""Normalization helpers that turn messy API values into comparable values.

Every function accepts ``None`` and returns ``None`` when the input is missing
or invalid, so client code can call them without extra checks.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

_WHITESPACE = re.compile(r"\s+")
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
# New-style arXiv IDs (2007+): 0704.0001, 2401.01234 ; optional version suffix v2.
_ARXIV_NEW = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
# Old-style arXiv IDs: cs/0112017, math.GT/0309136
# Not papers: exam-dump PDFs ("... Certified Administrator R80 PDF", unlike a
# real "Policy Algebras.pdf") and the front/back matter of proceedings.
_JUNK_TITLE = re.compile(
    r"(?:[\s,]pdf$|exam dumps?|^(?:front matter|back matter|table of contents|title page|editorial board|index)$)"
)
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE)


def clean_text(value: str | None) -> str | None:
    """Collapse whitespace (incl. newlines in arXiv titles) and strip."""
    if value is None:
        return None
    text = _WHITESPACE.sub(" ", str(value)).strip()
    return text or None


def normalize_doi(value: str | None) -> str | None:
    """Return a lower-case bare DOI such as ``10.1109/tifs.2023.1``.

    DOIs are case-insensitive, and APIs return them in different forms:
    ``https://doi.org/10.X/Y``, ``doi:10.X/Y`` or ``10.X/Y``.
    """
    text = clean_text(value)
    if not text:
        return None
    text = _DOI_PREFIX.sub("", text).strip().lower()
    return text if text.startswith("10.") and "/" in text else None


def normalize_arxiv_id(value: str | None) -> str | None:
    """Return a version-less arXiv ID (``2401.01234`` or ``cs/0112017``).

    Accepts abs/pdf URLs, ``arXiv:`` prefixes and versioned IDs.
    """
    text = clean_text(value)
    if not text:
        return None
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    if match := _ARXIV_NEW.fullmatch(text):
        return match.group(1)
    if match := _ARXIV_OLD.fullmatch(text):
        return match.group(1).lower()
    return None


def is_mostly_english(text: str) -> bool:
    """True if at least 95% of the letters are plain ASCII.

    Vietnamese, for example, uses many accented letters (ư, ộ, ể ...), so a
    Vietnamese question falls below the threshold.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    return sum(ch.isascii() for ch in letters) / len(letters) >= 0.95


def is_junk_title(title: str | None) -> bool:
    """True for a title that names a non-paper (see ``_JUNK_TITLE``)."""
    return bool(title and _JUNK_TITLE.search(title.strip().lower()))


def normalize_title(value: str | None) -> str | None:
    """Return a matching key for a title: lower-case ASCII words only.

    ``"Deep-Learning for  Intrusion Détection!"`` -> ``"deep learning for intrusion detection"``
    """
    text = clean_text(value)
    if not text:
        return None
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text or None


def parse_date(value: str | None) -> date | None:
    """Parse ``YYYY-MM-DD`` or ISO datetime strings; return None otherwise."""
    text = clean_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_year(value: object) -> int | None:
    """Return a plausible publication year (1900-2100) or None."""
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild an abstract from OpenAlex's ``abstract_inverted_index``.

    OpenAlex stores ``{"word": [positions]}`` instead of plain text.
    """
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, indexes in inverted_index.items():
        for index in indexes:
            positions[index] = word
    return clean_text(" ".join(positions[i] for i in sorted(positions)))
