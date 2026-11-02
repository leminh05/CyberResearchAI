"""Remove headers, footers and extraction noise from PDF text blocks."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from cyberresearch.processing.extract import PageText

# "arXiv:2401.01234v2 [cs.CR] 3 Jan 2024" stamped on the margin of arXiv PDFs.
_ARXIV_STAMP = re.compile(r"^arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]\s*\d{1,2}\s+\w{3,9}\s+\d{4}$", re.I)
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Only the first/last blocks of a page can be running headers or footers.
EDGE_BLOCKS = 2


def repeat_key(text: str) -> str:
    """Key for detecting repeated lines: digits ignored ("Page 3" == "Page 4")."""
    return re.sub(r"\d+", "#", _WHITESPACE.sub(" ", text.strip().lower()))


def find_boilerplate(pages: list[PageText], min_fraction: float = 0.5) -> set[str]:
    """Return keys of short blocks repeated at the top/bottom of many pages.

    Journal names, running titles and "Page N" footers appear on most pages;
    real content almost never repeats verbatim.
    """
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        edges = page.blocks[:EDGE_BLOCKS] + page.blocks[-EDGE_BLOCKS:]
        counts.update({repeat_key(b) for b in edges if len(b) <= 150})
    threshold = max(2, int(len(pages) * min_fraction))
    return {key for key, n in counts.items() if n >= threshold}


def clean_block(text: str) -> str:
    """Normalize one block into a single clean line of text.

    * NFKC turns ligatures into letters ("ﬁ" -> "fi")
    * "detec-\\ntion" -> "detection" (hyphenation at line breaks)
    * remaining line breaks and repeated spaces -> one space
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1", text)
    text = text.replace("\n", " ")
    return _WHITESPACE.sub(" ", text).strip()


def is_noise(text: str) -> bool:
    """True for blocks that carry no content: page numbers, stamps, symbols."""
    if not text:
        return True
    if _PAGE_NUMBER.match(text) or _ARXIV_STAMP.match(text):
        return True
    letters = sum(ch.isalpha() for ch in text)
    return letters < 3  # e.g. "•", "(3)", "—"


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """Return pages with boilerplate and noise removed and blocks normalized.

    Each block keeps its original line breaks *until* heading detection has
    looked at the first line, so this returns blocks with "\\n" preserved.
    """
    boilerplate = find_boilerplate(pages)
    cleaned: list[PageText] = []
    for page in pages:
        blocks = []
        for block in page.blocks:
            if repeat_key(block) in boilerplate:
                continue
            normalized = unicodedata.normalize("NFKC", _CONTROL.sub("", block)).strip()
            if not is_noise(clean_block(normalized)):
                blocks.append(normalized)
        cleaned.append(PageText(page.page_number, blocks))
    return cleaned
