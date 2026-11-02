"""Generate realistic test PDFs with PyMuPDF (no binary fixtures in the repo)."""

from __future__ import annotations

from pathlib import Path

import pymupdf

HEADER = "Journal of Testing Security, Vol. 1"
BODY_WORDS = (
    "network intrusion detection systems monitor traffic flows and raise alerts when malicious "
    "patterns appear in packets collected from enterprise sensors. "
)


def _paragraph(n_words: int, seed: str) -> str:
    words = (BODY_WORDS * (n_words // 20 + 1)).split()[:n_words]
    return f"{seed} " + " ".join(words) + "."


def _box(page: pymupdf.Page, y: float, text: str, height: float = 180, fontsize: float = 9) -> float:
    rect = pymupdf.Rect(60, y, 540, y + height)
    # insert_textbox silently writes nothing when the text does not fit,
    # so fail loudly instead of producing a misleading test PDF.
    if page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv") < 0:
        raise ValueError(f"text does not fit in its box: {text[:40]!r}")
    return y + height + 10


def make_paper_pdf(path: Path, arxiv_stamp: bool = True) -> Path:
    """Four pages: abstract, three numbered sections, references.

    Every page has a running header and a "Page N" footer; page 1 has an
    arXiv stamp; the introduction contains a hyphenated line break and the
    "Related Work" section continues from page 2 onto page 3.
    """
    doc = pymupdf.open()
    contents = [
        [
            ("Deep Learning for Network Intrusion Detection", 30, 14),
            ("Abstract - We study intrusion detection with deep learning and evaluate it on public datasets.", 40, 9),
            ("1 Introduction", 30, 12),
            ("Intrusion detec-\ntion is a core security task. " + _paragraph(120, "Introduction body."), 200, 9),
        ],
        [
            ("2 Related Work", 30, 12),
            (_paragraph(170, "Related work part one."), 260, 9),
        ],
        [
            (_paragraph(60, "Related work part two on page three."), 140, 9),
            ("3 Methodology", 30, 12),
            (_paragraph(100, "Method body."), 200, 9),
        ],
        [
            ("References", 30, 12),
            ("[1] A. Author. Some cited paper. In Proc. of Something, 2020. "
             "[2] B. Author. Another cited paper. Journal, 2021.", 60, 9),
        ],
    ]
    for number, blocks in enumerate(contents, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 40), HEADER, fontsize=8, fontname="helv")
        y = 70.0
        for text, height, size in blocks:
            y = _box(page, y, text, height, size)
        page.insert_text((280, 810), f"Page {number}", fontsize=8, fontname="helv")
        if number == 1 and arxiv_stamp:
            page.insert_text((20, 600), "arXiv:2401.01234v2 [cs.CR] 3 Jan 2024", fontsize=8, fontname="helv", rotate=90)
    doc.save(path)
    doc.close()
    return path
