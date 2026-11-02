"""PDF text extraction that keeps page numbers.

PyMuPDF returns text *blocks* per page (roughly paragraphs), in the order
they appear in the PDF content stream, which follows two-column layouts
better than sorting by coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class PdfExtractionError(Exception):
    """The PDF could not be opened or contains no extractable text."""


@dataclass
class PageText:
    """Text blocks of one page; ``page_number`` is 1-based like PDF viewers."""

    page_number: int
    blocks: list[str] = field(default_factory=list)


def extract_pages(path: Path) -> list[PageText]:
    """Return the text blocks of every page.

    Raises:
        PdfExtractionError: for corrupt/encrypted files or scanned PDFs
            without a text layer (OCR is out of scope for this project).
    """
    import pymupdf  # imported lazily: only needed when PDFs are processed

    try:
        # filetype="pdf": PyMuPDF can also open HTML/XPS/EPUB, and an HTML
        # error page saved as .pdf must not be treated as a paper.
        document = pymupdf.open(path, filetype="pdf")
    except Exception as exc:  # pymupdf raises several exception types
        raise PdfExtractionError(f"cannot open PDF: {exc}") from exc

    with document:
        if not document.is_pdf:  # PyMuPDF sniffs content and may ignore filetype
            raise PdfExtractionError("file is not a PDF")
        if document.needs_pass:
            raise PdfExtractionError("PDF is password protected")
        pages = []
        for index, page in enumerate(document):
            # Block tuple: (x0, y0, x1, y1, text, block_no, block_type); type 0 = text.
            blocks = [b[4] for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            pages.append(PageText(page_number=index + 1, blocks=blocks))

    if not any(page.blocks for page in pages):
        raise PdfExtractionError("no text layer (scanned PDF?)")
    return pages
