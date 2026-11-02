"""PDF extraction, page preservation, cleaning, sections and chunking."""

from pathlib import Path

import pytest

from cyberresearch.processing.chunking import chunk_paragraphs
from cyberresearch.processing.cleaning import clean_block, clean_pages, find_boilerplate, is_noise
from cyberresearch.processing.extract import PageText, PdfExtractionError, extract_pages
from cyberresearch.processing.sections import (
    FRONT_MATTER,
    Paragraph,
    detect_heading,
    heading_at_block_start,
    is_plausible_next,
    section_name,
    segment,
)
from pdf_factory import HEADER, make_paper_pdf


@pytest.fixture
def paper_pdf(tmp_path: Path) -> Path:
    return make_paper_pdf(tmp_path / "paper.pdf")


@pytest.fixture
def paragraphs(paper_pdf: Path) -> list[Paragraph]:
    return segment(clean_pages(extract_pages(paper_pdf)))


# --------------------------------------------------------------- extraction

def test_extract_keeps_page_numbers(paper_pdf: Path) -> None:
    pages = extract_pages(paper_pdf)
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    assert any("Methodology" in b for b in pages[2].blocks)
    assert any("References" in b for b in pages[3].blocks)


def test_corrupt_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"<html>not a pdf</html>")
    with pytest.raises(PdfExtractionError):
        extract_pages(bad)


def test_pdf_without_text_layer_raises(tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    doc.save(tmp_path / "blank.pdf")
    with pytest.raises(PdfExtractionError, match="no text layer"):
        extract_pages(tmp_path / "blank.pdf")


# ----------------------------------------------------------------- cleaning

def test_running_headers_and_page_footers_are_removed(paper_pdf: Path) -> None:
    pages = extract_pages(paper_pdf)
    assert find_boilerplate(pages)  # header detected as repeated
    cleaned = clean_pages(pages)
    all_text = " ".join(b for p in cleaned for b in p.blocks)
    assert HEADER not in all_text
    assert "Page 2" not in all_text
    assert "arXiv:2401.01234v2" not in all_text


def test_clean_block_fixes_hyphenation_ligatures_and_spaces() -> None:
    assert clean_block("Intrusion detec-\ntion is   hard") == "Intrusion detection is hard"
    assert clean_block("eﬃcient classiﬁer") == "efficient classifier"  # ffi / fi ligatures
    assert clean_block("state-of-the-art\nmodel") == "state-of-the-art model"
    assert clean_block("bad\x00char") == "badchar"


@pytest.mark.parametrize("text", ["12", "Page 3", "3 / 10", "•", "(4)", "arXiv:2401.01234v2 [cs.CR] 3 Jan 2024", ""])
def test_noise_blocks(text: str) -> None:
    assert is_noise(text)


def test_real_sentences_are_not_noise() -> None:
    assert not is_noise("Table 2 shows the results.")


def test_boilerplate_needs_three_pages() -> None:
    pages = [PageText(1, ["Header", "body a"]), PageText(2, ["Header", "body b"])]
    assert find_boilerplate(pages) == set()


# ----------------------------------------------------------------- sections

@pytest.mark.parametrize(
    "line, expected",
    [
        ("1 Introduction", "1 Introduction"),
        ("2.3. Feature Extraction", "2.3 Feature Extraction"),
        ("IV. EXPERIMENTS", "IV Experiments"),
        ("REFERENCES", "References"),
        ("Related Work", "Related Work"),
        ("A. Conclusion", "A Conclusion"),
        ("A model is trained on the data", None),
        ("We propose a new method for detecting attacks in large enterprise networks today", None),
        ("3 The results show that accuracy improves.", None),
        # false positives seen in real PDFs
        ("13 Construct Gsampled from (Vsub, Esub), remove self-loops, and relabel nodes", None),
        ("2 Global Observer: A centralized gateway or platform observes", None),
        ("2 Transfer error is sufficiently higher compared to baseline:", None),
    ],
)
def test_detect_heading(line: str, expected: str | None) -> None:
    heading = detect_heading(line)
    assert (heading.label if heading else None) == expected


@pytest.mark.parametrize(
    "lines, label, used",
    [
        (["1", "Introduction"], "1 Introduction", 2),        # LaTeX/arXiv layout
        (["1.", "INTRODUCTION"], "1 Introduction", 2),
        (["2.2", "Background", "Body text"], "2.2 Background", 2),
        (["References"], "References", 1),
        (["3", "and then the model converges."], None, 0),
    ],
)
def test_heading_split_across_lines(lines, label, used) -> None:
    heading, n = heading_at_block_start(lines)
    assert (heading.label if heading else None, n) == (label, used)


def test_numbering_must_progress() -> None:
    assert is_plausible_next(None, (1,)) and not is_plausible_next(None, (5,))
    assert is_plausible_next((2,), (2, 1)) and is_plausible_next((2, 3), (3,))
    assert is_plausible_next((2,), (4,))       # one heading may have been missed
    assert not is_plausible_next((3,), (13,))  # algorithm step "13 ..."
    assert not is_plausible_next((4,), (2,))   # numbered list item going backwards
    assert is_plausible_next((4,), None)       # unnumbered headings always allowed


def test_segment_ignores_out_of_order_numbers() -> None:
    pages = [PageText(1, ["1\nIntroduction", "Intro text here.", "13 Construct the graph", "2\nMethod", "Body."])]
    sections = [p.section for p in segment(pages)]
    assert sections == ["1 Introduction", "1 Introduction", "2 Method"]


def test_section_name_strips_numbers() -> None:
    assert section_name("3.1 Results") == "Results"
    assert section_name("IV Experiments") == "Experiments"
    assert section_name("Abstract") == "Abstract"


def test_segment_assigns_sections_and_pages(paragraphs: list[Paragraph]) -> None:
    sections = [p.section for p in paragraphs]
    assert sections[0] == FRONT_MATTER  # the title
    assert "Abstract" in sections  # IEEE "Abstract -" style detected
    assert {"1 Introduction", "2 Related Work", "3 Methodology", "References"} <= set(sections)
    related = [p for p in paragraphs if p.section == "2 Related Work"]
    assert {p.page_number for p in related} == {2, 3}  # section continues on page 3
    intro = next(p for p in paragraphs if p.section == "1 Introduction")
    assert intro.text.startswith("Intrusion detection is")  # de-hyphenated


# ----------------------------------------------------------------- chunking

def test_chunks_preserve_pages_and_never_cross_sections(paragraphs: list[Paragraph]) -> None:
    chunks = chunk_paragraphs(paragraphs, max_words=80, overlap_words=15, skip_sections=("References",))
    assert chunks
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        source = [p for p in paragraphs if p.section == chunk.section]
        assert source, "chunk section must exist in the document"
        assert {p.page_number for p in source} >= {chunk.page_start, chunk.page_end}
        assert 1 <= chunk.page_start <= chunk.page_end <= 4
        assert chunk.word_count <= 80
    assert not any(c.section == "References" for c in chunks)
    assert any(c.page_start == 2 and c.page_end == 3 for c in chunks)  # spans a page break


def test_chunk_overlap_and_sentence_boundaries() -> None:
    # 300 unique words; every 10th word ends a sentence.
    text = " ".join(f"w{i}." if i % 10 == 9 else f"w{i}" for i in range(300))
    chunks = chunk_paragraphs([Paragraph(1, "1 Intro", text)], max_words=95, overlap_words=20)
    assert len(chunks) >= 3
    assert chunks[0].word_count == 90  # moved back from 95 to the sentence end at word 90
    for first, second in zip(chunks, chunks[1:]):
        assert first.text.endswith(".")
        assert second.text.split()[:20] == first.text.split()[-20:]  # exact 20-word overlap


def test_chunking_tracks_page_per_word() -> None:
    paragraphs = [Paragraph(5, "2 Method", "alpha " * 30), Paragraph(6, "2 Method", "beta " * 30)]
    [chunk] = chunk_paragraphs(paragraphs, max_words=100, overlap_words=10)
    assert (chunk.page_start, chunk.page_end) == (5, 6)


def test_tiny_fragments_are_dropped_and_bad_overlap_rejected() -> None:
    assert chunk_paragraphs([Paragraph(1, "Front Matter", "Short title")]) == []
    with pytest.raises(ValueError):
        chunk_paragraphs([], max_words=50, overlap_words=50)
