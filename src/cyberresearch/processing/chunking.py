"""Split section-labelled paragraphs into overlapping chunks for retrieval.

Rules:
* chunks never cross a section boundary (a chunk is about one topic);
* each word remembers its page, so ``page_start``/``page_end`` are exact even
  when a chunk continues onto the next page;
* chunks end at a sentence boundary when one is close to the size limit;
* consecutive chunks overlap so that a sentence split between two chunks is
  still fully contained in one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from cyberresearch.processing.sections import Paragraph, section_name

MIN_CHUNK_WORDS = 15


@dataclass
class Chunk:
    """A retrievable piece of a paper with its exact location."""

    chunk_index: int
    text: str
    section: str
    page_start: int | None
    page_end: int | None
    word_count: int


def _sentence_end(words: list[tuple[str, int]], start: int, end: int) -> int:
    """Move ``end`` back to just after a sentence end, within the last 30%."""
    lower_bound = start + int((end - start) * 0.7)
    for i in range(end, lower_bound, -1):
        if words[i - 1][0].endswith((".", "?", "!")):
            return i
    return end


def chunk_paragraphs(
    paragraphs: list[Paragraph],
    max_words: int = 220,
    overlap_words: int = 40,
    skip_sections: tuple[str, ...] = (),
    min_words: int = MIN_CHUNK_WORDS,
) -> list[Chunk]:
    """Create chunks from paragraphs (see module docstring for the rules).

    ``min_words`` drops tiny PDF fragments (stray captions, affiliations).
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    skip = {s.lower() for s in skip_sections}
    chunks: list[Chunk] = []

    # groupby keeps document order and starts a new group at every heading.
    for section, group in groupby(paragraphs, key=lambda p: p.section):
        if section_name(section).lower() in skip:
            continue
        words = [(word, p.page_number) for p in group for word in p.text.split()]
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            if end < len(words):
                end = _sentence_end(words, start, end)
            piece = words[start:end]
            if len(piece) >= min_words:
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        text=" ".join(w for w, _ in piece),
                        section=section,
                        page_start=piece[0][1],
                        page_end=piece[-1][1],
                        word_count=len(piece),
                    )
                )
            if end >= len(words):
                break
            start = max(end - overlap_words, start + 1)  # always move forward
    return chunks
