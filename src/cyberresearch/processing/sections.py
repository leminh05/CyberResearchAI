"""Detect section headings and split pages into section-labelled paragraphs.

Heading formats seen in the downloaded papers (verified on real PDFs):
  * "1\\nIntroduction"      - number and title on separate lines (LaTeX/arXiv)
  * "1.\\nINTRODUCTION"     - same, upper case
  * "2.3 Feature Extraction", "IV. EXPERIMENTS"  - one line
  * "References", "REFERENCES"                    - unnumbered known names

Numbered lists and algorithm steps look like headings ("13 Construct the
graph ..."), so a numbered heading is accepted only if its number plausibly
follows the previous accepted heading (see ``is_plausible_next``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cyberresearch.processing.cleaning import clean_block
from cyberresearch.processing.extract import PageText

FRONT_MATTER = "Front Matter"

KNOWN_SECTIONS = (
    "abstract", "introduction", "background", "related work", "related works", "preliminaries",
    "motivation", "problem statement", "threat model", "system model", "methodology", "methods",
    "method", "approach", "proposed method", "design", "implementation", "evaluation",
    "experiments", "experimental setup", "experimental results", "results", "discussion",
    "limitations", "future work", "conclusion", "conclusions", "conclusion and future work",
    "acknowledgments", "acknowledgements", "acknowledgment", "references", "bibliography", "appendix",
)

_NUMBER = r"(?:\d{1,2}(?:\.\d{1,2}){0,2}|[IVX]{1,5}|[A-H])"
_BARE_NUMBER = re.compile(rf"^{_NUMBER}\.?$")
_NUMBERED = re.compile(rf"^({_NUMBER})\.?\s+(.{{2,90}})$")
# IEEE style: "Abstract—We propose ..." or "Abstract: ..."
_INLINE_ABSTRACT = re.compile(r"^(abstract)\s*[—–:.\-]\s*", re.I)
_ROMAN = {"I": 1, "V": 5, "X": 10}
MAX_TITLE_WORDS = 10


@dataclass(frozen=True)
class Heading:
    """A detected heading; ``number`` is e.g. (2, 3) for "2.3", None if unnumbered."""

    label: str
    number: tuple[int, ...] | None


def _roman_to_int(text: str) -> int:
    total = 0
    for i, ch in enumerate(text):
        value = _ROMAN[ch]
        total += -value if i + 1 < len(text) and _ROMAN[text[i + 1]] > value else value
    return total


def parse_number(token: str) -> tuple[int, ...] | None:
    """"2.3" -> (2, 3); "IV" -> (4,); single letters (appendix "A") -> None."""
    if token[0].isdigit():
        return tuple(int(part) for part in token.split("."))
    if set(token) <= set(_ROMAN):
        return (_roman_to_int(token),)
    return None


def _looks_like_title(title: str) -> bool:
    """Headings are short noun phrases, not sentences or list items."""
    return (
        title[:1].isupper()
        and len(title.split()) <= MAX_TITLE_WORDS
        and sum(ch.isalpha() for ch in title) >= 3
        and not title.endswith((".", ":", ";", ","))
        and not any(ch in title for ch in ",;=")
        # "Case Study: Botnets" is a heading; "Observer: A gateway that
        # monitors traffic" is a definition list item.
        and (":" not in title or len(title.split()) <= 6)
    )


def detect_heading(line: str) -> Heading | None:
    """Return a Heading if ``line`` looks like one, else None (ignores numbering order)."""
    text = " ".join(line.split())
    if not text or len(text) > 100:
        return None
    bare = text.rstrip(":").strip()
    if bare.lower() in KNOWN_SECTIONS:
        return Heading(bare.title() if bare.isupper() else bare, None)
    match = _NUMBERED.match(text)
    if not match:
        return None
    token, title = match.group(1), match.group(2).strip()
    number = parse_number(token)
    if number is None and title.lower() not in KNOWN_SECTIONS:
        return None  # "A model is ..." is a sentence, "A. Conclusion" is a heading
    if not _looks_like_title(title):
        return None
    return Heading(f"{token} {title.title() if title.isupper() else title}", number)


def is_plausible_next(previous: tuple[int, ...] | None, new: tuple[int, ...] | None) -> bool:
    """Accept a numbered heading only if it can follow the previous one.

    It must come after the previous number and not jump more than two
    top-level sections (tolerates one missed heading).
    """
    if new is None:
        return True
    if previous is None:
        return new[0] <= 2
    return new > previous and new[0] <= previous[0] + 2


def heading_at_block_start(lines: list[str]) -> tuple[Heading | None, int]:
    """Detect a heading at the start of a block; returns (heading, lines used)."""
    if len(lines) >= 2 and _BARE_NUMBER.match(lines[0].strip()):
        heading = detect_heading(f"{lines[0].strip()} {lines[1].strip()}")
        if heading:
            return heading, 2
    heading = detect_heading(lines[0])
    return (heading, 1) if heading else (None, 0)


def section_name(label: str) -> str:
    """Section label without its number: "3.1 Results" -> "Results"."""
    parts = label.split(" ", 1)
    if len(parts) == 2 and re.fullmatch(_NUMBER, parts[0]):
        return parts[1]
    return label


@dataclass
class Paragraph:
    """A cleaned paragraph with the page it is on and its section label."""

    page_number: int
    section: str
    text: str


def segment(pages: list[PageText]) -> list[Paragraph]:
    """Walk the cleaned blocks in order and attach the current section."""
    paragraphs: list[Paragraph] = []
    section = FRONT_MATTER
    last_number: tuple[int, ...] | None = None
    for page in pages:
        for block in page.blocks:
            lines = [ln for ln in block.split("\n") if ln.strip()]
            if not lines:
                continue
            heading, used = heading_at_block_start(lines)
            if heading and is_plausible_next(last_number, heading.number):
                section = heading.label
                last_number = heading.number or last_number
                lines = lines[used:]  # the rest of the block is body text
            elif _INLINE_ABSTRACT.match(lines[0]):
                section = "Abstract"
                lines[0] = _INLINE_ABSTRACT.sub("", lines[0])
            text = clean_block("\n".join(lines))
            if text:
                paragraphs.append(Paragraph(page.page_number, section, text))
    return paragraphs
