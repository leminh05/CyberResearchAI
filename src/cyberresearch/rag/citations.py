"""Citation mapping and validation.

A citation is accepted only if
  1. its evidence id was actually retrieved for this question, and
  2. its quote (>= min words) appears in that chunk's stored text.
A valid URL is never proof: Claude cannot supply URLs at all, and any URL,
DOI or arXiv id it writes is removed. References shown to the user are built
from the database row behind each accepted evidence id.

An "evidence" claim without at least one accepted citation is rejected.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from cyberresearch.rag.retrieval import Evidence

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_URL_LIKE = re.compile(
    r"https?://\S+|www\.\S+|\b10\.\d{4,9}/[^\s,;)]+|\barxiv:\s*\d{4}\.\d{4,5}(v\d+)?",
    re.IGNORECASE,
)
_EVIDENCE_MARKER = re.compile(r"\s*[\[(]\s*E\d+(\s*,\s*E\d+)*\s*[\])]")


def normalize_for_match(text: str) -> str:
    """Lower-case alphanumeric words only, so quotes survive punctuation/spacing changes."""
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_NON_ALNUM.sub(" ", text).split())


def strip_links(text: str) -> tuple[str, list[str]]:
    """Remove URLs, DOIs and arXiv ids written by the model; return what was removed."""
    removed = [m.group(0) for m in _URL_LIKE.finditer(text)]
    cleaned = _URL_LIKE.sub("", text)
    cleaned = _EVIDENCE_MARKER.sub("", cleaned)  # we render our own [n] markers
    return " ".join(cleaned.split()), removed


@dataclass
class Reference:
    """A numbered bibliography entry built from database metadata."""

    number: int
    evidence: Evidence

    def format(self) -> str:
        e = self.evidence
        authors = ", ".join(e.authors[:3]) + (" et al." if len(e.authors) > 3 else "")
        link = e.canonical_url or "no link available"
        section = f"section \"{e.section}\", " if e.section and e.content_source == "pdf" else ""
        return (
            f"[{self.number}] {e.title or 'Untitled'}. {authors or 'Unknown authors'} ({e.year or 'n.d.'}). "
            f"{link}\n     {section}{e.location}; evidence {e.evidence_id} = paper_id {e.paper_id}, "
            f"chunk_id {e.chunk_id} (chunk #{e.chunk_index})"
        )


@dataclass
class CheckedCitation:
    evidence_id: str
    quote: str
    valid: bool
    reason: str | None = None
    reference: Reference | None = None


@dataclass
class CheckedClaim:
    statement: str
    kind: str  # evidence | interpretation
    citations: list[CheckedCitation]

    @property
    def accepted(self) -> list[CheckedCitation]:
        return [c for c in self.citations if c.valid]

    @property
    def supported(self) -> bool:
        return self.kind == "interpretation" or bool(self.accepted)


@dataclass
class ValidatedAnswer:
    """What the user sees, after every citation was checked."""

    status: str  # answered | partial | insufficient_evidence | out_of_scope | refused | error
    summary: str
    claims: list[CheckedClaim] = field(default_factory=list)
    rejected_claims: list[CheckedClaim] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    ambiguity_note: str = ""
    limitations: str = ""
    follow_ups: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def invalid_citations(self) -> list[CheckedCitation]:
        return [c for claim in self.claims + self.rejected_claims for c in claim.citations if not c.valid]


def check_citation(raw: dict[str, Any], evidence_by_id: dict[str, Evidence], min_words: int) -> CheckedCitation:
    evidence_id = str(raw.get("evidence_id", "")).strip()
    quote = str(raw.get("quote", "")).strip()
    evidence = evidence_by_id.get(evidence_id)
    if evidence is None:
        return CheckedCitation(evidence_id, quote, False, "evidence id was not retrieved for this question")
    normalized_quote = normalize_for_match(quote)
    if len(normalized_quote.split()) < min_words:
        return CheckedCitation(evidence_id, quote, False, f"quote shorter than {min_words} words")
    if normalized_quote not in normalize_for_match(evidence.text):
        return CheckedCitation(evidence_id, quote, False, "quote not found in the cited chunk")
    return CheckedCitation(evidence_id, quote, True)


def validate_answer(data: dict[str, Any], evidence: list[Evidence], min_quote_words: int = 5) -> ValidatedAnswer:
    """Check every citation of Claude's JSON answer against the retrieved evidence."""
    evidence_by_id = {e.evidence_id: e for e in evidence}
    status = data.get("answer_status", "insufficient_evidence")
    warnings: list[str] = []

    def clean(text: str, where: str) -> str:
        cleaned, removed = strip_links(text or "")
        if removed:
            warnings.append(f"removed model-written link(s) from {where}: {', '.join(removed)}")
        return cleaned

    claims, rejected = [], []
    for raw_claim in data.get("claims") or []:
        citations = [check_citation(c, evidence_by_id, min_quote_words) for c in raw_claim.get("citations") or []]
        for citation in citations:
            if not citation.valid:
                warnings.append(f"rejected citation {citation.evidence_id or '?'}: {citation.reason}")
        claim = CheckedClaim(clean(raw_claim.get("statement", ""), "a claim"), raw_claim.get("kind", "evidence"), citations)
        (claims if claim.supported else rejected).append(claim)

    summary = clean(data.get("summary", ""), "the summary")
    if status in ("insufficient_evidence", "out_of_scope"):
        # Nothing may be presented as a finding when Claude says the evidence is missing.
        rejected += [c for c in claims if c.kind == "evidence"]
        claims = [c for c in claims if c.kind != "evidence"]
    elif not any(c.kind == "evidence" for c in claims):
        status = "insufficient_evidence"
        warnings.append("no evidence claim could be verified against the retrieved chunks")
        summary = ""
    elif rejected:
        status = "partial"
        # The summary may repeat a rejected claim, so only verified claims are shown.
        summary = ""
        warnings.append(f"{len(rejected)} claim(s) removed because their citations could not be verified")

    # Number references only now, so the list has no gaps from rejected claims.
    references: dict[str, Reference] = {}
    for claim in claims:
        for citation in claim.accepted:
            if citation.evidence_id not in references:
                references[citation.evidence_id] = Reference(len(references) + 1, evidence_by_id[citation.evidence_id])
            citation.reference = references[citation.evidence_id]

    return ValidatedAnswer(
        status=status,
        summary=summary,
        claims=claims,
        rejected_claims=rejected,
        references=list(references.values()),
        ambiguity_note=clean(data.get("ambiguity_note", ""), "the ambiguity note"),
        limitations=clean(data.get("limitations", ""), "the limitations"),
        follow_ups=[clean(s, "a suggestion") for s in data.get("follow_up_suggestions") or []],
        warnings=warnings,
    )
