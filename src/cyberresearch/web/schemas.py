"""Request validation and the citation response schema of the web API."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints

from cyberresearch.rag.citations import CheckedClaim, Reference
from cyberresearch.rag.engine import RAGResponse

MAX_QUESTION_CHARS = 2000


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)]
    conversation_id: UUID | None = None
    provider: Annotated[str, StringConstraints(max_length=20)] | None = None


class CitationOut(BaseModel):
    """A citation that passed validation: its quote is in the chunk of ``reference``."""

    reference: int
    quote: str


class ClaimOut(BaseModel):
    statement: str
    kind: str  # evidence | interpretation
    citations: list[CitationOut]


class ReferenceOut(BaseModel):
    """Built from the database row behind the citation, never from model output."""

    number: int
    title: str | None
    authors: list[str]
    year: int | None
    url: str | None
    doi: str | None
    arxiv_id: str | None
    section: str | None
    location: str
    content_source: str
    paper_id: int
    chunk_id: int
    similarity: float


class AnswerMeta(BaseModel):
    gated: bool  # True when no LLM was called (no relevant evidence)
    evidence_count: int
    best_similarity: float
    model: str | None
    retrieval_query: str


class ChatResponse(BaseModel):
    conversation_id: UUID
    question: str
    status: str  # answered | partial | insufficient_evidence | out_of_scope | refused
    summary: str
    claims: list[ClaimOut]
    references: list[ReferenceOut]
    ambiguity_note: str
    limitations: str
    follow_ups: list[str]
    warnings: list[str]
    meta: AnswerMeta


def safe_url(url: str | None) -> str | None:
    """Only http(s) links reach the browser: paper URLs come from external APIs."""
    return url if url and urlsplit(url).scheme.lower() in ("http", "https") else None


def _claim(claim: CheckedClaim) -> ClaimOut:
    citations = [CitationOut(reference=c.reference.number, quote=c.quote) for c in claim.accepted if c.reference]
    return ClaimOut(statement=claim.statement, kind=claim.kind, citations=citations)


def _reference(ref: Reference) -> ReferenceOut:
    e = ref.evidence
    return ReferenceOut(
        number=ref.number,
        title=e.title,
        authors=e.authors,
        year=e.year,
        url=safe_url(e.canonical_url),
        doi=e.doi,
        arxiv_id=e.arxiv_id,
        section=e.section if e.content_source == "pdf" else None,
        location=e.location,
        content_source=e.content_source,
        paper_id=e.paper_id,
        chunk_id=e.chunk_id,
        similarity=round(e.similarity, 3),
    )


def to_chat_response(response: RAGResponse, conversation_id: UUID) -> ChatResponse:
    a = response.answer
    return ChatResponse(
        conversation_id=conversation_id,
        question=response.question,
        status=a.status,
        summary=a.summary,
        claims=[_claim(c) for c in a.claims],
        references=[_reference(r) for r in a.references],
        ambiguity_note=a.ambiguity_note,
        limitations=a.limitations,
        follow_ups=a.follow_ups[:3],
        warnings=a.warnings,
        meta=AnswerMeta(
            gated=response.gated,
            evidence_count=len(response.evidence),
            best_similarity=round(response.best_similarity, 3),
            model=response.llm.model if response.llm else None,
            retrieval_query=response.retrieval_query,
        ),
    )
