"""RAG engine: retrieve -> gate -> generate -> validate, with conversation memory."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cyberresearch.config import RagSettings
from cyberresearch.ingestion.normalize import is_mostly_english
from cyberresearch.rag.citations import ValidatedAnswer, validate_answer
from cyberresearch.rag.llm import AnswerGenerator, LLMError, LLMResult, QueryTranslator
from cyberresearch.rag.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, build_user_message
from cyberresearch.rag.retrieval import Evidence, Retriever, SearchFilters

logger = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = (
    "The indexed papers contain no passage that matches this question closely enough "
    "(best similarity {best:.2f} < {threshold:.2f}). No answer is given rather than an unsupported one."
)


@dataclass
class RAGResponse:
    question: str
    retrieval_query: str
    filters: SearchFilters
    evidence: list[Evidence]
    answer: ValidatedAnswer
    llm: LLMResult | None = None
    gated: bool = False  # True when Claude was not called (no relevant evidence)

    @property
    def best_similarity(self) -> float:
        return max((e.similarity for e in self.evidence), default=0.0)


@dataclass
class RAGEngine:
    """Answers questions from retrieved evidence and keeps follow-up context."""

    retriever: Retriever
    generator: AnswerGenerator
    settings: RagSettings
    history: list[tuple[str, str]] = field(default_factory=list)
    # The embedding model is English-only (all-MiniLM-L6-v2): a Vietnamese
    # question retrieved an unrelated Indonesian paper in testing. Non-English
    # questions are therefore translated into an English search query; the
    # answer is still written in the language of the question.
    translator: QueryTranslator | None = None
    _translations: dict[str, str] = field(default_factory=dict, repr=False)

    def reset(self) -> None:
        self.history.clear()

    def _search_text(self, question: str) -> str:
        """English text to embed for ``question`` (translated once, then cached)."""
        if self.translator is None or is_mostly_english(question):
            return question
        if question not in self._translations:
            self._translations[question] = self.translator.translate_query(question)
        return self._translations[question]

    def retrieval_query(self, question: str) -> str:
        """Follow-ups ("what about its false positives?") are searched together
        with the previous question so pronouns still find the right papers."""
        current = self._search_text(question)
        if self.history:
            return f"{self._search_text(self.history[-1][0])} {current}"
        return current

    def ask(self, question: str, filters: SearchFilters | None = None) -> RAGResponse:
        filters = filters or SearchFilters()
        query = self.retrieval_query(question)
        evidence = self.retriever.retrieve(query, filters)
        response = RAGResponse(question, query, filters, evidence, ValidatedAnswer("insufficient_evidence", ""))

        # Gate: do not ask Claude when nothing relevant was retrieved; an LLM
        # given irrelevant text tends to answer from general knowledge anyway.
        if response.best_similarity < self.settings.min_similarity:
            response.gated = True
            response.answer.summary = NO_EVIDENCE_MESSAGE.format(
                best=response.best_similarity, threshold=self.settings.min_similarity
            )
            return response

        user_message = build_user_message(
            question, evidence, self.history[-self.settings.history_turns:], self.settings.weak_similarity
        )
        try:
            result = self.generator.generate(SYSTEM_PROMPT, user_message, ANSWER_SCHEMA)
        except LLMError as exc:
            logger.warning("answer generation failed", extra={"error": str(exc)})
            response.answer = ValidatedAnswer("error", f"Answer generation failed: {exc}")
            return response
        response.llm = result

        if result.stop_reason == "refusal":
            response.answer = ValidatedAnswer(
                "refused",
                "The model declined to answer this request"
                + (f" (category: {result.refusal_category})." if result.refusal_category else "."),
            )
        elif result.data is None:
            logger.warning("model output unusable", extra={"stop_reason": result.stop_reason, "model": result.model})
            response.answer = ValidatedAnswer(
                "error", f"No usable answer (stop reason: {result.stop_reason}; output was not valid JSON or was cut off)."
            )
        else:
            response.answer = validate_answer(result.data, evidence, self.settings.min_quote_words)
            if response.answer.status in ("answered", "partial"):
                memory = response.answer.summary or " ".join(c.statement for c in response.answer.claims)
                self.history.append((question, memory[:600]))

        logger.info(
            "question answered",
            extra={
                "status": response.answer.status,
                "evidence": len(evidence),
                "best_similarity": round(response.best_similarity, 3),
                "claims": len(response.answer.claims),
                "rejected_claims": len(response.answer.rejected_claims),
                "invalid_citations": len(response.answer.invalid_citations),
            },
        )
        return response
