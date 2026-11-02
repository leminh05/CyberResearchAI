"""Claude API client for structured, evidence-grounded answers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from cyberresearch.config import RagSettings

logger = logging.getLogger(__name__)

# Server-side fallback: if a safety classifier declines the request (cyber
# topics can trigger the "cyber" category), the API re-runs it on the model
# Anthropic recommends for that category instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """Claude could not produce a usable answer (network, auth, limits)."""


@dataclass
class LLMResult:
    """Parsed answer JSON plus metadata about how it was produced."""

    data: dict[str, Any] | None
    stop_reason: str | None
    model: str | None
    refusal_category: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class AnswerGenerator(Protocol):
    def generate(self, system: str, user_message: str, schema: dict) -> LLMResult:
        ...


class QueryTranslator(Protocol):
    def translate_query(self, question: str) -> str:
        ...


TRANSLATE_SYSTEM = (
    "Rewrite the user's question as a short English search query for a database of "
    "English cybersecurity papers. Keep technical terms. Do not answer the question."
)
TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"english_query": {"type": "string"}},
    "required": ["english_query"],
    "additionalProperties": False,
}


class ClaudeClient:
    """Calls the Messages API with structured output and refusal fallback.

    Credentials come from the environment (``ANTHROPIC_API_KEY`` in .env);
    the key is never logged or printed.
    """

    def __init__(self, settings: RagSettings, client: anthropic.Anthropic | None = None) -> None:
        self.settings = settings
        self.client = client or anthropic.Anthropic(max_retries=3)

    def translate_query(self, question: str) -> str:
        """English search query for a non-English question (the embedding model is English-only).

        Returns the original question if translation fails, so retrieval still runs.
        """
        try:
            result = self.generate(TRANSLATE_SYSTEM, question, TRANSLATE_SCHEMA, effort="low", max_tokens=1024)
        except LLMError:
            return question
        query = (result.data or {}).get("english_query", "").strip()
        return query or question

    def generate(
        self, system: str, user_message: str, schema: dict, effort: str | None = None, max_tokens: int | None = None
    ) -> LLMResult:
        try:
            response = self.client.beta.messages.create(
                model=self.settings.model,
                max_tokens=max_tokens or self.settings.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                output_config={
                    "effort": effort or self.settings.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude API authentication failed - check ANTHROPIC_API_KEY in .env") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("the API key is not allowed to use this model") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"model {self.settings.model!r} not found") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited by the Claude API - try again shortly") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"request rejected: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("cannot reach the Claude API - check the network connection") from exc

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        logger.info(
            "claude response",
            extra={"model": response.model, "stop_reason": response.stop_reason, **usage},
        )

        # Check the stop reason BEFORE reading content.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return LLMResult(None, "refusal", response.model, getattr(details, "category", None), usage)
        if response.stop_reason == "max_tokens":
            return LLMResult(None, "max_tokens", response.model, usage=usage)

        text = next((b.text for b in response.content if b.type == "text"), None)
        try:
            data = json.loads(text) if text else None
        except json.JSONDecodeError:
            data = None
        return LLMResult(data, response.stop_reason, response.model, usage=usage)
