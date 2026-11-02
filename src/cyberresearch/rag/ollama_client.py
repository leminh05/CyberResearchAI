"""Local answer generator using Ollama (free alternative to the Claude API).

Same interface as ``ClaudeClient``: identical prompt, JSON schema and,
downstream, identical citation validation. Ollama constrains the output to
the schema through its ``format`` parameter.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from cyberresearch.config import RagSettings
from cyberresearch.rag.llm import TRANSLATE_SCHEMA, TRANSLATE_SYSTEM, LLMError, LLMResult

logger = logging.getLogger(__name__)

# If the prompt fills (almost) the whole context window, Ollama has cut it:
# the model would answer without seeing all the evidence.
TRUNCATION_MARGIN = 0.98


class OllamaGenerator:
    """Calls ``POST /api/chat`` on a local Ollama server."""

    def __init__(self, settings: RagSettings, session: Any = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.url = settings.ollama_base_url.rstrip("/") + "/api/chat"

    def translate_query(self, question: str) -> str:
        """English search query for a non-English question; original on failure."""
        try:
            result = self.generate(TRANSLATE_SYSTEM, question, TRANSLATE_SCHEMA, max_tokens=256)
        except LLMError:
            return question
        query = (result.data or {}).get("english_query", "").strip()
        return query or question

    def generate(
        self, system: str, user_message: str, schema: dict, effort: str | None = None, max_tokens: int | None = None
    ) -> LLMResult:
        """``effort`` is accepted for interface compatibility; Ollama has no equivalent."""
        body = {
            "model": self.settings.ollama_model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_message}],
            "format": schema,
            "options": {
                "num_ctx": self.settings.ollama_num_ctx,
                "num_predict": max_tokens or self.settings.ollama_num_predict,
                "temperature": 0,  # deterministic answers for grounded QA
            },
        }
        try:
            response = self.session.post(self.url, json=body, timeout=self.settings.ollama_timeout_seconds)
        except requests.ConnectionError as exc:
            raise LLMError(f"Ollama server not reachable at {self.settings.ollama_base_url} - start Ollama") from exc
        except requests.Timeout as exc:
            raise LLMError(f"Ollama did not answer within {self.settings.ollama_timeout_seconds:.0f} s") from exc
        if response.status_code == 404:
            raise LLMError(f"model {self.settings.ollama_model!r} not found - run: ollama pull {self.settings.ollama_model}")
        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:200]}")

        payload = response.json()
        prompt_tokens = payload.get("prompt_eval_count") or 0
        usage = {"input_tokens": prompt_tokens, "output_tokens": payload.get("eval_count") or 0}
        logger.info("ollama response", extra={"model": payload.get("model"), **usage})
        if prompt_tokens >= self.settings.ollama_num_ctx * TRUNCATION_MARGIN:
            raise LLMError(
                f"prompt ({prompt_tokens} tokens) filled the {self.settings.ollama_num_ctx}-token context and was "
                "truncated - raise rag.ollama.num_ctx or lower rag.top_k"
            )
        if payload.get("done_reason") == "length":
            return LLMResult(None, "max_tokens", payload.get("model"), usage=usage)

        try:
            data = json.loads(payload.get("message", {}).get("content") or "")
        except json.JSONDecodeError:
            data = None
        return LLMResult(data, "end_turn", payload.get("model"), usage=usage)
