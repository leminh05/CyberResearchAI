"""Answer generator using the Groq API (free tier available).

Same interface as ``ClaudeClient``: identical prompt, JSON schema and,
downstream, identical citation validation. Questions and paper excerpts are
sent to Groq, unlike with the local Ollama model.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

import requests

from cyberresearch.config import RagSettings
from cyberresearch.rag.llm import LLMError, LLMResult
from cyberresearch.rag.ollama_client import OllamaGenerator

logger = logging.getLogger(__name__)

BASE_URL = "https://api.groq.com/openai/v1"
# A minute-limit wait is short; a longer one means the daily limit, so fail instead of hanging.
MAX_RETRY_WAIT_SECONDS = 90.0
MAX_RATE_LIMIT_RETRIES = 4


def key_is_set() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


class GroqGenerator:
    """Calls the OpenAI-compatible chat endpoint. The key is read from ``GROQ_API_KEY`` only."""

    translate_query = OllamaGenerator.translate_query

    def __init__(self, settings: RagSettings, session: Any = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.sleep = sleep

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}"}

    def is_ready(self) -> bool:
        """True if the key works and the configured model is available."""
        try:
            response = self.session.get(f"{BASE_URL}/models/{self.settings.groq_model}", headers=self._headers(), timeout=5)
            return response.status_code == 200
        except (requests.RequestException, KeyError):
            return False

    def _post(self, body: dict[str, Any]) -> Any:
        for _ in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self.session.post(
                    f"{BASE_URL}/chat/completions", json=body, headers=self._headers(),
                    timeout=self.settings.groq_timeout_seconds,
                )
            except requests.RequestException as exc:
                raise LLMError(f"Groq not reachable: {type(exc).__name__}") from exc
            if response.status_code != 429:
                return response
            wait = float(response.headers.get("retry-after") or MAX_RETRY_WAIT_SECONDS)
            if wait > MAX_RETRY_WAIT_SECONDS:
                raise LLMError(f"Groq limit reached, retry in {wait:.0f} s (probably the daily limit)")
            logger.info("groq rate limit, waiting", extra={"seconds": wait})
            self.sleep(wait)
        raise LLMError("Groq limit still reached after several waits")

    def generate(
        self, system: str, user_message: str, schema: dict, effort: str | None = None, max_tokens: int | None = None
    ) -> LLMResult:
        """``effort`` is accepted for interface compatibility."""
        model = self.settings.groq_model
        try:
            response = self._post({
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_message}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": False}},
                "reasoning_format": "hidden",
                "max_completion_tokens": max_tokens or self.settings.groq_max_tokens,
                "temperature": 0,
            })
        except KeyError as exc:
            raise LLMError("GROQ_API_KEY is not set") from exc
        if response.status_code == 400 and "json_validate_failed" in response.text:
            return LLMResult(None, "end_turn", model)  # the model broke the JSON format
        if response.status_code in (401, 403):
            raise LLMError("Groq rejected the API key - check GROQ_API_KEY in .env")
        if response.status_code >= 400:
            raise LLMError(f"Groq error {response.status_code}: {response.text[:200]}")

        payload = response.json()
        tokens = payload.get("usage") or {}
        usage = {"input_tokens": tokens.get("prompt_tokens") or 0, "output_tokens": tokens.get("completion_tokens") or 0}
        logger.info("groq response", extra={"model": payload.get("model"), **usage})
        choice = payload["choices"][0]
        if choice.get("finish_reason") == "length":
            return LLMResult(None, "max_tokens", payload.get("model"), usage=usage)
        try:
            data = json.loads(choice["message"].get("content") or "")
        except json.JSONDecodeError:
            data = None
        return LLMResult(data, "end_turn", payload.get("model"), usage=usage)
