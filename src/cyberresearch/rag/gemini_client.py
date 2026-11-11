"""Answer generator using the Google Gemini API (free tier available).

Same interface as ``ClaudeClient``: identical prompt, JSON schema and,
downstream, identical citation validation. Questions and paper excerpts are
sent to Google, unlike with the local Ollama model.
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

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# A minute-limit wait is short; a longer one means the daily limit, so fail instead of hanging.
MAX_RETRY_WAIT_SECONDS = 90.0
MAX_RATE_LIMIT_RETRIES = 4


def _keys() -> list[str]:
    """``GEMINI_API_KEY`` may hold several keys separated by commas."""
    return [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]


def key_is_set() -> bool:
    return bool(_keys())


def _retry_delay(response: Any) -> float:
    """Seconds from the RetryInfo detail of a 429 body; the full wait if missing."""
    try:
        for detail in response.json()["error"].get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s"))
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return MAX_RETRY_WAIT_SECONDS


class GeminiGenerator:
    """Calls ``models/{model}:generateContent``. The key is read from ``GEMINI_API_KEY`` only."""

    translate_query = OllamaGenerator.translate_query

    def __init__(self, settings: RagSettings, session: Any = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.sleep = sleep
        self.key_index = 0

    def _headers(self) -> dict[str, str]:
        keys = _keys()
        if not keys:
            raise KeyError("GEMINI_API_KEY")
        # Header, not ?key=: URLs end up in logs.
        return {"x-goog-api-key": keys[self.key_index % len(keys)]}

    def is_ready(self) -> bool:
        """True if the key works and the configured model is available."""
        try:
            response = self.session.get(f"{BASE_URL}/models/{self.settings.gemini_model}", headers=self._headers(), timeout=5)
            return response.status_code == 200
        except (requests.RequestException, KeyError):
            return False

    def _post(self, body: dict[str, Any]) -> Any:
        url = f"{BASE_URL}/models/{self.settings.gemini_model}:generateContent"
        if not _keys():
            raise KeyError("GEMINI_API_KEY")
        for _ in range(MAX_RATE_LIMIT_RETRIES + 1):
            waits = []
            for _ in range(len(_keys())):  # each key once, then wait
                try:
                    response = self.session.post(url, json=body, headers=self._headers(), timeout=self.settings.gemini_timeout_seconds)
                except requests.RequestException as exc:
                    raise LLMError(f"Gemini not reachable: {type(exc).__name__}") from exc
                if response.status_code != 429:
                    return response
                waits.append(_retry_delay(response))
                self.key_index += 1
                logger.info("gemini rate limit, next key", extra={"key_number": self.key_index % len(_keys()) + 1})
            wait = min(waits)
            if wait > MAX_RETRY_WAIT_SECONDS:
                raise LLMError(f"Gemini limit reached on every key, retry in {wait:.0f} s (probably the daily limit)")
            logger.info("gemini rate limit on every key, waiting", extra={"seconds": wait})
            self.sleep(wait)
        raise LLMError("Gemini limit still reached after several waits (probably the daily limit)")

    def generate(
        self, system: str, user_message: str, schema: dict, effort: str | None = None, max_tokens: int | None = None
    ) -> LLMResult:
        """``effort`` is accepted for interface compatibility."""
        model = self.settings.gemini_model
        try:
            response = self._post({
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user_message}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": max_tokens or self.settings.gemini_max_tokens,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                },
            })
        except KeyError as exc:
            raise LLMError("GEMINI_API_KEY is not set") from exc
        if response.status_code in (401, 403) or "API_KEY_INVALID" in response.text:
            raise LLMError("Gemini rejected the API key - check GEMINI_API_KEY in .env")
        if response.status_code >= 400:
            raise LLMError(f"Gemini error {response.status_code}: {response.text[:200]}")

        payload = response.json()
        tokens = payload.get("usageMetadata") or {}
        usage = {"input_tokens": tokens.get("promptTokenCount") or 0, "output_tokens": tokens.get("candidatesTokenCount") or 0}
        logger.info("gemini response", extra={"model": payload.get("modelVersion"), **usage})
        candidate = (payload.get("candidates") or [{}])[0]
        finish = candidate.get("finishReason")
        if finish == "MAX_TOKENS":
            return LLMResult(None, "max_tokens", payload.get("modelVersion"), usage=usage)
        if finish not in (None, "STOP"):  # SAFETY, RECITATION, ...: no usable answer
            return LLMResult(None, "refusal", payload.get("modelVersion"), refusal_category=finish, usage=usage)
        text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []) if not p.get("thought"))
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        return LLMResult(data, "end_turn", payload.get("modelVersion"), usage=usage)
