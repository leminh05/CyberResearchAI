"""Gemini answer generator (no real server: fake HTTP session)."""

import json
from types import SimpleNamespace

import pytest
import requests

from cyberresearch.config import RagSettings
from cyberresearch.rag.gemini_client import GeminiGenerator
from cyberresearch.rag.llm import LLMError

SETTINGS = RagSettings(provider="gemini", gemini_model="m", gemini_max_tokens=100)
SECRET = "AIza_test_not_real"


def response(status=200, payload=None, text=""):
    return SimpleNamespace(status_code=status, text=text, json=lambda: payload)


def answer(content='{"a": 1}', finish="STOP", thought=None):
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": content}]
    return response(payload={"modelVersion": "m", "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
                             "candidates": [{"finishReason": finish, "content": {"parts": parts}}]})


def rate_limited(delay):
    detail = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}
    return response(429, payload={"error": {"code": 429, "details": [detail]}})


class Session:
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def get(self, url, headers=None, timeout=None):
        return self.results.pop(0)


def generator(*results):
    sleeps = []
    return GeminiGenerator(SETTINGS, Session(*results), sleeps.append), sleeps


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)


def test_request_and_parsed_answer() -> None:
    gen, _ = generator(answer(thought="thinking..."))
    result = gen.generate("sys", "user", {"type": "object"})
    call = gen.session.calls[0]
    assert result.data == {"a": 1} and result.usage == {"input_tokens": 5, "output_tokens": 2}
    assert call["url"].endswith("/models/m:generateContent") and SECRET not in call["url"]
    assert call["headers"]["x-goog-api-key"] == SECRET
    config = call["json"]["generationConfig"]
    assert config["temperature"] == 0 and config["maxOutputTokens"] == 100
    assert config["responseJsonSchema"] == {"type": "object"}
    assert call["json"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_waits_for_the_minute_limit_then_succeeds() -> None:
    gen, sleeps = generator(rate_limited("7s"), answer())
    assert gen.generate("s", "u", {}).data == {"a": 1} and sleeps == [7.0]


def test_daily_limit_fails_instead_of_waiting() -> None:
    gen, sleeps = generator(rate_limited("3600s"))
    with pytest.raises(LLMError, match="daily"):
        gen.generate("s", "u", {})
    assert not sleeps


def test_unusable_output_gives_no_data() -> None:
    assert generator(answer("not json"))[0].generate("s", "u", {}).data is None
    assert generator(answer(finish="MAX_TOKENS"))[0].generate("s", "u", {}).stop_reason == "max_tokens"
    blocked = generator(answer(finish="SAFETY"))[0].generate("s", "u", {})
    assert blocked.stop_reason == "refusal" and blocked.refusal_category == "SAFETY"


@pytest.mark.parametrize("failure", [response(403, text="denied"), requests.ConnectionError(f"key={SECRET}")])
def test_errors_never_contain_the_key(failure) -> None:
    with pytest.raises(LLMError) as error:
        generator(failure)[0].generate("s", "u", {})
    assert SECRET not in str(error.value)


def test_missing_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        generator()[0].generate("s", "u", {})


def test_translate_query_falls_back_to_the_question() -> None:
    assert generator(answer(json.dumps({"english_query": "ids"})))[0].translate_query("câu hỏi") == "ids"
    assert generator(response(500, text="x"))[0].translate_query("câu hỏi") == "câu hỏi"


def test_is_ready() -> None:
    assert generator(response(200))[0].is_ready() is True
    assert generator(response(401))[0].is_ready() is False


def test_switches_to_the_next_key_before_waiting(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k1, k2,k3")
    gen, sleeps = generator(rate_limited("30s"), rate_limited("30s"), answer(), answer())
    assert gen.generate("s", "u", {}).data == {"a": 1} and not sleeps
    assert [c["headers"]["x-goog-api-key"] for c in gen.session.calls] == ["k1", "k2", "k3"]
    gen.generate("s", "u", {})  # the next question starts on the key that worked
    assert gen.session.calls[-1]["headers"]["x-goog-api-key"] == "k3"


def test_waits_only_when_every_key_is_limited(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k1,k2")
    gen, sleeps = generator(rate_limited("20s"), rate_limited("9s"), answer())
    assert gen.generate("s", "u", {}).data == {"a": 1} and sleeps == [9.0]
    gen, sleeps = generator(rate_limited("3600s"), rate_limited("7200s"))
    with pytest.raises(LLMError, match="every key"):
        gen.generate("s", "u", {})
    assert not sleeps
