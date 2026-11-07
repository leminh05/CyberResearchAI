"""Groq answer generator (no real server: fake HTTP session)."""

import json
from types import SimpleNamespace

import pytest
import requests

from cyberresearch.config import RagSettings
from cyberresearch.rag.groq_client import GroqGenerator
from cyberresearch.rag.llm import LLMError

SETTINGS = RagSettings(provider="groq", groq_model="m", groq_max_tokens=100)
SECRET = "gsk_test_not_real"


def response(status=200, payload=None, text="", headers=None):
    return SimpleNamespace(status_code=status, text=text, headers=headers or {}, json=lambda: payload)


def answer(content='{"a": 1}', finish="stop"):
    return response(payload={"model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                             "choices": [{"finish_reason": finish, "message": {"content": content}}]})


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
    return GroqGenerator(SETTINGS, Session(*results), sleeps.append), sleeps


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)


def test_request_and_parsed_answer() -> None:
    gen, _ = generator(answer())
    result = gen.generate("sys", "user", {"type": "object"})
    call = gen.session.calls[0]
    assert result.data == {"a": 1} and result.usage == {"input_tokens": 5, "output_tokens": 2}
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["json"]["temperature"] == 0 and call["json"]["max_completion_tokens"] == 100
    assert call["json"]["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_waits_for_the_minute_limit_then_succeeds() -> None:
    gen, sleeps = generator(response(429, headers={"retry-after": "7"}), answer())
    assert gen.generate("s", "u", {}).data == {"a": 1} and sleeps == [7.0]


def test_daily_limit_fails_instead_of_waiting() -> None:
    gen, sleeps = generator(response(429, headers={"retry-after": "3600"}))
    with pytest.raises(LLMError, match="daily"):
        gen.generate("s", "u", {})
    assert not sleeps


def test_unusable_output_gives_no_data() -> None:
    assert generator(answer("not json"))[0].generate("s", "u", {}).data is None
    assert generator(answer(finish="length"))[0].generate("s", "u", {}).stop_reason == "max_tokens"
    assert generator(response(400, text='{"code": "json_validate_failed"}'))[0].generate("s", "u", {}).data is None


@pytest.mark.parametrize("failure", [response(401, text=SECRET), requests.ConnectionError(f"Bearer {SECRET}")])
def test_errors_never_contain_the_key(failure) -> None:
    with pytest.raises(LLMError) as error:
        generator(failure)[0].generate("s", "u", {})
    assert SECRET not in str(error.value)


def test_missing_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        generator()[0].generate("s", "u", {})


def test_translate_query_falls_back_to_the_question() -> None:
    assert generator(answer(json.dumps({"english_query": "ids"})))[0].translate_query("câu hỏi") == "ids"
    assert generator(response(500, text="x"))[0].translate_query("câu hỏi") == "câu hỏi"


def test_is_ready() -> None:
    assert generator(response(200))[0].is_ready() is True
    assert generator(response(401))[0].is_ready() is False


def test_answer_language_is_explicit() -> None:
    from cyberresearch.rag.prompts import answer_language

    assert answer_language("How are GNNs used?") == "English"
    assert answer_language("Phát hiện xâm nhập là gì?") == "Vietnamese"
    assert answer_language("Qué es una intrusión?") == "the same language as the question"
