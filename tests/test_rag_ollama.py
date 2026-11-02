"""Ollama answer generator (no real server: fake HTTP session)."""

import json

import pytest
import requests

from cyberresearch.config import RagSettings
from cyberresearch.rag.llm import LLMError
from cyberresearch.rag.ollama_client import OllamaGenerator
from cyberresearch.rag.prompts import ANSWER_SCHEMA

SETTINGS = RagSettings(provider="ollama", ollama_num_ctx=8000, ollama_num_predict=2000)


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload or {}, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, result):
        self.result, self.calls = result, []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def ok(content, prompt_tokens=3000, done_reason="stop"):
    return FakeResponse(200, {"model": "qwen2.5:7b", "message": {"content": content},
                              "prompt_eval_count": prompt_tokens, "eval_count": 400, "done_reason": done_reason})


def test_request_shape_and_parsing() -> None:
    session = FakeSession(ok(json.dumps({"answer_status": "answered"})))
    result = OllamaGenerator(SETTINGS, session).generate("SYS", "USER", ANSWER_SCHEMA)
    call = session.calls[0]
    assert call["url"] == "http://127.0.0.1:11434/api/chat"
    body = call["json"]
    assert body["model"] == "qwen2.5:7b" and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["format"] == ANSWER_SCHEMA
    assert body["options"] == {"num_ctx": 8000, "num_predict": 2000, "temperature": 0}
    assert result.data == {"answer_status": "answered"} and result.model == "qwen2.5:7b"
    assert result.usage == {"input_tokens": 3000, "output_tokens": 400}


def test_truncated_prompt_is_an_error() -> None:
    session = FakeSession(ok("{}", prompt_tokens=7990))
    with pytest.raises(LLMError, match="truncated"):
        OllamaGenerator(SETTINGS, session).generate("s", "u", ANSWER_SCHEMA)


def test_length_limit_and_invalid_json() -> None:
    assert OllamaGenerator(SETTINGS, FakeSession(ok('{"a', done_reason="length"))).generate("s", "u", {}).stop_reason == "max_tokens"
    assert OllamaGenerator(SETTINGS, FakeSession(ok("not json"))).generate("s", "u", {}).data is None


@pytest.mark.parametrize(
    "result, message",
    [
        (requests.ConnectionError(), "not reachable"),
        (requests.Timeout(), "did not answer"),
        (FakeResponse(404), "ollama pull qwen2.5:7b"),
        (FakeResponse(500, text="out of memory"), "out of memory"),
    ],
)
def test_errors_are_explained(result, message) -> None:
    with pytest.raises(LLMError, match=message):
        OllamaGenerator(SETTINGS, FakeSession(result)).generate("s", "u", {})


def test_translate_query_and_fallback() -> None:
    session = FakeSession(ok('{"english_query": "LLM vulnerability detection"}'))
    assert OllamaGenerator(SETTINGS, session).translate_query("phát hiện lỗ hổng") == "LLM vulnerability detection"
    assert session.calls[0]["json"]["options"]["num_predict"] == 256
    failing = FakeSession(requests.ConnectionError())
    assert OllamaGenerator(SETTINGS, failing).translate_query("phát hiện lỗ hổng") == "phát hiện lỗ hổng"


def test_provider_setting_is_validated(tmp_path) -> None:
    from cyberresearch.config import _rag_settings

    settings = _rag_settings({"provider": "ollama", "ollama": {"model": "qwen2.5:3b", "num_ctx": 16384}})
    assert settings.provider == "ollama" and settings.ollama_model == "qwen2.5:3b" and settings.ollama_num_ctx == 16384
    with pytest.raises(ValueError):
        _rag_settings({"provider": "openai"})
