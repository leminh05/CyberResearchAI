"""Tests for structured logging and secret redaction."""

import json
import logging

from cyberresearch.logging_setup import JsonFormatter, redact


def test_redact_hides_api_keys() -> None:
    url = "https://api.openalex.org/works?search=x&api_key=SECRET123&per-page=10"
    assert "SECRET123" not in redact(url)
    assert "api_key=***" in redact(url)
    assert "SECRET" not in redact("x-api-key: SECRET")


def test_json_formatter_includes_extra_fields_and_redacts() -> None:
    record = logging.makeLogRecord(
        {"msg": "GET %s", "args": ("https://x?api_key=abc",), "levelname": "INFO", "name": "t"}
    )
    record.source = "arxiv"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["source"] == "arxiv"
    assert "abc" not in payload["message"]
    assert payload["level"] == "INFO"
