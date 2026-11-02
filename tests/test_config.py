"""Tests for configuration loading."""

from pathlib import Path

import pytest

from cyberresearch.config import DEFAULT_SETTINGS_PATH, load_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "SEMANTIC_SCHOLAR_API_KEY", "OPENALEX_API_KEY", "CONTACT_EMAIL", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)


def _load(tmp_path: Path):
    # An empty .env file keeps tests independent of the developer's real .env.
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    return load_settings(DEFAULT_SETTINGS_PATH, env_file=empty_env)


def test_default_settings_are_valid(tmp_path: Path) -> None:
    settings = _load(tmp_path)
    assert set(settings.sources) == {"arxiv", "openalex", "semantic_scholar"}
    assert [t.key for t in settings.topics] == [
        "network_security",
        "intrusion_detection",
        "ai_for_cybersecurity",
    ]
    for topic in settings.topics:
        assert set(topic.queries) == {"arxiv", "openalex", "semantic_scholar"}


def test_arxiv_respects_three_second_rule(tmp_path: Path) -> None:
    assert _load(tmp_path).sources["arxiv"].min_interval_seconds >= 3.0


def test_semantic_scholar_slower_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    without_key = _load(tmp_path).sources["semantic_scholar"].min_interval_seconds
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "dummy-test-value")
    with_key = _load(tmp_path).sources["semantic_scholar"].min_interval_seconds
    assert without_key > with_key


def test_empty_env_values_become_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "   ")
    assert _load(tmp_path).database_url is None
