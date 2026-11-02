"""Application configuration.

Non-secret settings come from ``config/settings.yaml``; secrets and
machine-specific values come from environment variables (loaded from ``.env``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class HttpSettings:
    """Timeout and retry policy shared by all API clients."""

    timeout_seconds: float = 30.0
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0


@dataclass(frozen=True)
class SourceSettings:
    """Per-source endpoint, page size and rate limit."""

    name: str
    base_url: str
    min_interval_seconds: float
    page_size: int
    enabled: bool = True


@dataclass(frozen=True)
class Topic:
    """A research topic with one query string per source."""

    key: str
    name: str
    queries: dict[str, str]


@dataclass(frozen=True)
class IngestionSettings:
    """Limits and windows for daily and backfill ingestion."""

    max_records_per_topic: int = 200
    daily_initial_lookback_days: int = 3
    daily_overlap_days: int = 1
    backfill_window_months: int = 12


@dataclass(frozen=True)
class Settings:
    """All settings needed by the ingestion pipeline."""

    database_url: str | None
    contact_email: str | None
    openalex_api_key: str | None
    semantic_scholar_api_key: str | None
    log_level: str
    log_dir: Path
    log_file_name: str
    http: HttpSettings
    ingestion: IngestionSettings
    sources: dict[str, SourceSettings] = field(default_factory=dict)
    topics: list[Topic] = field(default_factory=list)


def _env(name: str) -> str | None:
    """Return a stripped environment variable, or None when empty/missing."""
    value = os.getenv(name, "").strip()
    return value or None


def load_settings(settings_path: Path | None = None, env_file: Path | None = None) -> Settings:
    """Load settings from YAML and environment variables.

    Args:
        settings_path: YAML file to read (defaults to config/settings.yaml).
        env_file: .env file to load (defaults to PROJECT_ROOT/.env if present).
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    path = settings_path or DEFAULT_SETTINGS_PATH
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    s2_key = _env("SEMANTIC_SCHOLAR_API_KEY")
    sources: dict[str, SourceSettings] = {}
    for name, cfg in (raw.get("sources") or {}).items():
        interval = cfg["min_interval_seconds"]
        # Without an API key, Semantic Scholar's public pool needs a slower pace.
        if name == "semantic_scholar" and not s2_key:
            interval = cfg.get("min_interval_seconds_without_key", interval)
        sources[name] = SourceSettings(
            name=name,
            base_url=cfg["base_url"],
            min_interval_seconds=float(interval),
            page_size=int(cfg["page_size"]),
            enabled=bool(cfg.get("enabled", True)),
        )

    topics = [
        Topic(
            key=key,
            name=cfg["name"],
            queries={k: v for k, v in cfg.items() if k != "name"},
        )
        for key, cfg in (raw.get("topics") or {}).items()
    ]

    log_cfg = raw.get("logging") or {}
    log_dir = Path(log_cfg.get("directory", "logs"))
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    return Settings(
        database_url=_env("DATABASE_URL"),
        contact_email=_env("CONTACT_EMAIL"),
        openalex_api_key=_env("OPENALEX_API_KEY"),
        semantic_scholar_api_key=s2_key,
        log_level=(_env("LOG_LEVEL") or "INFO").upper(),
        log_dir=log_dir,
        log_file_name=log_cfg.get("file_name", "ingestion.log"),
        http=HttpSettings(**(raw.get("http") or {})),
        ingestion=IngestionSettings(**(raw.get("ingestion") or {})),
        sources=sources,
        topics=topics,
    )
