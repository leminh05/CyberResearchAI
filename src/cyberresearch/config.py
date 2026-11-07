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
class ProcessingSettings:
    """PDF access policy, download limits and chunking parameters."""

    pdf_directory: Path = PROJECT_ROOT / "data" / "pdfs"
    max_pdf_megabytes: float = 30.0
    allowed_licenses: tuple[str, ...] = ("cc0", "public-domain", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa")
    allow_unknown_license: bool = False
    host_min_interval_seconds: float = 1.0
    chunk_max_words: int = 220
    chunk_overlap_words: int = 40
    skip_sections: tuple[str, ...] = ("References", "Bibliography", "Acknowledgments", "Acknowledgements")


@dataclass(frozen=True)
class EmbeddingSettings:
    """Sentence Transformers model used for chunk embeddings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


@dataclass(frozen=True)
class RagSettings:
    """Retrieval and answer-generation settings (Phase 3)."""

    provider: str = "claude"  # claude | ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_num_ctx: int = 12288
    ollama_num_predict: int = 4096
    ollama_timeout_seconds: float = 900.0
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000
    top_k: int = 8
    candidate_k: int = 40
    exact_vector_search: bool = True
    hnsw_ef_search: int = 200
    hybrid_search: bool = True
    lexical_max_doc_fraction: float = 0.10
    rrf_k: int = 60
    max_chunks_per_paper: int = 3
    min_similarity: float = 0.30
    weak_similarity: float = 0.45
    exclude_sections: tuple[str, ...] = ("Front Matter",)
    history_turns: int = 3
    min_quote_words: int = 5


@dataclass(frozen=True)
class SparkSettings:
    """Local Spark session configuration."""

    master: str = "local[*]"
    driver_memory: str = "2g"
    shuffle_partitions: int = 8
    lake_root: str = "hdfs://namenode:8020/cyberresearch"
    hdfs_replication: int = 2


@dataclass(frozen=True)
class StreamingSettings:
    """Kafka and Spark Structured Streaming."""

    bootstrap_servers: str = "kafka:9092"
    topic: str = "papers.raw"
    max_offsets_per_trigger: int = 200
    trigger_seconds: int = 30


@dataclass(frozen=True)
class WebSettings:
    """Limits of the web API (Phase 5)."""

    rate_limit_per_minute: int = 10
    max_concurrent_requests: int = 2
    max_conversations: int = 1000
    conversation_ttl_minutes: int = 60


@dataclass(frozen=True)
class Settings:
    """All application settings."""

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
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    spark: SparkSettings = field(default_factory=SparkSettings)
    rag: RagSettings = field(default_factory=RagSettings)
    web: WebSettings = field(default_factory=WebSettings)
    streaming: StreamingSettings = field(default_factory=StreamingSettings)
    # Separate from database_url, so streamed papers cannot change the evaluated corpus.
    stream_database_url: str | None = None
    # Only whether a key exists; the SDK reads the key itself from the environment.
    anthropic_api_key_present: bool = False


def _rag_settings(raw: dict[str, Any]) -> RagSettings:
    values = dict(raw)
    for key, value in (values.pop("ollama", None) or {}).items():
        values[f"ollama_{key}"] = value
    if values.get("provider", "claude") not in ("claude", "ollama"):
        raise ValueError("rag.provider must be 'claude' or 'ollama'")
    if "exclude_sections" in values:
        values["exclude_sections"] = tuple(values["exclude_sections"])
    # Inside Docker, Ollama runs on the host (host.docker.internal), not localhost.
    if _env("OLLAMA_BASE_URL"):
        values["ollama_base_url"] = _env("OLLAMA_BASE_URL")
    return RagSettings(**values)


def _processing_settings(raw: dict[str, Any]) -> ProcessingSettings:
    """Build ProcessingSettings, resolving the PDF directory and list values."""
    values = dict(raw)
    pdf_dir = Path(values.pop("pdf_directory", "data/pdfs"))
    values["pdf_directory"] = pdf_dir if pdf_dir.is_absolute() else PROJECT_ROOT / pdf_dir
    for key in ("allowed_licenses", "skip_sections"):
        if key in values:
            values[key] = tuple(values[key])
    return ProcessingSettings(**values)


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
        stream_database_url=_env("STREAM_DATABASE_URL"),
        streaming=StreamingSettings(**(raw.get("streaming") or {})),
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
        processing=_processing_settings(raw.get("processing") or {}),
        embedding=EmbeddingSettings(**(raw.get("embedding") or {})),
        spark=SparkSettings(**(raw.get("spark") or {})),
        rag=_rag_settings(raw.get("rag") or {}),
        web=WebSettings(**(raw.get("web") or {})),
        anthropic_api_key_present=bool(_env("ANTHROPIC_API_KEY") or _env("ANTHROPIC_AUTH_TOKEN")),
    )
