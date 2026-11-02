"""Structured (JSON lines) logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Matches secrets in URLs or messages, e.g. "api_key=abc123" or "x-api-key: abc".
_SECRET_PATTERN = re.compile(r"(?i)((?:api[_-]?key|token|password)[\"']?\s*[=:]\s*[\"']?)([^&\s\"',}]+)")

# Attributes every LogRecord has; anything else was passed via ``extra=``.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def redact(text: str) -> str:
    """Replace secret values (API keys, tokens, passwords) with ``***``."""
    return _SECRET_PATTERN.sub(r"\1***", text)


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line.

    Fields passed with ``logger.info("msg", extra={"source": "arxiv"})`` are
    included as top-level keys, which makes logs easy to filter and parse.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS:
                payload[key] = redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", log_dir: Path | None = None, file_name: str = "ingestion.log") -> None:
    """Configure root logging: JSON lines to a rotating file and to the console."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / file_name, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # urllib3 logs full request URLs at DEBUG, which could contain API keys.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
