"""Resolve which PostgreSQL database to use.

Production / Docker: set ``DATABASE_URL`` in ``.env``.
Local development without PostgreSQL installed: an embedded PostgreSQL 16 +
pgvector server (``pgserver`` package) is started from ``data/pgdata/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cyberresearch.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

EMBEDDED_DATA_DIR = PROJECT_ROOT / "data" / "pgdata"

# Keep a reference so the embedded server lives as long as the process.
_embedded_server: Any = None


def start_embedded_postgres(data_dir: Path = EMBEDDED_DATA_DIR) -> str:
    """Start (or reuse) the embedded server and return its connection URI."""
    global _embedded_server
    try:
        import pgserver
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "DATABASE_URL is not set and pgserver is not installed. "
            "Run: python -m pip install -r requirements-dev.txt"
        ) from exc
    data_dir.mkdir(parents=True, exist_ok=True)
    if _embedded_server is None:
        # cleanup_mode=None keeps the server running for later commands,
        # which avoids a slow restart on every CLI call.
        _embedded_server = pgserver.get_server(data_dir, cleanup_mode=None)
        logger.info("embedded postgres ready", extra={"data_dir": str(data_dir)})
    return _embedded_server.get_uri()


def resolve_database_url(settings: Settings) -> str:
    """Return ``DATABASE_URL`` if configured, otherwise the embedded server URI."""
    return settings.database_url or start_embedded_postgres()
