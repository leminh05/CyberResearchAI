"""Copy the embedded development database (data/pgdata) into PostgreSQL in Docker.

Usage (PowerShell, project root, after `docker compose up -d`):
    python scripts/migrate_embedded_to_docker.py

Uses pg_dump / pg_restore bundled with the pgserver package, so nothing else
has to be installed. The target is DATABASE_URL from .env. Tables that
already contain rows in the target are refused, so nothing is overwritten.
Row counts of every table are compared at the end.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from cyberresearch.config import load_settings  # noqa: E402
from cyberresearch.db import start_embedded_postgres  # noqa: E402

TABLES = ["papers", "paper_sources", "ingestion_logs", "ingestion_checkpoints", "pdf_documents", "paper_chunks"]


def pg_bin(name: str) -> str:
    import pgserver

    return str(Path(pgserver.__file__).parent / "pginstall" / "bin" / f"{name}.exe" if os.name == "nt" else name)


def counts(url: str) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    with psycopg.connect(url) as conn:
        for table in TABLES:
            exists = conn.execute("SELECT to_regclass(%s)", (f"public.{table}",)).fetchone()[0]
            result[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] if exists else None
    return result


def libpq_env(url: str) -> dict[str, str]:
    """Connection settings as PG* variables, so the password never appears on a command line."""
    parts = urlsplit(url)
    env = dict(os.environ)
    env.update(
        PGHOST=parts.hostname or "localhost",
        PGPORT=str(parts.port or 5432),
        PGUSER=unquote(parts.username or ""),
        PGPASSWORD=unquote(parts.password or ""),
        PGDATABASE=parts.path.lstrip("/"),
    )
    return env


def main() -> int:
    settings = load_settings()
    target = settings.database_url
    if not target:
        print("DATABASE_URL is empty in .env - set it to the Docker database first.")
        return 2
    # SOURCE_DATABASE_URL: an already running embedded server (for example one
    # started by hand after a crash recovery); otherwise pgserver starts it.
    source = os.getenv("SOURCE_DATABASE_URL") or start_embedded_postgres()

    target_counts = counts(target)
    if any(target_counts.values()):
        print("The Docker database already has data; refusing to overwrite:", target_counts)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "cyberresearch.dump"
        subprocess.run(
            [pg_bin("pg_dump"), "--format=custom", "--no-owner", "--no-acl", "--schema=public", f"--file={dump}"],
            env=libpq_env(source), check=True,
        )
        print(f"dump: {dump.stat().st_size / 1024 / 1024:.1f} MB")
        # The vector extension already exists in the target, so pg_restore
        # reports that one statement as an error; the rest is restored.
        restore = subprocess.run(
            [pg_bin("pg_restore"), "--no-owner", "--no-acl", "--dbname", libpq_env(target)["PGDATABASE"], str(dump)],
            env=libpq_env(target), capture_output=True, text=True,
        )
        errors = [line for line in restore.stderr.splitlines() if "error" in line.lower()]
        unexpected = [e for e in errors if "already exists" not in e and "errors ignored" not in e]
        for line in errors:
            print("pg_restore:", line)

    source_counts, target_counts = counts(source), counts(target)
    print(f"{'table':<24}{'embedded':>10}{'docker':>10}")
    for table in TABLES:
        print(f"{table:<24}{source_counts[table] or 0:>10}{target_counts[table] or 0:>10}")
    ok = source_counts == target_counts and not unexpected
    print("migration OK" if ok else "MIGRATION MISMATCH - check the output above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
