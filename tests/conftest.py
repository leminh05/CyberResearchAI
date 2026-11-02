"""Shared fixtures: a PostgreSQL database for persistence tests.

Uses ``TEST_DATABASE_URL`` if set, otherwise starts a throw-away embedded
PostgreSQL (pgserver) in a temporary directory. Every test gets its own
schema, so tests never touch real tables in ``public``.
"""

from __future__ import annotations

import os
import uuid

import pytest

from cyberresearch.ingestion.storage import PostgresStore


@pytest.fixture(scope="session")
def database_url(tmp_path_factory: pytest.TempPathFactory):
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        yield url
        return
    pgserver = pytest.importorskip("pgserver", reason="pgserver not installed and TEST_DATABASE_URL not set")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop")
    yield server.get_uri()
    server.cleanup()


@pytest.fixture(scope="session")
def spark():
    """One local Spark session shared by all Spark tests (startup is slow)."""
    import shutil

    if shutil.which("java") is None:
        pytest.skip("Java is not installed")
    from cyberresearch.bigdata.spark_jobs import create_spark_session
    from cyberresearch.config import SparkSettings

    session = create_spark_session(SparkSettings(master="local[2]", shuffle_partitions=2), "tests")
    yield session
    session.stop()


@pytest.fixture
def store(database_url: str):
    """A PostgresStore whose tables live in a fresh, temporary schema."""
    schema = f"test_{uuid.uuid4().hex[:12]}"
    store = PostgresStore.connect(database_url)
    store.conn.execute(f"CREATE SCHEMA {schema}")
    store.conn.execute(f"SET search_path TO {schema}")
    store.init_schema()
    yield store
    store.conn.execute(f"DROP SCHEMA {schema} CASCADE")
    store.close()
