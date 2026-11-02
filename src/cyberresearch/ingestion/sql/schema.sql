-- CyberResearchAI ingestion schema (idempotent: safe to run many times).

-- One row per unique paper, after deduplication across sources.
CREATE TABLE IF NOT EXISTS papers (
    paper_id          BIGSERIAL PRIMARY KEY,
    title             TEXT,
    normalized_title  TEXT,
    abstract          TEXT,
    authors           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    year              INTEGER,
    publication_date  DATE,
    doi               TEXT UNIQUE,
    arxiv_id          TEXT UNIQUE,
    s2_paper_id       TEXT UNIQUE,
    openalex_id       TEXT UNIQUE,
    venue             TEXT,
    url               TEXT,
    pdf_url           TEXT,
    is_open_access    BOOLEAN,
    topics            TEXT[]      NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_papers_title_year ON papers (normalized_title, year);

-- One row per (source, source_id): keeps the original payload of every source.
CREATE TABLE IF NOT EXISTS paper_sources (
    id               BIGSERIAL PRIMARY KEY,
    paper_id         BIGINT      NOT NULL REFERENCES papers (paper_id) ON DELETE CASCADE,
    source           TEXT        NOT NULL,
    source_id        TEXT        NOT NULL,
    source_url       TEXT,
    content_hash     TEXT        NOT NULL,
    raw              JSONB       NOT NULL,
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_changed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS idx_paper_sources_paper ON paper_sources (paper_id);

-- One row per (run, source, topic, time window): what happened and how much.
CREATE TABLE IF NOT EXISTS ingestion_logs (
    log_id           BIGSERIAL PRIMARY KEY,
    run_id           UUID        NOT NULL,
    mode             TEXT        NOT NULL CHECK (mode IN ('daily', 'backfill')),
    source           TEXT        NOT NULL,
    topic            TEXT        NOT NULL,
    window_start     DATE        NOT NULL,
    window_end       DATE        NOT NULL,
    status           TEXT        NOT NULL CHECK (status IN ('running', 'completed', 'partial', 'failed')),
    pages            INTEGER     NOT NULL DEFAULT 0,
    fetched          INTEGER     NOT NULL DEFAULT 0,
    inserted         INTEGER     NOT NULL DEFAULT 0,
    merged           INTEGER     NOT NULL DEFAULT 0,
    updated          INTEGER     NOT NULL DEFAULT 0,
    unchanged        INTEGER     NOT NULL DEFAULT 0,
    skipped          INTEGER     NOT NULL DEFAULT 0,
    failed           INTEGER     NOT NULL DEFAULT 0,
    error_message    TEXT,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_run ON ingestion_logs (run_id);

-- Resume state: the API cursor reached for each (mode, source, topic, window).
CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
    checkpoint_key   TEXT PRIMARY KEY,
    mode             TEXT        NOT NULL CHECK (mode IN ('daily', 'backfill')),
    source           TEXT        NOT NULL,
    topic            TEXT        NOT NULL,
    window_start     DATE        NOT NULL,
    window_end       DATE        NOT NULL,
    cursor           TEXT,
    status           TEXT        NOT NULL CHECK (status IN ('in_progress', 'completed')),
    records_fetched  INTEGER     NOT NULL DEFAULT 0,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
