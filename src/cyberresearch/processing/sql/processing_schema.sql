-- Phase 2 schema: PDF status + chunks with pgvector embeddings (idempotent).
-- {dimension} is replaced by the embedding dimension from settings.yaml.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per processed paper: access decision, download and extraction status.
CREATE TABLE IF NOT EXISTS pdf_documents (
    paper_id           BIGINT PRIMARY KEY REFERENCES papers (paper_id) ON DELETE CASCADE,
    access_reason      TEXT        NOT NULL,  -- arxiv | license:cc-by | no_pdf_url | license_unknown | ...
    pdf_status         TEXT        NOT NULL,  -- downloaded | robots_disallowed | not_pdf | too_large | failed | not_allowed | skipped_download
    extraction_status  TEXT        NOT NULL DEFAULT 'not_attempted',  -- extracted | extraction_failed | not_attempted
    content_source     TEXT        NOT NULL DEFAULT 'none' CHECK (content_source IN ('pdf', 'abstract', 'none')),
    pdf_url            TEXT,
    pdf_origin         TEXT,
    license            TEXT,
    http_status        INTEGER,
    file_path          TEXT,
    sha256             TEXT,
    size_bytes         BIGINT,
    page_count         INTEGER,
    chunk_count        INTEGER     NOT NULL DEFAULT 0,
    error              TEXT,
    attempts           INTEGER     NOT NULL DEFAULT 1,
    processed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk. PDF chunks must carry a page number (CHECK below), so
-- every citation can point to an actual page of an actual paper.
CREATE TABLE IF NOT EXISTS paper_chunks (
    chunk_id         BIGSERIAL PRIMARY KEY,
    paper_id         BIGINT      NOT NULL REFERENCES papers (paper_id) ON DELETE CASCADE,
    chunk_index      INTEGER     NOT NULL,
    content_source   TEXT        NOT NULL CHECK (content_source IN ('pdf', 'abstract')),
    section          TEXT,
    page_start       INTEGER,
    page_end         INTEGER,
    text             TEXT        NOT NULL,
    word_count       INTEGER     NOT NULL,
    source_url       TEXT,
    embedding        vector({dimension}) NOT NULL,
    embedding_model  TEXT        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (paper_id, chunk_index),
    CHECK (content_source = 'abstract' OR (page_start IS NOT NULL AND page_end >= page_start))
);
CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper ON paper_chunks (paper_id);
-- HNSW = approximate nearest-neighbour index for fast cosine-similarity search.
CREATE INDEX IF NOT EXISTS idx_paper_chunks_embedding ON paper_chunks USING hnsw (embedding vector_cosine_ops);
-- Stored full-text vector: the lexical half of hybrid search.
ALTER TABLE paper_chunks ADD COLUMN IF NOT EXISTS text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS idx_paper_chunks_text_search ON paper_chunks USING gin (text_search);
