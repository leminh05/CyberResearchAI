-- Runs once, when the PostgreSQL volume is created.
-- The application also runs CREATE EXTENSION IF NOT EXISTS; doing it here makes
-- the extension available before the first connection.
CREATE EXTENSION IF NOT EXISTS vector;
