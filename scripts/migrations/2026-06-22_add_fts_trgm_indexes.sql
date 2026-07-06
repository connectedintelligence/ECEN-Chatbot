-- 2026-06-22 — Latency fix: index the keyword + fuzzy retrieval arms.
--
-- Before this, _keyword_search() recomputed to_tsvector('english', text) for
-- every row on every query, and _fuzzy_search() ran word_similarity() across
-- the whole corpus — both full sequential scans on each request. These two
-- indexes make both arms index lookups.
--
-- Run ONCE against the live database (the schema in crawler/ingest.py already
-- creates these for fresh rebuilds). CONCURRENTLY avoids locking the table, so
-- the service can keep serving while the indexes build.
--
--   psql "$PG_DSN" -f scripts/migrations/2026-06-22_add_fts_trgm_indexes.sql
--
-- NOTE: CREATE INDEX CONCURRENTLY cannot run inside a transaction block. If you
-- paste these into a tool that wraps statements in a transaction (e.g. some
-- Supabase SQL editors), drop the word CONCURRENTLY or run them one at a time.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Full-text keyword arm.
CREATE INDEX CONCURRENTLY IF NOT EXISTS ecen_docs_fts_idx
    ON ecen_docs USING gin (to_tsvector('english', text));

-- Fuzzy/typo arm (the `<%` word-similarity operator).
CREATE INDEX CONCURRENTLY IF NOT EXISTS ecen_docs_text_trgm_idx
    ON ecen_docs USING gin (text gin_trgm_ops);

-- Refresh planner statistics so the new indexes get used immediately.
ANALYZE ecen_docs;
