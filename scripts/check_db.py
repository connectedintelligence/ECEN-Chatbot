#!/usr/bin/env python3
"""
DB health check for the TAMU ECE RAG chatbot.

Verifies the Postgres + pgvector setup the backend depends on:
  - connection works (PG_DSN)
  - `vector` extension installed
  - `ecen_docs` table exists with the expected schema
  - row count, distinct URLs, per-section breakdown
  - embedding dimension == 384, no NULL/zero embeddings
  - HNSW + supporting indexes present

Run on the machine where Postgres is reachable (host or Docker host):
    python scripts/check_db.py
Reads PG_DSN from .env (falls back to the compose default).
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

import psycopg2

PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5433/ecen")
EXPECTED_DIM = 384

OK, WARN, FAIL = "  OK ", " WARN", " FAIL"


def line(status, msg):
    print(f"[{status}] {msg}")


def main():
    print(f"Connecting: {PG_DSN.split('@')[-1]}")
    try:
        conn = psycopg2.connect(PG_DSN)
    except Exception as e:
        line(FAIL, f"cannot connect: {e}")
        print("\n-> Is the Docker container running?  docker compose up postgres -d")
        sys.exit(1)
    line(OK, "connection established")

    cur = conn.cursor()

    # vector extension
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
    if cur.fetchone():
        line(OK, "pgvector extension installed")
    else:
        line(FAIL, "pgvector extension NOT installed  ->  CREATE EXTENSION vector;")

    # table exists
    cur.execute("SELECT to_regclass('public.ecen_docs');")
    if not cur.fetchone()[0]:
        line(FAIL, "table ecen_docs does not exist  ->  run: python crawler/ingest.py")
        conn.close()
        sys.exit(1)
    line(OK, "table ecen_docs exists")

    # row / url counts
    cur.execute("SELECT count(*), count(DISTINCT url) FROM ecen_docs;")
    rows, urls = cur.fetchone()
    if rows == 0:
        line(FAIL, "table is EMPTY  ->  run: python crawler/ingest.py")
    else:
        line(OK, f"{rows} chunks across {urls} distinct URLs")
        # memory note: expected ~1288 chunks / 491 pages
        if rows < 800:
            line(WARN, "fewer chunks than the ~1288 expected from a full crawl")

    # section breakdown
    cur.execute("SELECT section, count(*) FROM ecen_docs GROUP BY section ORDER BY 2 DESC;")
    secs = cur.fetchall()
    if secs:
        print("       sections: " + ", ".join(f"{s or 'NULL'}={c}" for s, c in secs))

    # embedding integrity
    cur.execute("SELECT count(*) FROM ecen_docs WHERE embedding IS NULL;")
    nulls = cur.fetchone()[0]
    if nulls:
        line(FAIL, f"{nulls} rows have NULL embedding")
    else:
        line(OK, "no NULL embeddings")

    cur.execute("SELECT vector_dims(embedding) FROM ecen_docs WHERE embedding IS NOT NULL LIMIT 1;")
    r = cur.fetchone()
    if r:
        dim = r[0]
        if dim == EXPECTED_DIM:
            line(OK, f"embedding dimension = {dim}")
        else:
            line(FAIL, f"embedding dimension = {dim}, expected {EXPECTED_DIM} (model mismatch)")

    # indexes
    cur.execute("""
        SELECT indexname, indexdef FROM pg_indexes
        WHERE tablename = 'ecen_docs';
    """)
    idx = {n: d for n, d in cur.fetchall()}
    if any("hnsw" in d.lower() for d in idx.values()):
        line(OK, "HNSW vector index present")
    else:
        line(WARN, "no HNSW index -> vector search will be slow (seq scan)")
    for want in ("ecen_docs_url_idx", "ecen_docs_section_idx"):
        line(OK if want in idx else WARN, f"index {want}: {'present' if want in idx else 'missing'}")

    # Full-text + trigram indexes back the keyword and fuzzy retrieval arms.
    # Without them those arms recompute to_tsvector / word_similarity over the
    # whole corpus on every query (seq scan) -> seconds of added latency.
    defs = " ".join(idx.values()).lower()
    if "ecen_docs_fts_idx" in idx or "to_tsvector" in defs:
        line(OK, "full-text GIN index present (keyword arm)")
    else:
        line(WARN, "no full-text GIN index -> keyword search seq-scans every query "
                   "(run scripts/migrations/2026-06-22_add_fts_trgm_indexes.sql)")
    if "ecen_docs_text_trgm_idx" in idx or "gin_trgm_ops" in defs:
        line(OK, "trigram GIN index present (fuzzy arm)")
    else:
        line(WARN, "no trigram GIN index -> fuzzy search seq-scans every query "
                   "(run scripts/migrations/2026-06-22_add_fts_trgm_indexes.sql)")

    # freshness
    cur.execute("SELECT max(last_indexed) FROM ecen_docs;")
    last = cur.fetchone()[0]
    if last:
        line(OK, f"last indexed: {last}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
