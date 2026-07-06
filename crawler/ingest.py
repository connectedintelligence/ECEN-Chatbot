"""
ingest.py — Embeds chunks and upserts them into PostgreSQL (pgvector).

Run:
    python ingest.py            # full crawl + ingest
    python ingest.py --diff     # only update changed/new pages
"""

from __future__ import annotations

import argparse
import logging
import os
import hashlib
from datetime import datetime, timezone

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from crawler import crawl
from chunker import chunk_docs, Chunk

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5433/ecen")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIM = 384
BATCH_SIZE = 64

# ── Security: PII / secret scrubbing before indexing ─────────────────────────
# The corpus is a public university site, so emails/phones are intentionally
# kept (public directory data). We scrub things that should never be indexed
# even if they appear on a compromised or misconfigured page.
import re as _re

_PII_PATTERNS = [
    ("SSN", _re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", _re.compile(
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[ -]?\d{4}[ -]?\d{4}[ -]?\d{2,4}\b")),
    ("API_KEY", _re.compile(
        r"(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"AKIA[0-9A-Z]{16}|xox[bporas]-[A-Za-z0-9-]{10,})")),
    ("PRIVATE_KEY", _re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("PASSWORD_ASSIGNMENT", _re.compile(r"(?i)\bpassword\s*[:=]\s*\S{6,}")),
]


def scrub_pii(pages) -> None:
    """Redact sensitive patterns in-place before chunking/embedding.
    Deterministic (same input -> same output) so diff hashes stay stable."""
    import hashlib
    total = 0
    for p in pages:
        hits = []
        text = p.text
        for label, pat in _PII_PATTERNS:
            text, n = pat.subn(f"[REDACTED_{label}]", text)
            if n:
                hits.append(f"{label}x{n}")
        if hits:
            total += len(hits)
            log.warning("PII scrubbed on %s: %s", p.url, ", ".join(hits))
            p.text = text
            p.content_hash = hashlib.md5(text.encode()).hexdigest()
    if total:
        log.warning("PII scrub: redactions on this crawl — review the URLs above.")


# ── Security: data-poisoning guard ───────────────────────────────────────────
# If a huge fraction of the corpus changed in one crawl, something is wrong —
# a compromised site, a broken crawler, or a poisoning attempt. Refuse to
# auto-index and require a human to re-run with FORCE_INGEST=1.
POISON_GUARD_THRESHOLD = float(os.getenv("POISON_GUARD_THRESHOLD", "0.5"))


# ── DB connection ─────────────────────────────────────────────────────────────
def get_conn():
    conn = psycopg2.connect(PG_DSN)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def get_embedder() -> SentenceTransformer:
    log.info(f"Loading local embedding model '{EMBED_MODEL}'...")
    return SentenceTransformer(EMBED_MODEL)


# ── Schema setup ─────────────────────────────────────────────────────────────
def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS ecen_docs (
                id          BIGINT PRIMARY KEY,
                chunk_id    TEXT UNIQUE NOT NULL,
                url         TEXT NOT NULL,
                title       TEXT,
                section     TEXT,
                text        TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_indexed TIMESTAMPTZ NOT NULL,
                embedding   vector({VECTOR_DIM})
            );
        """)
        # HNSW index for fast ANN search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS ecen_docs_embedding_idx
            ON ecen_docs USING hnsw (embedding vector_cosine_ops);
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS ecen_docs_url_idx ON ecen_docs (url);")
        cur.execute("CREATE INDEX IF NOT EXISTS ecen_docs_section_idx ON ecen_docs (section);")
        # Full-text keyword arm: a functional GIN index over the english
        # tsvector. Without it, `to_tsvector('english', text) @@ to_tsquery(...)`
        # recomputes the tsvector for EVERY row on EVERY query (sequential scan).
        cur.execute("""
            CREATE INDEX IF NOT EXISTS ecen_docs_fts_idx
            ON ecen_docs USING gin (to_tsvector('english', text));
        """)
        # Fuzzy/typo arm: trigram GIN index so the `<%` word-similarity operator
        # is index-accelerated instead of scanning the whole corpus per query.
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS ecen_docs_text_trgm_idx
            ON ecen_docs USING gin (text gin_trgm_ops);
        """)
    conn.commit()
    log.info("Table and indexes ready.")


# ── Embedding ────────────────────────────────────────────────────────────────
def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    all_vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i: i + BATCH_SIZE]
        vecs = embedder.encode(batch, normalize_embeddings=True)
        all_vectors.extend(vecs.tolist())
    return all_vectors


# ── Diff logic ───────────────────────────────────────────────────────────────
def get_existing_hashes(conn) -> dict[str, str]:
    """Returns {chunk_id: content_hash} for all stored chunks."""
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, content_hash FROM ecen_docs;")
        return {row[0]: row[1] for row in cur.fetchall()}


# ── Main ingest ───────────────────────────────────────────────────────────────
def ingest(diff_mode: bool = False) -> None:
    conn = get_conn()
    embedder = get_embedder()
    ensure_table(conn)

    log.info("Starting crawl...")
    pages = crawl()
    log.info(f"Crawled {len(pages)} pages. Scrubbing PII...")
    scrub_pii(pages)
    log.info("Chunking...")
    chunks = chunk_docs(pages)
    all_chunks = chunks  # full crawl result, pre-diff — used for stale pruning
    log.info(f"Produced {len(chunks)} chunks.")

    if diff_mode:
        existing = get_existing_hashes(conn)
        total_chunks = len(chunks)
        chunks = [c for c in chunks if existing.get(c.chunk_id) != c.content_hash]
        log.info(f"Diff mode: {len(chunks)} new/changed chunks to upsert.")

        # Poisoning guard: refuse a suspiciously large one-shot rewrite of the
        # corpus unless a human explicitly forces it.
        if existing and total_chunks:
            changed_ratio = len(chunks) / total_chunks
            if changed_ratio > POISON_GUARD_THRESHOLD and not os.getenv("FORCE_INGEST"):
                log.error(
                    "POISON GUARD: %.0f%% of the corpus changed in one crawl "
                    "(threshold %.0f%%). Refusing to auto-index — review the "
                    "site, then re-run with FORCE_INGEST=1 if legitimate.",
                    changed_ratio * 100, POISON_GUARD_THRESHOLD * 100)
                conn.close()
                return

    if not chunks:
        log.info("Nothing to upsert.")
        prune_stale(conn, all_chunks)
        conn.close()
        return

    texts = [c.text for c in chunks]
    log.info(f"Embedding {len(texts)} chunks...")
    vectors = embed_texts(embedder, texts)

    now = datetime.now(timezone.utc)

    log.info(f"Upserting {len(chunks)} chunks into PostgreSQL...")
    with conn.cursor() as cur:
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting"):
            batch_chunks = chunks[i: i + BATCH_SIZE]
            batch_vecs = vectors[i: i + BATCH_SIZE]
            for c, vec in zip(batch_chunks, batch_vecs):
                row_id = int(hashlib.md5(c.chunk_id.encode()).hexdigest(), 16) % (2**63)
                cur.execute("""
                    INSERT INTO ecen_docs
                        (id, chunk_id, url, title, section, text, content_hash, last_indexed, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        url          = EXCLUDED.url,
                        title        = EXCLUDED.title,
                        section      = EXCLUDED.section,
                        text         = EXCLUDED.text,
                        content_hash = EXCLUDED.content_hash,
                        last_indexed = EXCLUDED.last_indexed,
                        embedding    = EXCLUDED.embedding;
                """, (row_id, c.chunk_id, c.url, c.title, c.section,
                      c.text, c.content_hash, now, vec))
    conn.commit()
    prune_stale(conn, all_chunks)
    conn.close()
    log.info("Ingest complete.")


def prune_stale(conn, all_chunks) -> None:
    """Remove DB chunks whose chunk_id no longer exists in the current crawl
    (deleted pages, restructured content). Guard: a partial crawl (network
    issues, MAX_PAGES cut) must not wipe valid data — skip when more than half
    the DB would vanish, unless FORCE_INGEST=1."""
    current_ids = {c.chunk_id for c in all_chunks}
    if not current_ids:
        return
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM ecen_docs;")
        db_ids = {row[0] for row in cur.fetchall()}
    stale = db_ids - current_ids
    if not stale:
        return
    stale_ratio = len(stale) / max(1, len(db_ids))
    if stale_ratio > 0.5 and not os.getenv("FORCE_INGEST"):
        log.warning("Prune skipped: %d stale chunks (%.0f%% of DB) — crawl may "
                    "be incomplete. Re-run with FORCE_INGEST=1 to prune.",
                    len(stale), stale_ratio * 100)
        return
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ecen_docs WHERE chunk_id = ANY(%s);", (list(stale),))
    conn.commit()
    log.info("Pruned %d stale chunks no longer present on the site.", len(stale))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff", action="store_true", help="Only upsert new/changed chunks")
    args = parser.parse_args()
    ingest(diff_mode=args.diff)
