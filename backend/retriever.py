"""
retriever.py — Hybrid retrieval (dense + BM25) with cross-encoder re-ranking.

Pipeline:
  1. Dense search via pgvector cosine similarity (top-40)
  2. BM25 sparse search over the same candidate set
  3. Reciprocal Rank Fusion to merge both lists
  4. Cross-encoder re-ranker → final top-2
"""

from __future__ import annotations

import os
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv(override=True)

log = logging.getLogger(__name__)

TAMU_API_URL = os.getenv("OPENAI_BASE_URL", "https://chat-api.tamu.ai/openai")
TAMU_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAMU_MODEL = os.getenv("OPENAI_MODEL", "protected.gpt-5")

_REWRITE_PROMPT = (
    "You are a search query optimizer for the TAMU ECE department website. "
    "Rewrite the user's question into 2-3 keyword-rich search phrases that would match "
    "content on an engineering department website. "
    "Output only the rewritten query — no explanation, no quotes, no bullet points. "
    "Keep it under 30 words.\n\nQuestion: {question}\n\nRewritten query:"
)


async def _rewrite_query(question: str) -> str:
    """Use the LLM to expand the query into retrieval-friendly form."""
    import httpx, json as _json
    payload = {
        "model": TAMU_MODEL,
        "messages": [
            {"role": "user", "content": _REWRITE_PROMPT.format(question=question)}
        ],
        "temperature": 0.0,
        "stream": True,
    }
    parts = []
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            async with client.stream(
                "POST", f"{TAMU_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {TAMU_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data in ("[DONE]", ""):
                        continue
                    try:
                        obj = _json.loads(data)
                        text = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if text:
                            parts.append(text)
                    except Exception:
                        pass
        rewritten = "".join(parts).strip()
        log.info("Query rewrite: %r → %r", question, rewritten)
        return rewritten if rewritten else question
    except Exception as e:
        log.warning("Query rewrite failed: %s — using original query", e)
        return question


PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5433/ecen")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Candidate-pool sizes. These drive how many (query, passage) pairs the CPU
# cross-encoder scores (the dominant retrieval cost) — BUT cutting them too far
# drops correct chunks out of the top results and wrecks answer quality. Kept at
# the original recall-safe values; speed is bought from CPU/DB instead, not by
# starving recall. Env-tunable for careful experimentation.
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "40"))
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "8"))
# "List everything" questions (all degrees, every research area, etc.) need more
# chunks than a pointed factual lookup, or items past the 8th chunk are silently
# dropped. For list-intent queries we widen the cross-encoder cut-off.
LIST_TOP_K = 20
# Max sibling chunks pulled from the top result's page on a NON-list query.
# Bounds the extra cross-encoder rerank pass; list queries ignore this cap.
SIBLING_TOP_K = 6

# Words that signal the user wants a *complete enumeration*, not a single fact.
_LIST_INTENT_RE = re.compile(
    r"\b(all|every|list|each|complete|full|entire|overview|"
    r"what are|which|name the|how many)\b",
    re.IGNORECASE,
)


def _is_list_query(query: str) -> bool:
    """True when the question asks for a full list (so we shouldn't cap at 8)."""
    return bool(_LIST_INTENT_RE.search(query or ""))

# Lazy-loaded singletons
_embedder = None
_reranker: Optional[CrossEncoder] = None

# ── Connection pool ──────────────────────────────────────────────────────────
# Previously each request created a fresh ThreadPoolExecutor, so its worker
# threads were new every time and their thread-local connections were always
# None — meaning EVERY request opened 3-4 brand-new connections to a remote DB
# AND ran `CREATE EXTENSION` on each before it could query. Over a remote
# Postgres that connection+DDL churn was seconds of latency on every request.
#
# A process-wide pool fixes both: connections are opened once and reused, and
# the one-time extension setup happens once at pool creation, never on the hot
# path. register_vector() is applied per physical connection (tracked with an
# attribute) since the pool hands the same connection objects back repeatedly.
_POOL = None
_POOL_LOCK = threading.Lock()
POOL_MIN = int(os.getenv("PG_POOL_MIN", "2"))
POOL_MAX = int(os.getenv("PG_POOL_MAX", "16"))
# psycopg2 connection objects are C types that don't accept custom attributes,
# so we can't tag them directly. Track which physical connections have had the
# pgvector adapter registered by id(); pool connections are long-lived (the pool
# holds a reference) so their ids stay stable for the life of the process.
_registered_conn_ids: set[int] = set()


def _get_pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                from psycopg2 import pool as _pg_pool
                try:
                    host_part = PG_DSN.split("@")[-1].split("/")[0]
                except Exception:
                    host_part = "(unknown)"
                log.info("Creating PG connection pool at %s (min=%d max=%d)…",
                         host_part, POOL_MIN, POOL_MAX)
                _POOL = _pg_pool.ThreadedConnectionPool(
                    POOL_MIN, POOL_MAX, PG_DSN, connect_timeout=10)
                # One-time extension setup — NOT on the per-request hot path.
                conn = _POOL.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                    conn.commit()
                finally:
                    _POOL.putconn(conn)
    return _POOL


@contextmanager
def _conn():
    """Check out a pooled connection; guarantee it's returned to the pool.

    Each retrieval arm calls this independently, so parallel arms get distinct
    connections (no shared-connection concurrency bug) without paying a fresh
    connect()/DDL on every request.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        if id(conn) not in _registered_conn_ids:
            register_vector(conn)
            _registered_conn_ids.add(id(conn))
        yield conn
        # End the implicit read transaction cleanly (also discards any
        # SET LOCAL, e.g. the fuzzy threshold) so the next checkout is fresh.
        conn.rollback()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pool.putconn(conn)


def _get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            log.info("Using Apple Metal (MPS) for model inference.")
            return "mps"
        if torch.cuda.is_available():
            log.info("Using CUDA GPU for model inference.")
            return "cuda"
    except Exception:
        pass
    log.info("Using CPU for model inference.")
    return "cpu"


_torch_configured = False


def _configure_torch_threads() -> None:
    """Make CPU inference use all allocated cores.

    Cloud Run containers report the HOST cpu count via os.cpu_count(), not the
    `--cpu` limit, so PyTorch's default thread count is wrong — it under- or
    over-subscribes, which is why bumping `--cpu` didn't fully speed up the
    cross-encoder. TORCH_NUM_THREADS should match the Cloud Run --cpu value
    (default 4). Pure performance knob — does not affect outputs.
    """
    global _torch_configured
    if _torch_configured:
        return
    try:
        import torch
        n = int(os.getenv("TORCH_NUM_THREADS", "4"))
        torch.set_num_threads(n)
        log.info("torch num_threads set to %d (cores for CPU inference)", n)
    except Exception as e:  # noqa: BLE001
        log.warning("could not set torch threads: %s", e)
    _torch_configured = True


def _get_embedder():
    global _embedder
    if _embedder is None:
        _configure_torch_threads()
        from sentence_transformers import SentenceTransformer
        device = _get_device()
        log.info(f"Loading local embedding model '{EMBED_MODEL}' on {device}...")
        _embedder = SentenceTransformer(EMBED_MODEL, device=device)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _configure_torch_threads()
        # max_length caps how many tokens of each passage the cross-encoder
        # processes. Kept at the model default (512) so re-ranking sees the full
        # chunk — dropping to 256 measurably hurt ranking quality (correct
        # chunks fell out of the top results). Tune via RERANK_MAX_LEN.
        max_len = int(os.getenv("RERANK_MAX_LEN", "512"))
        log.info("Loading cross-encoder re-ranker (max_length=%d)...", max_len)
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                 max_length=max_len)
    return _reranker


def _embed_query(query: str) -> list[float]:
    vec = _get_embedder().encode([query], normalize_embeddings=True)
    return vec[0].tolist()


def _row_to_chunk(row) -> dict:
    return {
        "chunk_id": row[0],
        "url":      row[1],
        "title":    row[2],
        "section":  row[3],
        "text":     row[4],
        "score":    float(row[5]) if row[5] is not None else 0.0,
    }


def _dense_search(query_vec: list[float], section_filter: Optional[str] = None) -> list[dict]:
    with _conn() as conn, conn.cursor() as cur:
        if section_filter:
            cur.execute("""
                SELECT chunk_id, url, title, section, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM ecen_docs
                WHERE section = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vec, section_filter, query_vec, DENSE_TOP_K))
        else:
            cur.execute("""
                SELECT chunk_id, url, title, section, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM ecen_docs
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vec, query_vec, DENSE_TOP_K))
        return [_row_to_chunk(r) for r in cur.fetchall()]


_STOPWORDS = {
    "who", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "and", "or", "what", "which", "are", "was", "were", "this", "that",
    "with", "from", "by", "as", "be", "it", "do", "does", "did", "how",
    "me", "my", "i", "you", "your", "our", "we", "tell", "about", "give",
}


def _significant_words(query: str) -> list[str]:
    """Lowercased, de-duplicated, stopword-stripped alphanumeric tokens."""
    words = []
    seen = set()
    for raw in query.lower().split():
        w = "".join(ch for ch in raw if ch.isalnum())
        if len(w) < 2 or w in _STOPWORDS or w in seen:
            continue
        seen.add(w)
        words.append(w)
    return words


def _build_tsqueries(query: str) -> tuple[str, str]:
    """Return (or_query, phrase_query) tsquery strings.

    - or_query  ORs every significant word for maximum recall (a single
      missing term like the acronym "ece" shouldn't drop a perfect match).
    - phrase_query ORs the adjacent bigrams using the `<->` operator so chunks
      containing the actual phrase ("department head") can be ranked ABOVE
      pages that merely repeat the words scattered (e.g. "Departmental
      Committees"), which ts_rank's term-density scoring otherwise favors.
    """
    words = _significant_words(query)
    or_query = " | ".join(words)
    bigrams = [f"{a} <-> {b}" for a, b in zip(words, words[1:])]
    phrase_query = " | ".join(bigrams)
    return or_query, phrase_query


def _keyword_search(query: str, section_filter: Optional[str] = None) -> list[dict]:
    """Postgres full-text search over the ENTIRE corpus.

    Dense search can miss exact-phrase matches when many chunks are
    near-duplicates in embedding space (e.g. 80+ faculty profiles that all
    start "Professor, Electrical & Computer Engineering"). This independent
    keyword arm guarantees that lexical matches like "Department Head" enter
    the candidate pool regardless of their dense rank.
    """
    or_query, phrase_query = _build_tsqueries(query)
    if not or_query:
        return []
    # If there are no bigrams (single significant word), phrase matching is a
    # no-op; reuse the or_query so the query is always valid.
    phrase_query = phrase_query or or_query
    with _conn() as conn, conn.cursor() as cur:
        if section_filter:
            cur.execute("""
                SELECT chunk_id, url, title, section, text,
                       ts_rank(to_tsvector('english', text),
                               to_tsquery('english', %s)) AS score,
                       (to_tsvector('english', text)
                            @@ to_tsquery('english', %s))::int AS phrase_hit
                FROM ecen_docs
                WHERE section = %s
                  AND to_tsvector('english', text) @@ to_tsquery('english', %s)
                ORDER BY phrase_hit DESC, score DESC
                LIMIT %s;
            """, (or_query, phrase_query, section_filter, or_query, KEYWORD_TOP_K))
        else:
            # Exclude news: college-wide news repeats common phrases (e.g.
            # "department head") for OTHER departments and drowns out the ECE
            # page that actually answers the question. News is still reachable
            # via dense search and the News section filter.
            cur.execute("""
                SELECT chunk_id, url, title, section, text,
                       ts_rank(to_tsvector('english', text),
                               to_tsquery('english', %s)) AS score,
                       (to_tsvector('english', text)
                            @@ to_tsquery('english', %s))::int AS phrase_hit
                FROM ecen_docs
                WHERE section <> 'news'
                  AND to_tsvector('english', text) @@ to_tsquery('english', %s)
                ORDER BY phrase_hit DESC, score DESC
                LIMIT %s;
            """, (or_query, phrase_query, or_query, KEYWORD_TOP_K))
        # _row_to_chunk reads cols 0-5; the extra phrase_hit col is ignored.
        return [_row_to_chunk(r) for r in cur.fetchall()]


FUZZY_THRESHOLD = 0.45  # word_similarity cutoff; ~typo tolerance for 1-2 edits
FUZZY_TOP_K = int(os.getenv("FUZZY_TOP_K", "15"))


def _fuzzy_search(query: str, section_filter: Optional[str] = None) -> list[dict]:
    """Trigram (pg_trgm) fuzzy keyword search for typo tolerance.

    Full-text search needs exact lexical tokens, so a misspelling like
    'deparment hed' matches nothing. word_similarity(term, text) scores a term
    against the MOST similar word in the chunk, so a typo'd term still scores
    high against the correctly-spelled word in the document. We score each
    chunk by the best-matching query term and keep those above a threshold.
    """
    words = _significant_words(query)
    if not words:
        return []

    # GREATEST(word_similarity(w1, text), word_similarity(w2, text), ...)
    sim_terms = ", ".join(["word_similarity(%s, text)"] * len(words))
    score_expr = f"GREATEST({sim_terms})" if len(words) > 1 else sim_terms
    # Use the `<%` operator (term <% text) rather than the function form
    # `word_similarity(term, text) > k`. Only `<%` can use the trigram GIN
    # index (ecen_docs_text_trgm_idx); the function-with-comparison form forces
    # a full-table scan even when the index exists. `<%` is true when
    # word_similarity >= pg_trgm.word_similarity_threshold (set per-txn below),
    # so it matches the previous FUZZY_THRESHOLD semantics. The GREATEST(...)
    # scoring in SELECT now only runs on the few rows the index returned.
    where_sim = " OR ".join(["%s <%% text"] * len(words))

    score_params = list(words)
    where_params: list = list(words)

    with _conn() as conn, conn.cursor() as cur:
        # Session-local threshold so `<%` matches the old word_similarity > k cut.
        cur.execute("SET LOCAL pg_trgm.word_similarity_threshold = %s;",
                    (FUZZY_THRESHOLD,))
        if section_filter:
            sql = f"""
                SELECT chunk_id, url, title, section, text, {score_expr} AS score
                FROM ecen_docs
                WHERE section = %s AND ({where_sim})
                ORDER BY score DESC
                LIMIT %s;
            """
            params = score_params + [section_filter] + where_params + [FUZZY_TOP_K]
        else:
            sql = f"""
                SELECT chunk_id, url, title, section, text, {score_expr} AS score
                FROM ecen_docs
                WHERE section <> 'news' AND ({where_sim})
                ORDER BY score DESC
                LIMIT %s;
            """
            params = score_params + where_params + [FUZZY_TOP_K]
        cur.execute(sql, params)
        return [_row_to_chunk(r) for r in cur.fetchall()]


def _fetch_siblings(url: str, existing_ids: set, limit: Optional[int] = None) -> list[dict]:
    """Fetch chunks from the same URL, excluding already-seen chunk_ids.

    `limit` caps how many chunks are pulled — and therefore how many pairs the
    cross-encoder must score downstream. None means all chunks (used for
    list/enumeration pages, where completeness matters); a small cap is plenty
    when we're only enriching a single factual answer with adjacent context.
    """
    with _conn() as conn, conn.cursor() as cur:
        if limit is not None:
            cur.execute("""
                SELECT chunk_id, url, title, section, text, 0.0
                FROM ecen_docs
                WHERE url = %s
                ORDER BY chunk_id
                LIMIT %s;
            """, (url, limit))
        else:
            cur.execute("""
                SELECT chunk_id, url, title, section, text, 0.0
                FROM ecen_docs
                WHERE url = %s;
            """, (url,))
        return [
            _row_to_chunk(r)
            for r in cur.fetchall()
            if r[0] not in existing_ids
        ]


def _bm25_rerank(query: str, candidates: list[dict]) -> list[dict]:
    tokenized_corpus = [c["text"].lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    for i, c in enumerate(candidates):
        c["bm25_score"] = float(scores[i])
    return candidates


def _rrf_merge(dense: list[dict], bm25_scored: list[dict], k: int = 60) -> list[dict]:
    dense_rank = {c["chunk_id"]: i + 1 for i, c in enumerate(dense)}
    bm25_sorted = sorted(bm25_scored, key=lambda x: x["bm25_score"], reverse=True)
    bm25_rank = {c["chunk_id"]: i + 1 for i, c in enumerate(bm25_sorted)}

    all_ids = set(dense_rank) | set(bm25_rank)
    chunk_map = {c["chunk_id"]: c for c in dense + bm25_scored}
    merged = {}
    for cid in all_ids:
        d_rank = dense_rank.get(cid, DENSE_TOP_K + 1)
        b_rank = bm25_rank.get(cid, DENSE_TOP_K + 1)
        merged[cid] = 1 / (k + d_rank) + 1 / (k + b_rank)

    sorted_ids = sorted(merged, key=merged.get, reverse=True)[:DENSE_TOP_K]
    return [chunk_map[cid] for cid in sorted_ids if cid in chunk_map]


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "is", "are", "who",
    "what", "where", "which", "and", "or", "ece", "tamu", "department", "dept",
}
# Query intents that should favor a person's faculty profile over an org/list page.
_PERSON_INTENT = {
    "who", "head", "chair", "chairman", "director", "dean", "professor",
    "lead", "leader", "leadership", "contact", "name",
}


def _phrase_boost(query: str, candidates: list[dict]) -> None:
    """
    The cross-encoder rates topical similarity, so a list page ("Departmental
    Committees") can outscore the one page that literally answers the question
    ("...Department Head..."). Nudge scores up when a chunk contains exact query
    phrases, and when a person-seeking query hits a faculty-profile chunk.
    """
    ql = query.lower()
    words = [w for w in re.findall(r"[a-z]+", ql)]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    person_query = bool(set(words) & _PERSON_INTENT)

    for c in candidates:
        text = (c.get("text") or "").lower()
        boost = 0.0
        # Exact phrase (bigram) match — strongest lexical signal that this page
        # is the answer, not merely the same topic.
        for bg in bigrams:
            if bg not in _STOPWORDS and bg in text:
                boost += 2.0
        # Each distinct content keyword present adds a little.
        boost += 0.4 * sum(1 for kw in set(keywords) if kw in text)
        # A "who is the head/chair/..." query should prefer a person's profile.
        if person_query and c.get("section") == "people":
            boost += 2.5
        c["rerank_score"] = c.get("rerank_score", 0.0) + boost


def _cross_encode(query: str, candidates: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    _phrase_boost(query, candidates)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# Query keywords → URLs that should always be injected into candidates
_URL_INJECTIONS: list[tuple[list[str], str]] = [
    (["degree", "program", "degrees", "programs", "major", "offered"],
     "https://engineering.tamu.edu/electrical/academics/degrees/index.html"),
    (["graduate program", "graduate degree", "masters", "master of science", "phd", "doctoral"],
     "https://engineering.tamu.edu/electrical/academics/degrees/graduate/index.html"),
    (["undergraduate", "bachelor", "bs degree"],
     "https://engineering.tamu.edu/electrical/academics/degrees/undergraduate/index.html"),
    (["research area", "research group", "research program", "research areas", "what research"],
     "https://engineering.tamu.edu/electrical/research/research-areas.html"),
    (["power system", "power systems", "smart grid", "smart grids", "energy", "power engineering"],
     "https://engineering.tamu.edu/electrical/research/energy-and-power.html"),
    (["control system", "control systems", "controls", "linear control", "nonlinear control", "feedback", "ecen 420"],
     "https://catalog.tamu.edu/undergraduate/engineering/electrical-computer/#coursestext"),
    (["control system", "control systems", "controls", "linear control", "nonlinear control", "feedback",
      "adaptive control", "robust control", "optimal control", "ecen 605", "ecen 606", "ecen 608", "ecen 609", "ecen 628"],
     "https://catalog.tamu.edu/graduate/colleges-schools-interdisciplinary/engineering/electrical-computer/#coursestext"),
    (["course", "courses", "undergraduate course", "ecen", "take", "curriculum", "elective", "prerequisite"],
     "https://catalog.tamu.edu/undergraduate/engineering/electrical-computer/#coursestext"),
    (["graduate course", "graduate courses", "phd course", "ms course"],
     "https://catalog.tamu.edu/graduate/colleges-schools-interdisciplinary/engineering/electrical-computer/#coursestext"),
    (["about", "overview", "what is tamu ece", "tell me about", "mission", "department", "facts", "figures", "enrollment"],
     "https://engineering.tamu.edu/electrical/about/index.html"),
    (["facts", "figures", "enrollment", "students", "faculty count", "ranking"],
     "https://engineering.tamu.edu/electrical/about/facts.html"),
    (["contact", "phone", "email", "address", "location", "office"],
     "https://engineering.tamu.edu/electrical/contact.html"),
    (["academics", "study", "courses", "curriculum", "advising", "student resources"],
     "https://engineering.tamu.edu/electrical/academics/index.html"),
    (["certificate", "online degree", "distance", "distance learning", "online program"],
     "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/index.html"),
    (["honors", "engineering honors"],
     "https://engineering.tamu.edu/academics/eh/departments/ecen-track/index.html"),
    (["study abroad", "international", "global"],
     "https://engineering.tamu.edu/electrical/academics/index.html"),
    (["advising", "advisor", "degree plan"],
     "https://engineering.tamu.edu/electrical/advising/index.html"),
    (["short course", "professional development", "continuing education"],
     "https://engineering.tamu.edu/electrical/academics/professional-development-short-courses.html"),
    (["patent", "patents", "startup", "startups", "invention", "intellectual property", "commercialization", "duplexer", "innovation"],
     "https://engineering.tamu.edu/electrical/research/patents-and-startups.html"),
    (["admissions", "apply", "application", "how to apply", "admission"],
     "https://engineering.tamu.edu/electrical/admissions-and-aid/index.html"),
    (["undergraduate admission", "undergrad apply", "freshman", "transfer"],
     "https://engineering.tamu.edu/electrical/admissions-and-aid/undergraduate-admissions/index.html"),
    (["graduate admission", "grad apply", "ms admission", "phd admission", "graduate application"],
     "https://engineering.tamu.edu/electrical/admissions-and-aid/graduate-admissions/index.html"),
    (["scholarship", "financial aid", "funding", "fellowship", "tuition", "assistantship"],
     "https://engineering.tamu.edu/electrical/admissions-and-aid/scholarships-aid/index.html"),
]


def _inject_known_pages(query: str, existing_ids: set, section_filter: Optional[str]) -> list[dict]:
    q = query.lower()
    injected = []
    seen_urls = set()
    for keywords, url in _URL_INJECTIONS:
        if any(kw in q for kw in keywords) and url not in seen_urls:
            siblings = _fetch_siblings(url, existing_ids)
            if not siblings:
                continue
            for s in siblings:
                if s["chunk_id"] not in existing_ids:
                    injected.append(s)
                    existing_ids.add(s["chunk_id"])
            seen_urls.add(url)
    return injected


# ---------------------------------------------------------------------------
# Structured fallback: "which professors research <area>" enumeration.
#
# Top-k semantic retrieval returns the *few best* chunks, so an enumeration
# query is silently capped at FINAL_TOP_K faculty — any research area with more
# members than that is truncated. For "people in research area X" questions we
# bypass ranking entirely and pull EVERY people-section chunk whose profile
# matches the area, guaranteeing a complete roster.
# ---------------------------------------------------------------------------

ENUM_TOP_K = 50  # safety cap; a research area rarely has more faculty than this

# Words that are query scaffolding, not the research topic itself.
_SCAFFOLD = {
    "which", "what", "who", "whom", "list", "all", "every", "name", "names",
    "the", "a", "an", "of", "is", "are", "do", "does", "did", "doing", "done",
    "can", "could", "please", "tell", "show", "give", "me", "us", "there",
    "any", "some", "at", "tamu", "ece", "department", "dept", "university",
    "work", "works", "working", "research", "researches", "researching",
    "study", "studies", "studying", "specialize", "specializes", "specializing",
    "focus", "focuses", "focusing", "interested", "expert", "experts",
    "in", "on", "into", "field", "fields", "area", "areas", "topic", "topics",
    "professor", "professors", "prof", "profs", "faculty", "faculties",
    "researcher", "researchers", "scientist", "scientists", "people", "that",
    "with", "and", "their", "his", "her", "doing", "involved", "focused",
    "instructor", "instructors", "member", "members", "staff",
    "should", "i", "if", "am", "my", "reach", "out", "contact", "talk",
    "speak", "connect", "touch", "get", "someone", "anyone", "suggest",
    "recommend", "advisor", "adviser", "mentor", "email", "whom",
    # Intersection / bridging words — structural scaffolding, not topic words.
    "intersection", "cross", "bridge", "bridges", "between", "combine",
    "combines", "combining", "combined", "overlap", "overlaps", "overlapping",
    "both", "across", "span", "spans", "spanning",
}

# Common abbreviations -> the phrase actually used in faculty profiles.
_TOPIC_ALIASES = {
    "ai": ["artificial", "intelligence"],
    "ml": ["machine", "learning"],
    "ev": ["electric", "vehicles"],
    "rf": ["radio", "frequency"],
}

_PEOPLE_PLURAL = {
    "professors", "faculty", "faculties", "researchers", "scientists", "profs",
    "instructors", "members",
}
_RESEARCH_VERBS = {
    "research", "researches", "researching", "work", "works", "working",
    "study", "studies", "studying", "specialize", "specializes",
    "specializing", "focus", "focuses", "focusing", "interested",
}


def _people_area_topic(query: str) -> Optional[list[str]]:
    """If `query` asks to enumerate faculty by research area, return the topic
    words; otherwise None.

    Triggers on a clear enumeration signal — a plural people noun (or a
    "who ... research" phrasing) PLUS a concrete research topic. The topic can be
    signalled by a research verb, an in/on area preposition, OR simply by the
    topic words appearing as a modifier ("AI/ML researchers", "machine learning
    faculty"). Single-person lookups ("tell me about professor X") and bare
    people questions with no topic ("who are the faculty", "list all professors")
    still return None and fall through to the normal / full-roster paths.
    """
    ql = query.lower()
    tokens = set(re.findall(r"[a-z]+", ql))

    is_people_plural = bool(tokens & _PEOPLE_PLURAL)
    who_research = bool(tokens & {"who", "whom"}) and bool(tokens & _RESEARCH_VERBS)
    # "whom should I reach out to / contact / talk to about <area>" — a contact
    # question about a research topic is a faculty-enumeration question.
    contact_intent = bool(re.search(
        r"\b(reach out|contact|talk to|speak (to|with)|connect with|get in touch|"
        r"work with|advisor|adviser|mentor)\b", ql))
    has_research_verb = bool(tokens & _RESEARCH_VERBS)
    has_area_prep = bool(re.search(r"\b(in|on|area of|field of|working on|focused on)\b", ql))

    # Extract the topic first: everything that isn't query scaffolding.
    topic = []
    for w in re.findall(r"[a-z0-9+&-]+", ql):
        if w in _SCAFFOLD:
            continue
        topic.extend(_TOPIC_ALIASES.get(w, [w]))

    # A people-plural / who-research / contact signal enumerates faculty when
    # there's a topic to enumerate by. Previously this required a research verb
    # or an in/on preposition, so "AI/ML researchers" and "machine learning
    # faculty" fell through to capped retrieval and returned a TRUNCATED list.
    # Allowing a bare topic (bool(topic)) makes completeness phrasing-independent;
    # topic-less queries ("who are the faculty") still get topic == [] → None.
    has_people_signal = is_people_plural or who_research or contact_intent
    enumeration = has_people_signal and (has_research_verb or has_area_prep or bool(topic))
    if not enumeration:
        return None

    return topic or None


# ── Signal-term matching for "faculty by research area" (graded core tier) ────
# The core-researcher tier USED to require a faculty profile to contain the
# umbrella area name VERBATIM (one ILIKE on the full joined phrase). Almost no
# profile writes "artificial intelligence machine learning" literally — people
# describe their work as "reinforcement learning", "deep learning", "computer
# vision", etc. — so the core tier collapsed to the lone person who happened to
# use the umbrella phrase, and everyone else was dumped into an undifferentiated
# "also working in the area" bucket.
#
# Instead we match a set of SIGNAL TERMS (the umbrella phrase PLUS its concrete
# sub-fields) and SCORE each profile by how strongly it matches: umbrella terms
# weigh most, specific sub-fields a bit less, generic words least. A profile
# must clear AREA_MIN_SIGNAL_SCORE to count at all (precision floor — keeps out
# profiles that merely mention a term once in an unrelated sentence), and the
# score then drives the core-vs-adjacent grading in build_area_roster.
_AREA_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "artificial intelligence and machine learning": [
        ("artificial intelligence", 3.0), ("machine learning", 3.0),
        ("deep learning", 2.0), ("reinforcement learning", 2.0),
        ("neural network", 2.0), ("computer vision", 2.0),
        ("natural language processing", 2.0), ("statistical learning", 2.0),
        ("graphical model", 2.0), ("generative model", 2.0),
        ("representation learning", 2.0), ("learning algorithm", 2.0),
        ("bayesian", 1.0), ("data-driven", 1.0),
    ],
    "information science and learning systems": [
        ("information theory", 3.0), ("learning systems", 3.0),
        ("statistical learning", 2.0), ("coding theory", 2.0),
        ("estimation", 1.0), ("inference", 1.0), ("signal processing", 1.0),
    ],
    "security": [
        ("cybersecurity", 3.0), ("cyber security", 3.0),
        ("hardware security", 2.0), ("network security", 2.0),
        ("cryptography", 2.0), ("side-channel", 2.0), ("malware", 2.0),
        ("trustworthy", 1.0), ("privacy", 1.0), ("security", 2.0),
    ],
    "energy and power": [
        ("power systems", 3.0), ("power electronics", 3.0),
        ("smart grid", 2.0), ("power grid", 2.0), ("electric machine", 2.0),
        ("motor drive", 2.0), ("renewable energy", 2.0),
        ("electric vehicle", 2.0), ("energy conversion", 2.0),
        ("power converter", 2.0), ("energy", 1.0),
    ],
    "communications and networks": [
        ("wireless communication", 3.0), ("communication networks", 3.0),
        ("information theory", 2.0), ("coding theory", 2.0),
        ("networking", 2.0), ("wireless", 2.0), ("network protocol", 2.0),
        ("mimo", 2.0), ("5g", 2.0), ("signal processing", 1.0),
    ],
    "computer engineering and systems": [
        ("computer architecture", 3.0), ("embedded systems", 3.0),
        ("vlsi", 2.0), ("fpga", 2.0), ("real-time systems", 2.0),
        ("cyber-physical", 2.0), ("control systems", 2.0),
        ("hardware design", 2.0), ("operating systems", 2.0),
    ],
    "device science and nanotechnology": [
        ("nanotechnology", 3.0), ("semiconductor device", 3.0),
        ("nanoelectronics", 2.0), ("nanomaterials", 2.0),
        ("quantum device", 2.0), ("photonics", 2.0), ("transistor", 2.0),
        ("nanoscale", 2.0), ("2d materials", 2.0),
    ],
    "electromagnetics and microwaves": [
        ("electromagnetics", 3.0), ("microwave", 3.0), ("antenna", 2.0),
        ("radar", 2.0), ("wave propagation", 2.0), ("metamaterial", 2.0),
        ("terahertz", 2.0), ("rf", 1.0), ("photonics", 1.0),
    ],
    "analog and mixed signals": [
        ("analog circuit", 3.0), ("mixed-signal", 3.0),
        ("data converter", 2.0), ("integrated circuit", 2.0),
        ("rf circuit", 2.0), ("amplifier", 2.0),
        ("phase-locked loop", 2.0), ("low-power", 1.0),
    ],
    "biomedical imaging, sensing and genomic signal processing": [
        ("biomedical imaging", 3.0), ("genomic signal processing", 3.0),
        ("medical imaging", 2.0), ("bioinformatics", 2.0),
        ("computational biology", 2.0), ("biosensor", 2.0),
        ("genomics", 2.0), ("neural engineering", 2.0), ("mri", 2.0),
    ],
    "chip manufacturing": [
        ("semiconductor manufacturing", 3.0), ("chip manufacturing", 3.0),
        ("fabrication", 2.0), ("lithography", 2.0), ("vlsi", 2.0),
        ("process technology", 2.0), ("packaging", 2.0), ("yield", 1.0),
    ],
}

# Precision floor: minimum signal score for a profile to count as researching
# the topic at all. 2.0 = one specific sub-field term (or two generic words);
# a single passing mention of a generic word (weight 1) is not enough.
MIN_SIGNAL_SCORE = float(os.getenv("AREA_MIN_SIGNAL_SCORE", "2.0"))


def _signal_terms(topic_words: list[str]) -> list[tuple[str, float]]:
    """Map a query topic to (term, weight) signals to score profiles against.

    Uses the curated _AREA_SIGNALS table for the department's known research
    areas; for anything else, falls back to the full phrase (strong) plus each
    significant word (weak) so the behavior generalizes without a hand-built
    entry for every possible topic.
    """
    joined = " ".join(topic_words).lower().strip()
    if not joined:
        return []
    toks = set(joined.split())
    # 1) The topic names (or closely overlaps) a research-area name.
    for key, sigs in _AREA_SIGNALS.items():
        kt = set(key.split())
        if (joined in key or key in joined
                or len(toks & kt) >= 2
                or (len(toks) == 1 and toks <= kt)):
            return sigs
    # 2) The topic names a specific SUB-FIELD of an area (e.g. "power systems",
    #    "embedded systems", "antenna") — route to that area's signal set.
    for key, sigs in _AREA_SIGNALS.items():
        for term, _ in sigs:
            if (term == joined
                    or (len(joined) > 4 and joined in term)
                    or (len(term) > 4 and term in joined)):
                return sigs
    # 3) Generic fallback: the full phrase (strong) + each significant word.
    sigs: list[tuple[str, float]] = [(joined, 3.0)]
    for w in topic_words:
        wl = w.lower()
        if len(wl) > 3 and wl != joined:
            sigs.append((wl, 1.0))
    return sigs


def _people_by_area(topic_words: list[str]) -> list[dict]:
    """People-section chunks researching `topic_words`, scored by signal strength.

    Returns chunks that match at least one signal term and clear the precision
    floor (MIN_SIGNAL_SCORE). Each chunk carries `signal_score` (sum of matched
    term weights) and `signal_hits` (the matched terms) so build_area_roster can
    grade faculty into a core tier vs. an also-active tier instead of the old
    binary "exact-phrase or relegated" split.
    """
    signals = _signal_terms(topic_words)
    if not signals:
        return []
    where = " OR ".join(["text ILIKE %s"] * len(signals))
    params = ["%" + term + "%" for term, _ in signals] + [ENUM_TOP_K * 2]
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT chunk_id, url, title, section, text, 0.0
            FROM ecen_docs
            WHERE section = 'people'
              AND ({where})
            LIMIT %s;
        """, params)
        rows = [_row_to_chunk(r) for r in cur.fetchall()]

    scored: list[dict] = []
    for c in rows:
        text = (c.get("text") or "").lower()
        hits = [(term, w) for term, w in signals if term in text]
        if not hits:
            continue
        score = sum(w for _, w in hits)
        if score < MIN_SIGNAL_SCORE:
            continue
        c["signal_score"] = score
        # term -> weight, so build_area_roster can grade a person by the SUM of
        # their distinct sub-fields across ALL their profile chunks (a profile
        # split across chunks would otherwise be graded by its best single chunk).
        c["signal_hits"] = {term: w for term, w in hits}
        scored.append(c)
    scored.sort(key=lambda c: c["signal_score"], reverse=True)
    return scored[:ENUM_TOP_K]


async def retrieve_async(query: str, section_filter: Optional[str] = None) -> list[dict]:
    import asyncio
    # Query rewrite removed: it made a full OpenAI round-trip before retrieval
    # started, adding 0.5–1.5 s of latency on every request. The dense embedder
    # + cross-encoder reranker handle query–passage matching well without it.
    return await asyncio.to_thread(retrieve, query, section_filter, _original_query=query)


def retrieve(query: str, section_filter: Optional[str] = None, _original_query: Optional[str] = None) -> list[dict]:
    """
    Full hybrid retrieval pipeline.
    Returns top chunks most relevant to `query`.
    """
    # Structured fallback for "which professors research <area>" enumeration.
    # Detect on the user's ORIGINAL wording (the rewrite mangles structure), and
    # only when not filtered away from people. Returns the COMPLETE roster,
    # ordered by relevance but never truncated to FINAL_TOP_K.
    intent_query = _original_query or query
    if section_filter in (None, "people"):
        area_words = _people_area_topic(intent_query)
        if area_words:
            people = _people_by_area(area_words)
            if people:
                reranker = _get_reranker()
                scores = reranker.predict([(intent_query, c["text"]) for c in people])
                for c, s in zip(people, scores):
                    c["rerank_score"] = float(s)
                people.sort(key=lambda c: c["rerank_score"], reverse=True)
                log.info(
                    "people-area enumeration: topic=%r matched=%d faculty",
                    " ".join(area_words), len(people),
                )
                return people

    import time as _time
    _t: dict = {}
    _m0 = _time.perf_counter()
    query_vec = _embed_query(query)
    _t["embed"] = _time.perf_counter() - _m0

    # Run the three independent retrieval arms in parallel — each gets its own
    # pooled DB connection so there are no concurrency conflicts.
    kw_query = _original_query or query
    # Fuzzy (trigram typo-tolerance) is the least-essential arm and the most
    # likely to scan. Toggle it off (ENABLE_FUZZY=0) to isolate / cut db_arms.
    _enable_fuzzy = os.getenv("ENABLE_FUZZY", "1") != "0"
    _m1 = _time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        dense_fut  = ex.submit(_dense_search,   query_vec, section_filter)
        kw_fut     = ex.submit(_keyword_search,  kw_query,  section_filter)
        fuzzy_fut  = ex.submit(_fuzzy_search, kw_query, section_filter) if _enable_fuzzy else None
        dense_results   = dense_fut.result()
        keyword_results = kw_fut.result()
        fuzzy_results   = fuzzy_fut.result() if fuzzy_fut else []
    _t["db_arms"] = _time.perf_counter() - _m1

    # Rewrite keyword arm removed (rewrite itself was removed); no second arm needed.
    rewrite_keyword_results: list[dict] = []

    # Merge dense + all lexical arms, de-duplicating by chunk_id.
    candidates = list(dense_results)
    seen = {c["chunk_id"] for c in candidates}
    for arm in (keyword_results, rewrite_keyword_results, fuzzy_results):
        for c in arm:
            if c["chunk_id"] not in seen:
                candidates.append(c)
                seen.add(c["chunk_id"])

    if not candidates:
        return []

    # Diagnostic: did Reddy / any phrase- or fuzzy-matched chunk enter the pool?
    log.info(
        "retrieve(): query=%r | dense=%d keyword=%d rewrite_kw=%d fuzzy=%d candidates=%d",
        query, len(dense_results), len(keyword_results),
        len(rewrite_keyword_results), len(fuzzy_results), len(candidates),
    )
    log.info(
        "keyword candidates: %s",
        [f"{c['section']}:{c['title'][:30]}" for c in keyword_results[:10]],
    )
    log.info(
        "fuzzy candidates: %s",
        [f"{c['section']}:{c['title'][:30]}" for c in fuzzy_results[:10]],
    )

    _m2 = _time.perf_counter()
    bm25_results = _bm25_rerank(query, candidates)
    merged = _rrf_merge(dense_results, bm25_results)
    _t["bm25_rrf"] = _time.perf_counter() - _m2

    # RRF ranks using only the dense list, so keyword-only chunks (e.g. the
    # faculty profile that literally says "Department Head") can be trimmed
    # before the cross-encoder — the one component able to recognize them —
    # ever sees them. Add back any candidate RRF dropped so the cross-encoder
    # judges the FULL dense+keyword pool. It reorders by relevance anyway.
    merged_ids = {c["chunk_id"] for c in merged}
    for c in candidates:
        if c["chunk_id"] not in merged_ids:
            merged.append(c)
            merged_ids.add(c["chunk_id"])

    # Cross-encoders judge (query, passage) relevance best with natural
    # question phrasing, so score against the user's original wording rather
    # than the verbose keyword-style rewrite used for dense retrieval.
    rerank_query = _original_query or query
    # Widen the cut-off for list-style questions so complete enumerations
    # (all degrees, every research area, full course list) aren't truncated.
    effective_top_k = LIST_TOP_K if _is_list_query(intent_query) else FINAL_TOP_K
    _n_merged = len(merged)
    _m3 = _time.perf_counter()
    final = _cross_encode(rerank_query, merged, effective_top_k)
    _t["cross_encode"] = _time.perf_counter() - _m3

    log.info(
        "cross-encoder top %d: %s",
        len(final),
        [f"{c['section']}:{c['title'][:30]}={c.get('rerank_score'):.2f}" for c in final],
    )

    # Inject chunks from known pages that keyword-match the query
    _m4 = _time.perf_counter()
    existing_ids = {c["chunk_id"] for c in final}
    injected = _inject_known_pages(query, existing_ids, section_filter)
    _n_injected = len(injected)
    if injected:
        reranker = _get_reranker()
        inj_scores = reranker.predict([(rerank_query, s["text"]) for s in injected])
        for chunk, score in zip(injected, inj_scores):
            chunk["rerank_score"] = float(score)
        injected_sorted = sorted(zip(injected, inj_scores), key=lambda x: x[1], reverse=True)
        final = [c for c, _ in injected_sorted] + final
    _t["inject"] = _time.perf_counter() - _m4

    # Pull sibling chunks from the top result's page. For list/enumeration
    # queries we keep ALL of the page's chunks (completeness); for ordinary
    # factual queries a small cap bounds how many pairs the cross-encoder must
    # score — a large page otherwise adds a full extra rerank pass per query.
    _m5 = _time.perf_counter()
    _n_siblings = 0
    if final:
        top_url = final[0]["url"]
        existing_ids = {c["chunk_id"] for c in final}
        sibling_limit = None if _is_list_query(intent_query) else SIBLING_TOP_K
        siblings = _fetch_siblings(top_url, existing_ids, limit=sibling_limit)
        if siblings:
            _n_siblings = len(siblings)
            reranker = _get_reranker()
            sib_scores = reranker.predict([(rerank_query, s["text"]) for s in siblings])
            for sib, score in sorted(zip(siblings, sib_scores), key=lambda x: x[1], reverse=True):
                sib["rerank_score"] = float(score)
                final.append(sib)
    _t["siblings"] = _time.perf_counter() - _m5

    log.info(
        "TIMING retrieve breakdown: embed=%.2fs db_arms=%.2fs bm25_rrf=%.2fs "
        "cross_encode=%.2fs(%d pairs) inject=%.2fs(%d) siblings=%.2fs(%d)",
        _t.get("embed", 0), _t.get("db_arms", 0), _t.get("bm25_rrf", 0),
        _t.get("cross_encode", 0), _n_merged, _t.get("inject", 0), _n_injected,
        _t.get("siblings", 0), _n_siblings,
    )
    return final
