"""
main.py — FastAPI application for the TAMU ECE RAG chatbot.

Endpoints:
  POST /chat          — question → streamed answer
  POST /chat/sync     — question → full answer (non-streaming)
  GET  /health        — liveness check
  POST /admin/reindex — manually trigger re-index
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from generator import generate, generate_stream, route_question, generate as _generate_full
from retriever import retrieve_async, _people_area_topic, _people_by_area, _get_embedder, _get_reranker
from graph_retriever import (
    graph_query, build_area_roster, build_intersection_roster, is_full_faculty_query,
    build_full_faculty_roster, faculty_roster_sources, research_area_names,
    is_degree_list_query, build_degree_roster,
)
from scheduler import create_scheduler, run_reindex
import scheduler as _scheduler

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm ML models so the first request doesn't pay the load penalty.
    # SentenceTransformer + CrossEncoder each take 2-5s to load from disk;
    # doing it here means the first chat message is just as fast as the rest.
    import asyncio
    log.info("Pre-warming embedder and reranker…")
    await asyncio.gather(
        asyncio.to_thread(_get_embedder),
        asyncio.to_thread(_get_reranker),
    )
    log.info("Models ready.")

    # On Cloud Run the CPU is only allocated during requests, so the in-process
    # APScheduler won't fire reliably — set DISABLE_SCHEDULER=1 there and drive
    # /admin/reindex from Cloud Scheduler instead.
    if os.getenv("DISABLE_SCHEDULER"):
        log.info("Scheduler disabled (DISABLE_SCHEDULER set).")
        yield
        return
    scheduler = create_scheduler()
    scheduler.start()
    log.info("Scheduler started.")
    yield
    scheduler.shutdown()
    log.info("Scheduler stopped.")


def _client_ip(request: Request) -> str:
    """Real client IP. The Next.js proxy forwards X-Forwarded-For (Cloud Run
    sets it on the outer request); direct connections fall back to the peer."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Per-IP rate limits (tunable via env without redeploying the image).
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "10/minute")
REPORT_RATE_LIMIT = os.getenv("REPORT_RATE_LIMIT", "3/minute")
limiter = Limiter(key_func=_client_ip)

app = FastAPI(
    title="TAMU ECE Chatbot API",
    description="RAG-powered chatbot for the TAMU Electrical & Computer Engineering department.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class HistoryTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="Who are the AI/ML faculty?")
    section_filter: Optional[str] = Field(
        None,
        description="Restrict retrieval to a section: people | research | academics | admissions | news | events | about",
        example="research",
    )
    history: Optional[list[HistoryTurn]] = Field(
        None, max_length=10,
        description="Recent conversation turns (oldest first) so follow-up questions can be understood.",
    )


class Source(BaseModel):
    url: str
    title: str
    section: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class FeedbackRequest(BaseModel):
    rating: str = Field(..., pattern="^(up|down)$")
    question: str = Field(..., max_length=1000)
    answer: str = Field(..., max_length=8000)


class ReportRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=4000,
                            description="What went wrong, in the user's words.")
    context: Optional[str] = Field(
        None, max_length=8000,
        description="Optional context, e.g. the recent question and answer.")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "last_reindex": _scheduler.last_reindex}


# In-memory usage counters (reset on instance restart; durable analytics live
# in the AUDIT log lines in Cloud Logging).
from collections import Counter
_STATS: dict = {"questions": 0, "intents": Counter(), "flags": Counter(),
                "feedback_up": 0, "feedback_down": 0, "cache_hits": 0}


@app.post("/feedback")
@limiter.limit(REPORT_RATE_LIMIT)
async def feedback(request: Request, req: FeedbackRequest):
    """Thumbs up/down on an answer → audit log + counters."""
    import hashlib
    import json as _j
    if req.rating == "up":
        _STATS["feedback_up"] += 1
    else:
        _STATS["feedback_down"] += 1
    ip = _client_ip(request)
    log.info("FEEDBACK %s", _j.dumps({
        "rating": req.rating,
        "ip_hash": hashlib.sha256(ip.encode()).hexdigest()[:12],
        "question": req.question[:300],
        "answer_preview": req.answer[:300],
    }))
    return {"ok": True}


@app.get("/admin/stats")
async def stats():
    """Lightweight usage counters since this instance started."""
    return {
        "questions": _STATS["questions"],
        "intents": dict(_STATS["intents"]),
        "flags": dict(_STATS["flags"]),
        "feedback": {"up": _STATS["feedback_up"], "down": _STATS["feedback_down"]},
        "cache_hits": _STATS["cache_hits"],
    }


@app.post("/report-issue")
@limiter.limit(REPORT_RATE_LIMIT)
async def report_issue(request: Request, req: ReportRequest):
    """Create a GitHub issue from an in-app bug report so non-technical testers
    can report problems without a GitHub account. Labeled 'user-report' so the
    Codex triage workflow picks it up."""
    import github_issues

    if not github_issues.reporting_enabled():
        raise HTTPException(status_code=503,
                            detail="Issue reporting is not configured on the server.")

    first_line = req.description.strip().splitlines()[0][:80]
    title = f"[User report] {first_line}"
    body = (
        f"**Reported from the chat UI.**\n\n"
        f"### What went wrong\n{req.description.strip()}\n"
    )
    if req.context:
        body += f"\n### Context (recent conversation)\n{req.context.strip()}\n"
    body += "\n_Filed automatically by the app on a tester's behalf._"

    import asyncio

    try:
        # Run the blocking HTTP call off the event loop so it can't stall other
        # requests (e.g. an in-flight chat stream).
        issue = await asyncio.to_thread(
            github_issues.create_issue, title, body, ["user-report"]
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Failed to create GitHub issue")
        raise HTTPException(status_code=502, detail="Could not file the report.") from e

    return {"ok": True, "issue": issue["number"], "url": issue["html_url"]}


# URLs of synthetic, code-generated context chunks — excluded from cited sources.
_SYNTHETIC_URLS = {"knowledge-graph", "research-area-roster", "faculty-roster",
                   "degree-roster", "about:eira"}
# Roster paths supply their own already-curated, relevant source list — keep all
# of those; only the open-ended retrieval path needs trimming.
_ROSTER_URLS = {"research-area-roster", "faculty-roster", "degree-roster"}
# Max sources to cite on the normal retrieval path (we feed more chunks to the
# LLM for answer completeness, but only cite the few most relevant).
MAX_SOURCES = 6
# Cross-encoder relevance gate for CITED sources. ms-marco cross-encoder scores
# are logits (~ -11 weakly relevant … +11 highly relevant); chunks below this
# only padded the top-k and shouldn't be shown to users as "sources". The LLM
# context is NOT filtered — only what we display. Tune via env without deploy.
SOURCE_MIN_SCORE = float(os.getenv("SOURCE_MIN_SCORE", "0"))


def _select_sources(chunks: list[dict]) -> list[dict]:
    """De-duplicate by URL (keep the highest-scoring chunk per page), drop
    synthetic chunks, and — on the normal retrieval path — keep only the most
    relevant few above the cross-encoder relevance gate. Roster paths already
    supply a curated source set, so keep it whole."""
    best: dict[str, dict] = {}
    for c in chunks:
        url = c["url"]
        if url in _SYNTHETIC_URLS:
            continue
        score = c.get("rerank_score", 0.0) or 0.0
        if url not in best or score > (best[url].get("rerank_score", 0.0) or 0.0):
            best[url] = c
    ordered = sorted(best.values(),
                     key=lambda c: c.get("rerank_score", 0.0) or 0.0, reverse=True)
    if any(c["url"] in _ROSTER_URLS for c in chunks):
        return ordered
    # Only cite chunks that genuinely cleared the relevance gate. If nothing
    # did, cite NOTHING — showing the least-irrelevant junk as "sources" is
    # worse than an unsourced answer (the answer itself already says when it
    # doesn't have the details).
    relevant = [c for c in ordered
                if (c.get("rerank_score", 0.0) or 0.0) >= SOURCE_MIN_SCORE]
    return relevant[:MAX_SOURCES]


import re as _re

# "Who built this?" must mean the chatbot, not whatever page retrieval happens
# to surface (a Mars-construction news chunk once answered this). Catch
# creator-intent questions BEFORE retrieval and serve a creator context chunk.
_CREATOR_RE = _re.compile(
    r"\bwho\b.{0,40}?\b(built|created|made|developed|designed|coded|wrote|behind|author(?:ed)?)\b"
    r".{0,40}?\b(this|you|it|chatbot|chat\s*bot|bot|assistant|rag|tool)\b[\s?!.]*$",
    _re.IGNORECASE,
)

_CREATOR_CHUNK = {
    "url": "https://www.linkedin.com/in/aarohi-mohrir/",
    "title": "Aarohi Mohrir — let's connect on LinkedIn",
    "section": "about",
    "rerank_score": 10.0,
    "text": (
        "This assistant is a Retrieval-Augmented Generation (RAG) system over the "
        "TAMU ECE website, designed and built end to end by Aarohi Mohrir, a "
        "Master's student in Computer Science, under the guidance of Prof. Krishna "
        "Narayanan — the crawler, the hybrid retrieval pipeline, the knowledge "
        "graph, and the cloud deployment are all her work. "
        "Connect with her on LinkedIn: https://www.linkedin.com/in/aarohi-mohrir/"
    ),
}


def _is_creator_question(question: str) -> bool:
    return bool(_CREATOR_RE.search(question.strip()))


_CHITCHAT_RE = _re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings?|sup|yo|"
    r"thanks?|thank\s*you|ty|cheers|"
    r"bye|goodbye|see\s*you|"
    r"what\s+can\s+you\s+do|what\s+are\s+you|who\s+are\s+you|"
    r"help(\s+me)?|can\s+you\s+help)\s*[!?.]*\s*$",
    _re.IGNORECASE,
)


def _fast_route(req: "ChatRequest") -> dict:
    """Zero-latency heuristic router — no LLM call.

    Covers all intents the LLM router handled, using the deterministic
    helper functions that already exist in the codebase. For multi-turn
    follow-ups the standalone question is NOT rewritten (we lose pronoun
    resolution), but for fresh queries — the common case — this is identical
    in quality and saves 1–2 s on every single request.
    """
    q = req.question

    if _is_creator_question(q):
        return {"intent": "creator", "standalone_question": q, "topic": None,
                "suspicious": False}

    if _CHITCHAT_RE.match(q):
        return {"intent": "chitchat", "standalone_question": q, "topic": None,
                "suspicious": False}

    if is_full_faculty_query(q):
        return {"intent": "list_all_faculty", "standalone_question": q, "topic": None,
                "suspicious": False}

    topic_words = _people_area_topic(q)
    if topic_words:
        return {"intent": "people_by_area", "standalone_question": q,
                "topic": " ".join(topic_words), "suspicious": False}

    return {"intent": "general", "standalone_question": q, "topic": None,
            "suspicious": False}


# ── Security layer ───────────────────────────────────────────────────────────
# Fast regex screen for blatant injection phrasing (cheap first line; the LLM
# router's `suspicious` flag catches subtler attempts).
_INJECTION_RE = _re.compile(
    r"(ignore\s+(all\s+|the\s+)?(previous|prior|above)\s+(instructions?|prompts?)|"
    r"disregard\s+(your|the|all)\s+(instructions?|rules|system\s*prompt)|"
    r"(reveal|show|print|repeat|output)\s+(me\s+)?(your\s+)?(system|hidden|initial)\s*(prompt|instructions?)|"
    r"developer\s+(mode|instructions?)|jailbreak|do\s+anything\s+now|\bDAN\s+mode\b)",
    _re.IGNORECASE,
)

_REFUSAL_TEXT = ("I can't help with that — but I'm happy to answer questions about "
                 "TAMU ECE programs, courses, research, faculty, admissions, or events!")
_NO_INFO_TEXT = ("I couldn't find anything reliable on the department website about that. "
                 "Could you rephrase, or ask me about programs, courses, research areas, "
                 "faculty, admissions, or events?")

# Secrets that must never appear in output (defense in depth; the corpus is
# public web pages, but the model sees env-derived strings in error paths).
_SECRET_RE = _re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|xox[bporas]-[A-Za-z0-9-]{10,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}|"
    r"postgres(?:ql)?://[^\s'\"]+:[^\s'\"]+@)"
)


def _redact(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


# Minimum cross-encoder score for a chunk to be USED as LLM context (lower bar
# than the citation gate — recall matters more for answering than for citing).
CONTEXT_MIN_SCORE = float(os.getenv("CONTEXT_MIN_SCORE", "-4"))


def _gate_context(chunks: list[dict]) -> list[dict]:
    """Drop low-confidence retrievals from the LLM context. Synthetic chunks
    (rosters, creator, identity — rerank 10) and citation-only chunks always
    pass. Returns [] when nothing is reliable."""
    return [c for c in chunks
            if (c.get("rerank_score", 0.0) or 0.0) >= CONTEXT_MIN_SCORE]


def _canned_stream(text: str, sources: Optional[list] = None):
    """SSE response built from a ready answer (refusals, no-info, cache hits)."""
    import json as _j

    async def gen():
        yield f"event: sources\ndata: {_j.dumps(sources or [])}\n\n"
        yield f"data: {text.replace(chr(10), chr(92) + 'n')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Answer cache (in-memory, per instance) ───────────────────────────────────
# Only first questions (no history) are cacheable — follow-ups depend on the
# conversation. Durable across requests, reset on instance restart.
import time as _time

_ANSWER_CACHE: dict[str, tuple[float, str, list]] = {}
_CACHE_TTL = float(os.getenv("ANSWER_CACHE_TTL", "3600"))
_CACHE_MAX = 200


def _cache_key(search_req: "ChatRequest") -> str:
    return f"{search_req.question.strip().lower()}|{search_req.section_filter or ''}"


def _cache_get(key: str) -> Optional[tuple[str, list]]:
    entry = _ANSWER_CACHE.get(key)
    if not entry:
        return None
    ts, answer, sources = entry
    if _time.time() - ts > _CACHE_TTL:
        _ANSWER_CACHE.pop(key, None)
        return None
    return answer, sources


def _cache_put(key: str, answer: str, sources: list) -> None:
    if len(_ANSWER_CACHE) >= _CACHE_MAX:
        oldest = min(_ANSWER_CACHE, key=lambda k: _ANSWER_CACHE[k][0])
        _ANSWER_CACHE.pop(oldest, None)
    _ANSWER_CACHE[key] = (_time.time(), answer, sources)


def _audit(request: Request, question: str, resolved: str, route: Optional[dict],
           sources: list[dict], answer: str, flagged: str = "") -> None:
    """Structured audit record → Cloud Logging. IP is hashed (privacy), answer
    truncated. One line per request, greppable via 'AUDIT'."""
    import hashlib
    import json as _j
    _STATS["questions"] += 1
    _STATS["intents"][(route or {}).get("intent") or "fallback"] += 1
    if flagged:
        _STATS["flags"][flagged] += 1
    ip = _client_ip(request)
    log.info("AUDIT %s", _j.dumps({
        "ip_hash": hashlib.sha256(ip.encode()).hexdigest()[:12],
        "question": question[:300],
        "resolved": resolved[:300] if resolved != question else None,
        "intent": (route or {}).get("intent"),
        "flagged": flagged or None,
        "sources": [s.get("url", "") for s in sources][:10],
        "answer_chars": len(answer),
        "answer_preview": answer[:300],
    }))


# Context for small talk ("who are you", "hi", "what can you do") — answered
# from the persona alone, with NO cited sources (url is in _SYNTHETIC_URLS).
_IDENTITY_CHUNK = {
    "url": "about:eira",
    "title": "About EIRA",
    "section": "about",
    "rerank_score": 10.0,
    "text": (
        "EIRA (ECE Information & Resource Assistant) is the AI guide for the "
        "Texas A&M Department of Electrical and Computer Engineering. EIRA can "
        "answer questions about degree programs (BS/MS/PhD, online options, "
        "certificates), admissions and scholarships, courses, all research "
        "areas, faculty and staff (including who to contact for a given "
        "research interest), news, and events — all drawn from the "
        "department's official website."
    ),
}


async def _prepare_chunks(req: "ChatRequest", route: Optional[dict] = None,
                          prefetched: Optional[list] = None) -> list[dict]:
    """Assemble the context chunks for a question.

    Dispatch order: the LLM route (intent + topic) when available, falling back
    to the legacy keyword heuristics when it isn't. The executors themselves
    (graph rosters, exact-phrase people match, hybrid retrieval) are
    deterministic either way — only WHO decides the intent differs.
    """
    intent = (route or {}).get("intent")

    # "Who built this chatbot?" → creator context, no retrieval (which would
    # otherwise hijack "this" with an arbitrary page about building something).
    if intent == "creator" or (route is None and _is_creator_question(req.question)):
        log.info("Creator question detected: %r", req.question)
        return [_CREATOR_CHUNK]

    # Small talk needs no retrieval and must not cite random pages as sources.
    if intent == "chitchat":
        log.info("Chitchat detected: %r", req.question)
        return [_IDENTITY_CHUNK]

    # "Whom should I reach out to about <topic>?" — any phrasing, via router.
    if intent == "people_by_area" and (route or {}).get("topic"):
        # Two-area "works on BOTH X and Y" → intersect the two area rosters
        # instead of collapsing to a single area and dropping the 2nd constraint.
        inter = build_intersection_roster(req.question)
        if inter:
            log.info("Routed area INTERSECTION for %r", req.question)
            return [{"url": "research-area-roster",
                     "title": "Faculty across two research areas",
                     "section": "graph", "text": inter, "rerank_score": 10.0}]
        topic = [w for w in (route["topic"] or "").split() if w]
        precise = _people_by_area(topic)
        roster = build_area_roster(topic, precise)
        if roster:
            roster_chunk = {
                "url": "research-area-roster",
                "title": f"Faculty researching {' '.join(topic)}",
                "section": "graph", "text": roster, "rerank_score": 10.0,
            }
            log.info("Routed area roster: topic=%r, %d precise profiles",
                     " ".join(topic), len(precise))
            seen: set[str] = set()
            deduped = [c for c in precise
                       if c["url"] not in seen and not seen.add(c["url"])]
            return [roster_chunk] + deduped
        # No roster for the routed topic → fall through to normal retrieval.

    # Global "list all faculty" → serve the COMPLETE roster from the graph.
    # Retrieval can't enumerate the whole department (top-k cap + chunk splits),
    # so build it deterministically and keep the matching profile chunks as cites.
    if intent == "list_all_faculty" or (route is None and is_full_faculty_query(req.question)):
        roster = build_full_faculty_roster()
        if roster:
            roster_chunk = {
                "url": "faculty-roster",
                "title": "Complete TAMU ECE Faculty Roster",
                "section": "graph", "text": roster, "rerank_score": 10.0,
            }
            # Cite only the pages the roster is actually drawn from (the per-area
            # research pages) — not the whole retrieval set. text="" keeps them
            # out of the LLM context; they exist purely as citations.
            cite_chunks = [
                {**s, "text": "", "rerank_score": 0.0}
                for s in faculty_roster_sources()
            ]
            log.info("Full faculty roster injected (%d chars), %d sources",
                     len(roster), len(cite_chunks))
            return [roster_chunk] + cite_chunks

    # Legacy keyword heuristic for people-by-area — kept as the no-router
    # fallback (and as a safety net when the router says "general" but the
    # phrasing matches the classic enumeration pattern).
    topic = _people_area_topic(req.question)
    if topic:
        inter = build_intersection_roster(req.question)
        if inter:
            log.info("Legacy area INTERSECTION for %r", req.question)
            return [{"url": "research-area-roster",
                     "title": "Faculty across two research areas",
                     "section": "graph", "text": inter, "rerank_score": 10.0}]
        precise = _people_by_area(topic)
        roster = build_area_roster(topic, precise)
        if roster:
            roster_chunk = {
                "url": "research-area-roster",
                "title": f"Faculty researching {' '.join(topic)}",
                "section": "graph", "text": roster, "rerank_score": 10.0,
            }
            log.info("Layered roster built: topic=%r, %d precise profiles",
                     " ".join(topic), len(precise))
            # De-duplicate profiles by URL so a multi-chunk faculty page is
            # cited once (the roster text already lists each person once).
            seen: set[str] = set()
            deduped = [c for c in precise
                       if c["url"] not in seen and not seen.add(c["url"])]
            return [roster_chunk] + deduped

    # Authoritative complete degree-program list from the graph, so the answer
    # never drops newer programs (Microelectronics MS, certificates, minor) that
    # a stale crawled degrees page omits.
    if is_degree_list_query(req.question):
        roster = build_degree_roster()
        if roster:
            log.info("Degree roster injected for %r", req.question)
            return [
                {"url": "degree-roster", "title": "TAMU ECE Degree Programs",
                 "section": "graph", "text": roster, "rerank_score": 10.0},
                {"url": "https://engineering.tamu.edu/electrical/academics/degrees/index.html",
                 "title": "Degree Programs | Texas A&M University Engineering",
                 "section": "academics", "text": "", "rerank_score": 0.0},
            ]

    chunks = (prefetched if prefetched is not None
              else await retrieve_async(req.question, section_filter=req.section_filter))
    if not chunks:
        return []
    graph_ctx = graph_query(req.question)
    if graph_ctx:
        log.info("Graph context injected (%d chars)", len(graph_ctx))
        chunks = [{"url": "knowledge-graph", "title": "Knowledge Graph", "section": "graph",
                   "text": graph_ctx, "rerank_score": 10.0}] + chunks
    return chunks


def _history_dicts(req: "ChatRequest") -> list[dict]:
    return [{"role": t.role, "content": t.content} for t in (req.history or [])]


async def _route(req: "ChatRequest") -> tuple["ChatRequest", Optional[dict], Optional[list]]:
    """Classify intent with zero-latency heuristics and prefetch retrieval.

    Returns (search_req, route, prefetched_chunks).

    The LLM router has been replaced by _fast_route() — a pure in-process
    function with no network calls. Retrieval runs immediately after, so the
    only latency on the critical path is the DB round-trip (~0.3–0.5 s).
    """
    route = _fast_route(req)
    # For chitchat/creator we don't need retrieval at all — skip it.
    if route["intent"] in ("chitchat", "creator"):
        return req, route, None

    prefetched = await retrieve_async(req.question, req.section_filter)
    return req, route, prefetched


@app.post("/chat", summary="Streaming chat (Server-Sent Events)")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_stream(request: Request, req: ChatRequest):
    """Returns a streaming text/event-stream response."""
    # Input sanitization: blatant injection phrasing → polite canned refusal.
    if _INJECTION_RE.search(req.question):
        _audit(request, req.question, req.question, None, [], _REFUSAL_TEXT,
               flagged="injection_regex")
        return _canned_stream(_REFUSAL_TEXT)

    import time as _t
    _t_req = _t.perf_counter()
    history = _history_dicts(req)
    search_req, route, prefetched = await _route(req)
    log.info("TIMING route+retrieval: %.2fs (intent=%s) for %r",
             _t.perf_counter() - _t_req, (route or {}).get("intent"), req.question)

    # LLM-detected injection / jailbreak / exfiltration attempt.
    if route and route.get("suspicious"):
        _audit(request, req.question, search_req.question, route, [],
               _REFUSAL_TEXT, flagged="injection_llm")
        return _canned_stream(_REFUSAL_TEXT)

    # Cache: identical first questions replay the stored answer (no LLM call).
    ckey = _cache_key(search_req) if not history else None
    if ckey:
        hit = _cache_get(ckey)
        if hit:
            _STATS["cache_hits"] += 1
            answer, cached_sources = hit
            _audit(request, req.question, search_req.question, route,
                   cached_sources, answer, flagged="cache_hit")
            return _canned_stream(answer, cached_sources)

    chunks = _gate_context(await _prepare_chunks(search_req, route, prefetched=prefetched))
    log.info("TIMING through prepare_chunks: %.2fs (%d chunks)",
             _t.perf_counter() - _t_req, len(chunks))
    if not chunks:
        _audit(request, req.question, search_req.question, route, [],
               _NO_INFO_TEXT, flagged="no_reliable_context")
        return _canned_stream(_NO_INFO_TEXT)

    # Give the generator the resolved interpretation too, so the answer targets
    # what the follow-up actually meant — not a generic reading of it.
    gen_question = req.question if search_req.question == req.question else (
        f"{req.question}\n(In the context of this conversation, this means: "
        f"{search_req.question})")

    log.info("Retrieved %d chunks for query %r:", len(chunks), req.question)
    for c in chunks:
        log.info("  [%.3f] %s", c.get("rerank_score", 0), c.get("chunk_id", c["url"]))

    sources = [{"url": c["url"], "title": c["title"], "section": c["section"]}
               for c in _select_sources(chunks)]

    async def event_stream():
        import json
        # Emit sources first
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        # Stream LLM tokens with output redaction: hold back a small tail so
        # secret-shaped strings can't slip through a delta boundary, scrub each
        # released span, and audit the full answer at the end.
        emitted = 0
        answer_acc = ""
        pending = ""
        _HOLDBACK = 80

        def _sse(text: str) -> str:
            return f"data: {text.replace(chr(10), chr(92) + 'n')}\n\n"

        _llm_t0 = _t.perf_counter()
        _ttft_logged = False
        try:
            async for delta in generate_stream(gen_question, chunks, history=history):
                if not delta:
                    continue
                if not _ttft_logged:
                    log.info("TIMING LLM first token: %.2fs after retrieval done "
                             "(%.2fs total since request)",
                             _t.perf_counter() - _llm_t0, _t.perf_counter() - _t_req)
                    _ttft_logged = True
                pending += delta
                if len(pending) > 2 * _HOLDBACK:
                    release, pending = pending[:-_HOLDBACK], pending[-_HOLDBACK:]
                    release = _redact(release)
                    emitted += len(release)
                    answer_acc += release
                    yield _sse(release)
            tail = _redact(pending)
            pending = ""
            if tail:
                emitted += len(tail)
                answer_acc += tail
                yield _sse(tail)
        except Exception as e:
            log.warning("Streaming failed (%s); falling back to non-streaming", e)

        # If streaming produced nothing, fall back to a buffered generation
        # (and retry with a single chunk) so the user never sees an empty reply.
        if emitted == 0:
            answer = await _generate_full(gen_question, chunks, history=history)
            if not answer and len(chunks) > 1:
                log.warning("Empty answer with %d chunks, retrying with 1 chunk", len(chunks))
                answer = await _generate_full(req.question, chunks[:1])
            if not answer:
                answer = ("I don't have enough details to answer that — please "
                          "check the sources below or visit engineering.tamu.edu/electrical.")
            answer = _redact(answer)
            answer_acc += answer
            yield f"data: {answer.replace(chr(10), chr(92) + 'n')}\n\n"

        yield "data: [DONE]\n\n"
        log.info("TIMING LLM generation: %.2fs | total request: %.2fs (%d chars)",
                 _t.perf_counter() - _llm_t0, _t.perf_counter() - _t_req, len(answer_acc))
        if ckey and emitted > 0 and answer_acc:
            _cache_put(ckey, answer_acc, sources)
        _audit(request, req.question, search_req.question, route, sources, answer_acc)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/sync", response_model=ChatResponse, summary="Synchronous chat")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_sync(request: Request, req: ChatRequest):
    """Returns a complete JSON response (no streaming)."""
    if _INJECTION_RE.search(req.question):
        _audit(request, req.question, req.question, None, [], _REFUSAL_TEXT,
               flagged="injection_regex")
        return ChatResponse(answer=_REFUSAL_TEXT, sources=[])

    history = _history_dicts(req)
    search_req, route, prefetched = await _route(req)
    if route and route.get("suspicious"):
        _audit(request, req.question, search_req.question, route, [],
               _REFUSAL_TEXT, flagged="injection_llm")
        return ChatResponse(answer=_REFUSAL_TEXT, sources=[])

    chunks = _gate_context(await _prepare_chunks(search_req, route, prefetched=prefetched))
    if not chunks:
        _audit(request, req.question, search_req.question, route, [],
               _NO_INFO_TEXT, flagged="no_reliable_context")
        return ChatResponse(answer=_NO_INFO_TEXT, sources=[])

    gen_question = req.question if search_req.question == req.question else (
        f"{req.question}\n(In the context of this conversation, this means: "
        f"{search_req.question})")
    answer = _redact(await generate(gen_question, chunks, history=history))
    sources = [Source(url=c["url"], title=c["title"], section=c["section"])
               for c in _select_sources(chunks)]
    _audit(request, req.question, search_req.question, route,
           [s.model_dump() for s in sources], answer)
    return ChatResponse(answer=answer, sources=sources)


@app.get("/admin/test-llm", summary="Test LLM with a minimal prompt")
async def test_llm():
    """Sends a trivial prompt to the LLM and returns the answer plus timings.

    This isolates the model: no retrieval, no reranking, a one-line prompt. If
    `seconds_total` here is already large, the latency is the model/endpoint
    itself (e.g. a reasoning model, a cold serverless backend, or network), and
    no amount of retrieval tuning will help — the fix is the model choice.
    """
    import time, httpx, json as _json
    from generator import TAMU_API_URL, TAMU_API_KEY, TAMU_MODEL
    payload = {
        "model": TAMU_MODEL,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "stream": True,
    }
    parts = []
    raw_lines = []
    t0 = time.perf_counter()
    ttft = None  # time to first content token
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{TAMU_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {TAMU_API_KEY}", "Content-Type": "application/json"},
                json=payload) as resp:
            async for line in resp.aiter_lines():
                raw_lines.append(line)
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            obj = _json.loads(data)
                            text = obj["choices"][0]["delta"].get("content", "")
                            if text:
                                if ttft is None:
                                    ttft = round(time.perf_counter() - t0, 2)
                                parts.append(text)
                        except Exception:
                            pass
    return {
        "answer": "".join(parts),
        "model": TAMU_MODEL,
        "seconds_to_first_token": ttft,
        "seconds_total": round(time.perf_counter() - t0, 2),
        "raw_lines": raw_lines[:20],
    }


@app.post("/admin/reindex", summary="Manually trigger a site re-index")
async def manual_reindex():
    """Kicks off a diff re-index in the background."""
    import asyncio
    asyncio.create_task(run_reindex())
    return {"status": "re-index started"}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Reload is OFF by default: with auto-reload the watcher re-triggers on the
    # .pyc files Python writes and keeps reloading the heavy ML models, so the
    # server never stays up. Set RELOAD=1 to opt in during light dev, and we
    # exclude __pycache__ so it doesn't loop.
    reload = os.getenv("RELOAD") == "1"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=reload,
        reload_excludes=["*.pyc", "__pycache__/*", "*.bak"] if reload else None,
    )
