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

# Initialize Sentry as early as possible (no-op unless SENTRY_DSN is set), so the
# FastAPI integration can capture unhandled exceptions in request handlers.
from sentry_init import init_sentry
init_sentry()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from generator import generate, generate_stream, generate as _generate_full
from retriever import retrieve_async, _people_area_topic, _people_by_area
from graph_retriever import (
    graph_query, build_area_roster, is_full_faculty_query, build_full_faculty_roster,
    faculty_roster_sources,
)
from scheduler import create_scheduler, run_reindex
import scheduler as _scheduler

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(
    title="TAMU ECE Chatbot API",
    description="RAG-powered chatbot for the TAMU Electrical & Computer Engineering department.",
    version="1.0.0",
    lifespan=lifespan,
)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="Who are the AI/ML faculty?")
    section_filter: Optional[str] = Field(
        None,
        description="Restrict retrieval to a section: people | research | academics | admissions | news | events | about",
        example="research",
    )


class Source(BaseModel):
    url: str
    title: str
    section: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "last_reindex": _scheduler.last_reindex}


# URLs of synthetic, code-generated context chunks — excluded from cited sources.
_SYNTHETIC_URLS = {"knowledge-graph", "research-area-roster", "faculty-roster"}
# Roster paths supply their own already-curated, relevant source list — keep all
# of those; only the open-ended retrieval path needs trimming.
_ROSTER_URLS = {"research-area-roster", "faculty-roster"}
# Max sources to cite on the normal retrieval path (we feed more chunks to the
# LLM for answer completeness, but only cite the few most relevant).
MAX_SOURCES = 6


def _select_sources(chunks: list[dict]) -> list[dict]:
    """De-duplicate by URL (keep the highest-scoring chunk per page), drop
    synthetic chunks, and — on the normal retrieval path — keep only the most
    relevant few. Roster paths already supply a curated source set, so keep it
    whole."""
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
    return ordered[:MAX_SOURCES]


async def _prepare_chunks(req: "ChatRequest") -> list[dict]:
    """Assemble the context chunks for a question.

    For "which professors research <area>" enumeration queries, build a
    deterministic layered roster (precise profile matches + broader graph
    areas) so the answer can't be silently capped by top-k retrieval.
    Otherwise run the normal hybrid pipeline and prepend graph context.
    """
    # Global "list all faculty" → serve the COMPLETE roster from the graph.
    # Retrieval can't enumerate the whole department (top-k cap + chunk splits),
    # so build it deterministically and keep the matching profile chunks as cites.
    if is_full_faculty_query(req.question):
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
    topic = _people_area_topic(req.question)
    if topic:
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

    chunks = await retrieve_async(req.question, section_filter=req.section_filter)
    if not chunks:
        return []
    graph_ctx = graph_query(req.question)
    if graph_ctx:
        log.info("Graph context injected (%d chars)", len(graph_ctx))
        chunks = [{"url": "knowledge-graph", "title": "Knowledge Graph", "section": "graph",
                   "text": graph_ctx, "rerank_score": 10.0}] + chunks
    return chunks


@app.post("/chat", summary="Streaming chat (Server-Sent Events)")
async def chat_stream(req: ChatRequest):
    """Returns a streaming text/event-stream response."""
    chunks = await _prepare_chunks(req)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found for your question.")

    log.info("Retrieved %d chunks for query %r:", len(chunks), req.question)
    for c in chunks:
        log.info("  [%.3f] %s", c.get("rerank_score", 0), c.get("chunk_id", c["url"]))

    sources = [{"url": c["url"], "title": c["title"], "section": c["section"]}
               for c in _select_sources(chunks)]

    async def event_stream():
        import json
        # Emit sources first
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        # Stream real LLM tokens as they arrive (with auto-continuation on
        # length-truncation handled inside generate_stream).
        emitted = 0
        try:
            async for delta in generate_stream(req.question, chunks):
                if not delta:
                    continue
                emitted += len(delta)
                # SSE data lines can't contain raw newlines; the frontend
                # restores them by replacing the literal "\n" sequence.
                safe = delta.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
        except Exception as e:
            log.warning("Streaming failed (%s); falling back to non-streaming", e)

        # If streaming produced nothing, fall back to a buffered generation
        # (and retry with a single chunk) so the user never sees an empty reply.
        if emitted == 0:
            answer = await _generate_full(req.question, chunks)
            if not answer and len(chunks) > 1:
                log.warning("Empty answer with %d chunks, retrying with 1 chunk", len(chunks))
                answer = await _generate_full(req.question, chunks[:1])
            if not answer:
                answer = ("I don't have enough details to answer that — please "
                          "check the sources below or visit engineering.tamu.edu/electrical.")
            yield f"data: {answer.replace(chr(10), chr(92) + 'n')}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/sync", response_model=ChatResponse, summary="Synchronous chat")
async def chat_sync(req: ChatRequest):
    """Returns a complete JSON response (no streaming)."""
    chunks = await _prepare_chunks(req)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found for your question.")

    answer = await generate(req.question, chunks)
    sources = [Source(url=c["url"], title=c["title"], section=c["section"])
               for c in _select_sources(chunks)]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/admin/test-llm", summary="Test LLM with a minimal prompt")
async def test_llm():
    """Sends 'Say hello.' to the LLM and returns the raw response."""
    import httpx, json as _json
    from generator import TAMU_API_URL, TAMU_API_KEY, TAMU_MODEL
    payload = {
        "model": TAMU_MODEL,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "stream": True,
    }
    parts = []
    raw_lines = []
    async with httpx.AsyncClient(timeout=30) as client:
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
                                parts.append(text)
                        except Exception:
                            pass
    return {"answer": "".join(parts), "raw_lines": raw_lines[:20]}


@app.post("/admin/reindex", summary="Manually trigger a site re-index")
async def manual_reindex():
    """Kicks off a diff re-index in the background."""
    import asyncio
    asyncio.create_task(run_reindex())
    return {"status": "re-index started"}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
