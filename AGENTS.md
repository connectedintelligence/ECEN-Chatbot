# AGENTS.md — TAMU ECE RAG Chatbot

Context for the OpenAI Codex GitHub Action when it triages issues and writes fixes.
A Q&A chatbot over `engineering.tamu.edu/electrical/` (faculty, research, degree
programs, admissions, news, events).

## Stack
- **Backend:** Python, FastAPI + Uvicorn, APScheduler
- **Frontend:** Next.js 15 (App Router), TypeScript, Tailwind
- **Vector DB:** PostgreSQL + pgvector (`pgvector/pgvector:pg16`), table `ecen_docs`, HNSW index
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local)
- **LLM:** TAMU gateway `https://chat-api.tamu.ai/openai`, model `protected.gpt-5` (reasoning model)
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)

## Retrieval pipeline
dense search (pgvector cosine, top-40) → BM25 re-score + RRF fusion (top-20) →
cross-encoder re-rank → top-K chunks → TAMU LLM. List-intent queries widen to
`LIST_TOP_K=20`, else `FINAL_TOP_K=8`. A knowledge graph (`backend/graph.json`,
70 faculty) injects synthetic "roster" chunks for "list all faculty" / "professors
by research area" queries so enumerations aren't capped or cut off.

## File map
| File | Purpose |
|---|---|
| `crawler/crawler.py` | BFS crawler, extracts text, classifies sections |
| `crawler/chunker.py` | Section-aware chunking (600-token, 80 overlap) |
| `crawler/ingest.py` | Crawl → chunk → embed → upsert; `--diff` mode |
| `backend/retriever.py` | Hybrid retrieval (dense + BM25 + RRF + cross-encoder), roster routing |
| `backend/generator.py` | TAMU LLM calls, streaming + continuation loop |
| `backend/graph_retriever.py` | Builds faculty/area rosters from graph.json |
| `backend/graph_builder.py` | Rebuilds graph.json from Postgres chunks |
| `backend/main.py` | FastAPI: `/chat` (SSE), `/chat/sync`, `/health`, `/admin/reindex` |
| `backend/scheduler.py` | APScheduler daily re-index (2AM) |
| `frontend/components/ChatUI.tsx` | Streaming chat UI, section filter, source badges |
| `frontend/app/api/chat/route.ts` | Next.js proxy to FastAPI |
| `scripts/check_db.py` | DB health check |
| `scripts/rebuild.sh` | DB check → ingest → graph_builder |

## How to verify a change
- **Python compiles:** `python -m py_compile backend/*.py crawler/*.py`
- The Codex sandbox has **no network access** by default, so `npm install` /
  `npm run build` and any DB connection will not work in CI. Rely on static
  reasoning + `py_compile` to validate; note in your summary anything that needs
  a human to run locally.
- Changes to `retriever.py`, `generator.py`, or `graph_retriever.py` can affect
  roster injection / answer completeness — reason about those paths carefully.
- DB config lives in `.env` (NOT committed). Dockerized pgvector runs on host port
  **5433**. Don't hardcode credentials.

## Guardrails
- Make the **smallest correct fix**. Edit files in the working tree only.
- Do NOT run git, push, or open PRs — the workflow does that and a human merges.
- Never write `.env`, API keys, DSNs, or model weight files.

## Known open issues
- CORS is `allow_origins=["*"]` — needs tightening for production.
- Fine-tuned embedder may be incomplete; base model in use.
- `FINAL_TOP_K` / `LIST_TOP_K` tuning is ongoing.
