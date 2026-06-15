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
- **Frontend tests run as a CI gate (`npm test`).** A fix is not done until the
  suite passes. If your change is frontend-facing, add or update a test under
  `frontend/__tests__/` and make sure the whole suite is green — a fix that
  can't pass the gate will never reach a PR.
- DB connections and `npm run build` may not work in the sandbox (no DB, slow/no
  network for a full build). For those, rely on static reasoning + `py_compile`
  and note in your summary what a human must run locally. **This does NOT apply
  to `npm test`** — that gate runs, so treat it as a real check you must satisfy.
- Changes to `retriever.py`, `generator.py`, or `graph_retriever.py` can affect
  roster injection / answer completeness — reason about those paths carefully.
- DB config lives in `.env` (NOT committed). Dockerized pgvector runs on host port
  **5433**. Don't hardcode credentials.

## Frontend test harness (read before writing a frontend test)
- Config is **`frontend/jest.config.js`** (plain JS on purpose — a `.ts` config
  needs `ts-node`, which is not a dependency). Do not convert it back to `.ts`.
- The setup key is **`setupFilesAfterEnv`** (NOT `setupFilesAfterFramework`,
  which is silently ignored). It loads `jest.setup.ts` →
  `@testing-library/jest-dom`, which provides matchers like `toBeDisabled` and
  `toBeInTheDocument`. If those matchers are "not a function", the setup file
  isn't loading — fix the config, don't work around it.
- Component tests use `@testing-library/react` + `@testing-library/dom` (a
  required peer dep, kept in `devDependencies`). If a render test fails with
  "Cannot find module '@testing-library/dom'", add the missing dep — don't
  delete the test.
- Prefer testing small presentational components in their own file (e.g.
  `components/FeedbackButtons.tsx`) so tests don't drag in heavy ESM deps.

## Self-check before you finish
1. Run `npm test` in `frontend/` (and `py_compile` for backend changes). The
   suite must be green, including any test you added.
2. If the harness itself is broken or missing a dependency, **fixing it is in
   scope** — a "frontend-only, zero-risk" framing is not a reason to leave a
   broken test gate in place.

## Guardrails
- Make the **smallest correct fix** — but "smallest" includes any test-infra or
  dependency repair needed to make the fix verifiable. Don't scope yourself out
  of fixing the harness.
- Edit files in the working tree only.
- Do NOT run git, push, or open PRs — the workflow does that and a human merges.
- Never write `.env`, API keys, DSNs, or model weight files.

## Known open issues
- CORS is `allow_origins=["*"]` — needs tightening for production.
- Fine-tuned embedder may be incomplete; base model in use.
- `FINAL_TOP_K` / `LIST_TOP_K` tuning is ongoing.
