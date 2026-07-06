# EIRA Runbook — TAMU ECE RAG Chatbot

Operational and developer reference for **EIRA** (ECE Information & Resource
Assistant), the production RAG chatbot for the Texas A&M Department of
Electrical & Computer Engineering. This document covers the architecture,
every file in the repository, how to clone/run/test/deploy the system, and the
operational gotchas learned in production.

Built by **Aarohi Mohrir** (M.S. Computer Science) under the guidance of
**Prof. Krishna Narayanan**.

- **Live app:** https://ecen-chatbot-199137295144.us-central1.run.app
- **Repo:** https://github.com/Aa-Rho-Hi/ECEN-Chatbot
- Companion docs: [README.md](README.md) (overview), [SETUP.md](SETUP.md)
  (first-time local setup), [DEPLOY_GCP.md](DEPLOY_GCP.md) (cloud deploy from
  scratch), [AGENTIC_SETUP.md](AGENTIC_SETUP.md) + [AGENTS.md](AGENTS.md)
  (issue→fix→PR automation), [CLAUDE.md](CLAUDE.md) (agent context),
  [tests/test_questions.md](tests/test_questions.md) (test question bank).

---

## 1. What this project is

A question-answering chatbot over `engineering.tamu.edu/electrical/` —
faculty, research areas, degree programs, courses, admissions, scholarships,
news, and events. Answers are **grounded**: a crawler indexes the official
site into a vector database, retrieval finds the relevant passages, and the
LLM answers only from those passages, with cited sources.

Key design decisions:

- **Hybrid retrieval** (dense vectors + keyword + fuzzy + re-ranking) instead
  of vectors alone — robust to typos and exact-name lookups.
- **Knowledge graph** for enumerations — "list all faculty" is answered from
  a complete, deterministic roster, never truncated by retrieval top-k.
- **Deterministic guards** around the LLM — injection screens, context gates,
  course-number scrubbing, secret redaction — so the model cannot hallucinate
  courses, leak secrets, or answer from junk context.
- **Zero-latency routing** — intent classification and follow-up (pronoun)
  resolution are pure in-process heuristics, no extra LLM round-trip.
- **LLM-judged regression suite** (DeepEval) so retrieval/prompt changes are
  measured, not eyeballed.

## 2. Technologies

| Layer | Technology |
|---|---|
| Frontend | Next.js 15 (App Router), TypeScript, Tailwind-style CSS, streaming SSE UI, Jest tests |
| Backend | Python 3.11+, FastAPI + Uvicorn, slowapi (rate limiting), APScheduler |
| Vector DB | PostgreSQL + pgvector (HNSW index); Supabase in prod, `pgvector/pgvector:pg16` Docker locally; GIN indexes (full-text + pg_trgm) for keyword/fuzzy arms |
| Embeddings | sentence-transformers — fine-tuned `finetune/tamu-ece-embedder` (MiniLM-L6-v2 base, 384-dim, runs locally in-process) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | OpenAI `gpt-4o` / `gpt-4o-mini` via `OPENAI_BASE_URL` (TAMU gateway `protected.gpt-5` also supported — same env vars) |
| Evaluation | DeepEval 4.x (LLM-as-judge) + deterministic keyword harness + pytest |
| Hosting | GCP Cloud Run (combined frontend+backend image), Cloud Build, Artifact Registry, Cloud Scheduler (nightly re-index job) |
| Automation | GitHub Actions (Codex issue triage → fix → PR), in-app bug reports filed as GitHub issues |

## 3. Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              Cloud Run (one image)          │
 user ── HTTPS ──►  Next.js UI ── /api/chat proxy ──► FastAPI backend │
                        │  (port $PORT)                (internal :8000)│
                        └───────────────────────────┬─────────────────┘
                                                    │
              ┌───────────────┬─────────────────────┼────────────────────┐
              ▼               ▼                     ▼                    ▼
        fast router     hybrid retrieval     knowledge graph        OpenAI LLM
   (intent heuristics + dense (pgvector) +  (graph.json: faculty,  (streaming, persona,
    follow-up/pronoun    BM25/FTS + fuzzy +  areas, degrees →       auto-continuation,
    resolution)          RRF + cross-encoder) complete rosters)     course scrubber)
                                │
                                ▼
                     Supabase Postgres + pgvector (table: ecen_docs)
                                ▲
                                │  nightly 2AM (Cloud Scheduler → Cloud Run Job)
                     crawler → chunker → embedder → upsert (diff mode)
```

### Question pipeline (a request's life)

1. **Security screen** — per-IP rate limit (`CHAT_RATE_LIMIT`, default
   10/min), prompt-injection regex → canned refusal.
2. **Fast route** (`main._fast_route`) — zero-latency intent classification:
   `creator` / `chitchat` / `list_all_faculty` / `people_by_area` / `general`.
3. **Follow-up resolution** (`main._resolve_followup_question`) — if the
   question is anaphoric ("did **he** have collaborators on **this paper**?"),
   names nobody itself, and history exists, the retrieval query is anchored to
   the faculty member most recently mentioned in the conversation (fix for
   issue #18: retrieval used to latch onto an arbitrary professor who shared
   surface vocabulary).
4. **Intent dispatch** (`main._prepare_chunks`) — deterministic executors:
   - rosters (all faculty / by area / intersections / degrees) built complete
     from the knowledge graph, never top-k truncated;
   - chitchat/creator answered from persona chunks, no retrieval, no junk
     citations;
   - everything else → **hybrid retrieval**: dense search (pgvector cosine,
     top-40) + keyword (Postgres FTS) + fuzzy (pg_trgm) → BM25 re-score + RRF
     fusion (top-20) → cross-encoder re-rank → top-K (`FINAL_TOP_K=8`, list
     queries widen to `LIST_TOP_K=20`); graph context injected when relevant.
5. **Context gate** — chunks below `CONTEXT_MIN_SCORE` never reach the LLM;
   if nothing clears the bar the user gets an honest "couldn't find that".
6. **Generation** (`generator.py`) — streaming SSE, EIRA persona, history,
   auto-continuation on token-cap truncation, and a **deterministic course
   scrubber**: lines citing course numbers not present in the retrieved
   context are dropped mid-stream; if the scrubber empties the whole answer
   (user asked about a nonexistent course), an honest fallback is emitted
   instead of a blank.
7. **Output guard** — secret-pattern redaction applied mid-stream with a
   holdback buffer; relevance-gated source citations (`SOURCE_MIN_SCORE`);
   structured audit log (grep `AUDIT` in logs).
8. **Answer cache** — identical first questions (no history) replay a cached
   answer for `ANSWER_CACHE_TTL` (1h), per instance.

### Frontend flow

`ChatUI.tsx` posts to `/api/chat` (Next.js route) which proxies to the FastAPI
`/chat` SSE stream (with retry while the backend cold-starts). The stream
emits a `sources` event first, then answer tokens. `parseAnswer.ts` splits the
`|||SUGGEST: q1 | q2 | q3` trailer into follow-up chips. Thumbs up/down post
to `/api/feedback`; the "report a problem" flow posts to `/api/report`, which
files a GitHub issue labeled `user-report` (that's how issue #18 arrived).

## 4. Getting started (clone → run → open)

### 4.1 Prerequisites

- Python 3.11+ (3.13 works), Node 20+, Docker Desktop (for local Postgres)
- An OpenAI API key (LLM + eval judge), or TAMU gateway credentials

### 4.2 Clone and configure

```bash
git clone https://github.com/Aa-Rho-Hi/ECEN-Chatbot.git chatbot
cd chatbot
cp .env.example .env         # then fill in values
```

`.env` keys (read by every backend module via `load_dotenv(override=True)` —
see the gotcha in §8):

| Key | Meaning |
|---|---|
| `PG_DSN` | Postgres DSN. Local Docker: `postgresql://postgres:postgres@localhost:5433/ecen`. Prod: Supabase **session pooler** — `postgresql://postgres.<project-ref>:<password>@aws-1-us-east-2.pooler.supabase.com:5432/postgres` (username must be `postgres.<project-ref>`; the direct `db.*.supabase.co` host is IPv6-only) |
| `OPENAI_API_KEY` | LLM key (also used by embeddings API if configured) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` (or TAMU gateway `https://chat-api.tamu.ai/openai`) |
| `OPENAI_MODEL` | `gpt-4o` / `gpt-4o-mini` (or `protected.gpt-5`) |
| `EMBEDDING_MODEL` | Path or HF name of the SentenceTransformer, e.g. `finetune/tamu-ece-embedder`. **Must match the embedder used at ingest time** or retrieval silently degrades |
| `TARGET_URL`, `CRAWL_DELAY_SECONDS`, `MAX_PAGES` | Crawler config |
| `REINDEX_CRON` | APScheduler cron for in-process re-index (default 2AM) |
| `ALLOWED_ORIGINS` | CORS allowlist for the API |
| `GH_ISSUE_TOKEN`, `GH_REPO` | Token/repo for filing in-app bug reports as GitHub issues |

### 4.3 Database

Option A — local Docker (self-contained):

```bash
docker compose up -d ecen-postgres          # pgvector on host port 5433
python scripts/check_db.py                  # health check: table, indexes, counts
./scripts/rebuild.sh                        # crawl + chunk + embed + upsert + graph
```

Option B — Supabase (prod data): set `PG_DSN` to the session-pooler DSN and
skip ingestion; the data is already there.

### 4.4 Run the backend

```bash
cd backend
pip install -r requirements.txt
python main.py                              # http://localhost:8000
```

Wait for `Models ready.` then `Uvicorn running on http://0.0.0.0:8000`
**without** an "address already in use" error (see §8 if you get one).
Sanity check: `curl http://127.0.0.1:8000/health`.

Backend endpoints: `POST /chat` (SSE stream), `POST /chat/sync` (JSON),
`GET /health`, `POST /feedback`, `POST /report-issue`, `GET /admin/stats`,
`GET /admin/test-llm` (isolates the LLM from retrieval), `POST /admin/reindex`.

### 4.5 Run the frontend

```bash
cd frontend
npm install
npm run dev                                 # http://localhost:3000
```

Open http://localhost:3000 and chat. The UI proxies to the backend via
`BACKEND_URL` (defaults to `http://localhost:8000`).

### 4.6 Docker (whole stack)

`docker compose up -d` runs `ecen-postgres`, `ecen-backend`, and
`ecen-frontend` containers. **Do not run the containerized backend and a
hand-started `python main.py` at the same time — they fight over port 8000**
(`docker stop ecen-backend` first).

## 5. Testing & evaluation

### 5.1 Unit tests

```bash
pip install -r requirements-test.txt
pytest tests/                               # prompt/regression unit tests
cd frontend && npx jest                     # UI unit tests
```

### 5.2 Deterministic keyword harness (fast, free)

```bash
python scripts/eval.py --delay 0            # local backend
python scripts/eval.py --fast               # P0 smoke set only
BASE_URL=https://<service> python scripts/eval.py   # deployed (keep 7s delay)
```

~47 cases with required/forbidden substring checks across identity, security,
rosters, faculty facts, degrees, admissions, typos, and edge cases. Case
definitions in `scripts/eval.py` mirror `tests/test_questions.md`.

### 5.3 DeepEval LLM-judged regression suite (the real quality gate)

```bash
export OPENAI_API_KEY=sk-...                          # judge (gpt-4o-mini)
# terminal 1: backend with context echo + lifted rate limit
cd backend && CHAT_RATE_LIMIT=1000/minute EVAL_MODE=1 python main.py
# terminal 2:
python scripts/deepeval_eval.py --delay 0             # full 60-case suite
python scripts/deepeval_eval.py --tag multiturn       # issue-#18 class only
python scripts/deepeval_eval.py --tag extended        # deep probes
python scripts/deepeval_eval.py --no-llm              # keyword-only, no key
EVAL_JUDGE_MODEL=gpt-4o python scripts/deepeval_eval.py   # stricter judge
```

Every case runs the keyword gate first, then category-specific LLM judges:
**Conversational Grounding** (multi-turn follow-ups stay on the right
person), **Guardrail Adherence** (no prompt/model/credential leaks; creator
attribution is intended), **Roster Completeness & Grounding**, **Direct
Answer Relevancy** (house style — contact info, suggestions — is not
penalized), **Faithfulness** (answer vs retrieved context; needs `EVAL_MODE=1`
so `/chat/sync` returns context), **Scope & Hallucination Discipline**.

Reports land in `eval_reports/deepeval_report.md` + `.json` (score + judge's
written reason per case; overwritten each run; gitignored).

Judge calibration notes (learned the hard way): GEval scores jitter between
runs, so GEval metrics use `EVAL_SOFT_THRESHOLD` (default 0.55 — observed
true failures ≤ ~0.4, correct answers ≥ ~0.6); Faithfulness stays at 0.7.
The `|||SUGGEST` trailer is stripped before judging. **When the judge flags
something, verify against the official site before acting** — the judge errs
strict, and two of its faithfulness flags proved to be false positives.

Workflow: run `--fast` before every deploy; run the full suite after any
retrieval/prompt/embedder change; **turn every user bug report into a new
case** (issue #18 → case M.1) so fixed bugs can never silently return.

### 5.4 Bugs this suite has caught (and now guards)

1. **Issue #18** — anaphoric follow-ups retrieved an arbitrary professor
   (fixed via follow-up resolution; case M.1).
2. **"Are you a real person?"** returned the no-info fallback (chitchat regex
   extended; case 1.2).
3. **Blank answers for nonexistent courses** — the course scrubber deleted
   everything the model wrote, leaving only the suggestions trailer (honest
   fallback added in `generator.py`; case E.3).

## 6. Deployment (GCP)

Project `gcp-clen-ecen-ai-initiatives`, region `us-central1`, service
`ecen-chatbot` (single image: FastAPI + Next.js, entrypoint `start.sh`).

**Manual build + deploy (the proven path — the push-to-main Cloud Build
trigger exists but has not been observed to fire; investigate its repo
connection in the Cloud Build console):**

```bash
# from repo root, on main, after merging:
gcloud builds submit . --project gcp-clen-ecen-ai-initiatives \
  --tag us-central1-docker.pkg.dev/gcp-clen-ecen-ai-initiatives/cloud-run-source-deploy/ecen-chatbot:manual-$(date +%Y%m%d-%H%M%S)

gcloud run services update ecen-chatbot --region us-central1 \
  --project gcp-clen-ecen-ai-initiatives \
  --image <the tag printed by the build>
```

Build takes ~10–15 min (ML models are baked into the image; runtime sets
`HF_HUB_OFFLINE=1` so cold starts don't HEAD-check HuggingFace — that used to
add ~2.5 min per cold start via 429s). After deploying, verify:

```bash
BASE_URL=https://ecen-chatbot-199137295144.us-central1.run.app \
  python scripts/deepeval_eval.py --tag multiturn      # expect 5/5
```

Secrets (`OPENAI_API_KEY`, `EMBEDDING_API_KEY`, `PG_DSN`) live in Secret
Manager and are mounted by the service. Rotating the Supabase DB password
requires updating the `PG_DSN` secret in the same step or the service breaks;
`secretmanager.versions.access` on this project may require the admin.

Scaling: `_MIN_INSTANCES=0` (scale to zero, ~1 min cold start masked by proxy
retries) — flip to 1 in `cloudbuild.yaml` to keep one instance warm (~$40/mo).

## 7. Operations

```bash
# logs / audit trail (one AUDIT line per question: hashed IP, intent, sources)
gcloud run services logs read ecen-chatbot --region us-central1 --limit 50

# usage counters since instance start
curl https://<service-url>/admin/stats

# manual full re-index (nightly job does --diff automatically at 2AM Central)
gcloud run jobs execute ecen-reindex --region us-central1 --wait

# local index rebuild (DB check → crawl → embed → upsert → graph rebuild)
./scripts/rebuild.sh --diff
```

In-app bug reports (chat UI → "report a problem") file GitHub issues labeled
`user-report`; the Actions workflows in `.github/workflows/` triage them and
open fix PRs on `claude/fix-issue-<n>` / codex branches. **Agents never push
to `main`** — humans review and merge.

## 8. Gotchas / troubleshooting (each of these cost real time)

| Symptom | Cause / fix |
|---|---|
| `address already in use` on 8000, or requests hit stale code | A zombie `python main.py` (or the `ecen-backend` Docker container) holds the port and keeps serving while your new process dies at startup. `lsof -ti :8000 \| xargs kill -9` (and `docker stop ecen-backend`), then start ONE backend. |
| Env vars on the command line seem ignored | Every backend module calls `load_dotenv(override=True)` — **`.env` values overwrite shell/inline env vars.** To change `OPENAI_*`, `PG_DSN`, `EMBEDDING_*` locally, edit `.env` (keep a backup) and restart. Vars not present in `.env` (e.g. `EVAL_MODE`, `CHAT_RATE_LIMIT`) work inline. |
| `password authentication failed for user "postgres"` on Supabase | The pooler reports the base role name even when you sent `postgres.<ref>` — the username is fine; the **password** is wrong. Also make sure you actually restarted the backend after editing `.env`. |
| Supabase direct host unreachable | `db.<ref>.supabase.co` is IPv6-only; use the session pooler (`aws-1-us-east-2.pooler.supabase.com:5432`, username `postgres.<project-ref>`). |
| `/chat/sync` 200 but empty answers | Generation failed server-side (check `TAMU raw status:` in logs — 4xx/429 from the LLM API) or you're talking to a stale process (see zombie row). `GET /admin/test-llm` isolates the model from retrieval. |
| Cold start takes minutes | Should not happen anymore (`HF_HUB_OFFLINE=1` baked in). If it does, check logs for HuggingFace HEAD requests. |
| Retrieval suddenly bad after embedder change | Query embedder must match the embedder that ingested the DB (`EMBEDDING_MODEL`); re-ingest when switching. |
| Keyword/fuzzy search slow (seconds) | Missing GIN indexes — run `scripts/migrations/2026-06-22_add_fts_trgm_indexes.sql` (fresh ingests create them automatically). |
| Eval: all cases error "connection refused" | Backend isn't running / wrong port. `curl http://127.0.0.1:8000/health` first. |
| Eval: judge fails answers that look right | Judge errs strict. Read the reason in `eval_reports/deepeval_report.md`, verify against the official site, and only then change code (or calibrate the metric). |
| Full local eval run hits 429s | Rate limit applies locally too — start backend with `CHAT_RATE_LIMIT=1000/minute`. |

## 9. File-by-file reference

### Root

| File | Contents |
|---|---|
| `README.md` | Project overview, architecture diagram, stack, security hardening, ops quick-reference. |
| `RUNBOOK.md` | This document. |
| `SETUP.md` | First-time local development setup walkthrough. |
| `DEPLOY_GCP.md` | From-scratch GCP deployment guide (project, APIs, Artifact Registry, secrets, Cloud Run, scheduler). |
| `AGENTIC_SETUP.md` / `AGENTS.md` | The GitHub issue→fix→PR automation: how bug reports become PRs, agent guardrails. |
| `CLAUDE.md` | Context file for AI coding agents: stack summary, file map, verification commands, guardrails (never push to main, never commit `.env`). |
| `Dockerfile` | Combined production image: builds the Next.js standalone frontend, installs backend+crawler deps, **pre-bakes both ML models**, sets `HF_HUB_OFFLINE=1` for fast cold starts, entrypoint `start.sh`. |
| `start.sh` | Container entrypoint: uvicorn on :8000 + Next.js server on `$PORT`, wired via `BACKEND_URL`. |
| `docker-compose.yml` | Local stack: `ecen-postgres` (pgvector, host port 5433), `ecen-backend`, `ecen-frontend`. |
| `cloudbuild.yaml` | Cloud Build pipeline: docker build/push → `gcloud run services update` with pinned resources (4Gi/2CPU, `_MIN_INSTANCES`) → re-index job deploy. |
| `.env.example` | Template for `.env` (never commit the real one). |
| `.dockerignore` / `.gitignore` | Build/repo exclusions (`.env`, models, `eval_reports/`, caches). |
| `requirements-test.txt` | Test/eval deps: pytest, httpx, deepeval. |

### `backend/`

| File | Contents |
|---|---|
| `main.py` | FastAPI app — the orchestrator. Endpoints (`/chat` SSE, `/chat/sync`, `/health`, `/feedback`, `/report-issue`, `/admin/*`); security layer (rate limits, injection regex, secret redaction with streaming holdback); fast intent router; **follow-up/anaphora resolution** (issue #18 fix); intent dispatch to graph rosters vs retrieval; context gate; answer cache; source selection; audit logging; `EVAL_MODE` context echo for faithfulness testing. |
| `generator.py` | All LLM interaction: system prompt/persona (EIRA), streaming + non-streaming generation, auto-continuation on token-cap, conversation-history sanitization, **deterministic course-number scrubber** with honest fallback when it empties an answer, legacy LLM router (`route_question`, kept as reference). |
| `retriever.py` | Hybrid retrieval: local SentenceTransformer embedder + cross-encoder loading, pgvector dense search, Postgres FTS keyword arm, pg_trgm fuzzy arm, BM25 re-score + RRF fusion, cross-encoder re-rank, people-by-area enumeration with signal scoring, connection pooling. |
| `graph_retriever.py` | Knowledge-graph query layer: intent detection, research-area aliases (AI→"Artificial Intelligence and Machine Learning" etc.), faculty lookup, `find_faculty_mentions` (conversation-text name detection for follow-up resolution), complete faculty/area/degree rosters, cross-area intersection rosters, suggested-contact logic. |
| `graph_builder.py` | Rebuilds `graph.json` from the Postgres chunk store: parses faculty profiles, research-area pages, degree pages into nodes+edges. Run after re-ingest (`rebuild.sh` does it). |
| `graph.json` | The knowledge graph snapshot (~70 faculty, 11 research areas, degree programs, group-leader edges). Data artifact, regenerated by `graph_builder.py`. |
| `scheduler.py` | APScheduler daily re-index inside the FastAPI process (disabled on Cloud Run via `DISABLE_SCHEDULER=1`; the Cloud Run Job does it there). |
| `github_issues.py` | Files in-app bug reports as GitHub issues (`user-report` label) via `GH_ISSUE_TOKEN`/`GH_REPO`. |
| `requirements.txt` | Backend Python deps. |
| `Dockerfile` | Backend-only image (used by the split deploy path in DEPLOY_GCP.md). |

### `crawler/`

| File | Contents |
|---|---|
| `crawler.py` | BFS crawler of the `/electrical/` site: clean-text extraction, section classification (people/research/academics/admissions/news/events), people-directory JSON feed, crawl-delay + domain allowlist. |
| `chunker.py` | Section-aware chunking: per-person blocks on people pages, per-article for news, 600-token/80-overlap recursive splitting elsewhere. |
| `ingest.py` | Crawl → PII scrub → chunk → embed → upsert into `ecen_docs`; `--diff` mode embeds only changed pages; creates schema + HNSW/FTS/trigram indexes; stale-chunk pruning; mass-change poisoning guard (`POISON_GUARD_THRESHOLD`, `FORCE_INGEST=1` to override). |
| `generate_training_data.py` | Self-supervised (query, passage) pair generation from the corpus for embedder fine-tuning (no LLM calls). |
| `training_pairs.jsonl` | The generated fine-tuning pairs. |
| `requirements.txt` | Crawler deps. |

### `frontend/`

| File | Contents |
|---|---|
| `app/page.tsx` / `app/layout.tsx` / `app/globals.css` | Next.js App Router entry, layout, styles. |
| `components/ChatUI.tsx` | The chat interface: SSE streaming render, stop button, conversation history (sent with each request so follow-ups work), section filter, source badges, follow-up suggestion chips, report-a-problem flow. |
| `components/FeedbackButtons.tsx` | Thumbs up/down → `/api/feedback`. |
| `lib/parseAnswer.ts` | Splits answer text from the `\|\|\|SUGGEST:` trailer into body + suggestion chips. |
| `app/api/chat/route.ts` | Proxy to FastAPI `/chat` with cold-start retry (up to ~4.5 min) and real-client-IP forwarding for rate limiting. |
| `app/api/feedback/route.ts` / `app/api/report/route.ts` | Proxies to `/feedback` and `/report-issue`. |
| `__tests__/` + `jest.config.js` + `jest.setup.ts` | Jest unit tests (answer parsing, feedback buttons). |
| `index.html` / `widget.html` | Standalone/embeddable widget page. |
| `next.config.js`, `tsconfig.json`, `package.json` | Build config. |
| `public/ellie*.png/jpg` | Avatar assets. |

### `scripts/`

| File | Contents |
|---|---|
| `eval.py` | Deterministic keyword regression harness (~47 cases; required/forbidden/require_any per case; `--fast`, `--tag`, `--delay`; also the shared case dataset for DeepEval). |
| `deepeval_eval.py` | **LLM-judged regression suite** (60 cases = eval.py cases + multi-turn M.1–M.5 + extended probes E.1–E.8). Per-category judge metrics, calibrated thresholds, markdown+JSON reports with judge reasons. See §5.3. |
| `check_db.py` | DB health check: connectivity, table/index presence (including FTS/trigram), row counts, freshness. |
| `rebuild.sh` | One-command index rebuild: DB check → `ingest.py` → `graph_builder.py`. |
| `migrations/2026-06-22_add_fts_trgm_indexes.sql` | One-time CONCURRENTLY migration adding the FTS + trigram GIN indexes to a live DB. |
| `deploy_gcp.sh` | Scripted two-service GCP deploy (backend + frontend separately; the current prod uses the combined-image path instead). |
| `ingest_catalog.py` | Parses TAMU catalog ECEN course descriptions and upserts them (grounding for course questions). |
| `reingest_urls.py` | Re-crawl and re-ingest a hand-picked URL list. |
| `audit_collection.py` | Corpus coverage audit (counts, per-section, thin pages, duplicates). Written for the old Qdrant store — needs porting before use. |
| `check_patents_indexed.py` | Spot check that a specific page/term made it into the index. |

### `finetune/`

| File | Contents |
|---|---|
| `finetune_embedder.py` | Fine-tunes MiniLM on the domain pairs (MultipleNegativesRankingLoss); produces `finetune/tamu-ece-embedder` (the model directory itself is not committed). |
| `TAMU_ECE_Embedder_Finetune.ipynb` | Notebook version of the fine-tune. |
| `hprc_job.sh` | TAMU HPRC (SLURM) job script for GPU fine-tuning. |

### `tests/` and `.github/`

| File | Contents |
|---|---|
| `tests/test_generator_prompt.py` | Pytest unit tests for prompt construction/generator behavior. |
| `tests/test_questions.md` | Human-readable test question bank: every retrieval path, expected content, severity (P0–P2), including §12 multi-turn grounding cases. |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Structured bug-report template. |
| `.github/workflows/codex-triage.yml` / `codex.yml` / `codex-review.yml` | Issue triage → automated fix branch/PR → review workflows. |
| `.github/scripts/implement.py` | Helper the workflows use to drive the fix loop. |

### Generated / local-only (gitignored)

`eval_reports/` (DeepEval reports), `.env` (secrets), `finetune/tamu-ece-embedder/`
(model weights), `backend/graph.json.*.bak` (graph snapshots), `.deepeval/`
(judge cache), `__pycache__/`, `node_modules/`, `.next/`.

## 10. Quality status (as of 2026-07-06)

- Core suite: 49/52 passing; the 3 failures were adjudicated as judge false
  positives against the official site (certificates, deadlines all correct).
- Multi-turn (issue-#18 class): 5/5 against production.
- Extended probes: 8/8 against production.
- Open ops items: rotate the Supabase DB password together with the GCP
  `PG_DSN` secret (needs project admin); investigate why the Cloud Build
  push-to-main trigger never fires.
