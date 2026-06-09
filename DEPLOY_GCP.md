# Deploy to GCP — Cloud Run + Cloud SQL (from scratch)

Target architecture:

```
            ┌──────────────────────────────────────────────┐
  Browser → │ Cloud Run: ecen-frontend (Next.js)           │
            │   server route /api/chat ──BACKEND_URL──┐    │
            └─────────────────────────────────────────┼────┘
                                                       ▼
            ┌──────────────────────────────────────────────┐
            │ Cloud Run: ecen-backend (FastAPI + ML models)│
            │   PG_DSN ──Cloud SQL socket──┐               │
            └──────────────────────────────┼───────────────┘
                                           ▼
            ┌──────────────────────────────────────────────┐
            │ Cloud SQL: Postgres 16 + pgvector (db: ecen) │
            └──────────────────────────────────────────────┘
   Secret Manager → API keys / DB password / DSN   (mounted as env vars)
   Artifact Registry → container images
```

Set these shell variables once and reuse them throughout:

```bash
export PROJECT_ID="ecen-chatbot"          # pick a globally-unique id
export REGION="us-central1"
export REPO="ecen"                          # Artifact Registry repo
export SQL_INSTANCE="ecen-pg"
export DB_NAME="ecen"
export DB_USER="ecen_app"
```

---

## 0. Install gcloud + create the project (fresh start)

1. Install the CLI: https://cloud.google.com/sdk/docs/install (macOS: `brew install --cask google-cloud-sdk`).
2. Log in and create the project:

```bash
gcloud auth login
gcloud projects create "$PROJECT_ID"
gcloud config set project "$PROJECT_ID"
```

3. **Enable billing** — link a billing account in the Console
   (https://console.cloud.google.com/billing) or:

```bash
gcloud billing accounts list
gcloud billing projects link "$PROJECT_ID" --billing-account=XXXXXX-XXXXXX-XXXXXX
```

4. Enable the APIs you'll use:

```bash
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com
```

---

## 1. Artifact Registry (image storage)

```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker --location="$REGION"
gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

---

## 2. Cloud SQL (Postgres + pgvector)

```bash
# Create the instance (db-custom-1-3840 = 1 vCPU / ~3.75GB; bump if needed)
gcloud sql instances create "$SQL_INSTANCE" \
  --database-version=POSTGRES_16 --tier=db-custom-1-3840 --region="$REGION"

# Database + app user
gcloud sql databases create "$DB_NAME" --instance="$SQL_INSTANCE"
gcloud sql users create "$DB_USER" --instance="$SQL_INSTANCE" --password="CHANGE_ME_STRONG"

# Note the connection name (PROJECT:REGION:INSTANCE) — you'll need it
export CONN_NAME=$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(connectionName)')
echo "$CONN_NAME"
```

Enable the `vector` extension (the app also runs `CREATE EXTENSION IF NOT EXISTS vector`
on startup, but enable it once to be safe). Connect with:

```bash
gcloud sql connect "$SQL_INSTANCE" --user=postgres --database="$DB_NAME"
# then at the psql prompt:
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

---

## 3. Secret Manager (keys + DSN)

The backend reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `PG_DSN`,
and optionally `SENTRY_DSN`. Store the sensitive ones as secrets:

```bash
# LLM key (real OpenAI GPT key, or your TAMU gateway key)
printf 'sk-YOUR-OPENAI-KEY' | gcloud secrets create OPENAI_API_KEY --data-file=-

# Embeddings key (often the same OpenAI key)
printf 'sk-YOUR-OPENAI-KEY' | gcloud secrets create EMBEDDING_API_KEY --data-file=-

# DB connection string — Cloud Run reaches Cloud SQL over a unix socket:
printf 'postgresql://%s:%s@/%s?host=/cloudsql/%s' \
  "$DB_USER" "CHANGE_ME_STRONG" "$DB_NAME" "$CONN_NAME" \
  | gcloud secrets create PG_DSN --data-file=-

# Optional: Sentry backend DSN
printf 'https://...ingest.sentry.io/...' | gcloud secrets create SENTRY_DSN --data-file=-
```

Grant Cloud Run's runtime service account access to the secrets:

```bash
export PROJECT_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
export RUN_SA="$PROJECT_NUM-compute@developer.gserviceaccount.com"
for S in OPENAI_API_KEY EMBEDDING_API_KEY PG_DSN SENTRY_DSN; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:$RUN_SA" --role=roles/secretmanager.secretAccessor
done
```

---

## 4. Build + push images

From the repo root (`~/Documents/Claude/Projects/chatbot`):

```bash
export BACKEND_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/ecen-backend:latest"
export FRONTEND_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/ecen-frontend:latest"

# Backend (note: pre-bakes the ML models, so this build is large/slow the first time)
gcloud builds submit ./backend --tag "$BACKEND_IMG"

# Frontend
gcloud builds submit ./frontend --tag "$FRONTEND_IMG"
```

---

## 5. Deploy the backend (Cloud Run)

The ML models need memory and a slow first start; give it room and keep one warm
instance so users don't hit cold starts.

```bash
gcloud run deploy ecen-backend \
  --image "$BACKEND_IMG" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --add-cloudsql-instances "$CONN_NAME" \
  --memory 4Gi --cpu 2 \
  --min-instances 1 --max-instances 4 \
  --timeout 600 --cpu-boost \
  --set-secrets "OPENAI_API_KEY=OPENAI_API_KEY:latest,EMBEDDING_API_KEY=EMBEDDING_API_KEY:latest,PG_DSN=PG_DSN:latest,SENTRY_DSN=SENTRY_DSN:latest" \
  --set-env-vars "OPENAI_BASE_URL=https://api.openai.com/v1,OPENAI_MODEL=gpt-4o,EMBEDDING_API_URL=https://api.openai.com/v1,EMBEDDING_MODEL=text-embedding-3-small,DISABLE_SCHEDULER=1"

export BACKEND_URL=$(gcloud run services describe ecen-backend --region "$REGION" --format='value(status.url)')
echo "$BACKEND_URL"
```

> Using your **TAMU gateway** instead of OpenAI? Set
> `OPENAI_BASE_URL=https://chat-api.tamu.ai/openai` and `OPENAI_MODEL=protected.gpt-5`.

---

## 6. Deploy the frontend (Cloud Run)

The Next.js server route calls the backend via `BACKEND_URL` (server-to-server, no
CORS). Point it at the backend's URL:

```bash
gcloud run deploy ecen-frontend \
  --image "$FRONTEND_IMG" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1 \
  --set-env-vars "BACKEND_URL=$BACKEND_URL,NEXT_PUBLIC_SENTRY_DSN=https://...ingest.sentry.io/..."

export FRONTEND_URL=$(gcloud run services describe ecen-frontend --region "$REGION" --format='value(status.url)')
echo "Open: $FRONTEND_URL"
```

Then tighten the backend's CORS to the frontend URL (only matters if anything calls
it directly from the browser; the proxy path doesn't need it):

```bash
gcloud run services update ecen-backend --region "$REGION" \
  --update-env-vars "ALLOWED_ORIGINS=$FRONTEND_URL"
```

---

## 7. Load the data into Cloud SQL

Cloud SQL starts empty — you need the crawled chunks (~1,300) and the graph. Run
the ingest from your laptop against Cloud SQL through the Auth Proxy:

```bash
# 1. download + run the Cloud SQL Auth Proxy (https://cloud.google.com/sql/docs/postgres/sql-proxy)
./cloud-sql-proxy "$CONN_NAME" --port 5433 &

# 2. point the pipeline at the proxied DB and run the full rebuild
export PG_DSN="postgresql://$DB_USER:CHANGE_ME_STRONG@localhost:5433/$DB_NAME"
cd backend && pip install -r requirements.txt && cd ..
./scripts/rebuild.sh        # crawl → chunk → embed → upsert → graph_builder
```

Re-deploy isn't needed after loading data — the backend reads the DB live.

---

## 8. Scheduling the daily re-index

The in-process APScheduler (`backend/scheduler.py`) won't fire reliably on Cloud
Run, which only runs CPU during requests. Two options:
- **Recommended:** disable it in the container (the `DISABLE_SCHEDULER=1` env var
  above — see note below) and create a **Cloud Scheduler** job that POSTs to
  `"$BACKEND_URL"/admin/reindex` daily at 2 AM.
- Or set `--no-cpu-throttling` (CPU always allocated) so the background scheduler
  keeps running — costs more.

```bash
gcloud scheduler jobs create http ecen-reindex \
  --location "$REGION" --schedule "0 2 * * *" \
  --uri "$BACKEND_URL/admin/reindex" --http-method POST
```

> `DISABLE_SCHEDULER` is wired into `backend/main.py` lifespan: when set, the app
> skips the in-process scheduler so you can drive re-indexing from Cloud Scheduler.

---

## Costs & gotchas
- **`min-instances 1` is not free** — it keeps a 4Gi backend warm 24/7. Drop to 0
  to save money if cold starts (~30–60s to load models) are acceptable.
- **Every question bills your LLM key** when using OpenAI (TAMU gateway was free).
- **First backend build is slow/large** because it bakes the ML models in.
- **Never commit secrets** — they live only in Secret Manager; `.env` stays local
  and git-ignored.
- Custom domain + HTTPS: map one via Cloud Run → Manage Custom Domains, or put a
  load balancer in front.
```
