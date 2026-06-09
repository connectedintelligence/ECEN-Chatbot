#!/usr/bin/env bash
# Build + deploy ecen-backend and ecen-frontend to Cloud Run.
# Prereqs (one-time): see DEPLOY_GCP.md sections 0–3 (project, APIs, Artifact
# Registry, Cloud SQL, Secret Manager). Run from the repo root.
set -euo pipefail

# ── Config — edit these ──────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-ecen-chatbot}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-ecen}"
SQL_INSTANCE="${SQL_INSTANCE:-ecen-pg}"
# LLM target: OpenAI by default. For TAMU gateway, override these two env vars.
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
EMBEDDING_API_URL="${EMBEDDING_API_URL:-https://api.openai.com/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
NEXT_PUBLIC_SENTRY_DSN="${NEXT_PUBLIC_SENTRY_DSN:-}"
# ─────────────────────────────────────────────────────────────────────────────

gcloud config set project "$PROJECT_ID"
CONN_NAME="$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(connectionName)')"
BACKEND_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/ecen-backend:latest"
FRONTEND_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/ecen-frontend:latest"

echo ">> Building backend image (pre-bakes ML models — first build is slow)…"
gcloud builds submit ./backend --tag "$BACKEND_IMG"

echo ">> Deploying ecen-backend…"
gcloud run deploy ecen-backend \
  --image "$BACKEND_IMG" --region "$REGION" --platform managed --allow-unauthenticated \
  --add-cloudsql-instances "$CONN_NAME" \
  --memory 4Gi --cpu 2 --min-instances 1 --max-instances 4 --timeout 600 --cpu-boost \
  --set-secrets "OPENAI_API_KEY=OPENAI_API_KEY:latest,EMBEDDING_API_KEY=EMBEDDING_API_KEY:latest,PG_DSN=PG_DSN:latest,SENTRY_DSN=SENTRY_DSN:latest" \
  --set-env-vars "OPENAI_BASE_URL=$OPENAI_BASE_URL,OPENAI_MODEL=$OPENAI_MODEL,EMBEDDING_API_URL=$EMBEDDING_API_URL,EMBEDDING_MODEL=$EMBEDDING_MODEL,DISABLE_SCHEDULER=1"

BACKEND_URL="$(gcloud run services describe ecen-backend --region "$REGION" --format='value(status.url)')"
echo ">> Backend at: $BACKEND_URL"

echo ">> Building frontend image…"
gcloud builds submit ./frontend --tag "$FRONTEND_IMG"

echo ">> Deploying ecen-frontend…"
gcloud run deploy ecen-frontend \
  --image "$FRONTEND_IMG" --region "$REGION" --platform managed --allow-unauthenticated \
  --memory 512Mi --cpu 1 \
  --set-env-vars "BACKEND_URL=$BACKEND_URL,NEXT_PUBLIC_SENTRY_DSN=$NEXT_PUBLIC_SENTRY_DSN"

FRONTEND_URL="$(gcloud run services describe ecen-frontend --region "$REGION" --format='value(status.url)')"
echo ">> Tightening backend CORS to the frontend URL…"
gcloud run services update ecen-backend --region "$REGION" \
  --update-env-vars "ALLOWED_ORIGINS=$FRONTEND_URL"

echo ""
echo "✅ Done.  Open the app:  $FRONTEND_URL"
echo "   (If the DB is empty, load data per DEPLOY_GCP.md section 7.)"
