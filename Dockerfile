# ── Stage 1: Build Next.js frontend ─────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
# Install ALL deps (incl. devDependencies): next build needs typescript,
# @types/*, tailwindcss, postcss, autoprefixer at build time. Dev deps never
# reach the runtime image — this is a multi-stage build and only .next output
# is copied to stage 2, so Jest etc. are excluded regardless.
RUN npm install --legacy-peer-deps
COPY frontend/ .
RUN npm run build

# ── Stage 2: Combined runtime ────────────────────────────────────────────────
FROM python:3.11-slim

ENV HF_HOME=/app/.hf_cache \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=production

# System deps: Python build tools + Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (backend + crawler, so this image can also run the re-index job)
COPY backend/requirements.txt ./requirements-backend.txt
COPY crawler/requirements.txt ./requirements-crawler.txt
RUN pip install --no-cache-dir -r requirements-backend.txt -r requirements-crawler.txt

# Pre-bake ML models into the image so cold starts don't pay the download cost
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Models are baked into the image cache above. Force offline mode at RUNTIME so
# startup uses the local cache instead of HEAD-checking HuggingFace for updates.
# Those checks were returning HTTP 429 (rate-limited) and adding ~2.5 minutes to
# every cold start. Set AFTER the bake step so the download above still works.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Backend source + crawler (used by the ecen-reindex Cloud Run Job)
COPY backend/ ./backend/
COPY crawler/ ./crawler/

# Next.js standalone output
COPY --from=frontend-builder /app/.next/standalone ./frontend/
COPY --from=frontend-builder /app/.next/static     ./frontend/.next/static
COPY --from=frontend-builder /app/public           ./frontend/public

# Startup script
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

EXPOSE 8080
CMD ["./start.sh"]
