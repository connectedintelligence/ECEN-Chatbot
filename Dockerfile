# ── Stage 1: Build Next.js frontend ─────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
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

# Python deps
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake ML models into the image so cold starts don't pay the download cost
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Backend source
COPY backend/ ./backend/

# Next.js standalone output
COPY --from=frontend-builder /app/.next/standalone ./frontend/
COPY --from=frontend-builder /app/.next/static     ./frontend/.next/static
COPY --from=frontend-builder /app/public           ./frontend/public

# Startup script
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

EXPOSE 8080
CMD ["./start.sh"]
