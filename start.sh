#!/bin/sh
set -e

# Start FastAPI backend on fixed internal port 8000
cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Next.js frontend on Cloud Run's injected $PORT (default 8080)
# BACKEND_URL tells the Next.js /api/chat proxy where to reach FastAPI
cd /app/frontend
exec env BACKEND_URL=http://localhost:8000 PORT="${PORT:-8080}" node server.js
