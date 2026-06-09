FROM python:3.11-slim

WORKDIR /app

# Cache HuggingFace models inside the image so Cloud Run cold starts don't pay
# the model-download cost on first request.
ENV HF_HOME=/app/.hf_cache \
    PYTHONUNBUFFERED=1

# System deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedder + cross-encoder at build time (baked into the image).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Cloud Run injects $PORT (default 8080); fall back to 8000 for local/compose.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
