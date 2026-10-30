# syntax=docker/dockerfile:1

# Production image for the Document Intelligence Assistant.
#   docker build -t document-intelligence-assistant .
#   docker run -p 8000:8000 --env-file .env document-intelligence-assistant
# Three stages keep compilers, raw docs and build caches out of the final image:
#   deps    - Python packages in a virtualenv
#   index   - downloads the embedding model and builds the Chroma index from data/raw_docs
#   runtime - slim image with the venv, model, index, app code and frontend; runs as non-root

ARG PYTHON_VERSION=3.13

# ---- 1. deps: install Python packages -------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS deps
# HNSWLIB_NO_NATIVE: chroma-hnswlib has no prebuilt wheel for Python 3.13, so it's compiled here - and by
# default with -march=native, i.e. for the build machine's exact CPU. CI caches this layer, so a library
# built on a runner with a newer CPU crashed silently (no Python error) on runners and Cloud Run machines
# without those instructions. Build it for generic x86-64 instead, so the image runs on any machine.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HNSWLIB_NO_NATIVE=1
# Compilers only in this stage, in case a package has no prebuilt wheel for this platform
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# CPU-only PyTorch first: the default Linux wheel pulls ~3 GB of CUDA libraries this app never uses
# (embeddings run on CPU). The lock file's torch==2.14.0 is then already satisfied by 2.14.0+cpu.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

# ---- 2. index: embedding model + Chroma index, built once at image build time --------------
FROM python:${PYTHON_VERSION}-slim AS index
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False
COPY --from=deps /opt/venv /opt/venv
WORKDIR /app
# Only what ingestion needs, so editing main.py or the prompt doesn't trigger a re-index
COPY app/__init__.py app/config.py app/ingest.py app/
COPY data/raw_docs/ data/raw_docs/
RUN python -m app.ingest

# ---- 3. runtime ------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
# The model is baked in (offline at runtime: no download on start, no Hugging Face dependency).
# chroma_db must be writable by the app user: Chroma's SQLite writes lock/journal files on open.
COPY --from=index --chown=appuser:appuser /app/.cache/huggingface .cache/huggingface
COPY --from=index --chown=appuser:appuser /app/chroma_db chroma_db
COPY app/ app/
COPY frontend/ frontend/
USER appuser
EXPOSE 8000
# /health is 503 until the vector store is loaded and ANTHROPIC_API_KEY is set (no curl in slim images)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]
# One worker: each worker loads its own copy of the embedding model; FastAPI's threadpool
# already handles concurrent requests. The API key comes from the environment at run time.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
