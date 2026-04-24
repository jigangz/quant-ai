# Quant AI - Production Dockerfile
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Builder stage
FROM base as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN useradd --create-home --shell /bin/bash appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

RUN mkdir -p ./artifacts ./data && chown appuser:appuser ./artifacts ./data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage (with full deps)
# Use explicitly: docker build --target development .
# NOT used by Render (Render builds last stage = production below)
FROM production as development

USER root
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt || true
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================
# Default stage = production + trimmed runtime deps (Render free tier friendly)
# V4 P4 (second iteration): first fix tried requirements-full.txt but that
# pulled torch/faiss/catboost (~1 GB) and OOM'd on Render free build. This
# uses requirements-prod.txt — a curated subset with only the runtime deps
# we actually need: xgboost, lightgbm, optuna, shap, supabase, psycopg, redis.
#
# Cut from full: catboost (redundant), sentence-transformers+torch (RAG
# feature degrades gracefully — prod has never had this anyway), faiss-cpu
# (ditto), pytest/ruff (dev-only).
# ============================================================
FROM production

USER root
COPY requirements-prod.txt .
COPY requirements.txt .
# Fail-loud (no `|| true`) so missing deps show in Render build log
RUN pip install --no-cache-dir -r requirements-prod.txt
USER appuser
