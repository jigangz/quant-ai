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
# Default stage = production + full ML deps (Render builds the LAST stage)
# V4 P4 fix: add runtime deps (xgboost/lgbm/catboost/optuna/shap/redis/
# supabase/sentence-transformers/faiss) that were only in the unused
# `development` stage. Without this, prod 400s on xgboost model_type and
# silently degrades RAG/vol features.
# ============================================================
FROM production

USER root
COPY requirements-full.txt .
# Fail-loud (no `|| true`) so missing deps show in Render build log
RUN pip install --no-cache-dir -r requirements-full.txt
USER appuser
