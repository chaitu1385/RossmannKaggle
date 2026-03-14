# ── Stage 1: Base with dependencies ──────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies for LightGBM/XGBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY setup.py pyproject.toml ./

# ── Stage 2: Test runner (CI only — not shipped to production) ───────────────
FROM base AS test

COPY tests/ tests/
RUN pip install --no-cache-dir pytest pytest-cov httpx
RUN python -m pytest \
    --ignore=tests/test_metrics.py \
    --ignore=tests/test_feature_engineering.py \
    --ignore=tests/test_sku_mapping.py \
    -v --tb=short

# ── Stage 3: Production image ───────────────────────────────────────────────
FROM base AS production

RUN pip install --no-cache-dir gunicorn "uvicorn[standard]"

EXPOSE 8000

ENV API_DATA_DIR=/app/data \
    API_METRICS_DIR=/app/data/metrics \
    AUTH_ENABLED=false

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
