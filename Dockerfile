FROM python:3.10-slim AS base

WORKDIR /app

# System deps for LightGBM, XGBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY forecasting-product/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY forecasting-product/ forecasting-product/

# ---------------------------------------------------------------------------
#  FastAPI target
# ---------------------------------------------------------------------------
FROM base AS api
EXPOSE 8000
CMD ["python", "forecasting-product/scripts/serve.py", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--data-dir", "forecasting-product/data/"]
