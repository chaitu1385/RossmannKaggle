FROM python:3.10-slim AS base

WORKDIR /app

# System deps for LightGBM, XGBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY forecasting-platform/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY forecasting-platform/ forecasting-platform/

# ---------------------------------------------------------------------------
#  FastAPI target
# ---------------------------------------------------------------------------
FROM base AS api
EXPOSE 8000
CMD ["python", "forecasting-platform/scripts/serve.py", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--data-dir", "forecasting-platform/data/"]

# ---------------------------------------------------------------------------
#  Streamlit target
# ---------------------------------------------------------------------------
FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "forecasting-platform/streamlit/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
