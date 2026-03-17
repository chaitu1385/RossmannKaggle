"""
Forecasting Platform — Streamlit Dashboard

Main entry point.  Run with:
    streamlit run forecasting-platform/streamlit/app.py

Or via Docker Compose:
    docker compose up
"""

import sys
from pathlib import Path

# Ensure the forecasting-platform package is importable.
_PLATFORM_ROOT = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
#  Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
#  Landing page
# ---------------------------------------------------------------------------
st.title("Forecasting Platform")
st.markdown(
    """
    Weekly sales forecasting for retail S&OP — statistical, ML, neural, and
    foundation models with hierarchical reconciliation.

    Use the sidebar to navigate between pages:

    | Page | Purpose |
    |------|---------|
    | **Data Onboarding** | Upload data, detect schema & hierarchy, assess forecastability, get a recommended config |
    | **Backtest Results** | Model leaderboard, FVA cascade, champion map |
    | **Forecast Viewer** | Interactive forecast chart with confidence intervals and decomposition |
    | **Platform Health** | Pipeline manifests, drift alerts, data quality, compute cost |
    """
)

st.divider()
st.caption("Built with Streamlit · Powered by the Forecasting Platform engine")
