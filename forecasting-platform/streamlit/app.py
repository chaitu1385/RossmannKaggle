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
    "Weekly sales forecasting for retail S&OP — statistical, ML, neural, "
    "and foundation models with hierarchical reconciliation."
)

st.info(
    "**Start here:** Open **Data Onboarding** in the sidebar, "
    "then click **Use sample data** to see the platform in action."
)

st.markdown(
    """
**Pages**

1. **Data Onboarding** — Upload data or use the built-in sample to auto-detect schema, assess forecastability, and get a recommended config.
2. **Backtest Results** — Model leaderboard, FVA cascade showing which layers add or destroy value, per-series champion map.
3. **Forecast Viewer** — Interactive forecast chart with P10/P90 confidence intervals, actuals overlay, and seasonal decomposition.
4. **Platform Health** — Pipeline manifests, drift alerts, data quality summary, and compute cost.

**Need help?** See [QUICKSTART.md](https://github.com/chaitu1385/Forecasting-Platform/blob/master/QUICKSTART.md) for setup instructions.
    """
)

st.divider()
st.caption("Built with Streamlit")
