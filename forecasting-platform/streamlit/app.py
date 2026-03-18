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

# ---------------------------------------------------------------------------
#  Quick-start by persona
# ---------------------------------------------------------------------------
st.subheader("Quick Start")

col_ds, col_dp, col_eng = st.columns(3)

with col_ds:
    st.markdown("**Data Scientist**")
    st.markdown(
        "Upload your data on **Data Onboarding**, run a backtest, "
        "then compare models on **Backtest Results**. The platform "
        "auto-selects the best model per series and shows FVA analysis."
    )
    st.caption("Start with: Data Onboarding → Backtest Results")

with col_dp:
    st.markdown("**Demand Planner**")
    st.markdown(
        "Review forecasts on the **Forecast Viewer** with confidence "
        "intervals and actuals overlay. Check which series have "
        "accuracy issues on **Platform Health**."
    )
    st.caption("Start with: Forecast Viewer")

with col_eng:
    st.markdown("**Platform Engineer**")
    st.markdown(
        "Monitor pipeline runs, drift alerts, and compute costs on "
        "**Platform Health**. Each forecast run produces a provenance "
        "manifest for full traceability."
    )
    st.caption("Start with: Platform Health")

# ---------------------------------------------------------------------------
#  Pages overview
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Pages")

st.markdown(
    """
1. **Data Onboarding** — Upload one or more CSVs. The platform auto-detects
   schema, classifies file roles, assesses forecastability, and recommends a
   configuration. Supports multi-file upload with intelligent merge.
2. **Backtest Results** — Model leaderboard ranked by WMAPE, FVA cascade
   showing which model layers add or destroy value, per-series champion map.
3. **Forecast Viewer** — Interactive forecast chart with P10/P90 confidence
   intervals, actuals overlay, and seasonal decomposition with narrative.
4. **Platform Health** — Pipeline manifests with provenance, drift alerts
   with severity breakdown, data quality summary, and compute cost tracking.
    """
)

# ---------------------------------------------------------------------------
#  Glossary
# ---------------------------------------------------------------------------
with st.expander("Glossary — key terms explained"):
    st.markdown(
        """
| Term | Meaning |
|------|---------|
| **S&OP** | Sales & Operations Planning — the business process that aligns demand forecasts with supply plans. |
| **WMAPE** | Weighted Mean Absolute Percentage Error — forecast accuracy metric weighted by volume. Lower is better. |
| **FVA** | Forecast Value-Add — measures whether a model layer improves accuracy over the naive baseline. |
| **Hierarchical reconciliation** | Adjusting forecasts so they add up consistently across hierarchy levels (e.g., store totals match regional totals). |
| **Fan chart** | A forecast chart where the shaded area shows the prediction interval (e.g., P10-P90 = 80% confidence). |
| **Croston / TSB** | Intermittent demand models designed for sparse or lumpy time series where many periods have zero demand. |
| **Walk-forward backtest** | Evaluation method that trains on past data and tests on the next N periods, walking forward through time. |
| **Champion model** | The best-performing model for a given series, selected by backtest accuracy. |
        """
    )

st.markdown(
    "**Need help?** See [QUICKSTART.md]"
    "(https://github.com/chaitu1385/Forecasting-Platform/blob/master/QUICKSTART.md) "
    "for setup instructions."
)

st.divider()
st.caption("Built with Streamlit")
