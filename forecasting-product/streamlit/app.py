"""
Forecasting Product — Streamlit Dashboard

Main entry point.  Run with:
    streamlit run forecasting-product/streamlit/app.py

Or via Docker Compose:
    docker compose up
"""

import sys
from pathlib import Path

# Ensure the forecasting-product package is importable.
_PLATFORM_ROOT = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
#  Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forecasting Product",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
#  Landing page
# ---------------------------------------------------------------------------
st.title("Forecasting Product")

st.markdown(
    "Multi-frequency sales forecasting for retail S&OP — statistical, ML, neural, "
    "and foundation models with hierarchical reconciliation and AI-powered insights."
)

st.info(
    "**Start here:** Open **Data Onboarding** in the sidebar, "
    "then click **Use sample data** to see the platform in action."
)

# ---------------------------------------------------------------------------
#  Quick-start by persona
# ---------------------------------------------------------------------------
st.subheader("Quick Start")

col_ds, col_dp, col_sop, col_eng = st.columns(4)

with col_ds:
    st.markdown("**Data Scientist**")
    st.markdown(
        "Upload data on **Data Onboarding**, explore series quality on "
        "**Series Explorer**, run backtests, then compare models on "
        "**Backtest Results** with SHAP and AI config tuning."
    )
    st.caption("Pages 1 → 2 → 5 → 6")

with col_dp:
    st.markdown("**Demand Planner**")
    st.markdown(
        "Review forecasts on the **Forecast Viewer** with confidence "
        "intervals, AI Q&A, and comparison overlays. Manage SKU "
        "transitions and overrides on **SKU Transitions**."
    )
    st.caption("Pages 6 → 3 → 7")

with col_sop:
    st.markdown("**S&OP Leader**")
    st.markdown(
        "Generate executive commentary on **S&OP Meeting Prep** "
        "with AI-powered narratives. Review governance and export "
        "data for BI tools."
    )
    st.caption("Pages 6 → 7 → 8")

with col_eng:
    st.markdown("**Platform Engineer**")
    st.markdown(
        "Monitor pipelines, AI-triage drift alerts, and track compute "
        "costs on **Platform Health**. Review audit logs for full "
        "traceability."
    )
    st.caption("Pages 1 → 7")

# ---------------------------------------------------------------------------
#  Pages overview
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Pages")

st.markdown(
    """
**Data → Understand → Prepare → Structure → Model → Forecast → Monitor → Report**

1. **Data Onboarding** — Upload CSVs, auto-detect schema, assess forecastability,
   preview demand cleansing, screen regressors, and get a recommended config.
2. **Series Explorer** — Deep-dive into series quality: SBC demand classification,
   structural break detection, data quality profiling, cleansing audit, and AI Q&A.
3. **SKU Transitions** — Run SKU mapping pipeline (attribute matching, naming
   conventions), manage planner overrides with approval workflow.
4. **Hierarchy Manager** — Visualize hierarchy trees, explore aggregations,
   run forecast reconciliation (bottom-up, MinT, OLS, WLS).
5. **Backtest Results** — Model leaderboard, FVA cascade, champion map,
   prediction interval calibration, SHAP attribution, and AI config tuning.
6. **Forecast Viewer** — Interactive forecast chart with fan chart, decomposition,
   AI natural-language Q&A, forecast comparison, and constrained forecast toggle.
7. **Platform Health** — Pipeline manifests, drift alerts with AI triage,
   audit log viewer, data quality summary, and compute cost tracking.
8. **S&OP Meeting Prep** — AI-generated executive commentary, cross-run
   comparison, model governance, and BI export for Power BI / Tableau.
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
| **SBC matrix** | Syntetos-Boylan-Croston classification — categorizes demand patterns as Smooth, Intermittent, Erratic, or Lumpy based on ADI and CV-squared. |
| **Structural break** | A sudden level shift or trend change in a time series, detected using CUSUM or PELT algorithms. |
| **MinT reconciliation** | Minimum Trace — optimal reconciliation method using shrinkage covariance estimation. Best for large hierarchies. |
| **SHAP** | SHapley Additive exPlanations — explains ML model predictions by quantifying each feature's contribution. |
| **Conformal prediction** | A calibration technique that adjusts prediction intervals to achieve desired coverage using backtest residuals. |
        """
    )

st.markdown(
    "**Need help?** See [QUICKSTART.md]"
    "(https://github.com/chaitu1385/Forecasting-Platform/blob/master/QUICKSTART.md) "
    "for setup instructions."
)

st.divider()
st.caption("Built with Streamlit")
