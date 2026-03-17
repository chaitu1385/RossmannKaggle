"""
Page 1 — Data Onboarding

Upload CSV → run DataAnalyzer → view schema detection, hierarchy,
forecastability scores, hypotheses, and recommended config.
Optionally invoke LLM interpreter for executive-level narrative.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

_PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))

from src.analytics.analyzer import DataAnalyzer
from streamlit.utils import (
    COLORS,
    load_sample_data,
    load_uploaded_csv,
    polars_to_pandas,
    format_pct,
)

st.set_page_config(page_title="Data Onboarding", page_icon="📊", layout="wide")
st.title("Data Onboarding")
st.markdown("Upload a CSV to auto-detect schema, assess forecastability, and get a recommended configuration.")

# ---------------------------------------------------------------------------
#  Data source selection
# ---------------------------------------------------------------------------
col_upload, col_sample = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

with col_sample:
    use_sample = st.button("Use Rossmann sample data")

df = None
if uploaded_file is not None:
    df = load_uploaded_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from upload.")
elif use_sample or st.session_state.get("sample_loaded"):
    df = load_sample_data()
    if df is not None:
        st.session_state["sample_loaded"] = True
        st.success(f"Loaded Rossmann sample: {df.shape[0]:,} rows × {df.shape[1]} columns.")
    else:
        st.warning("Sample data not found. Place Rossmann train.csv in data/rossmann/.")

if df is None:
    st.info("Upload a CSV or load the sample data to get started.")
    st.stop()

# ---------------------------------------------------------------------------
#  Preview raw data
# ---------------------------------------------------------------------------
with st.expander("Preview raw data", expanded=False):
    st.dataframe(polars_to_pandas(df.head(500)), use_container_width=True)

# ---------------------------------------------------------------------------
#  Run DataAnalyzer
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Analyzing data...")
def run_analysis(_df):
    """Run DataAnalyzer on the uploaded DataFrame (cached)."""
    analyzer = DataAnalyzer(lob_name="uploaded")
    report = analyzer.analyze(_df)
    return report


report = run_analysis(df)

st.divider()

# ---------------------------------------------------------------------------
#  Schema Detection
# ---------------------------------------------------------------------------
st.header("Schema Detection")

schema = report.schema
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{schema.n_rows:,}")
col2.metric("Series", f"{schema.n_series:,}")
col3.metric("Frequency", schema.frequency_guess)
col4.metric("Confidence", format_pct(schema.confidence))

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Detected columns**")
    st.markdown(f"- Time: `{schema.time_column}`")
    st.markdown(f"- Target: `{schema.target_column}`")
    st.markdown(f"- ID columns: `{', '.join(schema.id_columns)}`")
    st.markdown(f"- Date range: `{schema.date_range[0]}` → `{schema.date_range[1]}`")

with col_b:
    if schema.dimension_columns:
        st.markdown("**Dimension columns**")
        for c in schema.dimension_columns:
            st.markdown(f"- `{c}`")
    if schema.numeric_columns:
        st.markdown("**Numeric columns** (regressor candidates)")
        for c in schema.numeric_columns:
            st.markdown(f"- `{c}`")

# ---------------------------------------------------------------------------
#  Hierarchy Detection
# ---------------------------------------------------------------------------
st.divider()
st.header("Hierarchy Detection")

hier = report.hierarchy
if hier.hierarchies:
    for h in hier.hierarchies:
        st.markdown(f"**{h.name}**: {' → '.join(h.levels)}")
    if hier.reasoning:
        with st.expander("Reasoning"):
            for r in hier.reasoning:
                st.markdown(f"- {r}")
else:
    st.info("No hierarchical structure detected.")

if hier.warnings:
    for w in hier.warnings:
        st.warning(w)

# ---------------------------------------------------------------------------
#  Forecastability Assessment
# ---------------------------------------------------------------------------
st.divider()
st.header("Forecastability Assessment")

fa = report.forecastability

col1, col2 = st.columns([1, 2])

with col1:
    # Overall score gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fa.overall_score,
        number={"suffix": "", "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 0.3], "color": "#fee2e2"},
                {"range": [0.3, 0.6], "color": "#fef9c3"},
                {"range": [0.6, 1.0], "color": "#dcfce7"},
            ],
        },
        title={"text": "Overall Forecastability"},
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=60, b=0, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    # Score distribution bar chart
    dist = fa.score_distribution
    fig_dist = go.Figure()
    categories = ["high", "medium", "low"]
    colors_dist = [COLORS["success"], COLORS["warning"], COLORS["danger"]]
    values = [dist.get(c, 0) for c in categories]

    fig_dist.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors_dist,
        text=values,
        textposition="auto",
    ))
    fig_dist.update_layout(
        title="Score Distribution",
        xaxis_title="Category",
        yaxis_title="Number of Series",
        height=250,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Demand class distribution
if fa.demand_class_distribution:
    st.subheader("Demand Classification")
    dcd = fa.demand_class_distribution
    fig_pie = px.pie(
        names=list(dcd.keys()),
        values=list(dcd.values()),
        color_discrete_sequence=[COLORS["primary"], COLORS["accent"],
                                 COLORS["warning"], COLORS["neutral"]],
        hole=0.4,
    )
    fig_pie.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

# Per-series detail table
if fa.per_series is not None and not fa.per_series.is_empty():
    with st.expander("Per-series forecastability detail"):
        st.dataframe(polars_to_pandas(fa.per_series), use_container_width=True)

# ---------------------------------------------------------------------------
#  Hypotheses & Warnings
# ---------------------------------------------------------------------------
st.divider()
st.header("Insights")

col_hyp, col_warn = st.columns(2)
with col_hyp:
    st.subheader("Hypotheses")
    for h in report.hypotheses:
        st.markdown(f"- {h}")

with col_warn:
    st.subheader("Warnings")
    if report.warnings:
        for w in report.warnings:
            st.warning(w)
    else:
        st.success("No warnings.")

# ---------------------------------------------------------------------------
#  Recommended Configuration
# ---------------------------------------------------------------------------
st.divider()
st.header("Recommended Configuration")

config = report.recommended_config

# Config reasoning
with st.expander("Why these settings?"):
    for r in report.config_reasoning:
        st.markdown(f"- {r}")

# Key config highlights
col_c1, col_c2, col_c3 = st.columns(3)
col_c1.metric("Models", len(config.forecast.forecasters))
col_c2.metric("Backtest folds", config.backtest.n_folds)
col_c3.metric("Horizon", f"{config.forecast.horizon_weeks} periods")

st.markdown(f"**Forecasters**: {', '.join(config.forecast.forecasters)}")
if config.forecast.intermittent_forecasters:
    st.markdown(
        f"**Intermittent models**: {', '.join(config.forecast.intermittent_forecasters)}"
    )

# YAML download
config_dict = asdict(config)
config_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

st.code(config_yaml, language="yaml")

col_dl, col_accept = st.columns(2)
with col_dl:
    st.download_button(
        label="Download Config YAML",
        data=config_yaml,
        file_name="platform_config.yaml",
        mime="text/yaml",
    )
with col_accept:
    if st.button("Accept Config"):
        st.session_state["accepted_config"] = config
        st.session_state["analysis_report"] = report
        st.success("Config accepted and stored in session.")

# ---------------------------------------------------------------------------
#  LLM Interpreter (optional)
# ---------------------------------------------------------------------------
st.divider()
st.header("AI Interpretation (optional)")
st.markdown(
    "Generate an executive-level narrative using Claude. "
    "Requires `ANTHROPIC_API_KEY` environment variable."
)

if st.button("Generate AI Interpretation"):
    try:
        from src.analytics.llm_analyzer import LLMAnalyzer

        llm = LLMAnalyzer()
        if not llm.available:
            st.warning(
                "LLM not available. Set the `ANTHROPIC_API_KEY` environment variable "
                "and install the `anthropic` package."
            )
        else:
            with st.spinner("Calling Claude..."):
                insight = llm.interpret(report)

            st.subheader("Narrative")
            st.markdown(insight.narrative)

            if insight.hypotheses:
                st.subheader("Key Hypotheses")
                for h in insight.hypotheses:
                    st.markdown(f"- {h}")

            if insight.model_rationale:
                st.subheader("Model Rationale")
                st.markdown(insight.model_rationale)

            if insight.risk_factors:
                st.subheader("Risk Factors")
                for r in insight.risk_factors:
                    st.warning(r)

            if insight.config_adjustments:
                st.subheader("Suggested Adjustments")
                for a in insight.config_adjustments:
                    st.info(a)
    except ImportError:
        st.warning("LLM module not available. Install `anthropic` package.")
