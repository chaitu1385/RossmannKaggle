"""
Page 1 — Data Onboarding

Upload one or more CSV files → auto-classify roles (time series, dimension,
regressor) → preview merge → run DataAnalyzer → view schema detection,
hierarchy, forecastability scores, hypotheses, and recommended config.
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
_STREAMLIT_DIR = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))
if str(_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_DIR))

from src.analytics.analyzer import DataAnalyzer
from src.data.file_classifier import FileClassifier
from src.data.file_merger import MultiFileMerger
from utils import (
    COLORS,
    load_sample_data,
    load_uploaded_csv,
    load_uploaded_csvs,
    polars_to_pandas,
    format_pct,
)

st.set_page_config(page_title="Data Onboarding", page_icon="📊", layout="wide")
st.title("Data Onboarding")
st.markdown(
    "Upload one or more CSVs to auto-detect schema, assess forecastability, "
    "and get a recommended configuration. Multiple files are automatically "
    "classified and merged."
)

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
_ROLE_LABELS = {
    "time_series": "Time Series (primary)",
    "dimension": "Dimension / Lookup",
    "regressor": "External Regressor",
    "unknown": "Unknown",
}
_ROLE_OPTIONS = list(_ROLE_LABELS.keys())
_CONFIDENCE_COLORS = {
    "high": "🟢",
    "medium": "🟡",
    "low": "🔴",
}


def _confidence_badge(score: float) -> str:
    if score >= 0.7:
        return f"{_CONFIDENCE_COLORS['high']} {score:.0%}"
    elif score >= 0.4:
        return f"{_CONFIDENCE_COLORS['medium']} {score:.0%}"
    return f"{_CONFIDENCE_COLORS['low']} {score:.0%}"


# --------------------------------------------------------------------------- #
#  Data source selection
# --------------------------------------------------------------------------- #
col_upload, col_sample = st.columns(2)

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

with col_sample:
    use_sample = st.button("Use sample data")

# --------------------------------------------------------------------------- #
#  Single-file fast path (backward compatible)
# --------------------------------------------------------------------------- #
df = None
multi_file_mode = False

if uploaded_files and len(uploaded_files) == 1:
    # Single file — original flow
    df = load_uploaded_csv(uploaded_files[0])
    st.success(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from upload.")

elif uploaded_files and len(uploaded_files) > 1:
    multi_file_mode = True

elif use_sample or st.session_state.get("sample_loaded"):
    df = load_sample_data()
    st.session_state["sample_loaded"] = True
    st.success(f"Loaded sample data: {df.shape[0]:,} rows x {df.shape[1]} columns.")

# --------------------------------------------------------------------------- #
#  Multi-file flow
# --------------------------------------------------------------------------- #
if multi_file_mode:
    # Step 1: Load and classify
    files = load_uploaded_csvs(uploaded_files)
    st.success(f"Loaded {len(files)} files: {', '.join(files.keys())}")

    classifier = FileClassifier()
    classification = classifier.classify_files(files)

    # Store in session for persistence across reruns
    st.session_state["classification"] = classification

    # Show classification results
    st.divider()
    st.header("Step 1: File Classification")

    if classification.warnings:
        for w in classification.warnings:
            st.warning(w)

    if classification.primary_file is None:
        st.error(
            "No file was classified as a time series. At least one file must "
            "contain a **date column**, a **numeric target** (e.g. 'quantity', "
            "'sales'), and **identifier columns** (e.g. 'store_id')."
        )
        st.stop()

    # Interactive confirmation table
    st.markdown("Review the auto-detected roles below. Override if needed.")

    role_overrides = {}
    for i, profile in enumerate(classification.profiles):
        col_name, col_role, col_conf, col_keys = st.columns([2, 2, 1, 3])
        with col_name:
            st.markdown(f"**{profile.filename}**")
            st.caption(f"{profile.n_rows:,} rows, {profile.n_columns} cols")
        with col_role:
            selected = st.selectbox(
                "Role",
                options=_ROLE_OPTIONS,
                index=_ROLE_OPTIONS.index(profile.role),
                format_func=lambda x: _ROLE_LABELS[x],
                key=f"role_{i}",
                label_visibility="collapsed",
            )
            role_overrides[profile.filename] = selected
        with col_conf:
            st.markdown(_confidence_badge(profile.confidence))
        with col_keys:
            key_cols = profile.id_columns[:3]
            time_info = f"time: {profile.time_column}" if profile.time_column else "no time col"
            st.caption(f"{time_info} | IDs: {', '.join(key_cols) if key_cols else 'none'}")

    # Apply overrides
    for profile in classification.profiles:
        override = role_overrides.get(profile.filename)
        if override and override != profile.role:
            profile.role = override

    # Re-bucket after overrides
    classification.primary_file = next(
        (p for p in classification.profiles if p.role == "time_series"), None,
    )
    classification.dimension_files = [
        p for p in classification.profiles if p.role == "dimension"
    ]
    classification.regressor_files = [
        p for p in classification.profiles if p.role == "regressor"
    ]
    classification.unknown_files = [
        p for p in classification.profiles if p.role == "unknown"
    ]

    if classification.primary_file is None:
        st.error("No file assigned the **Time Series** role. Please select one.")
        st.stop()

    # Step 2: Merge preview
    st.divider()
    st.header("Step 2: Merge Preview")

    merger = MultiFileMerger()
    preview = merger.preview_merge(classification)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Merged rows", f"{preview.total_rows:,}")
    col_s2.metric("Merged columns", preview.total_columns)
    col_s3.metric("Matched rows", f"{preview.matched_rows:,}")
    col_s4.metric("Unmatched keys", preview.unmatched_primary_keys)

    if preview.column_name_conflicts:
        with st.expander("Column name conflicts resolved"):
            for c in preview.column_name_conflicts:
                st.markdown(f"- {c}")

    if preview.null_fill_columns:
        st.caption(f"Null-filled regressor columns: {', '.join(preview.null_fill_columns)}")

    if preview.warnings:
        for w in preview.warnings:
            st.warning(w)

    st.markdown("**Sample of merged data:**")
    st.dataframe(polars_to_pandas(preview.sample_rows), use_container_width=True)

    # Step 3: Merge and analyze
    st.divider()
    if st.button("Run Analysis on Merged Data", type="primary"):
        merge_result = merger.merge(classification)
        df = merge_result.df
        st.session_state["merge_result"] = merge_result
        st.success(
            f"Merged {len(files)} files into {df.shape[0]:,} rows x "
            f"{df.shape[1]} columns."
        )
    elif "merge_result" in st.session_state:
        df = st.session_state["merge_result"].df
    else:
        st.info("Click **Run Analysis on Merged Data** to proceed.")
        st.stop()


if df is None:
    st.info("Upload a CSV or click **Use sample data** to get started.")
    st.stop()

# --------------------------------------------------------------------------- #
#  Preview raw data
# --------------------------------------------------------------------------- #
with st.expander("Preview raw data", expanded=False):
    st.dataframe(polars_to_pandas(df.head(500)), use_container_width=True)

# --------------------------------------------------------------------------- #
#  Run DataAnalyzer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Analyzing data...")
def run_analysis(_df):
    """Run DataAnalyzer on the uploaded DataFrame (cached)."""
    analyzer = DataAnalyzer(lob_name="uploaded")
    report = analyzer.analyze(_df)
    return report


report = run_analysis(df)

# Enrich report with multi-source info if available
if multi_file_mode and "classification" in st.session_state:
    cls = st.session_state["classification"]
    report.dimension_sources = [p.filename for p in cls.dimension_files]
    report.regressor_columns = [
        c for p in cls.regressor_files for c in p.numeric_columns
    ]

st.divider()

# --------------------------------------------------------------------------- #
#  Schema Detection
# --------------------------------------------------------------------------- #
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
    if report.dimension_sources:
        st.markdown("**Dimension sources**")
        for s in report.dimension_sources:
            st.markdown(f"- `{s}`")

# --------------------------------------------------------------------------- #
#  Hierarchy Detection
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
#  Forecastability Assessment
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
#  Hypotheses & Warnings
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
#  Recommended Configuration
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
#  LLM Interpreter (optional)
# --------------------------------------------------------------------------- #
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
