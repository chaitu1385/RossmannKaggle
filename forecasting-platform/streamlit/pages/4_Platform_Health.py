"""
Page 4 — Platform Health

Pipeline manifests, drift alerts, data quality summary, and compute cost.
"""

import json
import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
_STREAMLIT_DIR = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))
if str(_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_DIR))

import polars as pl

from src.metrics.drift import DriftConfig, ForecastDriftDetector
from src.metrics.store import MetricStore
from src.pipeline.manifest import PipelineManifest, read_manifest
from utils import (
    COLORS,
    DATA_DIR,
    SEVERITY_COLORS,
    polars_to_pandas,
    format_pct,
    format_number,
)

st.set_page_config(page_title="Platform Health", page_icon="🏥", layout="wide")
st.title("Platform Health")

# ---------------------------------------------------------------------------
#  Pipeline Manifests
# ---------------------------------------------------------------------------
st.header("Recent Pipeline Runs")

manifest_dir = DATA_DIR
manifest_files = sorted(manifest_dir.rglob("*_manifest.json"), reverse=True)[:20]

if manifest_files:
    manifests = []
    for mf in manifest_files:
        try:
            m = read_manifest(str(mf))
            manifests.append({
                "run_id": m.run_id[:12] + "..." if len(m.run_id) > 12 else m.run_id,
                "timestamp": m.timestamp,
                "lob": m.lob,
                "series": m.input_series_count,
                "rows": m.input_row_count,
                "champion": m.champion_model_id,
                "wmape": m.backtest_wmape,
                "horizon": m.forecast_horizon,
                "forecast_rows": m.forecast_row_count,
                "cleansing": m.cleansing_applied,
                "outliers_clipped": m.outliers_clipped,
                "validation_passed": m.validation_passed,
                "file": str(mf.name),
            })
        except Exception:
            continue

    if manifests:
        manifest_df = pl.DataFrame(manifests)
        st.dataframe(
            polars_to_pandas(manifest_df),
            use_container_width=True,
            column_config={
                "wmape": st.column_config.NumberColumn("WMAPE", format="%.4f"),
                "cleansing": st.column_config.CheckboxColumn("Cleansed"),
                "validation_passed": st.column_config.CheckboxColumn("Valid"),
            },
        )

        # Expandable manifest detail
        selected_manifest = st.selectbox(
            "View manifest detail",
            [mf.name for mf in manifest_files],
        )
        if selected_manifest:
            selected_path = next(
                (mf for mf in manifest_files if mf.name == selected_manifest), None
            )
            if selected_path:
                with st.expander("Full manifest JSON"):
                    raw = json.loads(selected_path.read_text())
                    st.json(raw)
    else:
        st.info("No valid manifests found.")
else:
    st.info(
        "No pipeline manifests found. Run `python scripts/run_forecast.py` "
        "to generate forecast manifests."
    )

# ---------------------------------------------------------------------------
#  Drift Alerts
# ---------------------------------------------------------------------------
st.divider()
st.header("Drift Alerts")

# Try to load metrics for drift detection
metrics_dir = str(DATA_DIR / "metrics")
lob = st.text_input("Line of Business", value="retail", key="health_lob")

drift_alerts = []

tab_auto, tab_upload = st.tabs(["From Metric Store", "Upload Metrics"])

with tab_auto:
    if st.button("Detect drift", key="detect_drift"):
        try:
            store = MetricStore(base_path=metrics_dir)
            metrics_df = store.read(run_type="backtest", lob=lob)
            if metrics_df is not None and not metrics_df.is_empty():
                detector = ForecastDriftDetector()
                drift_alerts = detector.detect(metrics_df)
                st.session_state["drift_alerts"] = drift_alerts
            else:
                st.warning("No metrics found for drift detection.")
        except Exception as e:
            st.error(f"Drift detection error: {e}")

with tab_upload:
    uploaded_metrics = st.file_uploader(
        "Upload metrics Parquet for drift detection", type=["parquet"], key="drift_upload"
    )
    if uploaded_metrics is not None:
        import io
        metrics_df = pl.read_parquet(io.BytesIO(uploaded_metrics.getvalue()))
        detector = ForecastDriftDetector()
        drift_alerts = detector.detect(metrics_df)
        st.session_state["drift_alerts"] = drift_alerts

# Display drift alerts
drift_alerts = drift_alerts or st.session_state.get("drift_alerts", [])

if drift_alerts:
    # Summary metrics
    critical = [a for a in drift_alerts if a.severity.value == "critical"]
    warnings = [a for a in drift_alerts if a.severity.value == "warning"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", len(drift_alerts))
    col2.metric("Critical", len(critical))
    col3.metric("Warning", len(warnings))

    # Alert table
    alert_data = []
    for a in drift_alerts:
        alert_data.append({
            "severity": a.severity.value,
            "series_id": a.series_id,
            "metric": a.metric,
            "current": round(a.current_value, 4),
            "baseline": round(a.baseline_value, 4),
            "message": a.message,
        })

    alert_df = pl.DataFrame(alert_data)
    alert_pd = polars_to_pandas(alert_df)

    def style_severity(val):
        if val == "critical":
            return f"background-color: {SEVERITY_COLORS['critical']}30; color: {SEVERITY_COLORS['critical']}"
        if val == "warning":
            return f"background-color: {SEVERITY_COLORS['warning']}30; color: #856404"
        return ""

    st.dataframe(
        alert_pd.style.map(style_severity, subset=["severity"]),
        use_container_width=True,
    )

    # Alert distribution by metric type
    fig_alert = px.histogram(
        alert_pd,
        x="metric",
        color="severity",
        color_discrete_map={"critical": SEVERITY_COLORS["critical"],
                           "warning": SEVERITY_COLORS["warning"]},
        barmode="group",
        title="Alerts by Metric Type",
    )
    fig_alert.update_layout(height=300, margin=dict(t=40, b=40))
    st.plotly_chart(fig_alert, use_container_width=True)
elif st.session_state.get("drift_alerts") is not None:
    st.success("No drift alerts detected.")
else:
    st.info("Click 'Detect drift' or upload metrics to run drift detection.")

# ---------------------------------------------------------------------------
#  Data Quality Summary
# ---------------------------------------------------------------------------
st.divider()
st.header("Data Quality Summary")

# Pull from analysis report if available
report = st.session_state.get("analysis_report")

if report is not None:
    schema = report.schema
    fa = report.forecastability

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Series", f"{schema.n_series:,}")
    col2.metric("Rows", f"{schema.n_rows:,}")
    col3.metric("Forecastability", f"{fa.overall_score:.2f}")
    col4.metric("Frequency", schema.frequency_guess)

    # Score distribution
    dist = fa.score_distribution
    fig_qual = go.Figure()
    cats = ["high", "medium", "low"]
    cols = [COLORS["success"], COLORS["warning"], COLORS["danger"]]
    vals = [dist.get(c, 0) for c in cats]

    fig_qual.add_trace(go.Bar(
        x=cats,
        y=vals,
        marker_color=cols,
        text=vals,
        textposition="auto",
    ))
    fig_qual.update_layout(
        title="Forecastability Distribution",
        height=250,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_qual, use_container_width=True)

    if report.warnings:
        st.subheader("Data Warnings")
        for w in report.warnings:
            st.warning(w)
else:
    st.info(
        "Data quality info is populated after running Data Onboarding. "
        "Go to the Data Onboarding page to analyze your data first."
    )

# ---------------------------------------------------------------------------
#  Compute Cost Summary
# ---------------------------------------------------------------------------
st.divider()
st.header("Compute Cost")

# Check manifests for cost data
if manifest_files:
    cost_data = []
    for mf in manifest_files:
        try:
            raw = json.loads(mf.read_text())
            if "total_seconds" in raw or "model_seconds" in raw:
                cost_data.append({
                    "run_id": raw.get("run_id", "")[:12],
                    "timestamp": raw.get("timestamp", ""),
                    "total_seconds": raw.get("total_seconds", 0),
                    "series_count": raw.get("input_series_count", 0),
                })
        except Exception:
            continue

    if cost_data:
        cost_df = pl.DataFrame(cost_data)
        st.dataframe(polars_to_pandas(cost_df), use_container_width=True)

        fig_cost = go.Figure()
        cost_pd = polars_to_pandas(cost_df)
        fig_cost.add_trace(go.Bar(
            x=cost_pd["run_id"],
            y=cost_pd["total_seconds"],
            marker_color=COLORS["primary"],
        ))
        fig_cost.update_layout(
            title="Compute Time by Run",
            xaxis_title="Run ID",
            yaxis_title="Seconds",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("No compute cost data found in manifests.")
else:
    st.info("No pipeline runs found. Cost data is collected from pipeline manifests.")
