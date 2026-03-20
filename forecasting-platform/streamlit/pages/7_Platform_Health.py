"""
Page 7 — Platform Health

Pipeline manifests, drift alerts with AI triage, audit log,
data quality summary, and compute cost tracking.
"""

import json
import sys
from datetime import date, timedelta
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
    format_duration,
    ai_available,
    render_ai_unavailable_notice,
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
    parse_errors = 0
    with st.spinner("Loading manifests..."):
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
                parse_errors += 1
                continue

    if parse_errors > 0:
        st.caption(
            f"Loaded {len(manifests)} of {len(manifest_files)} manifests "
            f"({parse_errors} could not be parsed)."
        )

    if manifests:
        manifest_df = pl.DataFrame(manifests)
        st.dataframe(
            polars_to_pandas(manifest_df),
            use_container_width=True,
            column_config={
                "run_id": st.column_config.TextColumn(
                    "Run ID", help="Unique identifier for this pipeline run."
                ),
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
                st.warning(
                    f"Manifest file `{selected_manifest}` not found. "
                    "It may have been moved or deleted."
                )
    else:
        st.info("No valid manifests found. Check that manifest JSON files are not corrupted.")
else:
    st.info(
        "No pipeline manifests found. Manifests are generated automatically "
        "when you run a forecast from the **Data Onboarding** page."
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
        with st.spinner("Detecting drift..."):
            try:
                store = MetricStore(base_path=metrics_dir)
                metrics_df = store.read(run_type="backtest", lob=lob)
                if metrics_df is not None and not metrics_df.is_empty():
                    detector = ForecastDriftDetector()
                    drift_alerts = detector.detect(metrics_df)
                    st.session_state["drift_alerts"] = drift_alerts
                else:
                    st.warning(
                        f"No metrics found for LOB '{lob}'. "
                        f"Run a backtest from the **Data Onboarding** page first."
                    )
            except Exception as e:
                st.error(
                    f"Drift detection error: {e}\n\n"
                    f"Check that the metric store exists at: `{metrics_dir}`"
                )

with tab_upload:
    uploaded_metrics = st.file_uploader(
        "Upload metrics Parquet for drift detection", type=["parquet"], key="drift_upload"
    )
    if uploaded_metrics is not None:
        try:
            import io
            metrics_df = pl.read_parquet(io.BytesIO(uploaded_metrics.getvalue()))
            with st.spinner("Detecting drift..."):
                detector = ForecastDriftDetector()
                drift_alerts = detector.detect(metrics_df)
            st.session_state["drift_alerts"] = drift_alerts
        except Exception as e:
            st.error(f"Failed to process metrics file: {e}")

# Display drift alerts
drift_alerts = drift_alerts or st.session_state.get("drift_alerts", [])

if drift_alerts:
    # Summary metrics
    critical = [a for a in drift_alerts if a.severity.value == "critical"]
    warnings = [a for a in drift_alerts if a.severity.value == "warning"]

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Alerts", len(drift_alerts),
        help="Total number of drift alerts detected across all series.",
    )
    col2.metric(
        "Critical", len(critical),
        help="Alerts indicating significant accuracy degradation that requires attention.",
    )
    col3.metric(
        "Warning", len(warnings),
        help="Alerts indicating minor accuracy changes worth monitoring.",
    )

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

    # Cross-page: view a drifting series in Forecast Viewer
    drifting_series = alert_df["series_id"].unique().to_list()
    selected_drift = st.selectbox(
        "View drifting series in Forecast Viewer",
        drifting_series,
        index=None,
        placeholder="Choose a series...",
    )
    if selected_drift:
        st.session_state["selected_series_id"] = selected_drift
        st.success(
            f"Series `{selected_drift}` selected. Go to **Forecast Viewer** "
            f"in the sidebar to investigate."
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

    # -------------------------------------------------------------------
    #  AI Anomaly Triage
    # -------------------------------------------------------------------
    st.subheader("AI Anomaly Triage")
    st.caption(
        "Rank alerts by business impact and get AI-suggested corrective actions."
    )

    if ai_available():
        if st.button("Triage Alerts with AI", type="primary", key="triage_btn"):
            try:
                from src.ai.anomaly_triage import AnomalyTriageEngine

                engine = AnomalyTriageEngine()
                with st.spinner("Triaging alerts with Claude..."):
                    triage_result = engine.query(
                        lob=lob,
                        drift_alerts=drift_alerts,
                        max_alerts=50,
                    )

                # Executive summary
                st.info(triage_result.executive_summary)

                # Ranked alerts table
                if triage_result.ranked_alerts:
                    triage_data = []
                    for ta in triage_result.ranked_alerts:
                        triage_data.append({
                            "impact_score": ta.business_impact_score,
                            "series_id": ta.series_id,
                            "metric": ta.metric,
                            "severity": ta.severity,
                            "suggested_action": ta.suggested_action,
                            "reasoning": ta.reasoning,
                        })
                    triage_df = pl.DataFrame(triage_data).sort(
                        "impact_score", descending=True
                    )
                    st.dataframe(
                        polars_to_pandas(triage_df),
                        use_container_width=True,
                        column_config={
                            "impact_score": st.column_config.NumberColumn(
                                "Impact Score", format="%.0f",
                                help="Business impact score (0-100). Higher = more urgent.",
                            ),
                        },
                    )

                    # Select triaged series to view in Forecast Viewer
                    triaged_series = triage_df["series_id"].unique().to_list()
                    selected_triage = st.selectbox(
                        "Investigate series in Forecast Viewer",
                        triaged_series,
                        index=None,
                        placeholder="Choose a series...",
                        key="triage_series_select",
                    )
                    if selected_triage:
                        st.session_state["selected_series_id"] = selected_triage
                        st.success(
                            f"Series `{selected_triage}` selected. "
                            f"Go to **Forecast Viewer** to investigate."
                        )

            except Exception as exc:
                st.warning(f"AI triage failed: {exc}")
    else:
        render_ai_unavailable_notice()

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
    col1.metric(
        "Series", f"{schema.n_series:,}",
        help="Number of unique time series in the analyzed data.",
    )
    col2.metric(
        "Rows", f"{schema.n_rows:,}",
        help="Total number of data rows.",
    )
    col3.metric(
        "Forecastability", f"{fa.overall_score:.2f}",
        help="Overall forecastability score (0-1). Higher means easier to forecast accurately.",
    )
    col4.metric(
        "Frequency", schema.frequency_guess,
        help="Detected data frequency: D=daily, W=weekly, M=monthly, Q=quarterly.",
    )

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
        "Go to the **Data Onboarding** page to analyze your data first."
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
                    "duration": format_duration(raw.get("total_seconds", 0)),
                    "total_seconds": raw.get("total_seconds", 0),
                    "series_count": raw.get("input_series_count", 0),
                })
        except Exception:
            continue

    if cost_data:
        cost_df = pl.DataFrame(cost_data)
        st.dataframe(
            polars_to_pandas(cost_df.select(["run_id", "timestamp", "duration", "series_count"])),
            use_container_width=True,
        )

        fig_cost = go.Figure()
        cost_pd = polars_to_pandas(cost_df)
        fig_cost.add_trace(go.Bar(
            x=cost_pd["run_id"],
            y=cost_pd["total_seconds"],
            marker_color=COLORS["primary"],
            text=cost_pd["duration"],
            textposition="auto",
        ))
        fig_cost.update_layout(
            title="Compute Time by Run",
            xaxis_title="Run ID",
            yaxis_title="Seconds",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Cost per series
        if any(d.get("series_count", 0) > 0 for d in cost_data):
            st.subheader("Cost Efficiency")
            efficiency_data = [
                {
                    "run_id": d["run_id"],
                    "seconds_per_series": round(
                        d["total_seconds"] / d["series_count"], 2
                    )
                    if d.get("series_count", 0) > 0
                    else 0,
                }
                for d in cost_data
                if d.get("series_count", 0) > 0
            ]
            if efficiency_data:
                eff_df = pl.DataFrame(efficiency_data)
                eff_pd = polars_to_pandas(eff_df)
                fig_eff = go.Figure()
                fig_eff.add_trace(go.Bar(
                    x=eff_pd["run_id"],
                    y=eff_pd["seconds_per_series"],
                    marker_color=COLORS["accent"],
                    text=[f"{v:.2f}s" for v in eff_pd["seconds_per_series"]],
                    textposition="auto",
                ))
                fig_eff.update_layout(
                    title="Seconds per Series",
                    xaxis_title="Run ID",
                    yaxis_title="Seconds / Series",
                    height=300,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_eff, use_container_width=True)
    else:
        st.info(
            "No compute cost data found in manifests. "
            "Cost tracking requires a `total_seconds` field in the manifest JSON."
        )
else:
    st.info("No pipeline runs found. Cost data is collected from pipeline manifests.")

# ---------------------------------------------------------------------------
#  Audit Log
# ---------------------------------------------------------------------------
st.divider()
st.header("Audit Log")

try:
    from src.audit.logger import AuditLogger

    audit_path = str(DATA_DIR / "audit_log")
    logger = AuditLogger(base_path=audit_path)

    col_action, col_limit = st.columns(2)
    with col_action:
        action_filter = st.text_input(
            "Filter by action", value="", key="audit_action_filter",
            help="Leave blank to show all actions.",
        )
    with col_limit:
        audit_limit = st.number_input(
            "Max entries", min_value=10, max_value=1000, value=100, key="audit_limit"
        )

    if st.button("Load Audit Log", key="load_audit"):
        with st.spinner("Querying audit log..."):
            audit_df = logger.query(
                action=action_filter or None,
                limit=audit_limit,
            )
        if audit_df is not None and not audit_df.is_empty():
            st.dataframe(polars_to_pandas(audit_df), use_container_width=True)
            st.caption(f"Showing {audit_df.shape[0]} entries.")
        else:
            st.info("No audit log entries found.")

except Exception as exc:
    st.info(
        f"Audit log not available: {exc}\n\n"
        "Audit entries are generated when pipeline actions are performed via the API."
    )
