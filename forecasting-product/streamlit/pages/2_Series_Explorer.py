"""
Page 2 — Series Explorer

Deep-dive into series-level data quality, demand classification,
structural breaks, cleansing audit, and regressor screening.
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

from utils import (
    COLORS,
    DEMAND_CLASS_COLORS,
    CONFIDENCE_BADGE_COLORS,
    polars_to_pandas,
    format_pct,
    format_number,
    ai_available,
    render_ai_unavailable_notice,
    render_ai_confidence_badge,
    render_api_key_sidebar,
    load_uploaded_csv,
)

st.set_page_config(page_title="Series Explorer", page_icon="🔍", layout="wide")
render_api_key_sidebar()
st.title("Series Explorer")
st.markdown(
    "Deep-dive into series-level data quality, demand classification, "
    "structural breaks, cleansing, and regressor screening."
)

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

_TIME_COL_CANDIDATES = ["week", "date", "ds", "time", "timestamp", "period"]
_TARGET_COL_CANDIDATES = ["quantity", "sales", "demand", "value", "y", "target"]
_ID_COL_CANDIDATES = ["series_id", "store_id", "product_id", "sku", "item_id", "id"]


def _detect_col(df: pl.DataFrame, candidates: list, fallback_dtype=None) -> str | None:
    """Detect the first matching column from a list of candidates."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # Fallback: pick first column of the expected dtype
    if fallback_dtype is not None:
        for c in df.columns:
            if df[c].dtype == fallback_dtype:
                return c
    return None


def _build_series_id_col(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Ensure a 'series_id' column exists; create from id_col if needed."""
    if "series_id" in df.columns:
        return df
    if id_col and id_col in df.columns:
        return df.with_columns(pl.col(id_col).cast(pl.Utf8).alias("series_id"))
    # Combine all string columns as a composite key
    str_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    if str_cols:
        return df.with_columns(
            pl.concat_str(str_cols, separator="_").alias("series_id")
        )
    return df.with_columns(pl.lit("all").alias("series_id"))


# --------------------------------------------------------------------------- #
#  Section 1: Data Loading
# --------------------------------------------------------------------------- #
st.divider()
st.header("Data Loading")

df = st.session_state.get("actuals_df")

if df is None:
    st.info(
        "No data found in session. Upload a CSV below, or go to "
        "**Data Onboarding** to load data first."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="series_explorer_upload")
    if uploaded is not None:
        try:
            df = load_uploaded_csv(uploaded)
            st.session_state["actuals_df"] = df
            st.success(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.")
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")
            st.stop()

if df is None:
    st.stop()

# Detect columns
time_col = _detect_col(df, _TIME_COL_CANDIDATES, fallback_dtype=pl.Date)
target_col = _detect_col(df, _TARGET_COL_CANDIDATES, fallback_dtype=pl.Float64)
id_col = _detect_col(df, _ID_COL_CANDIDATES, fallback_dtype=pl.Utf8)

if time_col is None:
    st.warning("Could not auto-detect a time/date column. Check your data.")
    time_col = st.selectbox("Select time column", options=df.columns, key="time_col_sel")
if target_col is None:
    st.warning("Could not auto-detect a target/quantity column.")
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    target_col = st.selectbox("Select target column", options=numeric_cols or df.columns, key="target_col_sel")
if id_col is None:
    st.warning("Could not auto-detect an ID column.")
    id_col = st.selectbox("Select ID column", options=df.columns, key="id_col_sel")

st.caption(f"Detected columns — time: `{time_col}`, target: `{target_col}`, id: `{id_col}`")

# Ensure series_id exists for backend modules
work_df = _build_series_id_col(df, id_col)
sid_col = "series_id"

# --------------------------------------------------------------------------- #
#  Section 2: Series Overview + SBC Classification
# --------------------------------------------------------------------------- #
st.divider()
st.header("Series Overview & Demand Classification")

series_list = sorted(work_df[sid_col].unique().to_list())
selected_series = st.selectbox(
    "Select a series to inspect",
    options=series_list,
    key="series_picker",
)

# Metric cards for selected series
series_df = work_df.filter(pl.col(sid_col) == selected_series)
n_periods = len(series_df)
mean_demand = series_df[target_col].mean() if target_col in series_df.columns else 0
zero_pct = (
    (series_df[target_col] == 0).sum() / max(n_periods, 1)
    if target_col in series_df.columns
    else 0
)

# Run SBC classification
try:
    from src.series.sparse_detector import SparseDetector

    detector = SparseDetector()
    classification_df = detector.classify(work_df, target_col=target_col, id_col=sid_col)
    st.session_state["sparse_classification"] = classification_df

    # Get demand class for selected series
    sel_class_row = classification_df.filter(pl.col(sid_col) == selected_series)
    demand_class = sel_class_row["demand_class"][0] if len(sel_class_row) > 0 else "Unknown"
except Exception as exc:
    classification_df = None
    demand_class = "N/A"
    st.warning(f"SBC classification unavailable: {exc}")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Length (periods)", f"{n_periods:,}")
col_m2.metric("Mean Demand", format_number(mean_demand) if mean_demand is not None else "N/A")
col_m3.metric("Zero %", format_pct(zero_pct) if zero_pct is not None else "N/A")
col_m4.metric("Demand Class", demand_class)

# SBC scatter plot
if classification_df is not None:
    st.subheader("SBC Demand Classification Scatter")
    try:
        cls_pd = polars_to_pandas(classification_df)

        fig_sbc = px.scatter(
            cls_pd,
            x="adi",
            y="cv2",
            color="demand_class",
            color_discrete_map=DEMAND_CLASS_COLORS,
            hover_data=[sid_col],
            labels={"adi": "ADI (Average Demand Interval)", "cv2": "CV\u00b2 (Coefficient of Variation\u00b2)"},
        )
        # Quadrant boundary lines
        fig_sbc.add_vline(x=1.32, line_dash="dash", line_color=COLORS["neutral"], annotation_text="ADI=1.32")
        fig_sbc.add_hline(y=0.49, line_dash="dash", line_color=COLORS["neutral"], annotation_text="CV\u00b2=0.49")
        fig_sbc.update_layout(
            height=450,
            margin=dict(t=30, b=40, l=40, r=20),
            legend_title_text="Demand Class",
        )
        st.plotly_chart(fig_sbc, use_container_width=True)

        # Summary counts
        summary = classification_df.group_by("demand_class").agg(pl.count().alias("count"))
        st.dataframe(polars_to_pandas(summary), use_container_width=True)

    except Exception as exc:
        st.error(f"Failed to render SBC scatter: {exc}")


# --------------------------------------------------------------------------- #
#  Section 3: Structural Break Detection
# --------------------------------------------------------------------------- #
st.divider()
st.header("Structural Break Detection")

try:
    from src.series.break_detector import StructuralBreakDetector
    from src.config.schema import StructuralBreakConfig

    # Sidebar controls
    with st.sidebar:
        st.subheader("Break Detection Settings")
        break_method = st.selectbox(
            "Method",
            options=["cusum", "pelt"],
            index=0,
            key="break_method",
            help="CUSUM is zero-dependency. PELT requires the `ruptures` package.",
        )
        break_penalty = st.slider(
            "Penalty",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            key="break_penalty",
            help="Higher penalty = fewer detected breaks.",
        )

    break_config = StructuralBreakConfig(
        enabled=True,
        method=break_method,
        penalty=break_penalty,
    )
    break_detector = StructuralBreakDetector(break_config)

    with st.spinner("Detecting structural breaks..."):
        break_report = break_detector.detect(
            work_df,
            target_col=target_col,
            time_col=time_col,
            id_col=sid_col,
        )

    # Summary metrics
    col_b1, col_b2 = st.columns(2)
    col_b1.metric("Total Breaks", break_report.total_breaks)
    col_b2.metric("Series with Breaks", break_report.series_with_breaks)

    # Time series chart for selected series with break markers
    series_ts = series_df.sort(time_col)
    fig_breaks = go.Figure()
    fig_breaks.add_trace(go.Scatter(
        x=polars_to_pandas(series_ts)[time_col],
        y=polars_to_pandas(series_ts)[target_col],
        mode="lines",
        name="Actual",
        line=dict(color=COLORS["primary"]),
    ))

    # Add vertical lines for break dates
    if break_report.per_series is not None and not break_report.per_series.is_empty():
        sel_breaks = break_report.per_series.filter(pl.col(sid_col) == selected_series)
        if len(sel_breaks) > 0 and "break_dates" in sel_breaks.columns:
            break_dates_val = sel_breaks["break_dates"][0]
            if break_dates_val is not None:
                # break_dates may be a list or comma-separated string
                if isinstance(break_dates_val, list):
                    dates_list = break_dates_val
                elif isinstance(break_dates_val, str):
                    dates_list = [d.strip() for d in break_dates_val.split(",") if d.strip()]
                else:
                    dates_list = []
                for bd in dates_list:
                    fig_breaks.add_vline(
                        x=str(bd),
                        line_dash="dash",
                        line_color=COLORS["danger"],
                        annotation_text="Break",
                    )

    fig_breaks.update_layout(
        title=f"Series: {selected_series} — Structural Breaks",
        xaxis_title="Date",
        yaxis_title=target_col,
        height=400,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    st.plotly_chart(fig_breaks, use_container_width=True)

    # Per-series detail
    if break_report.per_series is not None and not break_report.per_series.is_empty():
        with st.expander("Per-series break detail"):
            st.dataframe(polars_to_pandas(break_report.per_series), use_container_width=True)

    if break_report.warnings:
        for w in break_report.warnings:
            st.warning(w)

except ImportError as exc:
    st.info(
        f"Structural break detection requires additional dependencies: {exc}. "
        "Install the `ruptures` package for PELT, or the CUSUM method "
        "should work out of the box."
    )
except Exception as exc:
    st.warning(f"Structural break detection failed: {exc}")


# --------------------------------------------------------------------------- #
#  Section 4: Data Quality Deep Dive
# --------------------------------------------------------------------------- #
st.divider()
st.header("Data Quality Deep Dive")

try:
    from src.data.quality_report import DataQualityAnalyzer
    from src.config.schema import PlatformConfig

    # Try to use accepted config from session, otherwise create a minimal one
    platform_config = st.session_state.get("accepted_config")
    if platform_config is None:
        platform_config = PlatformConfig()

    analyzer = DataQualityAnalyzer(platform_config)
    quality_report = analyzer.analyze(
        work_df,
        time_col=time_col,
        value_col=target_col,
        sid_col=sid_col,
        cleansing_report=None,
        break_report=break_report if "break_report" in dir() else None,
    )

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Total Series", f"{quality_report.total_series:,}")
    col_q2.metric("Missing %", format_pct(quality_report.missing_week_pct))
    col_q3.metric("Zero Inflation", format_pct(quality_report.zero_inflation_rate))
    col_q4.metric("Short Series", f"{quality_report.short_series_count:,}")

    if quality_report.warnings:
        st.subheader("Warnings")
        for w in quality_report.warnings:
            st.warning(w)
    else:
        st.success("No data quality warnings.")

    if quality_report.per_series is not None and not quality_report.per_series.is_empty():
        with st.expander("Per-series quality detail"):
            st.dataframe(polars_to_pandas(quality_report.per_series), use_container_width=True)

except Exception:
    # Fallback: show basic stats from sparse classification
    st.caption("Full quality analysis unavailable. Showing basic stats from classification.")
    if classification_df is not None:
        n_series = len(classification_df)
        zero_series = (
            classification_df.filter(pl.col("demand_class") == "insufficient_data").shape[0]
            if "demand_class" in classification_df.columns
            else 0
        )
        mean_cv = (
            classification_df["cv2"].mean()
            if "cv2" in classification_df.columns
            else 0
        )
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Total Series", f"{n_series:,}")
        col_f2.metric("Insufficient Data", f"{zero_series:,}")
        col_f3.metric("Mean CV\u00b2", format_number(mean_cv) if mean_cv is not None else "N/A")
    else:
        st.info("No classification data available for quality summary.")


# --------------------------------------------------------------------------- #
#  Section 5: Demand Cleansing Audit
# --------------------------------------------------------------------------- #
st.divider()
st.header("Demand Cleansing Audit")

try:
    from src.data.cleanser import DemandCleanser
    from src.config.schema import CleansingConfig

    with st.sidebar:
        st.subheader("Cleansing Settings")
        cleansing_method = st.selectbox(
            "Outlier Method",
            options=["iqr", "zscore"],
            index=0,
            key="cleansing_method",
        )
        iqr_mult = st.slider(
            "IQR Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            key="iqr_multiplier",
            help="Higher = fewer outliers flagged. Only applies to IQR method.",
        )

    cleansing_config = CleansingConfig(
        enabled=True,
        outlier_method=cleansing_method,
        iqr_multiplier=iqr_mult,
    )
    cleanser = DemandCleanser(cleansing_config)

    with st.spinner("Running demand cleansing..."):
        cleansing_result = cleanser.cleanse(
            work_df,
            time_col=time_col,
            value_col=target_col,
            sid_col=sid_col,
        )

    report = cleansing_result.report

    # Summary metrics
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Outliers Clipped", f"{report.total_outliers:,}")
    col_c2.metric("Stockout Periods Imputed", f"{report.total_stockout_periods:,}")
    col_c3.metric("Rows Modified", f"{report.rows_modified:,}")

    # Before/after overlay chart
    st.subheader(f"Before / After — {selected_series}")
    original_series = work_df.filter(pl.col(sid_col) == selected_series).sort(time_col)
    cleaned_series = cleansing_result.df.filter(pl.col(sid_col) == selected_series).sort(time_col)

    orig_pd = polars_to_pandas(original_series)
    clean_pd = polars_to_pandas(cleaned_series)

    fig_cleanse = go.Figure()
    fig_cleanse.add_trace(go.Scatter(
        x=orig_pd[time_col],
        y=orig_pd[target_col],
        mode="lines",
        name="Original",
        line=dict(color=COLORS["neutral"], dash="dot"),
    ))
    fig_cleanse.add_trace(go.Scatter(
        x=clean_pd[time_col],
        y=clean_pd[target_col],
        mode="lines",
        name="Cleaned",
        line=dict(color=COLORS["primary"]),
    ))

    # Highlight outlier points (where values differ)
    if len(orig_pd) == len(clean_pd):
        import numpy as np
        diff_mask = np.array(orig_pd[target_col]) != np.array(clean_pd[target_col])
        if diff_mask.any():
            outlier_pts = orig_pd[diff_mask]
            fig_cleanse.add_trace(go.Scatter(
                x=outlier_pts[time_col],
                y=outlier_pts[target_col],
                mode="markers",
                name="Outlier / Modified",
                marker=dict(color=COLORS["danger"], size=8, symbol="x"),
            ))

    fig_cleanse.update_layout(
        xaxis_title="Date",
        yaxis_title=target_col,
        height=400,
        margin=dict(t=30, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cleanse, use_container_width=True)

    # Per-series cleansing detail
    if report.per_series is not None and not report.per_series.is_empty():
        with st.expander("Per-series cleansing detail"):
            st.dataframe(polars_to_pandas(report.per_series), use_container_width=True)

except Exception as exc:
    st.warning(f"Demand cleansing audit unavailable: {exc}")


# --------------------------------------------------------------------------- #
#  Section 6: Regressor Screening
# --------------------------------------------------------------------------- #
st.divider()
st.header("Regressor Screening")

try:
    # Detect potential regressor columns (numeric, not target or time)
    numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32)
    regressor_candidates = [
        c for c in work_df.columns
        if work_df[c].dtype in numeric_types
        and c != target_col
        and c != time_col
        and c != sid_col
        and c != id_col
    ]

    if not regressor_candidates:
        st.info(
            "No numeric regressor columns detected beyond the target. "
            "Upload data with external regressors (e.g. promo_intensity, temperature) "
            "to use this feature."
        )
    else:
        st.markdown(f"Detected **{len(regressor_candidates)}** potential regressor column(s): "
                    f"`{'`, `'.join(regressor_candidates)}`")

        from src.data.regressor_screen import screen_regressors

        with st.spinner("Screening regressors..."):
            screen_report = screen_regressors(
                work_df,
                feature_columns=regressor_candidates,
                target_col=target_col,
            )

        # Passed vs dropped
        passed = [c for c in regressor_candidates if c not in screen_report.dropped_columns]
        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Passed", len(passed))
        col_r2.metric("Dropped", len(screen_report.dropped_columns))

        if passed:
            st.success(f"Passed columns: `{'`, `'.join(passed)}`")
        if screen_report.dropped_columns:
            st.warning(f"Dropped columns: `{'`, `'.join(screen_report.dropped_columns)}`")

        # Detailed reasons
        if screen_report.low_variance_columns:
            st.caption(f"Low variance: `{'`, `'.join(screen_report.low_variance_columns)}`")
        if screen_report.low_mi_columns:
            st.caption(f"Low mutual information: `{'`, `'.join(screen_report.low_mi_columns)}`")
        if screen_report.high_correlation_pairs:
            with st.expander("Highly correlated pairs"):
                for pair in screen_report.high_correlation_pairs:
                    st.markdown(f"- {pair}")

        # Warnings
        if screen_report.warnings:
            for w in screen_report.warnings:
                st.warning(w)

        # Per-column stats
        if screen_report.per_column_stats:
            with st.expander("Per-column statistics"):
                stats_rows = []
                for col_name, stats in screen_report.per_column_stats.items():
                    row = {"column": col_name}
                    row.update(stats)
                    stats_rows.append(row)
                if stats_rows:
                    st.dataframe(pl.DataFrame(stats_rows).to_pandas(), use_container_width=True)

except Exception as exc:
    st.warning(f"Regressor screening failed: {exc}")


# --------------------------------------------------------------------------- #
#  Section 7: Ask About This Series (AI)
# --------------------------------------------------------------------------- #
st.divider()
st.header("Ask About This Series")

if not ai_available():
    render_ai_unavailable_notice()
else:
    try:
        from src.ai.nl_query import NaturalLanguageQueryEngine

        st.markdown(f"Ask a question about **{selected_series}** and get an AI-powered answer.")

        # Suggested question buttons
        st.caption("Suggested questions:")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("Why is demand volatile?", key="q_volatile"):
                st.session_state["ai_question"] = "Why is demand volatile for this series?"
        with col_s2:
            if st.button("Is there a trend?", key="q_trend"):
                st.session_state["ai_question"] = "Is there an underlying trend in this series?"
        with col_s3:
            if st.button("Any anomalies?", key="q_anomaly"):
                st.session_state["ai_question"] = "Are there any anomalies or unusual patterns in this series?"

        question = st.text_input(
            "Your question",
            value=st.session_state.get("ai_question", ""),
            key="ai_question_input",
            placeholder="e.g. Why did demand spike in Q3?",
        )

        if question and st.button("Ask", type="primary", key="ask_ai_btn"):
            with st.spinner("Querying AI..."):
                engine = NaturalLanguageQueryEngine()
                history_df = work_df.filter(pl.col(sid_col) == selected_series)
                forecast_df = st.session_state.get("forecast_df")

                result = engine.query(
                    series_id=selected_series,
                    question=question,
                    lob="explorer",
                    history=history_df,
                    forecast=forecast_df,
                )

            st.subheader("Answer")
            st.markdown(result.answer)

            render_ai_confidence_badge(result.confidence)

            if result.sources_used:
                st.caption(f"Sources: {', '.join(result.sources_used)}")

            if result.supporting_data:
                with st.expander("Supporting data"):
                    st.json(result.supporting_data)

    except ImportError as exc:
        st.info(f"AI query engine not available: {exc}")
    except Exception as exc:
        st.error(f"AI query failed: {exc}")
