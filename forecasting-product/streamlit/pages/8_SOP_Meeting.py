"""
Page 8 — S&OP Meeting Prep

Executive-level output — AI commentary, cross-run comparison, model
governance, and BI export for Power BI / Tableau consumption.
"""

import io
import sys
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

_PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
_STREAMLIT_DIR = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))
if str(_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_DIR))

from src.metrics.store import MetricStore
from utils import (
    COLORS,
    DATA_DIR,
    TREND_ICONS,
    ai_available,
    format_number,
    format_pct,
    polars_to_pandas,
    render_ai_unavailable_notice,
    render_api_key_sidebar,
    render_metric_card_with_trend,
)

st.set_page_config(page_title="S&OP Meeting Prep", page_icon="📋", layout="wide")
render_api_key_sidebar()
st.title("S&OP Meeting Prep")

metrics_dir = str(DATA_DIR / "metrics")

# =========================================================================== #
#  Section 1: AI Commentary Generation
# =========================================================================== #
st.header("Executive Commentary")
st.caption("Generate VP-level S&OP meeting narratives powered by Claude.")

col_lob, col_start, col_end = st.columns(3)
with col_lob:
    commentary_lob = st.text_input("Line of Business", value="retail", key="commentary_lob")
with col_start:
    period_start = st.date_input("Period start", value=None, key="commentary_start")
with col_end:
    period_end = st.date_input("Period end", value=None, key="commentary_end")

# Resolve metrics_df from session state or MetricStore
commentary_metrics_df = st.session_state.get("backtest_metrics")
if commentary_metrics_df is None:
    try:
        store = MetricStore(base_path=metrics_dir)
        commentary_metrics_df = store.read(run_type="backtest", lob=commentary_lob)
    except Exception:
        commentary_metrics_df = None

if st.button("Generate Commentary", type="primary", key="gen_commentary"):
    if not ai_available():
        render_ai_unavailable_notice()
    elif commentary_metrics_df is None or commentary_metrics_df.is_empty():
        st.warning(
            f"No metrics data available for LOB '{commentary_lob}'. "
            "Run a backtest first or load data on the Backtest Results page."
        )
    else:
        try:
            from src.ai.commentary import CommentaryEngine

            engine = CommentaryEngine()
            if not engine.available:
                render_ai_unavailable_notice()
            else:
                drift_alerts = st.session_state.get("drift_alerts")
                with st.spinner("Generating executive commentary..."):
                    result = engine.generate(
                        lob=commentary_lob,
                        metrics_df=commentary_metrics_df,
                        drift_alerts=drift_alerts,
                        period_start=period_start,
                        period_end=period_end,
                    )

                # Executive summary
                st.info(result.executive_summary)

                # Key metrics as cards
                if result.key_metrics:
                    metric_cols = st.columns(min(len(result.key_metrics), 4))
                    for i, km in enumerate(result.key_metrics):
                        with metric_cols[i % len(metric_cols)]:
                            render_metric_card_with_trend(
                                name=km.get("name", ""),
                                value=km.get("value", ""),
                                unit=km.get("unit", ""),
                                trend=km.get("trend", ""),
                            )

                # Exceptions
                if result.exceptions:
                    st.subheader("Exceptions")
                    for exc_item in result.exceptions:
                        st.warning(exc_item)

                # Action items
                if result.action_items:
                    st.subheader("Action Items")
                    for j, action in enumerate(result.action_items):
                        st.checkbox(action, value=False, key=f"action_{j}")

        except ImportError:
            st.warning("Commentary module not available. Install the `anthropic` package.")
        except Exception as exc:
            st.error(f"Commentary generation failed: {exc}")
            st.caption("This may be a transient API issue. Try again in a moment.")

st.divider()

# =========================================================================== #
#  Section 2: Cross-Run Forecast Comparison
# =========================================================================== #
st.header("Forecast Comparison")

uploaded_external = st.file_uploader(
    "Upload external forecast (CSV or Parquet)",
    type=["csv", "parquet"],
    key="external_forecast_upload",
)

model_forecast = st.session_state.get("forecast_df")

external_df = None
if uploaded_external is not None:
    try:
        raw = uploaded_external.getvalue()
        if uploaded_external.name.endswith(".parquet"):
            external_df = pl.read_parquet(io.BytesIO(raw))
        else:
            external_df = pl.read_csv(io.BytesIO(raw), try_parse_dates=True)
        st.success(
            f"Loaded external forecast: {external_df.shape[0]:,} rows x "
            f"{external_df.shape[1]} columns."
        )
    except Exception as exc:
        st.error(f"Failed to load external forecast: {exc}")

if model_forecast is not None and external_df is not None:
    try:
        from src.analytics.comparator import ForecastComparator

        comparator = ForecastComparator()
        with st.spinner("Comparing forecasts..."):
            comparison = comparator.compare(
                model_forecast=model_forecast,
                external_forecasts=external_df,
            )
            summary = comparator.summary(comparison)

        # Overlay chart
        st.subheader("Forecast Overlay")
        comp_pd = polars_to_pandas(comparison)
        time_col = "week" if "week" in comp_pd.columns else comp_pd.columns[0]

        fig = go.Figure()
        if "forecast" in comp_pd.columns:
            fig.add_trace(go.Scatter(
                x=comp_pd[time_col],
                y=comp_pd["forecast"],
                mode="lines",
                name="Model Forecast",
                line=dict(color=COLORS["primary"]),
            ))
        if "external_forecast" in comp_pd.columns:
            fig.add_trace(go.Scatter(
                x=comp_pd[time_col],
                y=comp_pd["external_forecast"],
                mode="lines",
                name="External Forecast",
                line=dict(color=COLORS["accent"]),
            ))
        fig.update_layout(
            xaxis_title="Period",
            yaxis_title="Forecast",
            height=400,
            margin=dict(t=30, b=40, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("Comparison Summary")
        st.dataframe(polars_to_pandas(summary), use_container_width=True)

    except ImportError:
        st.warning("ForecastComparator module not available.")
    except Exception as exc:
        st.error(f"Forecast comparison failed: {exc}")
        st.caption("Check that both forecasts have matching columns and time ranges.")

elif model_forecast is not None:
    st.info(
        "Model forecast loaded from session. Upload an external forecast above "
        "to generate a comparison."
    )
elif external_df is not None:
    st.info(
        "External forecast loaded. Run a forecast on the Data Onboarding page "
        "to generate a model forecast for comparison."
    )
else:
    st.info(
        "No forecasts available. Run a forecast on the Data Onboarding page "
        "and/or upload an external forecast to compare."
    )

st.divider()

# =========================================================================== #
#  Section 3: Model Governance
# =========================================================================== #
st.header("Model Governance")

try:
    from src.analytics.governance import ForecastLineage, ModelCardRegistry

    # Model Cards
    st.subheader("Model Cards")
    registry = ModelCardRegistry(base_path=str(DATA_DIR / "model_cards"))
    cards_df = registry.all_cards()

    if cards_df is not None and not cards_df.is_empty():
        st.dataframe(polars_to_pandas(cards_df), use_container_width=True)

        model_names = cards_df["model_name"].to_list() if "model_name" in cards_df.columns else []
        if model_names:
            selected_model = st.selectbox(
                "Select a model to view details",
                options=model_names,
                key="governance_model_select",
            )
            card = registry.get(selected_model)
            if card is not None:
                with st.expander(f"Model Card: {card.model_name}", expanded=True):
                    col_v, col_w, col_s = st.columns(3)
                    col_v.metric("Version", card.version)
                    col_w.metric("Training Window", card.training_window)
                    col_s.metric("Series Count", format_number(card.n_series, 0))

                    if card.metrics:
                        st.markdown("**Metrics**")
                        metric_cols = st.columns(min(len(card.metrics), 4))
                        for i, (k, v) in enumerate(card.metrics.items()):
                            with metric_cols[i % len(metric_cols)]:
                                st.metric(k, format_number(v))

                    if card.features:
                        st.markdown(f"**Features**: {', '.join(card.features)}")

                    if card.config_hash:
                        st.caption(f"Config hash: `{card.config_hash}`")
    else:
        st.info(
            "No model cards found. Model cards are generated during backtest "
            "and forecast pipeline runs."
        )

    # Forecast Lineage
    st.subheader("Forecast Lineage")
    lineage = ForecastLineage(base_path=str(DATA_DIR / "lineage"))
    lineage_df = lineage.history()

    if lineage_df is not None and not lineage_df.is_empty():
        st.dataframe(polars_to_pandas(lineage_df), use_container_width=True)
    else:
        st.info("No lineage records found. Run a forecast pipeline to generate lineage.")

except ImportError:
    st.warning("Governance module not available. Check that `src.analytics.governance` is installed.")
except Exception as exc:
    st.error(f"Failed to load governance data: {exc}")
    st.caption("This is expected if no pipeline runs have been completed yet.")

st.divider()

# =========================================================================== #
#  Section 4: BI Export
# =========================================================================== #
st.header("BI Export")
st.caption("Export data in Hive-partitioned Parquet format for Power BI / Tableau.")

export_lob = st.text_input("Line of Business", value="retail", key="export_lob")

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    if st.button("Export Forecast vs Actual", key="export_fva"):
        try:
            from src.analytics.bi_export import BIExporter

            forecast_df = st.session_state.get("forecast_df")
            actuals_df = st.session_state.get("actuals_df")

            if forecast_df is None or actuals_df is None:
                st.warning(
                    "Both forecast and actuals data are required. "
                    "Run a forecast on the Data Onboarding page first."
                )
            else:
                exporter = BIExporter(base_path=str(DATA_DIR / "bi_exports"))
                with st.spinner("Exporting forecast vs actual..."):
                    out_path = exporter.export_forecast_vs_actual(
                        forecasts=forecast_df,
                        actuals=actuals_df,
                        lob=export_lob,
                    )
                st.success(f"Exported to: `{out_path}`")
        except ImportError:
            st.warning("BIExporter module not available.")
        except Exception as exc:
            st.error(f"Export failed: {exc}")

with col_e2:
    if st.button("Export Leaderboard", key="export_lb"):
        try:
            from src.analytics.bi_export import BIExporter

            exporter = BIExporter(base_path=str(DATA_DIR / "bi_exports"))
            store = MetricStore(base_path=metrics_dir)
            with st.spinner("Exporting leaderboard..."):
                out_path = exporter.export_leaderboard(
                    metric_store=store,
                    lob=export_lob,
                )
            st.success(f"Exported to: `{out_path}`")
        except ImportError:
            st.warning("BIExporter module not available.")
        except Exception as exc:
            st.error(f"Export failed: {exc}")

with col_e3:
    if st.button("Export Bias Report", key="export_bias"):
        try:
            from src.analytics.bi_export import BIExporter

            exporter = BIExporter(base_path=str(DATA_DIR / "bi_exports"))
            store = MetricStore(base_path=metrics_dir)
            with st.spinner("Exporting bias report..."):
                out_path = exporter.export_bias_report(
                    metric_store=store,
                    lob=export_lob,
                )
            st.success(f"Exported to: `{out_path}`")
        except ImportError:
            st.warning("BIExporter module not available.")
        except Exception as exc:
            st.error(f"Export failed: {exc}")
