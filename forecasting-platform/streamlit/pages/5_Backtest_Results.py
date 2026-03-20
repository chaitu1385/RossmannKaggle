"""
Page 5 — Backtest Results

Model leaderboard, FVA cascade chart, per-series champion map,
layer leaderboard, prediction interval calibration, SHAP feature
attribution, and AI-powered configuration recommendations.
"""

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

from src.analytics.fva_analyzer import FVAAnalyzer
from src.metrics.store import MetricStore
from utils import (
    COLORS,
    DATA_DIR,
    FVA_COLORS,
    METRIC_TOOLTIPS,
    MODEL_LAYER_COLORS,
    RISK_COLORS,
    model_display_name,
    polars_to_pandas,
    format_pct,
    ai_available,
    render_ai_unavailable_notice,
    render_api_key_sidebar,
)

st.set_page_config(page_title="Backtest Results", page_icon="🏆", layout="wide")
render_api_key_sidebar()
st.title("Backtest Results")

# ---------------------------------------------------------------------------
#  Cross-page: show config from Data Onboarding if available
# ---------------------------------------------------------------------------
accepted_config = st.session_state.get("accepted_config")
if accepted_config:
    n_models = len(accepted_config.forecast.forecasters)
    horizon = accepted_config.forecast.horizon_weeks
    st.info(
        f"Using config from Data Onboarding: **{n_models} models**, "
        f"**{horizon}-period horizon**, "
        f"**{accepted_config.backtest.n_folds} folds**."
    )

# ---------------------------------------------------------------------------
#  Data source: MetricStore or uploaded Parquet
# ---------------------------------------------------------------------------
metrics_dir = str(DATA_DIR / "metrics")

tab_store, tab_upload = st.tabs(["From Metric Store", "Upload Parquet"])

metrics_df = None

with tab_store:
    lob = st.text_input("Line of Business", value="retail", key="lob_store")
    if st.button("Load from metric store", key="load_store"):
        with st.spinner("Loading metrics..."):
            try:
                store = MetricStore(base_path=metrics_dir)
                metrics_df = store.read(run_type="backtest", lob=lob)
                if metrics_df is not None and not metrics_df.is_empty():
                    st.session_state["backtest_metrics"] = metrics_df
                    st.success(f"Loaded {metrics_df.shape[0]:,} metric records.")
                else:
                    st.warning(
                        f"No backtest results found for LOB '{lob}'. "
                        f"Run `python scripts/run_backtest.py --lob {lob}` first."
                    )
            except Exception as e:
                st.error(
                    f"Error loading metrics: {e}\n\n"
                    f"Check that the metric store directory exists: `{metrics_dir}`"
                )

with tab_upload:
    uploaded = st.file_uploader("Upload backtest metrics Parquet", type=["parquet"])
    if uploaded is not None:
        try:
            import io
            metrics_df = pl.read_parquet(io.BytesIO(uploaded.getvalue()))
            st.session_state["backtest_metrics"] = metrics_df
            st.success(f"Loaded {metrics_df.shape[0]:,} records from upload.")
        except Exception as e:
            st.error(f"Failed to read Parquet file: {e}")

# Retrieve from session if previously loaded
if metrics_df is None:
    metrics_df = st.session_state.get("backtest_metrics")

if metrics_df is None:
    st.info(
        "No backtest results loaded yet. You can either:\n\n"
        "- **Run a backtest** from the **Data Onboarding** page (upload data \u2192 accept config \u2192 click *Run Backtest*)\n"
        "- **Load from metric store** or **upload a Parquet file** using the tabs above"
    )
    st.stop()

# ---------------------------------------------------------------------------
#  Model Leaderboard
# ---------------------------------------------------------------------------
st.divider()
st.header("Model Leaderboard")

# Aggregate: mean WMAPE and bias per model
if "model_id" in metrics_df.columns and "wmape" in metrics_df.columns:
    leaderboard = (
        metrics_df
        .group_by("model_id")
        .agg([
            pl.col("wmape").mean().alias("mean_wmape"),
            pl.col("normalized_bias").mean().alias("mean_bias"),
            pl.col("series_id").n_unique().alias("n_series"),
        ])
        .sort("mean_wmape")
        .with_row_index("rank", offset=1)
        .with_columns(
            pl.col("model_id").map_elements(
                model_display_name, return_dtype=pl.Utf8,
            ).alias("model_name")
        )
    )

    st.dataframe(
        polars_to_pandas(leaderboard),
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "model_id": "Model ID",
            "model_name": "Model",
            "mean_wmape": st.column_config.NumberColumn(
                "WMAPE", format="%.4f",
                help=METRIC_TOOLTIPS.get("wmape", ""),
            ),
            "mean_bias": st.column_config.NumberColumn(
                "Bias", format="%.4f",
                help=METRIC_TOOLTIPS.get("normalized_bias", ""),
            ),
            "n_series": st.column_config.NumberColumn("Series"),
        },
    )

    # Bar chart
    lb_pd = polars_to_pandas(leaderboard)
    fig_lb = go.Figure()
    fig_lb.add_trace(go.Bar(
        x=lb_pd["model_name"],
        y=lb_pd["mean_wmape"],
        marker_color=COLORS["primary"],
        text=[f"{v:.4f}" for v in lb_pd["mean_wmape"]],
        textposition="auto",
    ))
    fig_lb.update_layout(
        title="Mean WMAPE by Model",
        xaxis_title="Model",
        yaxis_title="WMAPE (lower is better)",
        height=350,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_lb, use_container_width=True)
else:
    available = ", ".join(metrics_df.columns)
    st.warning(
        f"Expected columns `model_id` and `wmape` not found in data.\n\n"
        f"Available columns: {available}"
    )

# ---------------------------------------------------------------------------
#  FVA Cascade
# ---------------------------------------------------------------------------
st.divider()
st.header("Forecast Value-Add (FVA) Cascade")
st.caption(
    "FVA measures whether each model layer improves accuracy over the naive baseline. "
    "Green = adds value, gray = neutral, red = destroys value."
)

try:
    with st.spinner("Computing FVA cascade..."):
        fva = FVAAnalyzer()
        fva_detail = fva.compute_fva_detail(metrics_df)

    if not fva_detail.is_empty():
        # Summary by layer
        fva_summary = fva.summarize(fva_detail)
        fva_summary_pd = polars_to_pandas(fva_summary)

        # FVA cascade bar chart
        if "forecast_layer" in fva_summary.columns and "mean_wmape" in fva_summary.columns:
            fig_fva = go.Figure()

            for _, row in fva_summary_pd.iterrows():
                layer = row["forecast_layer"]
                color = MODEL_LAYER_COLORS.get(layer, COLORS["neutral"])
                fig_fva.add_trace(go.Bar(
                    x=[layer],
                    y=[row["mean_wmape"]],
                    name=layer,
                    marker_color=color,
                    text=[f"{row['mean_wmape']:.4f}"],
                    textposition="auto",
                ))

            fig_fva.update_layout(
                title="FVA Cascade — Mean WMAPE by Layer",
                xaxis_title="Forecast Layer",
                yaxis_title="WMAPE",
                showlegend=False,
                height=350,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_fva, use_container_width=True)

        # FVA class distribution (stacked bar)
        if "n_adds_value" in fva_summary.columns:
            fva_class_cols = ["n_adds_value", "n_neutral", "n_destroys_value"]
            available_cols = [c for c in fva_class_cols if c in fva_summary.columns]

            if available_cols:
                fig_stack = go.Figure()
                label_map = {
                    "n_adds_value": "Adds Value",
                    "n_neutral": "Neutral",
                    "n_destroys_value": "Destroys Value",
                }
                color_map = {
                    "n_adds_value": FVA_COLORS["ADDS_VALUE"],
                    "n_neutral": FVA_COLORS["NEUTRAL"],
                    "n_destroys_value": FVA_COLORS["DESTROYS_VALUE"],
                }

                for col in available_cols:
                    fig_stack.add_trace(go.Bar(
                        x=fva_summary_pd["forecast_layer"],
                        y=fva_summary_pd[col],
                        name=label_map.get(col, col),
                        marker_color=color_map.get(col, COLORS["neutral"]),
                    ))

                fig_stack.update_layout(
                    barmode="stack",
                    title="FVA Classification by Layer",
                    xaxis_title="Forecast Layer",
                    yaxis_title="Number of Series",
                    height=350,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_stack, use_container_width=True)

        # Summary table
        with st.expander("FVA Summary Table"):
            st.dataframe(fva_summary_pd, use_container_width=True)

        # -------------------------------------------------------------------
        #  Layer Leaderboard
        # -------------------------------------------------------------------
        st.subheader("Layer Leaderboard")
        layer_lb = fva.layer_leaderboard(fva_detail)

        if not layer_lb.is_empty():
            layer_lb_pd = polars_to_pandas(layer_lb)

            # Colour the recommendation column
            def style_recommendation(val):
                colors = {
                    "Keep": f"background-color: {FVA_COLORS['ADDS_VALUE']}40; font-weight: bold",
                    "Review": f"background-color: {FVA_COLORS['NEUTRAL']}40",
                    "Remove": f"background-color: {FVA_COLORS['DESTROYS_VALUE']}30; font-weight: bold",
                }
                return colors.get(val, "")

            st.dataframe(
                layer_lb_pd.style.map(
                    style_recommendation, subset=["recommendation"]
                ),
                use_container_width=True,
            )
    else:
        st.info(
            "No FVA data available. FVA requires backtest results with "
            "multiple model layers (e.g., naive + statistical + ML)."
        )
except Exception as e:
    st.warning(f"FVA analysis not available: {e}")

# ---------------------------------------------------------------------------
#  Per-Series Champion Map
# ---------------------------------------------------------------------------
st.divider()
st.header("Per-Series Champion Map")

if "model_id" in metrics_df.columns and "series_id" in metrics_df.columns:
    # Find best model per series by WMAPE
    champion_map = (
        metrics_df
        .group_by(["series_id", "model_id"])
        .agg(pl.col("wmape").mean().alias("mean_wmape"))
        .sort("mean_wmape")
        .group_by("series_id")
        .first()
        .sort("series_id")
        .with_columns(
            pl.col("model_id").map_elements(
                model_display_name, return_dtype=pl.Utf8,
            ).alias("model_name")
        )
    )

    # Champion distribution
    champ_dist = (
        champion_map
        .group_by("model_name")
        .len()
        .sort("len", descending=True)
        .rename({"len": "n_series"})
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(polars_to_pandas(champ_dist), use_container_width=True)

    with col2:
        dist_pd = polars_to_pandas(champ_dist)
        fig_champ = px.pie(
            dist_pd,
            names="model_name",
            values="n_series",
            title="Champion Model Distribution",
            hole=0.4,
        )
        fig_champ.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_champ, use_container_width=True)

    with st.expander("Full champion map"):
        st.dataframe(polars_to_pandas(champion_map), use_container_width=True)

    # Cross-page link: view a series in Forecast Viewer
    st.markdown("---")
    series_list = champion_map["series_id"].to_list()
    selected = st.selectbox(
        "Select a series to view in Forecast Viewer",
        series_list,
        index=None,
        placeholder="Choose a series...",
    )
    if selected:
        st.session_state["selected_series_id"] = selected
        st.success(
            f"Series `{selected}` selected. Go to **Forecast Viewer** in the "
            f"sidebar to see the forecast chart."
        )
else:
    available = ", ".join(metrics_df.columns)
    st.info(
        f"Champion map requires `series_id` and `model_id` columns.\n\n"
        f"Available columns: {available}"
    )

# ---------------------------------------------------------------------------
#  Prediction Interval Calibration
# ---------------------------------------------------------------------------
st.divider()
st.header("Prediction Interval Calibration")

# Check if quantile columns exist in the metrics
quantile_cols = [c for c in metrics_df.columns if c.startswith("forecast_p") or c.startswith("p")]
has_quantiles = bool(quantile_cols) or ("actual" in metrics_df.columns and "forecast" in metrics_df.columns)

if has_quantiles:
    try:
        from src.evaluation.calibration import (
            compute_calibration_report,
            compute_conformal_residuals,
        )

        quantiles = [0.1, 0.5, 0.9]
        coverage_targets = {"p10_p90": 0.80}

        with st.spinner("Computing calibration report..."):
            cal_report = compute_calibration_report(
                backtest_results=metrics_df,
                quantiles=quantiles,
                coverage_targets=coverage_targets,
            )

        if cal_report is not None:
            # Per-model coverage
            if hasattr(cal_report, "per_model") and cal_report.per_model is not None:
                st.subheader("Coverage by Model")
                cal_pd = polars_to_pandas(cal_report.per_model) if isinstance(
                    cal_report.per_model, pl.DataFrame
                ) else cal_report.per_model
                st.dataframe(cal_pd, use_container_width=True)

            # Calibration plot
            if hasattr(cal_report, "nominal_vs_empirical"):
                nve = cal_report.nominal_vs_empirical
                if nve:
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines", line=dict(dash="dash", color=COLORS["neutral"]),
                        name="Perfect calibration",
                    ))
                    nominals = list(nve.keys())
                    empiricals = list(nve.values())
                    fig_cal.add_trace(go.Scatter(
                        x=nominals, y=empiricals,
                        mode="markers+lines",
                        marker=dict(size=8, color=COLORS["primary"]),
                        name="Empirical",
                    ))
                    fig_cal.update_layout(
                        title="Calibration Plot",
                        xaxis_title="Nominal Coverage",
                        yaxis_title="Empirical Coverage",
                        height=350, margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)

            st.session_state["calibration_report"] = cal_report
        else:
            st.info("Calibration report could not be computed with the available data.")

    except Exception as exc:
        st.info(
            f"Calibration analysis not available: {exc}\n\n"
            "Calibration requires quantile forecast columns (e.g., forecast_p10, forecast_p90) "
            "and actual values in the backtest results."
        )
else:
    st.info(
        "Calibration requires prediction interval columns in the backtest results. "
        "Enable quantile forecasting in your config to use this feature."
    )

# ---------------------------------------------------------------------------
#  SHAP Feature Attribution
# ---------------------------------------------------------------------------
st.divider()
st.header("SHAP Feature Attribution")

# Check if ML models are in the leaderboard
ml_models = {"lgbm_direct", "xgboost_direct"}
model_ids_in_data = set(metrics_df["model_id"].unique().to_list()) if "model_id" in metrics_df.columns else set()
ml_in_leaderboard = ml_models & model_ids_in_data

if ml_in_leaderboard:
    st.caption(
        "View feature importance for tree-based models using SHAP values."
    )

    try:
        from src.analytics.explainer import ForecastExplainer

        selected_ml = st.selectbox(
            "Select ML model",
            sorted(ml_in_leaderboard),
            format_func=model_display_name,
            key="shap_model",
        )

        if st.button("Compute SHAP", key="compute_shap"):
            with st.spinner("Computing SHAP values..."):
                explainer = ForecastExplainer(season_length=7)
                actuals_df = st.session_state.get("actuals_df")
                if actuals_df is not None:
                    shap_result = explainer.explain_ml(
                        model_name=selected_ml,
                        history=actuals_df,
                    )

                    if shap_result is not None and not shap_result.is_empty():
                        shap_pd = polars_to_pandas(shap_result.head(10))
                        fig_shap = go.Figure()
                        fig_shap.add_trace(go.Bar(
                            x=shap_pd.iloc[:, 1] if shap_pd.shape[1] > 1 else [],
                            y=shap_pd.iloc[:, 0] if shap_pd.shape[0] > 0 else [],
                            orientation="h",
                            marker_color=COLORS["accent"],
                        ))
                        fig_shap.update_layout(
                            title=f"Top Features — {model_display_name(selected_ml)}",
                            xaxis_title="Mean |SHAP|",
                            height=350, margin=dict(t=40, b=40, l=150),
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.info("SHAP values could not be computed.")
                else:
                    st.warning("Upload actuals data on the Data Onboarding page first.")

    except ImportError:
        st.info(
            "SHAP analysis requires the `shap` package. "
            "Install it with: `pip install shap`"
        )
    except Exception as exc:
        st.info(f"SHAP analysis not available: {exc}")
else:
    st.info(
        "SHAP attribution is available for ML models (LightGBM, XGBoost). "
        "Include `lgbm_direct` or `xgboost_direct` in your backtest to enable."
    )

# ---------------------------------------------------------------------------
#  AI Config Tuner
# ---------------------------------------------------------------------------
st.divider()
st.header("AI Configuration Recommendations")
st.caption(
    "Get AI-powered suggestions to improve your forecasting configuration "
    "based on backtest performance."
)

if ai_available():
    accepted_config = st.session_state.get("accepted_config")
    if accepted_config is None:
        st.info("Accept a config on the Data Onboarding page first.")
    elif st.button("Get AI Recommendations", type="primary", key="config_tune_btn"):
        try:
            from src.ai.config_tuner import ConfigTunerEngine

            engine = ConfigTunerEngine()
            with st.spinner("Analyzing configuration with Claude..."):
                tune_result = engine.recommend(
                    lob="uploaded",
                    current_config=accepted_config,
                    leaderboard=leaderboard if "leaderboard" in dir() else None,
                )

            # Overall assessment
            st.info(tune_result.overall_assessment)
            if tune_result.risk_summary:
                st.caption(f"Risk: {tune_result.risk_summary}")

            # Recommendations
            if tune_result.recommendations:
                for i, rec in enumerate(tune_result.recommendations):
                    with st.expander(
                        f"{rec.field_path} — {rec.expected_impact}",
                        expanded=(i == 0),
                    ):
                        col_cur, col_arrow, col_new = st.columns([2, 1, 2])
                        with col_cur:
                            st.markdown(f"**Current:** `{rec.current_value}`")
                        with col_arrow:
                            st.markdown("**→**")
                        with col_new:
                            st.markdown(f"**Recommended:** `{rec.recommended_value}`")

                        st.markdown(rec.reasoning)

                        risk_color = RISK_COLORS.get(rec.risk, COLORS["neutral"])
                        st.markdown(
                            f'Risk: <span style="color:{risk_color};font-weight:bold">'
                            f'{rec.risk.upper()}</span>',
                            unsafe_allow_html=True,
                        )
            else:
                st.success("No configuration changes recommended — current config looks good.")

        except Exception as exc:
            st.warning(f"AI config tuning failed: {exc}")
else:
    render_ai_unavailable_notice()
