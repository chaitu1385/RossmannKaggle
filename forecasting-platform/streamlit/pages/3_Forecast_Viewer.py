"""
Page 3 — Forecast Viewer

Interactive forecast chart with P10/P90 fan chart, actuals overlay,
seasonal decomposition, and explainer narrative.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

_PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
_STREAMLIT_DIR = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))
if str(_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_DIR))

import polars as pl

from src.analytics.explainer import ForecastExplainer
from utils import COLORS, DATA_DIR, polars_to_pandas

st.set_page_config(page_title="Forecast Viewer", page_icon="📈", layout="wide")
st.title("Forecast Viewer")

# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------
st.sidebar.header("Data Sources")

forecast_file = st.sidebar.file_uploader(
    "Forecast Parquet/CSV", type=["parquet", "csv"], key="fc_file",
)
actuals_file = st.sidebar.file_uploader(
    "Actuals Parquet/CSV (optional)",
    type=["parquet", "csv"],
    key="act_file",
    help="Upload actuals to overlay on the forecast chart and enable decomposition. Leave blank to view forecasts only.",
)

forecast_df = None
actuals_df = None


def _load_file(uploaded):
    """Load a Parquet or CSV file into Polars."""
    import io
    raw = uploaded.getvalue()
    name = uploaded.name.lower()
    try:
        if name.endswith(".parquet"):
            return pl.read_parquet(io.BytesIO(raw))
        return pl.read_csv(io.BytesIO(raw), try_parse_dates=True)
    except Exception as exc:
        st.error(
            f"Failed to parse `{uploaded.name}`: {exc}\n\n"
            "Ensure the file is a valid Parquet or CSV."
        )
        return None


if forecast_file is not None:
    forecast_df = _load_file(forecast_file)
    if forecast_df is not None:
        st.session_state["forecast_df"] = forecast_df

if actuals_file is not None:
    actuals_df = _load_file(actuals_file)
    if actuals_df is not None:
        st.session_state["actuals_df"] = actuals_df

forecast_df = forecast_df or st.session_state.get("forecast_df")
actuals_df = actuals_df or st.session_state.get("actuals_df")

if forecast_df is None:
    st.info(
        "Upload forecast output (Parquet or CSV) in the sidebar."
    )
    st.markdown(
        "**To generate forecasts:**\n"
        "```bash\n"
        "python scripts/run_forecast.py \\\n"
        "  --config configs/platform_config.yaml \\\n"
        "  --lob retail\n"
        "```\n\n"
        "The output should contain columns like: "
        "`series_id`, `week`, `forecast`, `forecast_p10`, `forecast_p90`."
    )
    st.stop()

# ---------------------------------------------------------------------------
#  Detect column names
# ---------------------------------------------------------------------------
def _detect_col(df, candidates, fallback_idx=0):
    """Find the first matching column name."""
    for c in candidates:
        if c in df.columns:
            return c
        # case-insensitive
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return df.columns[fallback_idx] if df.columns else None


id_col = _detect_col(forecast_df, ["series_id", "Store", "store", "id", "sku"])
time_col = _detect_col(forecast_df, ["week", "date", "ds", "Date", "target_week"])
value_col = _detect_col(forecast_df, ["forecast", "prediction", "yhat", "y_hat"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Detected columns**")
id_col = st.sidebar.text_input(
    "Series ID column", value=id_col or "",
    help="Column that identifies each time series (e.g., store_id, sku).",
)
time_col = st.sidebar.text_input(
    "Time column", value=time_col or "",
    help="Column containing dates or timestamps.",
)
value_col = st.sidebar.text_input(
    "Forecast column", value=value_col or "",
    help="Column containing the point forecast values.",
)

if not id_col or not time_col or not value_col:
    st.warning(
        "Could not auto-detect all required columns. "
        "Please specify them in the sidebar."
    )
    st.caption(f"Available columns: {', '.join(forecast_df.columns)}")
    st.stop()

# ---------------------------------------------------------------------------
#  Series selector
# ---------------------------------------------------------------------------
series_list = forecast_df[id_col].unique().sort().to_list()

# Cross-page: pre-select series from Backtest Results or Platform Health
pre_selected = st.session_state.get("selected_series_id")

# Check for drift warnings
drift_alerts = st.session_state.get("drift_alerts", [])
drift_series = {a.series_id for a in drift_alerts} if drift_alerts else set()

# Build display labels with drift indicators
def _series_label(s):
    if s in drift_series:
        return f"{s} (drift alert)"
    return str(s)

default_idx = 0
if pre_selected and pre_selected in series_list:
    default_idx = series_list.index(pre_selected)

selected_series = st.selectbox(
    "Select series",
    series_list,
    index=default_idx,
    format_func=_series_label,
)

if selected_series is None:
    st.stop()

# Clear the pre-selection after use
if pre_selected:
    del st.session_state["selected_series_id"]

# Filter to selected series
fc_series = forecast_df.filter(pl.col(id_col) == selected_series).sort(time_col)

if fc_series.is_empty():
    st.warning(f"No data found for series `{selected_series}`.")
    st.stop()

# ---------------------------------------------------------------------------
#  Forecast chart with fan chart
# ---------------------------------------------------------------------------
st.header(f"Forecast — {selected_series}")

fig = go.Figure()

# Quantile fan chart (P10/P90)
p10_col = _detect_col(fc_series, ["forecast_p10", "p10", "lower"])
p90_col = _detect_col(fc_series, ["forecast_p90", "p90", "upper"])

fc_pd = polars_to_pandas(fc_series)

if p10_col and p90_col and p10_col in fc_pd.columns and p90_col in fc_pd.columns:
    # Upper bound
    fig.add_trace(go.Scatter(
        x=fc_pd[time_col],
        y=fc_pd[p90_col],
        mode="lines",
        line=dict(width=0),
        name="P90",
        showlegend=False,
    ))
    # Lower bound with fill
    fig.add_trace(go.Scatter(
        x=fc_pd[time_col],
        y=fc_pd[p10_col],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=f"rgba(67, 97, 238, 0.15)",
        name="P10-P90",
    ))

# P50 / median line
p50_col = _detect_col(fc_series, ["forecast_p50", "p50"])
if p50_col and p50_col in fc_pd.columns:
    fig.add_trace(go.Scatter(
        x=fc_pd[time_col],
        y=fc_pd[p50_col],
        mode="lines",
        line=dict(color=COLORS["secondary"], width=2, dash="dash"),
        name="P50 (median)",
    ))

# Point forecast
fig.add_trace(go.Scatter(
    x=fc_pd[time_col],
    y=fc_pd[value_col],
    mode="lines+markers",
    line=dict(color=COLORS["primary"], width=2),
    marker=dict(size=4),
    name="Forecast",
))

# Actuals overlay
if actuals_df is not None:
    target_col = _detect_col(actuals_df, ["quantity", "sales", "demand", "Sales",
                                           "actual", "target", "value", "y"])
    act_id_col = _detect_col(actuals_df, [id_col, "series_id", "Store", "id"])
    act_time_col = _detect_col(actuals_df, [time_col, "week", "date", "Date", "ds"])

    if target_col and act_id_col and act_time_col:
        act_series = (
            actuals_df
            .filter(pl.col(act_id_col) == selected_series)
            .sort(act_time_col)
        )
        act_pd = polars_to_pandas(act_series)

        if not act_pd.empty:
            fig.add_trace(go.Scatter(
                x=act_pd[act_time_col],
                y=act_pd[target_col],
                mode="lines+markers",
                line=dict(color=COLORS["neutral"], width=1.5),
                marker=dict(size=3),
                name="Actuals",
            ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Value",
    hovermode="x unified",
    height=450,
    margin=dict(t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# Chart explanation
if p10_col and p90_col and p10_col in fc_pd.columns and p90_col in fc_pd.columns:
    st.caption("Shaded area shows the 80% prediction interval (P10-P90).")

# ---------------------------------------------------------------------------
#  Seasonal Decomposition
# ---------------------------------------------------------------------------
st.divider()
st.header("Seasonal Decomposition")

if actuals_df is not None:
    try:
        target_col = _detect_col(actuals_df, ["quantity", "sales", "demand", "Sales",
                                               "actual", "target", "value", "y"])
        act_id_col = _detect_col(actuals_df, [id_col, "series_id", "Store", "id"])
        act_time_col = _detect_col(actuals_df, [time_col, "week", "date", "Date", "ds"])

        if target_col and act_id_col and act_time_col:
            act_series = actuals_df.filter(pl.col(act_id_col) == selected_series)

            # Determine season_length from data frequency
            explainer = ForecastExplainer(season_length=7)

            with st.spinner("Computing decomposition..."):
                decomp = explainer.decompose(
                    history=act_series,
                    forecast=fc_series,
                    id_col=act_id_col,
                    time_col=act_time_col,
                    target_col=target_col,
                    value_col=value_col,
                )

            if not decomp.is_empty():
                decomp_pd = polars_to_pandas(decomp)

                components = ["trend", "seasonal", "residual"]
                available = [c for c in components if c in decomp_pd.columns]

                for comp in available:
                    fig_comp = go.Figure()
                    time_c = act_time_col if act_time_col in decomp_pd.columns else time_col
                    fig_comp.add_trace(go.Scatter(
                        x=decomp_pd[time_c],
                        y=decomp_pd[comp],
                        mode="lines",
                        line=dict(
                            color={"trend": COLORS["primary"],
                                   "seasonal": COLORS["accent"],
                                   "residual": COLORS["neutral"]}[comp],
                            width=1.5,
                        ),
                        name=comp.title(),
                    ))
                    fig_comp.update_layout(
                        title=comp.title(),
                        height=200,
                        margin=dict(t=30, b=20, l=40, r=20),
                        xaxis_title="",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Narrative
            try:
                with st.spinner("Generating narrative..."):
                    narratives = explainer.narrative(
                        decomposition=decomp,
                        id_col=act_id_col,
                        time_col=act_time_col,
                    )
                series_key = str(selected_series)
                if series_key in narratives:
                    st.subheader("Explainer Narrative")
                    st.info(narratives[series_key])
                else:
                    st.caption("Narrative not available for this series.")
            except Exception:
                st.caption("Narrative not available for this series.")
    except Exception as e:
        st.warning(
            f"Decomposition not available: {e}\n\n"
            "This can happen if the series is too short for seasonal decomposition "
            "or if the data contains too many missing values."
        )
else:
    st.info(
        "Upload actuals data in the sidebar to enable seasonal decomposition. "
        "Actuals are optional — leave blank to view forecasts only."
    )

# ---------------------------------------------------------------------------
#  Raw data preview
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Raw forecast data"):
    st.dataframe(fc_pd, use_container_width=True)
    st.download_button(
        "Download series data as CSV",
        data=fc_series.write_csv(),
        file_name=f"forecast_{selected_series}.csv",
        mime="text/csv",
    )
