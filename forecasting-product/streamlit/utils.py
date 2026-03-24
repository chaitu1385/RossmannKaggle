"""
Shared utilities for the Streamlit dashboard.

Provides helpers for data loading, Polars ↔ Pandas conversion,
colour palettes, and reusable chart fragments.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import streamlit as st

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
PLATFORM_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PLATFORM_ROOT / "data"
SAMPLE_DATA = DATA_DIR / "rossmann" / "train.csv"

# ---------------------------------------------------------------------------
#  Colour palette (consistent across pages)
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#4361ee",
    "secondary": "#3a0ca3",
    "accent": "#f72585",
    "success": "#06d6a0",
    "warning": "#ffd166",
    "danger": "#ef476f",
    "neutral": "#8d99ae",
    "bg_light": "#f8f9fa",
}

SEVERITY_COLORS = {
    "critical": "#ef476f",
    "warning": "#ffd166",
    "info": "#4361ee",
}

FVA_COLORS = {
    "ADDS_VALUE": "#06d6a0",
    "NEUTRAL": "#8d99ae",
    "DESTROYS_VALUE": "#ef476f",
    "BASELINE": "#4361ee",
}

MODEL_LAYER_COLORS = {
    "naive": "#8d99ae",
    "statistical": "#4361ee",
    "ml": "#06d6a0",
    "neural": "#f72585",
    "foundation": "#3a0ca3",
    "intermittent": "#ffd166",
    "ensemble": "#7209b7",
    "override": "#ff6b35",
}


# ---------------------------------------------------------------------------
#  Data helpers
# ---------------------------------------------------------------------------
def _read_csv_robust(source: io.BytesIO) -> pl.DataFrame:
    """Read CSV with fallback for mixed-type columns.

    First attempts with ``infer_schema_length=10_000`` so Polars samples
    enough rows to detect mixed-type columns (e.g. StateHoliday containing
    both integers and strings).  If that still fails, retries with
    ``infer_schema_length=None`` which scans *every* row for type inference.
    """
    try:
        return pl.read_csv(source, try_parse_dates=True, infer_schema_length=10_000)
    except Exception:
        source.seek(0)
        return pl.read_csv(source, try_parse_dates=True, infer_schema_length=None)


def load_uploaded_csv(uploaded_file) -> pl.DataFrame:
    """Read an uploaded CSV file into a Polars DataFrame."""
    raw = uploaded_file.getvalue()
    return _read_csv_robust(io.BytesIO(raw))


def load_uploaded_csvs(uploaded_files: List) -> Dict[str, pl.DataFrame]:
    """Read multiple uploaded CSV files into a dict of {filename: DataFrame}."""
    result: Dict[str, pl.DataFrame] = {}
    for f in uploaded_files:
        raw = f.getvalue()
        result[f.name] = _read_csv_robust(io.BytesIO(raw))
    return result


def polars_to_pandas(df: pl.DataFrame):
    """Convert Polars DF to Pandas for Streamlit / Plotly compatibility."""
    return df.to_pandas()


def load_sample_data() -> pl.DataFrame:
    """Load the bundled Rossmann sample dataset, or generate synthetic data.

    Tries Rossmann CSV first. If not found, generates a synthetic retail
    dataset so the sample button always works on first click.
    """
    if SAMPLE_DATA.exists():
        try:
            return _read_csv_robust(io.BytesIO(SAMPLE_DATA.read_bytes()))
        except Exception:
            pass  # fall through to synthetic
    return _generate_synthetic_retail()


def _generate_synthetic_retail() -> pl.DataFrame:
    """Generate a synthetic retail dataset for demo purposes.

    20 stores x 3 categories x 104 weeks = 6,240 rows with trend,
    seasonality, noise, and a promo regressor.
    """
    import numpy as np
    from datetime import date, timedelta

    rng = np.random.RandomState(42)
    rows = []
    base = date(2020, 1, 6)
    n_weeks = 104

    stores = {
        f"store_{i}": rng.choice(["North", "South", "East", "West"])
        for i in range(20)
    }
    categories = ["Food", "Electronics", "Clothing"]

    for store_id, region in stores.items():
        for cat in categories:
            base_demand = rng.uniform(50, 200)
            for w in range(n_weeks):
                seasonal = 30 * np.sin(2 * np.pi * w / 52)
                trend = 0.2 * w
                noise = rng.normal(0, 10)
                qty = max(0, base_demand + seasonal + trend + noise)
                rows.append({
                    "week": base + timedelta(weeks=w),
                    "store_id": store_id,
                    "region": region,
                    "category": cat,
                    "quantity": round(qty, 2),
                    "promo_intensity": round(rng.uniform(0, 1), 2),
                })

    return pl.DataFrame(rows)


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    return f"{value:,.{decimals}f}"



# ---------------------------------------------------------------------------
#  Model & metric display helpers
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "naive_seasonal": "Seasonal Naive",
    "auto_arima": "AutoARIMA",
    "auto_ets": "AutoETS",
    "auto_theta": "AutoTheta",
    "mstl": "MSTL",
    "lgbm_direct": "LightGBM",
    "xgboost_direct": "XGBoost",
    "nhits": "N-HiTS",
    "nbeats": "N-BEATS",
    "tft": "TFT",
    "chronos": "Chronos",
    "timegpt": "TimeGPT",
    "croston": "Croston",
    "croston_sba": "Croston SBA",
    "tsb": "TSB",
    "weighted_ensemble": "Weighted Ensemble",
}

METRIC_TOOLTIPS: Dict[str, str] = {
    "wmape": "Weighted Mean Absolute Percentage Error — lower is better. Measures forecast accuracy weighted by actual volume.",
    "normalized_bias": "Forecast bias — negative means under-forecasting, positive means over-forecasting. Close to 0 is ideal.",
    "mase": "Mean Absolute Scaled Error — compares forecast error to a naive seasonal baseline. Below 1.0 means better than naive.",
    "rmspe": "Root Mean Squared Percentage Error — penalizes large errors more heavily than WMAPE.",
}


def model_display_name(model_id: str) -> str:
    """Return a human-friendly name for a model ID."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string.

    Examples: ``45s``, ``2m 30s``, ``1h 15m 30s``.
    """
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ---------------------------------------------------------------------------
#  Domain-specific colour maps
# ---------------------------------------------------------------------------
DEMAND_CLASS_COLORS = {
    "Smooth": COLORS["success"],
    "Intermittent": COLORS["warning"],
    "Erratic": COLORS["accent"],
    "Lumpy": COLORS["danger"],
    "insufficient_data": COLORS["neutral"],
}

RISK_COLORS = {
    "low": COLORS["success"],
    "medium": COLORS["warning"],
    "high": COLORS["danger"],
}

CONFIDENCE_BADGE_COLORS = {
    "high": COLORS["success"],
    "medium": COLORS["warning"],
    "low": COLORS["danger"],
}

TREND_ICONS = {
    "improving": "\u2191",
    "stable": "\u2192",
    "degrading": "\u2193",
}


# ---------------------------------------------------------------------------
#  AI helpers
# ---------------------------------------------------------------------------
def _sync_api_key_to_env():
    """Push the session-state API key into ``os.environ`` so backend classes pick it up."""
    import os
    key = st.session_state.get("anthropic_api_key", "").strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    elif "ANTHROPIC_API_KEY" not in os.environ:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def render_api_key_sidebar():
    """Render a sidebar widget for entering the Anthropic API key.

    The key is stored in ``st.session_state["anthropic_api_key"]`` and
    automatically synced into ``os.environ["ANTHROPIC_API_KEY"]`` so that
    every downstream consumer (``AIFeatureBase``, ``LLMAnalyzer``,
    ``ai_available()``) picks it up without any extra wiring.

    Call this once at the top of any page that uses AI features.
    """
    import os

    with st.sidebar:
        st.markdown("---")
        st.markdown("**AI Settings**")
        # Pre-fill from env var if user hasn't entered one yet
        default = st.session_state.get(
            "anthropic_api_key",
            os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        st.text_input(
            "Anthropic API Key",
            value=default,
            type="password",
            key="anthropic_api_key",
            help="Entered key is used for this session only and is never stored to disk.",
        )
    _sync_api_key_to_env()


def ai_available() -> bool:
    """Check whether the Anthropic API key is configured.

    Checks both ``os.environ`` and Streamlit session state so it works
    whether the key was set via env var, ``.env`` file, or the sidebar widget.
    """
    import os
    if st.session_state.get("anthropic_api_key", "").strip():
        return True
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def render_ai_unavailable_notice():
    """Show a standard info box when AI features are unavailable."""
    st.info(
        "AI features require an Anthropic API key. "
        "Enter it in the sidebar or set the `ANTHROPIC_API_KEY` environment variable."
    )


def render_ai_confidence_badge(confidence: str):
    """Render a coloured confidence badge."""
    color = CONFIDENCE_BADGE_COLORS.get(confidence, COLORS["neutral"])
    st.markdown(
        f'<span style="background-color:{color}20;color:{color};'
        f'padding:2px 8px;border-radius:4px;font-weight:bold">'
        f'{confidence.upper()}</span>',
        unsafe_allow_html=True,
    )


def render_metric_card_with_trend(name: str, value, unit: str = "", trend: str = ""):
    """Render a metric card with optional trend arrow."""
    arrow = TREND_ICONS.get(trend, "")
    trend_color = {
        "improving": COLORS["success"],
        "degrading": COLORS["danger"],
        "stable": COLORS["neutral"],
    }.get(trend, COLORS["neutral"])
    display = f"{value}{unit}"
    if arrow:
        display += f" {arrow}"
    st.metric(name, display)


# Sample CSV template for data onboarding guidance
CSV_TEMPLATE = """\
week,store_id,product_id,quantity
2023-01-02,store_1,prod_A,150
2023-01-02,store_1,prod_B,80
2023-01-09,store_1,prod_A,145
"""
