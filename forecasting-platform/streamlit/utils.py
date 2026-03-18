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
def load_uploaded_csv(uploaded_file) -> pl.DataFrame:
    """Read an uploaded CSV file into a Polars DataFrame."""
    raw = uploaded_file.getvalue()
    return pl.read_csv(io.BytesIO(raw), try_parse_dates=True)


def load_uploaded_csvs(uploaded_files: List) -> Dict[str, pl.DataFrame]:
    """Read multiple uploaded CSV files into a dict of {filename: DataFrame}."""
    result: Dict[str, pl.DataFrame] = {}
    for f in uploaded_files:
        raw = f.getvalue()
        result[f.name] = pl.read_csv(io.BytesIO(raw), try_parse_dates=True)
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
            return pl.read_csv(
                str(SAMPLE_DATA),
                try_parse_dates=True,
                infer_schema_length=10000,
            )
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


def severity_badge(severity: str) -> str:
    """Return a coloured markdown badge for alert severity."""
    colour = SEVERITY_COLORS.get(severity.lower(), "#8d99ae")
    return f":{severity.upper()}: `{severity}`"


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


# Sample CSV template for data onboarding guidance
CSV_TEMPLATE = """\
week,store_id,product_id,quantity
2023-01-02,store_1,prod_A,150
2023-01-02,store_1,prod_B,80
2023-01-09,store_1,prod_A,145
"""
