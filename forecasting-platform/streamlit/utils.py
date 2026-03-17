"""
Shared utilities for the Streamlit dashboard.

Provides helpers for data loading, Polars ↔ Pandas conversion,
colour palettes, and reusable chart fragments.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

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


def polars_to_pandas(df: pl.DataFrame):
    """Convert Polars DF to Pandas for Streamlit / Plotly compatibility."""
    return df.to_pandas()


def load_sample_data() -> Optional[pl.DataFrame]:
    """Load the bundled Rossmann sample dataset (if present)."""
    if SAMPLE_DATA.exists():
        return pl.read_csv(str(SAMPLE_DATA), try_parse_dates=True)
    return None


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
