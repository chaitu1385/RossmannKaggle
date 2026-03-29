"""Shared pytest fixtures for the forecasting-product test suite."""

from __future__ import annotations

import random
from datetime import date, timedelta

import polars as pl
import pytest


# ── Data generators ──────────────────────────────────────────────────────────

@pytest.fixture()
def weekly_actuals() -> pl.DataFrame:
    """Synthetic weekly actuals (3 series, 104 weeks) with seasonal pattern."""
    return make_weekly_actuals()


@pytest.fixture()
def hierarchy_data() -> pl.DataFrame:
    """Small geography hierarchy for testing."""
    return make_hierarchy_data()


# ── Reusable factory functions (importable by individual test files) ─────────

def make_weekly_actuals(
    n_series: int = 3,
    n_weeks: int = 104,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic weekly actuals with a Q4 seasonal bump.

    Produces a Polars DataFrame with columns ``series_id``, ``week``,
    ``quantity``.  Series are named ``SKU-000``, ``SKU-001``, …
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n_series):
        sid = f"SKU-{i:03d}"
        base = 100 + i * 50
        for w in range(n_weeks):
            week_date = start_date + timedelta(weeks=w)
            seasonal = 1.3 if week_date.month >= 10 else 1.0
            noise = rng.gauss(0, base * 0.1)
            value = max(0, base * seasonal + noise)
            rows.append({
                "series_id": sid,
                "week": week_date,
                "quantity": round(value, 2),
            })
    return pl.DataFrame(rows)


def make_weekly_series(
    n_series: int = 2,
    n_weeks: int = 104,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a simple synthetic weekly panel with Gaussian noise.

    Produces a Polars DataFrame with columns ``series_id``, ``week``,
    ``quantity``.  Series are named ``SKU-000``, ``SKU-001``, …  Values are
    non-negative floats drawn from a normal distribution around a per-series
    base level.  Suitable for forecaster fit/predict smoke tests that only
    verify output shape and column names.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n_series):
        sid = f"SKU-{i:03d}"
        base = 50.0 + i * 30
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": start_date + timedelta(weeks=w),
                "quantity": max(0.0, base + rng.gauss(0, 5)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


def make_actuals(
    n_series: int = 2,
    n_weeks: int = 52,
    start_date: date = date(2024, 1, 1),
) -> pl.DataFrame:
    """Generate deterministic synthetic actuals (no randomness).

    Produces a Polars DataFrame with columns ``series_id``, ``week``,
    ``quantity``.  Series are named ``S000``, ``S001``, …  Values follow a
    simple deterministic formula so tests that inspect exact values remain
    stable across runs.
    """
    rows = []
    for s in range(n_series):
        for w in range(n_weeks):
            rows.append({
                "series_id": f"S{s:03d}",
                "week": start_date + timedelta(weeks=w),
                "quantity": float(100 + s * 10 + (w % 13) * 5),
            })
    return pl.DataFrame(rows)


def make_hierarchy_data() -> pl.DataFrame:
    """Small geography hierarchy for testing."""
    return pl.DataFrame({
        "global": ["Global"] * 6,
        "region": ["Americas", "Americas", "Americas", "EMEA", "EMEA", "EMEA"],
        "subregion": ["NA", "NA", "LATAM", "WE", "WE", "NE"],
        "country": ["USA", "CAN", "BRA", "GBR", "DEU", "NOR"],
    })


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Temporary data directory pre-populated with standard sub-directories."""
    for sub in ("forecasts/retail", "metrics", "history/retail", "audit_log"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path
