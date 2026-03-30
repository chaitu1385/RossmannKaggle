"""
Tests for the MASE (Mean Absolute Scaled Error) metric.

MASE = mean(|actual - forecast|) / mean(|y_t - y_{t-m}|)

where the denominator is computed on in-sample (training) data.
"""

import math

import polars as pl
import pytest

from src.metrics.definitions import (
    CONTEXT_METRICS,
    _naive_seasonal_mae,
    compute_all_metrics,
    make_mase,
)

pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_insample(n: int = 104, base: float = 100.0, seasonal_amp: float = 20.0) -> pl.Series:
    """Create a seasonal in-sample series (2 years of weekly data)."""
    import math as _m
    values = [base + seasonal_amp * _m.sin(2 * _m.pi * i / 52) for i in range(n)]
    return pl.Series("quantity", values)


def _make_flat_insample(n: int = 104, value: float = 50.0) -> pl.Series:
    """Flat in-sample series — naive seasonal error = 0."""
    return pl.Series("quantity", [value] * n)


# ── Unit tests: _naive_seasonal_mae ──────────────────────────────────────────

class TestNaiveSeasonalMae:
    def test_flat_series_zero_mae(self):
        insample = _make_flat_insample(104, 50.0)
        assert _naive_seasonal_mae(insample, m=52) == 0.0

    def test_short_series_returns_inf(self):
        insample = pl.Series("quantity", list(range(20)))
        assert _naive_seasonal_mae(insample, m=52) == float("inf")

    def test_seasonal_series_positive_mae(self):
        insample = _make_insample(104, 100.0, 20.0)
        mae = _naive_seasonal_mae(insample, m=52)
        # For a perfectly periodic series with period 52, naive seasonal MAE ≈ 0
        # but with 104 points (exactly 2 periods), diffs are tiny (float precision)
        assert mae >= 0.0


# ── Unit tests: make_mase factory ────────────────────────────────────────────

class TestMaseFactory:
    def test_perfect_forecast_returns_zero(self):
        insample = _make_insample(104)
        actual = pl.Series("a", [10.0, 20.0, 30.0])
        forecast = pl.Series("f", [10.0, 20.0, 30.0])
        mase_fn = make_mase(insample)
        assert mase_fn(actual, forecast) == 0.0

    def test_naive_equivalent_returns_one(self):
        """When forecast errors equal the naive seasonal MAE, MASE = 1.0."""
        insample = _make_insample(104, base=100.0, seasonal_amp=0.0)
        # Add a known shift so naive seasonal MAE is non-zero
        vals = insample.to_list()
        for i in range(52, 104):
            vals[i] = vals[i] + 10.0  # shift second year up by 10
        insample = pl.Series("quantity", vals)
        denom = _naive_seasonal_mae(insample, 52)
        assert denom > 0

        # Construct actual/forecast with same absolute error as denom
        actual = pl.Series("a", [100.0])
        forecast = pl.Series("f", [100.0 + denom])
        mase_fn = make_mase(insample)
        result = mase_fn(actual, forecast)
        assert abs(result - 1.0) < 1e-9

    def test_worse_than_naive(self):
        """MASE > 1 when errors exceed naive seasonal."""
        insample = _make_insample(104, base=100.0, seasonal_amp=0.0)
        vals = insample.to_list()
        for i in range(52, 104):
            vals[i] = vals[i] + 5.0
        insample = pl.Series("quantity", vals)
        denom = _naive_seasonal_mae(insample, 52)

        actual = pl.Series("a", [100.0])
        forecast = pl.Series("f", [100.0 + denom * 3])  # 3x worse
        mase_fn = make_mase(insample)
        result = mase_fn(actual, forecast)
        assert result > 1.0
        assert abs(result - 3.0) < 1e-9

    def test_short_insample_returns_inf(self):
        """When insample is too short, MASE returns inf."""
        insample = pl.Series("quantity", [10.0] * 20)
        actual = pl.Series("a", [10.0, 20.0])
        forecast = pl.Series("f", [15.0, 25.0])
        mase_fn = make_mase(insample)
        assert mase_fn(actual, forecast) == float("inf")

    def test_flat_insample_returns_inf(self):
        """When insample is flat (denom=0), MASE returns inf."""
        insample = _make_flat_insample(104)
        actual = pl.Series("a", [10.0, 20.0])
        forecast = pl.Series("f", [15.0, 25.0])
        mase_fn = make_mase(insample)
        assert mase_fn(actual, forecast) == float("inf")


# ── Integration tests: compute_all_metrics ───────────────────────────────────

class TestComputeAllMetricsWithMase:
    def test_mase_with_context(self):
        insample = _make_insample(104, base=100.0, seasonal_amp=0.0)
        vals = insample.to_list()
        for i in range(52, 104):
            vals[i] += 10.0
        insample = pl.Series("quantity", vals)

        actual = pl.Series("a", [100.0, 200.0])
        forecast = pl.Series("f", [100.0, 200.0])  # perfect

        result = compute_all_metrics(
            actual, forecast,
            metric_names=["mase"],
            context={"insample": insample},
        )
        assert "mase" in result
        assert result["mase"] == 0.0

    def test_mase_without_context_returns_nan(self):
        actual = pl.Series("a", [100.0])
        forecast = pl.Series("f", [110.0])

        result = compute_all_metrics(
            actual, forecast,
            metric_names=["mase"],
        )
        assert "mase" in result
        assert math.isnan(result["mase"])

    def test_mase_alongside_standard_metrics(self):
        insample = _make_insample(104, base=100.0, seasonal_amp=0.0)
        vals = insample.to_list()
        for i in range(52, 104):
            vals[i] += 10.0
        insample = pl.Series("quantity", vals)

        actual = pl.Series("a", [100.0, 200.0])
        forecast = pl.Series("f", [110.0, 190.0])

        result = compute_all_metrics(
            actual, forecast,
            metric_names=["wmape", "mae", "mase"],
            context={"insample": insample},
        )
        assert "wmape" in result
        assert "mae" in result
        assert "mase" in result
        assert result["mae"] == 10.0

    def test_unknown_metric_still_raises(self):
        actual = pl.Series("a", [100.0])
        forecast = pl.Series("f", [110.0])
        with pytest.raises(KeyError, match="nonexistent"):
            compute_all_metrics(actual, forecast, metric_names=["nonexistent"])

    def test_context_metrics_registry(self):
        assert "mase" in CONTEXT_METRICS
        assert CONTEXT_METRICS["mase"] == "insample"
