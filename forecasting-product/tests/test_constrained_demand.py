"""Tests for ConstrainedDemandEstimator — capacity and business-rule constraints."""

from datetime import date, timedelta
from typing import List

import polars as pl
import pytest

from src.config.schema import ConstraintConfig
from src.forecasting.base import BaseForecaster
from src.forecasting.constrained import (
    ConstrainedDemandEstimator,
)
from src.forecasting.registry import registry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake base forecaster for deterministic testing
# ---------------------------------------------------------------------------


class _StubForecaster(BaseForecaster):
    """Returns pre-set forecast values for testing."""

    name = "stub"

    def __init__(self, forecast_values: pl.DataFrame):
        self._forecast = forecast_values

    def fit(self, df, target_col="quantity", time_col="week", id_col="series_id"):
        pass  # no-op

    def predict(self, horizon, id_col="series_id", time_col="week"):
        return self._forecast

    def predict_quantiles(self, horizon, quantiles, id_col="series_id", time_col="week"):
        df = self._forecast.clone()
        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            # Scale forecast by quantile for deterministic spread
            df = df.with_columns(
                (pl.col("forecast") * (0.5 + q)).alias(col)
            )
        return df.drop("forecast")


def _make_forecast_df(
    series: List[str],
    n_weeks: int = 4,
    base_value: float = 100.0,
    start_date: date = date(2024, 1, 1),
) -> pl.DataFrame:
    """Build a stub forecast DataFrame."""
    rows = []
    for sid in series:
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": start_date + timedelta(weeks=w),
                "forecast": base_value,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("week").cast(pl.Date),
        pl.col("forecast").cast(pl.Float64),
    )


# ===========================================================================
# TestElementWiseConstraints
# ===========================================================================


class TestElementWiseConstraints:
    """Test per-row clipping: non-negativity and capacity."""

    def test_non_negativity_clipping(self):
        """Negative forecasts are clipped to min_demand (default 0)."""
        fdf = _make_forecast_df(["a"], base_value=-50.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=True, min_demand=0.0)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=4)
        assert result["forecast"].min() >= 0.0

    def test_global_max_capacity(self):
        """Forecasts exceeding max_capacity are clipped down."""
        fdf = _make_forecast_df(["a", "b"], base_value=200.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=True, max_capacity=150.0)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=4)
        assert result["forecast"].max() <= 150.0

    def test_per_series_capacity_column(self):
        """Capacity extracted from data column applies per-series."""
        # Training data with capacity column
        train = pl.DataFrame({
            "series_id": ["a", "a", "b", "b"],
            "week": [date(2023, 1, 2)] * 4,
            "quantity": [10.0, 10.0, 10.0, 10.0],
            "max_cap": [80.0, 80.0, 120.0, 120.0],
        }).with_columns(
            pl.col("week").cast(pl.Date),
            pl.col("quantity").cast(pl.Float64),
        )

        fdf = _make_forecast_df(["a", "b"], base_value=100.0)
        stub = _StubForecaster(fdf)
        cfg = ConstraintConfig(enabled=True, capacity_column="max_cap")
        cde = ConstrainedDemandEstimator(stub, cfg)
        cde.validate_and_prepare(train, "quantity", "week", "series_id")
        cde.base.fit(pl.DataFrame())

        result = cde.predict(horizon=4)
        a_max = result.filter(pl.col("series_id") == "a")["forecast"].max()
        b_max = result.filter(pl.col("series_id") == "b")["forecast"].max()
        assert a_max <= 80.0
        assert b_max <= 100.0  # b's cap is 120, forecast is 100

    def test_disabled_passthrough(self):
        """When enabled=False, forecasts pass through unchanged."""
        fdf = _make_forecast_df(["a"], base_value=-50.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=False)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=4)
        # Negatives should remain
        assert result["forecast"].min() < 0.0


# ===========================================================================
# TestAggregateConstraints
# ===========================================================================


class TestAggregateConstraints:
    """Test per-period aggregate budget constraints."""

    def test_proportional_redistribution(self):
        """When sum exceeds budget, forecasts scale down proportionally."""
        # 4 series × 100 = 400 per period, budget = 300
        fdf = _make_forecast_df(["a", "b", "c", "d"], base_value=100.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(
                enabled=True,
                aggregate_max=300.0,
                proportional_redistribution=True,
            ),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        period_sum = result["forecast"].sum()
        assert abs(period_sum - 300.0) < 0.01

    def test_clip_largest_strategy(self):
        """Non-proportional redistribution also respects budget."""
        fdf = _make_forecast_df(["a", "b"], base_value=200.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(
                enabled=True,
                aggregate_max=300.0,
                proportional_redistribution=False,
            ),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        period_sum = result["forecast"].sum()
        assert abs(period_sum - 300.0) < 0.01

    def test_under_budget_unchanged(self):
        """No adjustment when total is within budget."""
        fdf = _make_forecast_df(["a", "b"], base_value=100.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, aggregate_max=500.0),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        # 200 < 500, so no change
        assert result["forecast"].sum() == 200.0

    def test_aggregate_plus_element_wise(self):
        """Aggregate and element-wise constraints work together."""
        fdf = _make_forecast_df(["a", "b"], base_value=200.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(
                enabled=True,
                max_capacity=180.0,
                aggregate_max=300.0,
            ),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        # Element-wise clips to 180 first (360 total), then aggregate scales to 300
        assert result["forecast"].max() <= 180.0
        assert result["forecast"].sum() <= 300.01


# ===========================================================================
# TestQuantileConstraints
# ===========================================================================


class TestQuantileConstraints:
    """Test quantile clipping and monotonicity."""

    def test_quantiles_clipped_to_bounds(self):
        """All quantile values respect [min_demand, max_capacity]."""
        fdf = _make_forecast_df(["a"], base_value=200.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, min_demand=0.0, max_capacity=250.0),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])

        for col in ["forecast_p10", "forecast_p50", "forecast_p90"]:
            assert result[col].min() >= 0.0
            assert result[col].max() <= 250.0

    def test_monotonicity_preserved(self):
        """After clipping, p10 ≤ p50 ≤ p90 holds for every row."""
        fdf = _make_forecast_df(["a"], base_value=200.0)
        stub = _StubForecaster(fdf)
        # Tight cap forces clipping that could break order
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, min_demand=0.0, max_capacity=250.0),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])

        for row in result.iter_rows(named=True):
            assert row["forecast_p10"] <= row["forecast_p50"]
            assert row["forecast_p50"] <= row["forecast_p90"]

    def test_quantile_disabled_passthrough(self):
        """When disabled, quantiles pass through unchanged."""
        fdf = _make_forecast_df(["a"], base_value=1000.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=False, max_capacity=100.0)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict_quantiles(horizon=4, quantiles=[0.1, 0.9])
        # Should exceed 100 since constraints are disabled
        assert result["forecast_p90"].max() > 100.0


# ===========================================================================
# TestRegistryIntegration
# ===========================================================================


class TestRegistryIntegration:
    """Verify the forecaster is registered."""

    def test_in_registry(self):
        assert "constrained_demand" in registry.available

    def test_get_from_registry(self):
        cls = registry.get("constrained_demand")
        assert cls is ConstrainedDemandEstimator


# ===========================================================================
# TestGetParams
# ===========================================================================


class TestGetParams:
    """Verify parameter reporting."""

    def test_params_include_base_and_constraints(self):
        fdf = _make_forecast_df(["a"])
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, max_capacity=500.0),
        )
        params = cde.get_params()
        assert params["base"] == "stub"
        assert params["max_capacity"] == 500.0


# ===========================================================================
# TestWithDifferentBases
# ===========================================================================


class TestWithDifferentBases:
    """Verify composition works with real forecasters."""

    def test_with_seasonal_naive(self):
        from src.forecasting.naive import SeasonalNaiveForecaster

        naive = SeasonalNaiveForecaster()
        cde = ConstrainedDemandEstimator(
            naive, ConstraintConfig(enabled=True, max_capacity=500.0)
        )
        # Just verify construction — no fit needed for this test
        assert cde.base.name == "naive_seasonal"
        assert repr(cde).startswith("ConstrainedDemandEstimator")

    def test_with_lgbm(self):
        from src.forecasting.ml import LGBMDirectForecaster

        lgbm = LGBMDirectForecaster()
        cde = ConstrainedDemandEstimator(
            lgbm, ConstraintConfig(enabled=True, min_demand=10.0)
        )
        assert cde.base.name == "lgbm_direct"
        params = cde.get_params()
        assert params["min_demand"] == 10.0
